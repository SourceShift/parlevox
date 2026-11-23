//! Contract tests for the pure turn arbiter.
//!
//! These tests link only `parlevox_core` and `proptest`: the point of the
//! arbiter is that it needs no provider, task, clock or runtime to decide what
//! to do, so its tests must not need them either.

use std::collections::BTreeSet;

use parlevox_core::{
    ArbiterModel, Command, ErrorClass, Observation, ObservationKind, RecordKind, TurnArbiter,
    TurnEventKind, TurnState, ALL_EVENT_KINDS, ALL_STATES, DUCK_GAIN,
};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Strategies for the purity property
// ---------------------------------------------------------------------------

fn state_strategy() -> impl Strategy<Value = TurnState> {
    prop::sample::select(ALL_STATES.to_vec())
}

fn event_kind_strategy() -> impl Strategy<Value = TurnEventKind> {
    prop::sample::select(ALL_EVENT_KINDS.to_vec())
}

fn error_class_strategy() -> impl Strategy<Value = ErrorClass> {
    prop::sample::select(vec![
        ErrorClass::Transient,
        ErrorClass::Fatal,
        ErrorClass::Protocol,
        ErrorClass::Cancelled,
        ErrorClass::Capacity,
    ])
}

fn observation_kind_strategy() -> impl Strategy<Value = ObservationKind> {
    prop_oneof![
        any::<String>().prop_map(|text| ObservationKind::FinalTranscript { text }),
        any::<String>().prop_map(|text| ObservationKind::PartialTranscript { text }),
        Just(ObservationKind::VadStarted),
        Just(ObservationKind::VadStopped),
        any::<String>().prop_map(|text| ObservationKind::ReplyDelta { text }),
        (any::<u32>(), any::<u16>(), any::<u64>()).prop_map(
            |(sample_rate, channels, pcm_samples)| ObservationKind::AudioFrameQueued {
                sample_rate,
                channels,
                pcm_samples,
            }
        ),
        Just(ObservationKind::ReplyFinished),
        Just(ObservationKind::BargeInReported),
        Just(ObservationKind::CancellationObserved),
        (error_class_strategy(), any::<String>())
            .prop_map(|(class, message)| ObservationKind::Failure { class, message }),
        event_kind_strategy().prop_map(|event| ObservationKind::Control { event }),
    ]
}

fn observation_strategy() -> impl Strategy<Value = Observation> {
    (
        any::<u64>(),
        any::<u64>(),
        any::<u64>(),
        any::<u64>(),
        observation_kind_strategy(),
    )
        .prop_map(|(t_ms, event_id, turn, generation, kind)| Observation {
            t_ms,
            event_id,
            turn,
            generation,
            kind,
        })
}

fn model_strategy() -> impl Strategy<Value = ArbiterModel> {
    (
        state_strategy(),
        any::<u64>(),
        any::<u64>(),
        any::<bool>(),
        any::<u64>(),
        any::<u64>(),
        any::<u64>(),
    )
        .prop_map(
            |(state, turn, generation, ducked, accepted, ignored, rejected)| ArbiterModel {
                state,
                turn,
                generation,
                ducked,
                accepted,
                ignored,
                rejected,
            },
        )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// `step` is a pure function: the borrowed model is untouched, and the same
    /// inputs always produce the same outputs.
    #[test]
    fn arbiter_step_is_pure_for_any_model_and_observation(
        model in model_strategy(),
        observation in observation_strategy(),
    ) {
        // Arrange
        let snapshot = model;

        // Act
        let first = TurnArbiter::step(&model, observation.clone());
        let second = TurnArbiter::step(&model, observation);

        // Assert
        prop_assert_eq!(model, snapshot, "step must not mutate its borrowed model");
        prop_assert_eq!(first, second, "step must be deterministic for equal inputs");
    }
}

// ---------------------------------------------------------------------------
// Deterministic 10,000-observation invariant drive
// ---------------------------------------------------------------------------

/// A tiny deterministic xorshift generator, so the drive needs no `rand`
/// dependency and prints the same summary on every run.
struct Xorshift64(u64);

impl Xorshift64 {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

const DRIVE_STEPS: u64 = 10_000;

/// One representative observation of every kind.
fn kind_pool() -> Vec<ObservationKind> {
    vec![
        ObservationKind::FinalTranscript {
            text: "hello".to_string(),
        },
        ObservationKind::PartialTranscript {
            text: "hel".to_string(),
        },
        ObservationKind::VadStarted,
        ObservationKind::VadStopped,
        ObservationKind::ReplyDelta {
            text: "wor".to_string(),
        },
        ObservationKind::AudioFrameQueued {
            sample_rate: 16_000,
            channels: 1,
            pcm_samples: 320,
        },
        ObservationKind::ReplyFinished,
        ObservationKind::BargeInReported,
        ObservationKind::CancellationObserved,
        ObservationKind::Failure {
            class: ErrorClass::Transient,
            message: "boom".to_string(),
        },
        ObservationKind::Control {
            event: TurnEventKind::Start,
        },
    ]
}

#[test]
fn ten_thousand_observations_keep_model_invariants() {
    // Arrange
    let pool = kind_pool();
    let pool_len = u64::try_from(pool.len()).expect("kind pool fits in u64");
    let mut rng = Xorshift64(0x2545_f491_4f6c_dd1d);
    let mut model = ArbiterModel::default();
    let mut saw_closed = false;

    // Act + Assert: invariants after every single step.
    for step_number in 0..DRIVE_STEPS {
        let index = usize::try_from(rng.next_u64() % pool_len).expect("index fits in usize");
        let observation = Observation {
            t_ms: step_number,
            event_id: step_number,
            turn: model.turn,
            generation: model.generation,
            kind: pool[index].clone(),
        };

        let (next, _commands) = TurnArbiter::step(&model, observation);

        assert!(
            next.turn >= model.turn,
            "turn decreased at step {step_number}: {} -> {}",
            model.turn,
            next.turn
        );
        assert!(
            next.generation >= model.generation,
            "generation decreased at step {step_number}: {} -> {}",
            model.generation,
            next.generation
        );
        if next.ducked {
            assert_eq!(
                next.state,
                TurnState::EvaluatingBargeIn,
                "a ducked model must be in the contested window at step {step_number}"
            );
        }
        if saw_closed {
            assert_eq!(
                next.state,
                TurnState::Closed,
                "a closed session must stay closed at step {step_number}"
            );
        }
        if next.state == TurnState::Closed {
            saw_closed = true;
        }

        model = next;
    }

    // Assert: every observation is accounted for exactly once.
    let total = model
        .accepted
        .saturating_add(model.ignored)
        .saturating_add(model.rejected);
    assert_eq!(
        total, DRIVE_STEPS,
        "accepted + ignored + rejected must equal the step count"
    );

    println!(
        "steps={DRIVE_STEPS} accepted={} ignored={} rejected={} state={:?} turn={} generation={}",
        model.accepted, model.ignored, model.rejected, model.state, model.turn, model.generation
    );
}

// ---------------------------------------------------------------------------
// Command-coverage table
// ---------------------------------------------------------------------------

fn model_in(state: TurnState, turn: u64, generation: u64, ducked: bool) -> ArbiterModel {
    ArbiterModel {
        state,
        turn,
        generation,
        ducked,
        accepted: 0,
        ignored: 0,
        rejected: 0,
    }
}

fn observed(kind: ObservationKind) -> Observation {
    Observation {
        t_ms: 0,
        event_id: 0,
        turn: 0,
        generation: 0,
        kind,
    }
}

fn command_tag(command: Command) -> u8 {
    match command {
        Command::StartGeneration { .. } => 0,
        Command::CancelGeneration { .. } => 1,
        Command::StartSynthesis { .. } => 2,
        Command::FlushAudioBuffer { .. } => 3,
        Command::DuckAudio { .. } => 4,
        Command::RestoreAudio => 5,
        Command::EmitTurnEvent { .. } => 6,
        Command::Record { .. } => 7,
    }
}

fn record_tag(kind: RecordKind) -> u8 {
    match kind {
        RecordKind::ObservationIgnored => 0,
        RecordKind::IllegalTransition => 1,
        RecordKind::ObservationAfterClose => 2,
        RecordKind::TurnOpened => 3,
        RecordKind::GenerationSuperseded => 4,
    }
}

/// Case name, starting model, observation, and the exact ordered commands.
fn coverage_table() -> Vec<(&'static str, ArbiterModel, Observation, Vec<Command>)> {
    vec![
        (
            "final transcript from listening opens a turn",
            model_in(TurnState::Listening, 3, 4, false),
            observed(ObservationKind::FinalTranscript {
                text: "hi".to_string(),
            }),
            vec![
                Command::StartGeneration {
                    turn: 4,
                    generation: 5,
                },
                Command::EmitTurnEvent {
                    kind: TurnEventKind::FinalTranscript,
                },
                Command::Record {
                    kind: RecordKind::TurnOpened,
                },
            ],
        ),
        (
            "first audio from thinking starts synthesis",
            model_in(TurnState::Thinking, 1, 1, false),
            observed(ObservationKind::Control {
                event: TurnEventKind::ReplyStarted,
            }),
            vec![
                Command::StartSynthesis {
                    turn: 1,
                    generation: 1,
                },
                Command::EmitTurnEvent {
                    kind: TurnEventKind::ReplyStarted,
                },
            ],
        ),
        (
            "barge-in from speaking cancels and flushes",
            model_in(TurnState::Speaking, 2, 7, false),
            observed(ObservationKind::BargeInReported),
            vec![
                Command::CancelGeneration { generation: 7 },
                Command::FlushAudioBuffer { turn: 2 },
                Command::EmitTurnEvent {
                    kind: TurnEventKind::BargeIn,
                },
                Command::Record {
                    kind: RecordKind::GenerationSuperseded,
                },
            ],
        ),
        (
            "speech detection ducks the audio",
            model_in(TurnState::Speaking, 2, 7, false),
            observed(ObservationKind::Control {
                event: TurnEventKind::SpeechDetected,
            }),
            vec![
                Command::DuckAudio { gain: DUCK_GAIN },
                Command::EmitTurnEvent {
                    kind: TurnEventKind::SpeechDetected,
                },
            ],
        ),
        (
            "rejected barge-in restores the audio",
            model_in(TurnState::EvaluatingBargeIn, 2, 7, true),
            observed(ObservationKind::Control {
                event: TurnEventKind::BargeInRejected,
            }),
            vec![
                Command::RestoreAudio,
                Command::EmitTurnEvent {
                    kind: TurnEventKind::BargeInRejected,
                },
            ],
        ),
        (
            "partial transcript is ignored",
            model_in(TurnState::Listening, 2, 7, false),
            observed(ObservationKind::PartialTranscript {
                text: "h".to_string(),
            }),
            vec![Command::Record {
                kind: RecordKind::ObservationIgnored,
            }],
        ),
        (
            "reply started from idle is illegal",
            model_in(TurnState::Idle, 0, 0, false),
            observed(ObservationKind::Control {
                event: TurnEventKind::ReplyStarted,
            }),
            vec![Command::Record {
                kind: RecordKind::IllegalTransition,
            }],
        ),
        (
            "start after close is rejected",
            model_in(TurnState::Closed, 1, 1, false),
            observed(ObservationKind::Control {
                event: TurnEventKind::Start,
            }),
            vec![Command::Record {
                kind: RecordKind::ObservationAfterClose,
            }],
        ),
    ]
}

#[test]
fn every_command_variant_is_producible_from_a_named_observation() {
    let table = coverage_table();
    let mut covered_commands = BTreeSet::new();
    let mut covered_records = BTreeSet::new();

    for (name, model, observation, expected) in table {
        let (next, commands) = TurnArbiter::step(&model, observation);

        assert_eq!(commands, expected, "command sequence for case: {name}");
        assert_eq!(
            next.accepted
                .saturating_add(next.ignored)
                .saturating_add(next.rejected),
            1,
            "every step must account for exactly one observation in case: {name}"
        );

        for command in &commands {
            covered_commands.insert(command_tag(*command));
            if let Command::Record { kind } = command {
                covered_records.insert(record_tag(*kind));
            }
        }
    }

    let expected_commands: BTreeSet<u8> = (0u8..=7).collect();
    assert_eq!(
        covered_commands, expected_commands,
        "every Command variant must be produced by a named observation"
    );

    let expected_records: BTreeSet<u8> = (0u8..=4).collect();
    assert_eq!(
        covered_records, expected_records,
        "every RecordKind variant must be produced by a named observation"
    );
}
