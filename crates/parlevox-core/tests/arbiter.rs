//! Contract tests for the pure turn arbiter.
//!
//! These tests link only `parlevox_core` and `proptest`: the point of the
//! arbiter is that it needs no provider, task, clock or runtime to decide what
//! to do, so its tests must not need them either.

use std::collections::BTreeSet;

use parlevox_core::{
    ArbiterModel, Command, ErrorClass, FencePosition, Observation, ObservationKind, RecordKind,
    TurnArbiter, TurnEventKind, TurnState, ALL_EVENT_KINDS, ALL_STATES, DUCK_GAIN,
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
        any::<u64>(),
    )
        .prop_map(
            |(state, turn, generation, ducked, accepted, ignored, rejected, dropped)| {
                ArbiterModel {
                    state,
                    turn,
                    generation,
                    ducked,
                    accepted,
                    ignored,
                    rejected,
                    dropped,
                }
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
        .saturating_add(model.rejected)
        .saturating_add(model.dropped);
    assert_eq!(
        total, DRIVE_STEPS,
        "accepted + ignored + rejected + dropped must equal the step count"
    );

    println!(
        "steps={DRIVE_STEPS} accepted={} ignored={} rejected={} dropped={} state={:?} turn={} generation={}",
        model.accepted, model.ignored, model.rejected, model.dropped, model.state, model.turn,
        model.generation
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
        dropped: 0,
    }
}

fn observation_at(turn: u64, generation: u64, kind: ObservationKind) -> Observation {
    Observation {
        t_ms: 0,
        event_id: 0,
        turn,
        generation,
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
        RecordKind::ObservationDropped => 5,
        RecordKind::ObservationAhead => 6,
    }
}

/// Case name, starting model, observation, and the exact ordered commands.
///
/// Every observation is stamped with the model's current `(turn, generation)`,
/// so each case exercises its named branch rather than being dropped by the
/// fence. The superseded and ahead rows pin an explicit pair on purpose.
fn coverage_table() -> Vec<(&'static str, ArbiterModel, Observation, Vec<Command>)> {
    let opens = model_in(TurnState::Listening, 3, 4, false);
    let synthesis = model_in(TurnState::Thinking, 1, 1, false);
    let barge_in = model_in(TurnState::Speaking, 2, 7, false);
    let ducks = model_in(TurnState::Speaking, 2, 7, false);
    let restores = model_in(TurnState::EvaluatingBargeIn, 2, 7, true);
    let ignored = model_in(TurnState::Listening, 2, 7, false);
    let illegal = model_in(TurnState::Idle, 0, 0, false);
    let closed = model_in(TurnState::Closed, 1, 1, false);
    let superseded = model_in(TurnState::Listening, 3, 4, false);
    let ahead = model_in(TurnState::Listening, 2, 7, false);

    vec![
        (
            "final transcript from listening opens a turn",
            opens,
            observation_at(
                opens.turn,
                opens.generation,
                ObservationKind::FinalTranscript {
                    text: "hi".to_string(),
                },
            ),
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
            synthesis,
            observation_at(
                synthesis.turn,
                synthesis.generation,
                ObservationKind::Control {
                    event: TurnEventKind::ReplyStarted,
                },
            ),
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
            barge_in,
            observation_at(
                barge_in.turn,
                barge_in.generation,
                ObservationKind::BargeInReported,
            ),
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
            ducks,
            observation_at(
                ducks.turn,
                ducks.generation,
                ObservationKind::Control {
                    event: TurnEventKind::SpeechDetected,
                },
            ),
            vec![
                Command::DuckAudio { gain: DUCK_GAIN },
                Command::EmitTurnEvent {
                    kind: TurnEventKind::SpeechDetected,
                },
            ],
        ),
        (
            "rejected barge-in restores the audio",
            restores,
            observation_at(
                restores.turn,
                restores.generation,
                ObservationKind::Control {
                    event: TurnEventKind::BargeInRejected,
                },
            ),
            vec![
                Command::RestoreAudio,
                Command::EmitTurnEvent {
                    kind: TurnEventKind::BargeInRejected,
                },
            ],
        ),
        (
            "partial transcript is ignored",
            ignored,
            observation_at(
                ignored.turn,
                ignored.generation,
                ObservationKind::PartialTranscript {
                    text: "h".to_string(),
                },
            ),
            vec![Command::Record {
                kind: RecordKind::ObservationIgnored,
            }],
        ),
        (
            "reply started from idle is illegal",
            illegal,
            observation_at(
                illegal.turn,
                illegal.generation,
                ObservationKind::Control {
                    event: TurnEventKind::ReplyStarted,
                },
            ),
            vec![Command::Record {
                kind: RecordKind::IllegalTransition,
            }],
        ),
        (
            "start after close is rejected",
            closed,
            observation_at(
                closed.turn,
                closed.generation,
                ObservationKind::Control {
                    event: TurnEventKind::Start,
                },
            ),
            vec![Command::Record {
                kind: RecordKind::ObservationAfterClose,
            }],
        ),
        (
            "superseded observation is dropped",
            superseded,
            observation_at(
                superseded.turn.saturating_sub(1),
                superseded.generation,
                ObservationKind::FinalTranscript {
                    text: "late".to_string(),
                },
            ),
            vec![Command::Record {
                kind: RecordKind::ObservationDropped,
            }],
        ),
        (
            "newer observation is recorded as rejected",
            ahead,
            observation_at(
                ahead.turn.saturating_add(1),
                ahead.generation.saturating_add(2),
                ObservationKind::FinalTranscript {
                    text: "future".to_string(),
                },
            ),
            vec![Command::Record {
                kind: RecordKind::ObservationAhead,
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
                .saturating_add(next.rejected)
                .saturating_add(next.dropped),
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

    let expected_records: BTreeSet<u8> = (0u8..=6).collect();
    assert_eq!(
        covered_records, expected_records,
        "every RecordKind variant must be produced by a named observation"
    );
}

// ---------------------------------------------------------------------------
// Fence law: late finals for the previous turn
// ---------------------------------------------------------------------------

/// A scripted observation: either stamped with whatever `(turn, generation)`
/// the model holds when it is stepped, or pinned to an explicit pair.
#[derive(Debug, Clone)]
struct Scripted {
    kind: ObservationKind,
    stamp: Stamp,
}

/// How a [`Scripted`] observation is stamped.
#[derive(Debug, Clone, Copy)]
enum Stamp {
    /// Use the model's current `(turn, generation)`, so the observation is
    /// classified as [`FencePosition::Current`].
    Current,
    /// Use this explicit pair, which the fence classifies.
    Pinned { turn: u64, generation: u64 },
}

/// One stepped observation, the model it was applied to, and its commands.
struct StepRecord {
    observation: Observation,
    before: ArbiterModel,
    commands: Vec<Command>,
}

/// The result of replaying a script.
struct Replay {
    model: ArbiterModel,
    steps: Vec<StepRecord>,
}

fn current(kind: ObservationKind) -> Scripted {
    Scripted {
        kind,
        stamp: Stamp::Current,
    }
}

/// An eleven-observation two-turn conversation: start, a voice turn that is
/// interrupted mid-reply, then a text turn that finishes.
///
/// 1. `Start`
/// 2. `VadStarted`
/// 3. `VadStopped`
/// 4. `FinalTranscript` opening turn 1
/// 5. `ReplyStarted`
/// 6. `SpeechDetected`
/// 7. `BargeInConfirmed`
/// 8. `ResumeListening`
/// 9. `FinalTranscript` opening turn 2
/// 10. `ReplyStarted`
/// 11. `ReplyFinished`
fn scripted_trace() -> Vec<Scripted> {
    vec![
        current(ObservationKind::Control {
            event: TurnEventKind::Start,
        }),
        current(ObservationKind::VadStarted),
        current(ObservationKind::VadStopped),
        current(ObservationKind::FinalTranscript {
            text: "first".to_string(),
        }),
        current(ObservationKind::Control {
            event: TurnEventKind::ReplyStarted,
        }),
        current(ObservationKind::Control {
            event: TurnEventKind::SpeechDetected,
        }),
        current(ObservationKind::Control {
            event: TurnEventKind::BargeInConfirmed,
        }),
        current(ObservationKind::Control {
            event: TurnEventKind::ResumeListening,
        }),
        current(ObservationKind::FinalTranscript {
            text: "second".to_string(),
        }),
        current(ObservationKind::Control {
            event: TurnEventKind::ReplyStarted,
        }),
        current(ObservationKind::Control {
            event: TurnEventKind::ReplyFinished,
        }),
    ]
}

/// Step every scripted observation, stamping current ones from the live model.
fn replay(script: &[Scripted]) -> Replay {
    let mut model = ArbiterModel::default();
    let mut steps = Vec::with_capacity(script.len());

    for (index, item) in script.iter().enumerate() {
        let event_id = u64::try_from(index).expect("a script index fits in u64");
        let (turn, generation) = match item.stamp {
            Stamp::Current => (model.turn, model.generation),
            Stamp::Pinned { turn, generation } => (turn, generation),
        };
        let observation = Observation {
            t_ms: event_id,
            event_id,
            turn,
            generation,
            kind: item.kind.clone(),
        };

        let before = model;
        let (next, commands) = TurnArbiter::step(&model, observation.clone());
        steps.push(StepRecord {
            observation,
            before,
            commands,
        });
        model = next;
    }

    Replay { model, steps }
}

/// Insert one pinned `FinalTranscript` before each `(position, turn,
/// generation)` entry. Equal positions keep the order they were given in.
fn inject_late_finals(trace: &[Scripted], injections: &[(usize, u64, u64)]) -> Vec<Scripted> {
    let mut ordered = injections.to_vec();
    ordered.sort_by_key(|(position, _, _)| *position);

    let mut script = Vec::with_capacity(trace.len() + ordered.len());
    let mut pending = ordered.iter().peekable();

    for (index, item) in trace.iter().enumerate() {
        while let Some((position, turn, generation)) = pending.peek().copied().copied() {
            if position != index {
                break;
            }
            script.push(Scripted {
                kind: ObservationKind::FinalTranscript {
                    text: "late".to_string(),
                },
                stamp: Stamp::Pinned { turn, generation },
            });
            pending.next();
        }
        script.push(item.clone());
    }

    script
}

/// Base-step indices whose model has already opened a turn, i.e. the points
/// where an older `(turn, generation)` pair actually exists to inject.
fn eligible_injection_positions(run: &Replay) -> Vec<usize> {
    run.steps
        .iter()
        .enumerate()
        .filter(|(_, step)| step.before.turn >= 1)
        .map(|(index, _)| index)
        .collect()
}

fn flattened_commands(steps: &[StepRecord]) -> Vec<Command> {
    steps
        .iter()
        .flat_map(|step| step.commands.iter().copied())
        .collect()
}

fn without_drop_records(commands: &[Command]) -> Vec<Command> {
    commands
        .iter()
        .copied()
        .filter(|command| {
            !matches!(
                command,
                Command::Record {
                    kind: RecordKind::ObservationDropped
                }
            )
        })
        .collect()
}

fn drop_record_count(steps: &[StepRecord]) -> usize {
    flattened_commands(steps)
        .iter()
        .filter(|command| {
            matches!(
                command,
                Command::Record {
                    kind: RecordKind::ObservationDropped
                }
            )
        })
        .count()
}

/// Whether this step was one of the injected stale observations: the model had
/// opened a turn and the observation was stamped older.
fn is_injected_stale(step: &StepRecord) -> bool {
    step.before.turn >= 1 && step.observation.turn < step.before.turn
}

#[test]
fn late_finals_replay_to_the_same_commands_and_count_as_dropped() {
    // Arrange: the base run, then the same run with three late finals for the
    // previous turn inserted.
    let trace = scripted_trace();
    let base = replay(&trace);
    let injections = [
        (4, 0, 0), // before ReplyStarted on turn 1: turn 0 is now superseded
        (8, 0, 0), // before the second FinalTranscript: turn 0 still superseded
        (9, 1, 3), // before ReplyStarted on turn 2: turn 1 is now superseded
    ];
    let injected = replay(&inject_late_finals(&trace, &injections));

    // Act
    let base_commands = without_drop_records(&flattened_commands(&base.steps));
    let injected_commands = without_drop_records(&flattened_commands(&injected.steps));
    let drops = drop_record_count(&injected.steps);

    // Assert
    assert_eq!(
        injected_commands, base_commands,
        "filtering the drop records must leave the base command sequence intact"
    );
    assert_eq!(
        drops,
        injections.len(),
        "each injected late final must contribute exactly one ObservationDropped"
    );
    assert_eq!(
        injected.model.dropped,
        u64::try_from(injections.len()).expect("the injection count fits in u64"),
        "the final dropped counter must equal the number of late finals"
    );
    assert_eq!(
        injected.model.state, base.model.state,
        "state must be unchanged"
    );
    assert_eq!(
        injected.model.turn, base.model.turn,
        "turn must be unchanged"
    );
    assert_eq!(
        injected.model.generation, base.model.generation,
        "generation must be unchanged"
    );
    assert_eq!(
        injected.model.accepted, base.model.accepted,
        "accepted must be unchanged"
    );
    assert_eq!(
        injected.model.ignored, base.model.ignored,
        "ignored must be unchanged"
    );
    assert_eq!(
        injected.model.rejected, base.model.rejected,
        "rejected must be unchanged"
    );
    assert_eq!(
        injected.model.ducked, base.model.ducked,
        "duck flag must be unchanged"
    );

    // Every injected step is drop-only: no state-transition command escapes it.
    for step in &injected.steps {
        if is_injected_stale(step) {
            assert_eq!(
                step.commands,
                vec![Command::Record {
                    kind: RecordKind::ObservationDropped
                }],
                "a late final must yield only its accounting record"
            );
        }
    }

    println!(
        "replay: 11 base observations, 3 late finals injected, \
         filtered command sequences identical ({} commands), dropped == {}",
        base_commands.len(),
        injected.model.dropped
    );
    for (index, step) in base.steps.iter().enumerate() {
        println!(
            "base[{index}] {:?} at ({}, {}) -> {:?}",
            step.observation.kind,
            step.observation.turn,
            step.observation.generation,
            step.commands
        );
    }
}

proptest! {
    /// K late finals injected at any point where a turn is already open remove
    /// exactly the K drop records from the command stream and leave every
    /// other command and every other model field untouched.
    #[test]
    fn any_number_of_late_finals_replays_equivalently(
        raw_positions in prop::collection::vec(any::<u64>(), 0..=8),
    ) {
        // Arrange
        let trace = scripted_trace();
        let base = replay(&trace);
        let eligible = eligible_injection_positions(&base);
        let eligible_len = u64::try_from(eligible.len()).expect("eligible count fits in u64");

        let injections: Vec<(usize, u64, u64)> = raw_positions
            .iter()
            .map(|raw| {
                let offset = usize::try_from(raw % eligible_len)
                    .expect("an offset below the eligible count fits in usize");
                let position = eligible[offset];
                let before = base.steps[position].before;
                (position, before.turn.saturating_sub(1), 0)
            })
            .collect();
        let k = injections.len();

        // Act
        let injected = replay(&inject_late_finals(&trace, &injections));

        // Assert
        prop_assert_eq!(
            without_drop_records(&flattened_commands(&injected.steps)),
            without_drop_records(&flattened_commands(&base.steps)),
            "the filtered command sequence must not depend on how many late finals arrive"
        );
        prop_assert_eq!(drop_record_count(&injected.steps), k);
        prop_assert_eq!(
            injected.model.dropped,
            u64::try_from(k).expect("the injection count fits in u64")
        );
        prop_assert_eq!(injected.model.state, base.model.state);
        prop_assert_eq!(injected.model.turn, base.model.turn);
        prop_assert_eq!(injected.model.generation, base.model.generation);
        prop_assert_eq!(injected.model.accepted, base.model.accepted);
        prop_assert_eq!(injected.model.ignored, base.model.ignored);
        prop_assert_eq!(injected.model.rejected, base.model.rejected);
    }
}

#[test]
fn a_late_final_for_the_previous_turn_is_dropped_without_a_transition() {
    // Arrange: the model is mid-reply on turn 5, generation 8, when the final
    // transcript for turn 4 generation 6 arrives.
    let model = model_in(TurnState::Speaking, 5, 8, false);
    let observation = observation_at(
        4,
        6,
        ObservationKind::FinalTranscript {
            text: "late".to_string(),
        },
    );

    // Act
    let (next, commands) = TurnArbiter::step(&model, observation.clone());

    // Assert: the fence classifies it as superseded and only the drop counter
    // moves; the machine does not leave `Speaking`.
    assert_eq!(
        model.fence_position(&observation),
        FencePosition::Superseded,
        "an older (turn, generation) pair must be superseded"
    );
    assert_eq!(
        commands,
        vec![Command::Record {
            kind: RecordKind::ObservationDropped
        }],
        "a superseded observation emits only its accounting record"
    );
    assert_eq!(next.state, model.state, "state must not change");
    assert_eq!(next.turn, model.turn, "turn must not change");
    assert_eq!(
        next.generation, model.generation,
        "generation must not change"
    );
    assert_eq!(next.ducked, model.ducked, "duck flag must not change");
    assert_eq!(next.accepted, model.accepted, "accepted must not change");
    assert_eq!(next.ignored, model.ignored, "ignored must not change");
    assert_eq!(next.rejected, model.rejected, "rejected must not change");
    assert_eq!(
        next.dropped,
        model.dropped.saturating_add(1),
        "only the drop counter moves"
    );
}

#[test]
fn the_fence_not_the_observation_kind_decides_a_drop() {
    // Arrange: same model, same observation kind twice — once current, once
    // stale. `FinalTranscript` has no legal row from `Speaking`, so the current
    // observation reaches the legality branch and the stale one does not.
    let model = model_in(TurnState::Speaking, 5, 8, false);
    let kind = ObservationKind::FinalTranscript {
        text: "transcript".to_string(),
    };
    let current = observation_at(model.turn, model.generation, kind.clone());
    let stale = observation_at(4, 6, kind);

    // Act
    let (_, current_commands) = TurnArbiter::step(&model, current);
    let (_, stale_commands) = TurnArbiter::step(&model, stale);

    // Assert
    assert_eq!(
        current_commands,
        vec![Command::Record {
            kind: RecordKind::IllegalTransition
        }],
        "a current transcript in `Speaking` is rejected by the legality branch"
    );
    assert_eq!(
        stale_commands,
        vec![Command::Record {
            kind: RecordKind::ObservationDropped
        }],
        "the same kind is dropped when the fence says it is stale"
    );
}

#[test]
fn a_current_final_transcript_still_opens_a_turn() {
    // Arrange: the same `(turn, generation)` fence as the late-final test, but
    // in a state where `FinalTranscript` is legal, so the observation kind is
    // what produces `StartGeneration`.
    let model = model_in(TurnState::Listening, 5, 8, false);
    let observation = observation_at(
        5,
        8,
        ObservationKind::FinalTranscript {
            text: "fresh".to_string(),
        },
    );

    // Act
    let (next, commands) = TurnArbiter::step(&model, observation.clone());

    // Assert
    assert_eq!(
        model.fence_position(&observation),
        FencePosition::Current,
        "an equal (turn, generation) pair is current"
    );
    assert_eq!(
        commands,
        vec![
            Command::StartGeneration {
                turn: 6,
                generation: 9,
            },
            Command::EmitTurnEvent {
                kind: TurnEventKind::FinalTranscript,
            },
            Command::Record {
                kind: RecordKind::TurnOpened,
            },
        ],
        "a current final transcript opens the next turn"
    );
    assert_eq!(next.turn, 6, "the turn must advance");
    assert_eq!(next.generation, 9, "the generation must advance");
    assert_eq!(next.dropped, 0, "nothing was dropped");
}

#[test]
fn a_newer_observation_is_recorded_and_rejected_without_a_transition() {
    // Arrange: the model is on turn 2 generation 7; the observation belongs to
    // turn 3 generation 9.
    let model = model_in(TurnState::Listening, 2, 7, false);
    let observation = observation_at(
        3,
        9,
        ObservationKind::FinalTranscript {
            text: "future".to_string(),
        },
    );

    // Act
    let (next, commands) = TurnArbiter::step(&model, observation.clone());

    // Assert: a newer observation is recorded, never silently dropped, and the
    // machine stays put.
    assert_eq!(
        model.fence_position(&observation),
        FencePosition::Ahead,
        "a larger (turn, generation) pair is ahead"
    );
    assert_eq!(
        commands,
        vec![Command::Record {
            kind: RecordKind::ObservationAhead
        }],
        "a newer observation is recorded as ahead"
    );
    assert_eq!(
        next.rejected,
        model.rejected.saturating_add(1),
        "a newer observation counts as rejected"
    );
    assert_eq!(
        next.dropped, model.dropped,
        "a newer observation is not dropped"
    );
    assert_eq!(next.state, model.state, "state must not change");
    assert_eq!(next.turn, model.turn, "turn must not change");
    assert_eq!(
        next.generation, model.generation,
        "generation must not change"
    );
}

#[test]
fn mixed_current_and_stale_observations_keep_the_counts() {
    // Arrange: the same deterministic drive, but every observation that can be
    // made stale is stamped one turn behind.
    let pool = kind_pool();
    let pool_len = u64::try_from(pool.len()).expect("kind pool fits in u64");
    let mut rng = Xorshift64(0x9e37_79b9_7f4a_7c15);
    let mut model = ArbiterModel::default();

    // Act + Assert
    for step_number in 0..DRIVE_STEPS {
        let index = usize::try_from(rng.next_u64() % pool_len).expect("index fits in usize");
        let stale = rng.next_u64() % 2 == 1 && model.turn >= 1;
        let (turn, generation) = if stale {
            (model.turn.saturating_sub(1), 0)
        } else {
            (model.turn, model.generation)
        };
        let observation = Observation {
            t_ms: step_number,
            event_id: step_number,
            turn,
            generation,
            kind: pool[index].clone(),
        };

        let (next, _commands) = TurnArbiter::step(&model, observation);

        assert!(
            next.dropped >= model.dropped,
            "the drop counter must never decrease at step {step_number}"
        );
        assert_eq!(
            next.accepted
                .saturating_add(next.ignored)
                .saturating_add(next.rejected)
                .saturating_add(next.dropped),
            step_number.saturating_add(1),
            "every observation must be counted exactly once at step {step_number}"
        );

        model = next;
    }

    // Assert
    assert!(
        model.dropped > 0,
        "the mixed drive must exercise the drop path at least once"
    );
    println!(
        "mixed steps={DRIVE_STEPS} accepted={} ignored={} rejected={} dropped={}",
        model.accepted, model.ignored, model.rejected, model.dropped
    );
}
