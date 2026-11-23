//! Wire-contract tests for the observation envelope.
//!
//! These live in `tests/` rather than inline because they assert the serialized
//! form with `serde_json` and `proptest`, which the crate already declares as
//! dev-dependencies.

use parlevox_core::{
    ErrorClass, Observation, ObservationKind, ObservationSequencer, ALL_EVENT_KINDS,
    OBSERVATION_SCHEMA_VERSION,
};

/// Every error class, in declaration order.
const ERROR_CLASSES: [ErrorClass; 5] = [
    ErrorClass::Transient,
    ErrorClass::Fatal,
    ErrorClass::Protocol,
    ErrorClass::Cancelled,
    ErrorClass::Capacity,
];

/// Number of observation-kind variants produced by [`kind_for`].
const VARIANT_COUNT: usize = 11;

fn sample_observation(kind: ObservationKind) -> Observation {
    Observation {
        t_ms: 1_234,
        event_id: 7,
        turn: 3,
        generation: 2,
        kind,
    }
}

/// One representative observation of every [`ObservationKind`] variant,
/// including all turn event kinds wrapped in [`ObservationKind::Control`].
fn sample_kinds() -> Vec<ObservationKind> {
    let mut kinds = vec![
        ObservationKind::FinalTranscript {
            text: "hello".into(),
        },
        ObservationKind::PartialTranscript { text: "hel".into() },
        ObservationKind::VadStarted,
        ObservationKind::VadStopped,
        ObservationKind::ReplyDelta { text: "hi".into() },
        ObservationKind::AudioFrameQueued {
            sample_rate: 16_000,
            channels: 1,
            pcm_samples: 160,
        },
        ObservationKind::ReplyFinished,
        ObservationKind::BargeInReported,
        ObservationKind::CancellationObserved,
        ObservationKind::Failure {
            class: ErrorClass::Protocol,
            message: "bad frame".into(),
        },
    ];
    for kind in ALL_EVENT_KINDS {
        kinds.push(ObservationKind::Control { event: *kind });
    }
    kinds
}

#[test]
fn jsonl_round_trip_preserves_every_observation_kind() {
    for kind in sample_kinds() {
        let observation = sample_observation(kind);

        let line = serde_json::to_string(&observation).expect("observation serializes to JSON");
        assert!(
            !line.contains('\n'),
            "a trace line must not contain an embedded newline: {line}"
        );

        // Field order is declaration order and is part of the stable wire form.
        let positions: Vec<usize> = ["t_ms", "event_id", "turn", "generation", "kind"]
            .iter()
            .map(|field| {
                line.find(&format!("\"{field}\":"))
                    .unwrap_or_else(|| panic!("serialized line is missing field {field}: {line}"))
            })
            .collect();
        assert!(
            positions.windows(2).all(|pair| pair[0] < pair[1]),
            "fields are out of declaration order: {line}"
        );

        let parsed: Observation = serde_json::from_str(&line).expect("trace line parses back");
        println!("serialize -> parse -> equal: {line}");
        assert_eq!(parsed, observation);
    }
}

#[test]
fn unknown_field_is_rejected() {
    let line =
        r#"{"t_ms":1,"event_id":2,"turn":0,"generation":0,"kind":"vad_started","extra":true}"#;

    let error = serde_json::from_str::<Observation>(line)
        .expect_err("a field outside the envelope must be rejected");
    let message = error.to_string();
    println!("unknown-field rejection: {message}");

    assert!(
        message.contains("unknown field"),
        "rejection must name the problem: {message}"
    );
    assert!(
        message.contains("extra"),
        "rejection must name the unknown field: {message}"
    );
}

#[test]
fn observation_module_never_names_system_time() {
    const SOURCE: &str = include_str!("../src/observation.rs");
    assert!(
        !SOURCE.contains("SystemTime"),
        "observation.rs must not name wall-clock time"
    );
    assert!(
        !SOURCE.contains("std::time"),
        "observation.rs must not import the wall-clock module"
    );
}

#[test]
fn schema_version_is_the_documented_constant() {
    assert_eq!(OBSERVATION_SCHEMA_VERSION, 1);

    let header = serde_json::json!({ "observation_schema_version": OBSERVATION_SCHEMA_VERSION });
    assert_eq!(header.to_string(), r#"{"observation_schema_version":1}"#);
}

#[test]
fn sequencer_assigns_monotonic_ids_and_timestamps() {
    let mut sequencer = ObservationSequencer::default();
    let mut previous_t_ms = None;

    for expected_id in 0u64..8 {
        let observation = sequencer.stamp(2, 1, ObservationKind::VadStarted);

        assert_eq!(observation.event_id, expected_id);
        assert_eq!(observation.turn, 2);
        assert_eq!(observation.generation, 1);
        if let Some(previous_t_ms) = previous_t_ms {
            assert!(
                observation.t_ms >= previous_t_ms,
                "timestamps must never move backwards"
            );
        }
        previous_t_ms = Some(observation.t_ms);
    }
}

#[test]
fn every_turn_machine_event_kind_round_trips_as_a_control_observation() {
    assert_eq!(ALL_EVENT_KINDS.len(), 16);

    for kind in ALL_EVENT_KINDS {
        let observation = sample_observation(ObservationKind::Control { event: *kind });
        let line = serde_json::to_string(&observation).expect("control observation serializes");
        let parsed: Observation =
            serde_json::from_str(&line).expect("control observation parses back");
        assert_eq!(parsed, observation, "control round trip for {kind:?}");
    }
}

fn kind_for(
    variant: usize,
    text: String,
    sample_rate: u32,
    channels: u16,
    class_index: usize,
    event_kind_index: usize,
) -> ObservationKind {
    match variant {
        0 => ObservationKind::FinalTranscript { text },
        1 => ObservationKind::PartialTranscript { text },
        2 => ObservationKind::VadStarted,
        3 => ObservationKind::VadStopped,
        4 => ObservationKind::ReplyDelta { text },
        5 => ObservationKind::AudioFrameQueued {
            sample_rate,
            channels,
            pcm_samples: u64::from(sample_rate),
        },
        6 => ObservationKind::ReplyFinished,
        7 => ObservationKind::BargeInReported,
        8 => ObservationKind::CancellationObserved,
        9 => ObservationKind::Failure {
            class: ERROR_CLASSES[class_index],
            message: text,
        },
        10 => ObservationKind::Control {
            event: ALL_EVENT_KINDS[event_kind_index],
        },
        _ => unreachable!("variant index is bounded by the strategy"),
    }
}

use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn trace_lines_round_trip_for_arbitrary_observations(
        variant in 0usize..VARIANT_COUNT,
        t_ms in any::<u64>(),
        event_id in any::<u64>(),
        turn in any::<u64>(),
        generation in any::<u64>(),
        text in "[a-zA-Z0-9 ]{0,16}",
        sample_rate in 0u32..48_000,
        channels in 0u16..8,
        class_index in 0usize..ERROR_CLASSES.len(),
        event_kind_index in 0usize..ALL_EVENT_KINDS.len(),
    ) {
        let observation = Observation {
            t_ms,
            event_id,
            turn,
            generation,
            kind: kind_for(variant, text, sample_rate, channels, class_index, event_kind_index),
        };

        let line = serde_json::to_string(&observation).expect("observation serializes");
        prop_assert!(!line.contains('\n'), "a trace line must stay single-line: {}", line);
        prop_assert_eq!(line.lines().count(), 1);

        let parsed: Observation = serde_json::from_str(&line).expect("trace line parses back");
        prop_assert_eq!(parsed, observation);
    }
}
