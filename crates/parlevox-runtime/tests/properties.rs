//! Property checks for the bounded session event queue.
//!
//! These mirror `crates/parlevox-core/tests/properties.rs` for the audio ring:
//! the queue is always bounded, every push is accounted for exactly once, the
//! control lane is never silently lost, the terminal slot is always available,
//! and draining preserves survivor order.
//!
//! Keep the case count at 128 so this suite is cheap enough to run next to the
//! rest of the workspace.

use parlevox_core::{OverflowPolicy, ParlevoxError};
use parlevox_runtime::{EventPushOutcome, EventQueue, SentenceSplitter, SessionEvent};
use proptest::prelude::*;

#[derive(Debug, Clone, Copy)]
enum Kind {
    Control,
    Droppable,
    Terminal,
}

fn arb_kind() -> impl Strategy<Value = Kind> {
    prop_oneof![
        Just(Kind::Control),
        Just(Kind::Droppable),
        Just(Kind::Terminal),
    ]
}

fn non_terminal_kind() -> impl Strategy<Value = Kind> {
    prop_oneof![Just(Kind::Control), Just(Kind::Droppable)]
}

fn kinds() -> impl Strategy<Value = Vec<Kind>> {
    prop::collection::vec(arb_kind(), 0..64)
}

/// Build an event whose identity encodes `idx`, so a survivor-order check can
/// tell duplicate kinds apart without relying on hashing or timing.
fn event_for(kind: Kind, idx: usize) -> SessionEvent {
    let id = idx as u64;
    match kind {
        Kind::Control => {
            if idx % 2 == 0 {
                SessionEvent::TurnStarted { turn: id }
            } else {
                SessionEvent::TranscriptFinal {
                    turn: id,
                    text: format!("c{idx}"),
                }
            }
        }
        Kind::Droppable => SessionEvent::ReplyDelta {
            turn: id,
            text: format!("d{idx}"),
        },
        Kind::Terminal => SessionEvent::TurnDone {
            turn: id,
            audio_frames: 0,
            dropped_frames: 0,
            dropped_ms: 0,
            dropped_events: 0,
            sidecar_dropped_frames: 0,
            interrupted: false,
        },
    }
}

fn is_control(event: &SessionEvent) -> bool {
    matches!(
        event,
        SessionEvent::TurnStarted { .. }
            | SessionEvent::TranscriptFinal { .. }
            | SessionEvent::TurnDone { .. }
            | SessionEvent::Failed { .. }
    )
}

/// Text with no terminal punctuation, so the splitter can only emit through
/// its bounded force-split path.
fn punctuation_free() -> impl Strategy<Value = String> {
    prop::collection::vec(
        prop_oneof![
            prop::char::range('a', 'z'),
            prop::char::range('0', '9'),
            Just(' '),
        ],
        0..200,
    )
    .prop_map(|chars| chars.into_iter().collect())
}

fn split_all(max_chars: usize, deltas: &[String]) -> Vec<String> {
    let mut splitter = SentenceSplitter::new(max_chars).expect("max_chars is at least 1");
    let mut emitted = Vec::new();
    for delta in deltas {
        emitted.extend(splitter.push_delta(delta));
    }
    if let Some(rest) = splitter.flush() {
        emitted.push(rest);
    }
    emitted
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn every_push_is_bounded_by_capacity(
        capacity in 0usize..17,
        kinds in kinds(),
    ) {
        let mut queue = EventQueue::new(capacity, OverflowPolicy::DropOldest);
        for (idx, kind) in kinds.iter().enumerate() {
            let _ = queue.push(event_for(*kind, idx));
            prop_assert!(queue.len() <= capacity);
            let stats = queue.stats();
            prop_assert_eq!(stats.capacity, capacity);
            prop_assert!(stats.depth <= stats.capacity);
        }
    }

    #[test]
    fn outcomes_account_for_every_push_exactly_once(
        capacity in 0usize..17,
        policy in prop_oneof![
            Just(OverflowPolicy::DropOldest),
            Just(OverflowPolicy::DropNewest),
        ],
        kinds in kinds(),
    ) {
        let mut queue = EventQueue::new(capacity, policy);
        let mut pushes = 0u64;
        let mut accepted = 0u64;
        let mut replaced = 0u64;
        let mut rejected = 0u64;
        let mut refused = 0u64;
        let mut previous_dropped = 0u64;

        for (idx, kind) in kinds.iter().enumerate() {
            pushes += 1;
            match queue.push(event_for(*kind, idx)) {
                Ok(EventPushOutcome::Accepted) => accepted += 1,
                Ok(EventPushOutcome::Replaced) => {
                    accepted += 1;
                    replaced += 1;
                }
                Ok(EventPushOutcome::Dropped) => rejected += 1,
                Err(err) => {
                    prop_assert!(matches!(err, ParlevoxError::Protocol(_)));
                    refused += 1;
                }
            }

            let stats = queue.stats();
            prop_assert_eq!(stats.accepted, accepted);
            prop_assert_eq!(stats.control_overflow, refused);
            prop_assert_eq!(stats.dropped, replaced + rejected + refused);
            // Replaced events are counted as both an accept and a drop.
            prop_assert_eq!(stats.accepted + stats.dropped, pushes + replaced);
            prop_assert!(stats.dropped >= previous_dropped);
            previous_dropped = stats.dropped;
        }
    }

    #[test]
    fn accepted_control_events_survive_take(
        capacity in 0usize..17,
        kinds in kinds(),
    ) {
        let mut queue = EventQueue::new(capacity, OverflowPolicy::DropOldest);
        let mut expected = Vec::new();
        for (idx, kind) in kinds.iter().enumerate() {
            let event = event_for(*kind, idx);
            let outcome = queue.push(event.clone());
            if matches!(
                outcome,
                Ok(EventPushOutcome::Accepted) | Ok(EventPushOutcome::Replaced)
            ) && is_control(&event)
            {
                expected.push(event);
            }
        }

        let actual: Vec<SessionEvent> = queue
            .take()
            .into_iter()
            .filter(is_control)
            .collect();
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn control_lane_refuses_and_counts_on_overflow(
        capacity in 0usize..17,
        n in 0usize..64,
    ) {
        let mut queue = EventQueue::new(capacity, OverflowPolicy::DropOldest);
        let mut refused = 0u64;
        for i in 0..n {
            let event = SessionEvent::TranscriptFinal {
                turn: i as u64,
                text: format!("t{i}"),
            };
            match queue.push(event) {
                Ok(EventPushOutcome::Accepted) => {}
                Ok(other) => prop_assert!(false, "control push returned {other:?}"),
                Err(err) => {
                    prop_assert!(matches!(err, ParlevoxError::Protocol(_)));
                    refused += 1;
                }
            }
        }

        // Non-terminal control events reserve the terminal slot, so at most
        // `capacity - 1` are accepted.
        let accepted = (n.min(capacity.saturating_sub(1))) as u64;
        let stats = queue.stats();
        prop_assert_eq!(stats.accepted, accepted);
        prop_assert_eq!(stats.control_overflow, refused);
        prop_assert_eq!(stats.dropped, refused);
        prop_assert_eq!(queue.len(), accepted as usize);
    }

    #[test]
    fn terminal_is_admitted_when_capacity_allows_and_none_is_queued(
        capacity in 1usize..17,
        kinds in prop::collection::vec(non_terminal_kind(), 0..64),
    ) {
        let mut queue = EventQueue::new(capacity, OverflowPolicy::DropOldest);
        for (idx, kind) in kinds.iter().enumerate() {
            let _ = queue.push(event_for(*kind, idx));
        }
        // Non-terminal pushes can leave at most `capacity - 1` buffered, so the
        // reserved slot is always available for the terminal event.
        let terminal = event_for(Kind::Terminal, 999);
        prop_assert_eq!(
            queue.push(terminal).unwrap(),
            EventPushOutcome::Accepted
        );
    }

    #[test]
    fn splitter_partition_is_byte_exact(
        max_chars in 1usize..64,
        deltas in prop::collection::vec(any::<String>(), 0..16),
    ) {
        let emitted = split_all(max_chars, &deltas);
        prop_assert_eq!(emitted.concat(), deltas.concat());
        prop_assert!(emitted.iter().all(|sentence| !sentence.is_empty()));
    }

    #[test]
    fn splitter_forced_chunks_respect_max_chars(
        max_chars in 1usize..64,
        text in punctuation_free(),
    ) {
        let emitted = split_all(max_chars, std::slice::from_ref(&text));
        prop_assert_eq!(emitted.concat(), text);
        prop_assert!(
            emitted
                .iter()
                .all(|sentence| sentence.chars().count() <= max_chars),
            "a forced chunk exceeded {max_chars} characters: {emitted:?}"
        );
    }

    #[test]
    fn take_preserves_survivor_order(
        capacity in 0usize..17,
        kinds in kinds(),
    ) {
        let mut queue = EventQueue::new(capacity, OverflowPolicy::DropOldest);
        let mut pushed = Vec::new();
        for (idx, kind) in kinds.iter().enumerate() {
            let event = event_for(*kind, idx);
            pushed.push(event.clone());
            let _ = queue.push(event);
        }

        let depth = queue.len();
        let taken = queue.take();
        prop_assert_eq!(taken.len(), depth);

        // Every taken event must appear in the pushed order (a subsequence).
        let mut cursor = 0usize;
        for event in &taken {
            while cursor < pushed.len() && &pushed[cursor] != event {
                cursor += 1;
            }
            prop_assert!(
                cursor < pushed.len(),
                "taken event is not a pushed survivor in order"
            );
            cursor += 1;
        }
    }
}
