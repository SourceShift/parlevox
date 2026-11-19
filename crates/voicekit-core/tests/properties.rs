//! Property checks for the bounded ring: size never exceeds capacity and every
//! dropped frame is accounted for by frame and duration.

use proptest::prelude::*;
use voicekit_core::{AudioFrame, AudioRing, DurationMs, OverflowPolicy, Timestamp};

fn frame(ms: u64) -> AudioFrame {
    // 16 kHz mono → 16 samples per millisecond.
    AudioFrame::new(
        Timestamp::from_millis(0),
        16_000,
        1,
        vec![1; (ms * 16) as usize],
    )
}

fn total_duration(frames: &[u64]) -> DurationMs {
    DurationMs::from_millis(frames.iter().sum())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn drop_oldest_is_bounded_and_accounts_every_eviction(
        capacity in 0usize..17,
        durations in prop::collection::vec(1u64..=20, 0..48),
    ) {
        let mut ring = AudioRing::new(capacity, OverflowPolicy::DropOldest);
        for ms in &durations {
            ring.push(frame(*ms));
            prop_assert!(ring.len() <= capacity);
        }

        let stats = ring.stats();
        prop_assert_eq!(ring.len(), capacity.min(durations.len()));
        prop_assert!(stats.depth <= stats.capacity);

        if capacity == 0 {
            prop_assert_eq!(stats.accepted, 0);
            prop_assert_eq!(stats.dropped, durations.len() as u64);
            prop_assert_eq!(stats.dropped_duration, total_duration(&durations));
        } else {
            prop_assert_eq!(stats.accepted, durations.len() as u64);
            prop_assert_eq!(
                stats.dropped,
                durations.len().saturating_sub(capacity) as u64
            );
            let evicted = &durations[..durations.len().saturating_sub(capacity)];
            prop_assert_eq!(stats.dropped_duration, total_duration(evicted));
        }
    }

    #[test]
    fn drop_newest_rejects_the_tail_and_accounts_it(
        capacity in 1usize..17,
        durations in prop::collection::vec(1u64..=20, 0..48),
    ) {
        let mut ring = AudioRing::new(capacity, OverflowPolicy::DropNewest);
        for ms in &durations {
            ring.push(frame(*ms));
            prop_assert!(ring.len() <= capacity);
        }

        let stats = ring.stats();
        prop_assert_eq!(ring.len(), capacity.min(durations.len()));
        prop_assert_eq!(stats.accepted, capacity.min(durations.len()) as u64);
        let rejected = durations.iter().skip(capacity).copied().collect::<Vec<_>>();
        prop_assert_eq!(stats.dropped, rejected.len() as u64);
        prop_assert_eq!(stats.dropped_duration, total_duration(&rejected));
    }
}
