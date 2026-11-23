//! Property tests for the bounded ingress ring.
//!
//! These invariants are the crate's memory-safety contract for client audio:
//! the ring never exceeds its configured capacity, every push is accounted as
//! accepted or rejected, evictions and rejections are counted exactly once, the
//! cumulative counters never move backwards, and no payload can panic the
//! decoder.

use parlevox_core::OverflowPolicy;
use parlevox_transport_ws::IngressRing;
use proptest::prelude::*;

fn any_policy() -> impl Strategy<Value = OverflowPolicy> {
    prop_oneof![
        Just(OverflowPolicy::DropOldest),
        Just(OverflowPolicy::DropNewest),
    ]
}

proptest! {
    #[test]
    fn capacity_is_never_exceeded(
        capacity in 1usize..=4096,
        messages in prop::collection::vec(
            prop::collection::vec(any::<i16>(), 0..=512),
            0..64,
        ),
        policy in any_policy(),
    ) {
        let mut ring = IngressRing::new(24_000, 1, capacity, policy)
            .expect("a positive capacity is valid");
        for message in &messages {
            ring.push_samples(message);
            prop_assert!(ring.stats().depth_samples <= capacity);
            prop_assert!(ring.stats().depth_samples <= ring.stats().capacity_samples);
        }
    }

    #[test]
    fn every_push_is_accepted_or_rejected(
        capacity in 1usize..=2048,
        messages in prop::collection::vec(
            prop::collection::vec(any::<i16>(), 0..=256),
            0..64,
        ),
        policy in any_policy(),
    ) {
        let mut ring = IngressRing::new(16_000, 1, capacity, policy)
            .expect("a positive capacity is valid");
        for message in &messages {
            ring.push_samples(message);
        }
        let stats = ring.stats();
        prop_assert_eq!(stats.pushes, stats.accepted + stats.rejected);
        prop_assert_eq!(stats.dropped_frames, stats.rejected + stats.evicted);
        prop_assert!(stats.evicted <= stats.accepted);
        prop_assert!(stats.dropped_frames <= stats.pushes);
    }

    #[test]
    fn cumulative_drop_counters_never_decrease(
        capacity in 1usize..=512,
        messages in prop::collection::vec(
            prop::collection::vec(any::<i16>(), 0..=128),
            0..64,
        ),
        policy in any_policy(),
    ) {
        let mut ring = IngressRing::new(8_000, 1, capacity, policy)
            .expect("a positive capacity is valid");
        let mut previous_frames = 0;
        let mut previous_ms = 0;
        for message in &messages {
            ring.push_samples(message);
            let stats = ring.stats();
            prop_assert!(stats.dropped_frames >= previous_frames);
            prop_assert!(stats.dropped_ms >= previous_ms);
            previous_frames = stats.dropped_frames;
            previous_ms = stats.dropped_ms;
        }
        // Draining audio never rewinds the cumulative accounting.
        let _ = ring.drain();
        let stats = ring.stats();
        prop_assert!(stats.dropped_frames >= previous_frames);
        prop_assert!(stats.dropped_ms >= previous_ms);
    }

    #[test]
    fn arbitrary_payloads_never_panic(
        capacity in 1usize..=1024,
        payloads in prop::collection::vec(
            prop::collection::vec(any::<u8>(), 0..=2048),
            0..32,
        ),
        policy in any_policy(),
    ) {
        let mut ring = IngressRing::new(24_000, 2, capacity.max(2), policy)
            .expect("a capacity of at least one frame is valid");
        for payload in &payloads {
            let odd = payload.len() % 2 != 0;
            let result = ring.push_bytes(payload);
            prop_assert_eq!(result.is_err(), odd);
            prop_assert!(ring.stats().depth_samples <= ring.stats().capacity_samples);
        }
    }
}
