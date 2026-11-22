//! Audio frames and the bounded ring every audio path must go through.
//!
//! Real-time audio cannot wait for a slow consumer: a provider that stalls for
//! two seconds must not turn into two seconds of unbounded heap growth. The
//! ring makes the choice explicit — drop the oldest frame (keep the present) or
//! drop the newest (keep the past) — and accounts for every dropped millisecond.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::Waker;

use crate::clock::{DurationMs, Timestamp};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioFrame {
    pub timestamp: Timestamp,
    pub sample_rate: u32,
    pub channels: u16,
    /// Interleaved signed 16-bit PCM.
    pub pcm: Vec<i16>,
}

impl AudioFrame {
    pub fn new(timestamp: Timestamp, sample_rate: u32, channels: u16, pcm: Vec<i16>) -> Self {
        AudioFrame {
            timestamp,
            sample_rate,
            channels,
            pcm,
        }
    }

    /// Frames of silence with the same format as `other`.
    pub fn silence_like(other: &AudioFrame, timestamp: Timestamp) -> Self {
        AudioFrame {
            timestamp,
            sample_rate: other.sample_rate,
            channels: other.channels,
            pcm: vec![0; other.pcm.len()],
        }
    }

    pub fn samples_per_channel(&self) -> usize {
        if self.channels == 0 {
            0
        } else {
            self.pcm.len() / self.channels as usize
        }
    }

    pub fn duration(&self) -> DurationMs {
        if self.sample_rate == 0 {
            return DurationMs::ZERO;
        }
        // Whole-millisecond accounting: integer division truncates any
        // sub-millisecond remainder, and `saturating_mul` keeps a
        // pathological sample count from wrapping.
        let samples = u64::try_from(self.samples_per_channel()).unwrap_or(u64::MAX);
        DurationMs::from_millis(samples.saturating_mul(1000) / u64::from(self.sample_rate))
    }

    pub fn is_empty(&self) -> bool {
        self.pcm.is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverflowPolicy {
    /// Keep the present: evict the oldest queued frame to admit the new one.
    DropOldest,
    /// Keep the past: reject the incoming frame.
    DropNewest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PushOutcome {
    Accepted,
    /// The oldest frame was evicted to make room.
    Replaced,
    /// The incoming frame was rejected.
    Dropped,
}

/// A point-in-time view of a ring's capacity pressure.
///
/// `slack` is `capacity - depth` computed with saturating arithmetic, so it is
/// never negative and never underflows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct CreditSnapshot {
    pub capacity: usize,
    pub depth: usize,
    pub slack: usize,
}

/// Cumulative outcomes of opt-in credit waits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct CreditStats {
    pub granted: u64,
    pub cancelled: u64,
}

/// How a credit wait resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
#[non_exhaustive]
pub enum CreditWait {
    /// Slack opened: the ring has room for one more frame.
    Granted,
    /// The caller's cancellation token fired before any slack opened.
    Cancelled,
}

/// A cloneable, non-consuming view of one ring's credit state.
///
/// It is deliberately std-only: `parlevox-core` has no async runtime
/// dependency, so the handle owns the snapshot and the single waiter slot
/// while the asynchronous wait lives in `parlevox-runtime`.
#[derive(Clone)]
pub struct CreditHandle {
    inner: Arc<CreditInner>,
}

struct CreditInner {
    capacity: usize,
    depth: AtomicUsize,
    granted: AtomicU64,
    cancelled: AtomicU64,
    /// Single waker slot. The same mutex serializes the waiter's
    /// check-and-register with the ring's check-and-wake, so the two sides
    /// cannot both observe "no waiter" and "no slack". The handshake therefore
    /// does not depend on ordering inferred from `depth`; the depth lifetime is
    /// only a mirrored counter for lock-free readers.
    waker: Mutex<Option<Waker>>,
}

impl std::fmt::Debug for CreditHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CreditHandle")
            .field("snapshot", &self.snapshot())
            .field("stats", &self.stats())
            .finish()
    }
}

impl CreditHandle {
    fn new(capacity: usize) -> Self {
        CreditHandle {
            inner: Arc::new(CreditInner {
                capacity,
                depth: AtomicUsize::new(0),
                granted: AtomicU64::new(0),
                cancelled: AtomicU64::new(0),
                waker: Mutex::new(None),
            }),
        }
    }

    /// Current capacity, depth, and slack. Reading it never consumes or
    /// mutates ring state.
    pub fn snapshot(&self) -> CreditSnapshot {
        let depth = self.inner.depth.load(Ordering::Acquire);
        CreditSnapshot {
            capacity: self.inner.capacity,
            depth,
            slack: self.inner.capacity.saturating_sub(depth),
        }
    }

    /// Cumulative `granted` and `cancelled` wait outcomes.
    pub fn stats(&self) -> CreditStats {
        CreditStats {
            granted: self.inner.granted.load(Ordering::Relaxed),
            cancelled: self.inner.cancelled.load(Ordering::Relaxed),
        }
    }

    /// Record one wait that resolved because slack opened.
    pub fn record_granted(&self) {
        self.inner.granted.fetch_add(1, Ordering::Relaxed);
    }

    /// Record one wait that resolved because the caller cancelled.
    pub fn record_cancelled(&self) {
        self.inner.cancelled.fetch_add(1, Ordering::Relaxed);
    }

    /// Atomically test for slack and, when there is none, register `waker`.
    ///
    /// Returns `true` when slack is already available (the caller should
    /// resolve immediately) and `false` once the waker is registered and will
    /// be woken by the next pop that opens room.
    ///
    /// The test and the registration happen under the same mutex `pop` holds
    /// while it takes the waker. A mutex critical section is totally ordered,
    /// so at least one side sees the other: either the waiter sees the popped
    /// slack, or the pop sees the registered waker. That removes the
    /// check-then-register lost-wakeup window without relying on cross-variable
    /// atomic ordering.
    ///
    /// One slot is intentional: this is a capacity signal, not a queue of
    /// consumers. Registering again replaces the previous waker.
    pub fn register_or_grant(&self, waker: &Waker) -> bool {
        let mut slot = self.lock_waker();
        if self.snapshot().slack > 0 {
            return true;
        }
        *slot = Some(waker.clone());
        false
    }

    fn set_depth(&self, depth: usize) {
        // Release pairs with the Acquire load in `snapshot`, so a lock-free
        // reader sees a fully published depth. The waiter handshake does not
        // derive ordering from this counter; it is serialized by `lock_waker`.
        self.inner.depth.store(depth, Ordering::Release);
    }

    /// Wake the registered waiter if the ring currently has slack.
    ///
    /// Called after a pop that may have opened room. The slack check and the
    /// waker take share one critical section with `register_or_grant`; the
    /// guard is dropped before waking, so no lock is held across the wake
    /// callback.
    fn notify_if_slack(&self) {
        let waker = {
            let mut slot = self.lock_waker();
            if self.snapshot().slack > 0 {
                slot.take()
            } else {
                None
            }
        };
        if let Some(waker) = waker {
            waker.wake();
        }
    }

    /// Poisoning is recovered rather than panicked on: a panicking waker must
    /// not wedge the ring or strand a waiter.
    fn lock_waker(&self) -> std::sync::MutexGuard<'_, Option<Waker>> {
        match self.inner.waker.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingStats {
    pub accepted: u64,
    pub dropped: u64,
    pub dropped_duration: DurationMs,
    pub depth: usize,
    pub capacity: usize,
}

pub struct AudioRing {
    capacity: usize,
    policy: OverflowPolicy,
    queue: VecDeque<AudioFrame>,
    accepted: u64,
    dropped: u64,
    dropped_duration: DurationMs,
    /// Present only for rings built with [`AudioRing::new_with_credits`]. The
    /// default ring pays nothing for the opt-in signal.
    credits: Option<CreditHandle>,
}

impl AudioRing {
    /// `capacity` is in frames. A capacity of zero is legal and means "no
    /// buffering": every frame is dropped and counted, which is the honest
    /// configuration for a path that must never delay its consumer.
    ///
    /// This constructor is credit-free; use [`AudioRing::new_with_credits`] to
    /// also publish a [`CreditHandle`].
    pub fn new(capacity: usize, policy: OverflowPolicy) -> Self {
        AudioRing {
            capacity,
            policy,
            queue: VecDeque::with_capacity(capacity.min(4096)),
            accepted: 0,
            dropped: 0,
            dropped_duration: DurationMs::ZERO,
            credits: None,
        }
    }

    /// Build a ring that also publishes an opt-in [`CreditHandle`].
    ///
    /// The ring's drop-not-throttle behaviour is unchanged: the handle only
    /// mirrors capacity, depth, and slack and carries a single waiter slot.
    pub fn new_with_credits(capacity: usize, policy: OverflowPolicy) -> (AudioRing, CreditHandle) {
        let handle = CreditHandle::new(capacity);
        let ring = AudioRing {
            capacity,
            policy,
            queue: VecDeque::with_capacity(capacity.min(4096)),
            accepted: 0,
            dropped: 0,
            dropped_duration: DurationMs::ZERO,
            credits: Some(handle.clone()),
        };
        (ring, handle)
    }

    /// The credit handle this ring publishes, if it was built with
    /// [`AudioRing::new_with_credits`].
    pub fn credit_handle(&self) -> Option<CreditHandle> {
        self.credits.clone()
    }

    pub fn push(&mut self, frame: AudioFrame) -> PushOutcome {
        let outcome = if self.capacity == 0 {
            self.record_drop(&frame);
            PushOutcome::Dropped
        } else if self.queue.len() < self.capacity {
            self.queue.push_back(frame);
            self.accepted += 1;
            PushOutcome::Accepted
        } else {
            match self.policy {
                OverflowPolicy::DropOldest => {
                    if let Some(old) = self.queue.pop_front() {
                        self.record_drop(&old);
                    }
                    self.queue.push_back(frame);
                    self.accepted += 1;
                    PushOutcome::Replaced
                }
                OverflowPolicy::DropNewest => {
                    self.record_drop(&frame);
                    PushOutcome::Dropped
                }
            }
        };
        self.sync_credit_depth();
        outcome
    }

    pub fn pop(&mut self) -> Option<AudioFrame> {
        let frame = self.queue.pop_front();
        self.sync_credit_depth();
        if frame.is_some() {
            if let Some(credits) = &self.credits {
                credits.notify_if_slack();
            }
        }
        frame
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    pub fn is_full(&self) -> bool {
        self.queue.len() >= self.capacity
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn stats(&self) -> RingStats {
        RingStats {
            accepted: self.accepted,
            dropped: self.dropped,
            dropped_duration: self.dropped_duration,
            depth: self.queue.len(),
            capacity: self.capacity,
        }
    }

    /// Recompute the credit mirror from the queue. This is the single place
    /// the published depth is written, called at the end of every push path
    /// and after every pop, so it cannot drift from `len()`.
    fn sync_credit_depth(&self) {
        if let Some(credits) = &self.credits {
            credits.set_depth(self.queue.len());
        }
    }

    fn record_drop(&mut self, frame: &AudioFrame) {
        self.dropped += 1;
        self.dropped_duration = self.dropped_duration.sat_dur_add(frame.duration());
    }
}

impl DurationMs {
    /// Saturating add; exposed because ring accounting must not panic on a
    /// pathological session length.
    pub(crate) fn sat_dur_add(self, other: DurationMs) -> DurationMs {
        DurationMs::from_millis(self.as_millis().saturating_add(other.as_millis()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(ms: u64) -> AudioFrame {
        // 16 kHz mono → 16 samples per ms.
        AudioFrame::new(
            Timestamp::from_millis(0),
            16_000,
            1,
            vec![1; (ms * 16) as usize],
        )
    }

    #[test]
    fn frame_duration_is_derived_from_format() {
        assert_eq!(frame(20).duration().as_millis(), 20);
        let zero_rate = AudioFrame::new(Timestamp::from_millis(0), 0, 1, vec![0; 10]);
        assert_eq!(zero_rate.duration(), DurationMs::ZERO);
    }

    #[test]
    fn drop_oldest_keeps_the_present_and_counts_the_past() {
        let mut ring = AudioRing::new(2, OverflowPolicy::DropOldest);
        assert_eq!(ring.push(frame(10)), PushOutcome::Accepted);
        assert_eq!(ring.push(frame(20)), PushOutcome::Accepted);
        assert_eq!(ring.push(frame(30)), PushOutcome::Replaced);

        assert_eq!(ring.len(), 2);
        assert_eq!(ring.pop().unwrap().duration().as_millis(), 20);
        assert_eq!(ring.pop().unwrap().duration().as_millis(), 30);

        let stats = ring.stats();
        assert_eq!(stats.accepted, 3);
        assert_eq!(stats.dropped, 1);
        assert_eq!(stats.dropped_duration.as_millis(), 10);
    }

    #[test]
    fn drop_newest_rejects_the_incoming_frame() {
        let mut ring = AudioRing::new(1, OverflowPolicy::DropNewest);
        assert_eq!(ring.push(frame(10)), PushOutcome::Accepted);
        assert_eq!(ring.push(frame(99)), PushOutcome::Dropped);

        assert_eq!(ring.pop().unwrap().duration().as_millis(), 10);
        assert_eq!(ring.stats().dropped_duration.as_millis(), 99);
    }

    #[test]
    fn zero_capacity_never_buffers() {
        let mut ring = AudioRing::new(0, OverflowPolicy::DropOldest);
        assert_eq!(ring.push(frame(5)), PushOutcome::Dropped);
        assert!(ring.is_empty());
        assert_eq!(ring.stats().dropped, 1);
    }

    // --- opt-in credit signal ------------------------------------------

    /// Assert the published snapshot mirrors the ring exactly.
    fn assert_credit_mirror(ring: &AudioRing, handle: &CreditHandle) {
        assert_eq!(
            handle.snapshot(),
            CreditSnapshot {
                capacity: ring.capacity(),
                depth: ring.len(),
                slack: ring.capacity() - ring.len(),
            }
        );
    }

    #[test]
    fn credit_snapshot_matches_ring_after_every_op() {
        // Zero capacity: every push drops, slack stays zero, never underflows.
        let (mut ring, handle) = AudioRing::new_with_credits(0, OverflowPolicy::DropOldest);
        assert_credit_mirror(&ring, &handle);
        assert_eq!(ring.push(frame(5)), PushOutcome::Dropped);
        assert_credit_mirror(&ring, &handle);
        assert!(ring.pop().is_none());
        assert_credit_mirror(&ring, &handle);

        // DropOldest: a full ring evicts the oldest and keeps depth pinned.
        let (mut ring, handle) = AudioRing::new_with_credits(2, OverflowPolicy::DropOldest);
        assert_credit_mirror(&ring, &handle);
        assert_eq!(ring.push(frame(10)), PushOutcome::Accepted);
        assert_credit_mirror(&ring, &handle);
        assert_eq!(ring.push(frame(20)), PushOutcome::Accepted);
        assert_credit_mirror(&ring, &handle);
        assert_eq!(ring.push(frame(30)), PushOutcome::Replaced);
        assert_credit_mirror(&ring, &handle);
        assert!(ring.pop().is_some());
        assert_credit_mirror(&ring, &handle);

        // DropNewest: a full ring rejects the incoming frame.
        let (mut ring, handle) = AudioRing::new_with_credits(1, OverflowPolicy::DropNewest);
        assert_eq!(ring.push(frame(10)), PushOutcome::Accepted);
        assert_credit_mirror(&ring, &handle);
        assert_eq!(ring.push(frame(99)), PushOutcome::Dropped);
        assert_credit_mirror(&ring, &handle);
        assert!(ring.pop().is_some());
        assert_credit_mirror(&ring, &handle);
    }

    #[test]
    fn drop_not_throttle_survives_credit_signal() {
        for policy in [OverflowPolicy::DropOldest, OverflowPolicy::DropNewest] {
            let (mut ring, handle) = AudioRing::new_with_credits(1, policy);
            assert_eq!(ring.push(frame(10)), PushOutcome::Accepted);
            let before = ring.stats();
            eprintln!(
                "credits policy={policy:?} before: depth={} capacity={} accepted={} dropped={}",
                before.depth, before.capacity, before.accepted, before.dropped
            );

            let outcome = ring.push(frame(20));
            let after = ring.stats();
            eprintln!(
                "credits policy={policy:?} after: depth={} capacity={} accepted={} dropped={}",
                after.depth, after.capacity, after.accepted, after.dropped
            );

            match policy {
                OverflowPolicy::DropOldest => assert_eq!(outcome, PushOutcome::Replaced),
                OverflowPolicy::DropNewest => assert_eq!(outcome, PushOutcome::Dropped),
            }
            assert!(
                after.dropped > before.dropped,
                "a drop under credit must still be counted"
            );
            assert_eq!(handle.snapshot().slack, 0, "the ring is still full");
        }
    }

    use proptest::prelude::*;

    #[derive(Debug, Clone, Copy)]
    enum RingOp {
        Push(u64),
        Pop,
    }

    fn ring_ops() -> impl Strategy<Value = Vec<RingOp>> {
        prop::collection::vec(
            prop_oneof![(0u64..50).prop_map(RingOp::Push), Just(RingOp::Pop)],
            0..48,
        )
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn credit_snapshot_property_holds_under_arbitrary_ops(
            ops in ring_ops(),
            oldest in any::<bool>(),
            capacity in 0usize..=4,
        ) {
            let policy = if oldest {
                OverflowPolicy::DropOldest
            } else {
                OverflowPolicy::DropNewest
            };
            let (mut ring, handle) = AudioRing::new_with_credits(capacity, policy);
            for (index, op) in ops.into_iter().enumerate() {
                match op {
                    RingOp::Push(ms) => {
                        let before = ring.stats().dropped;
                        let outcome = ring.push(frame(ms));
                        if matches!(outcome, PushOutcome::Dropped | PushOutcome::Replaced) {
                            prop_assert!(
                                ring.stats().dropped > before,
                                "op {}: a drop was not counted",
                                index
                            );
                        }
                    }
                    RingOp::Pop => {
                        let _ = ring.pop();
                    }
                }
                prop_assert!(
                    ring.capacity() >= ring.len(),
                    "op {}: depth exceeds capacity",
                    index
                );
                prop_assert_eq!(
                    handle.snapshot(),
                    CreditSnapshot {
                        capacity: ring.capacity(),
                        depth: ring.len(),
                        slack: ring.capacity() - ring.len(),
                    }
                );
            }
        }
    }
}
