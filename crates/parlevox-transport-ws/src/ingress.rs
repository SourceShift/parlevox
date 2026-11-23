//! Bounded, message-granular PCM ingress.
//!
//! Client audio arrives as whole WebSocket messages. [`IngressRing`] keeps
//! those messages — never individual samples — in a ring bounded by a fixed
//! sample capacity, so one connection can never grow its buffer without limit.
//!
//! Every overflow is accounted: an eviction or a rejection increments the
//! cumulative drop counters and a per-commit pending counter. The next
//! `utterance_end` reports that pending count in `turn_done`, so loss caused by
//! backpressure is visible to the client instead of disappearing.

use std::collections::VecDeque;

use parlevox_core::{OverflowPolicy, ParlevoxError, PushOutcome};
use parlevox_providers::AudioChunk;

/// Drop counters accumulated since the previous commit.
///
/// The server drains this at `utterance_end` and reports it in the matching
/// `turn_done`. Drops that happen while a turn is in flight belong to the next
/// commit and are reported then.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct IngressDrops {
    /// Whole ingress messages lost since the last drain.
    pub frames: u64,
    /// Duration of those messages, in whole milliseconds.
    pub ms: u64,
}

/// A point-in-time view of an [`IngressRing`].
///
/// `pushes == accepted + rejected` and
/// `dropped_frames == rejected + evicted` always hold; both are exercised by
/// the property tests.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct IngressStats {
    /// Messages pushed into the ring, including rejected ones.
    pub pushes: u64,
    /// Messages admitted into the ring.
    pub accepted: u64,
    /// Messages refused outright because they did not fit.
    pub rejected: u64,
    /// Previously admitted messages evicted to make room.
    pub evicted: u64,
    /// Cumulative messages lost: `rejected + evicted`.
    pub dropped_frames: u64,
    /// Cumulative duration of the lost messages, in whole milliseconds.
    pub dropped_ms: u64,
    /// Interleaved samples currently retained.
    pub depth_samples: usize,
    /// Configured capacity, in interleaved samples.
    pub capacity_samples: usize,
}

/// A bounded ring of whole PCM messages.
///
/// One binary WebSocket message is one ingress frame. The ring evicts or
/// rejects whole messages under its [`OverflowPolicy`], never splitting one,
/// and it never allocates an unbounded intermediate buffer: the sum of the
/// retained message lengths is at most `capacity_samples`.
#[derive(Debug)]
pub struct IngressRing {
    sample_rate: u32,
    channels: u16,
    capacity_samples: usize,
    policy: OverflowPolicy,
    messages: VecDeque<Vec<i16>>,
    samples: usize,
    pushes: u64,
    accepted: u64,
    rejected: u64,
    evicted: u64,
    dropped_frames: u64,
    dropped_ms: u64,
    pending_frames: u64,
    pending_ms: u64,
}

impl IngressRing {
    /// Create an empty bounded ring.
    ///
    /// # Errors
    ///
    /// Returns a protocol-class [`ParlevoxError`] when the sample rate or
    /// channel count is zero, when the capacity is zero, or when the capacity
    /// cannot hold one interleaved sample frame.
    pub fn new(
        sample_rate: u32,
        channels: u16,
        capacity_samples: usize,
        policy: OverflowPolicy,
    ) -> Result<Self, ParlevoxError> {
        if sample_rate == 0 {
            return Err(ParlevoxError::Protocol(
                "sample rate must be non-zero".into(),
            ));
        }
        if channels == 0 {
            return Err(ParlevoxError::Protocol(
                "channel count must be non-zero".into(),
            ));
        }
        if capacity_samples < usize::from(channels) {
            return Err(ParlevoxError::Protocol(
                "ingress capacity must hold at least one sample frame".into(),
            ));
        }
        Ok(IngressRing {
            sample_rate,
            channels,
            capacity_samples,
            policy,
            messages: VecDeque::new(),
            samples: 0,
            pushes: 0,
            accepted: 0,
            rejected: 0,
            evicted: 0,
            dropped_frames: 0,
            dropped_ms: 0,
            pending_frames: 0,
            pending_ms: 0,
        })
    }

    /// Decode one little-endian signed 16-bit PCM message and push it.
    ///
    /// # Errors
    ///
    /// Returns a protocol-class [`ParlevoxError`] when the payload has an odd
    /// byte count: signed 16-bit samples cannot be split across a byte
    /// boundary, and silently discarding the trailing byte would corrupt the
    /// stream. Such a message is not pushed and no counter moves.
    pub fn push_bytes(&mut self, bytes: &[u8]) -> Result<PushOutcome, ParlevoxError> {
        if bytes.len() % 2 != 0 {
            return Err(ParlevoxError::Protocol(
                "pcm payload must have an even byte count".into(),
            ));
        }
        let mut samples = Vec::with_capacity(bytes.len() / 2);
        for pair in bytes.chunks_exact(2) {
            samples.push(i16::from_le_bytes([pair[0], pair[1]]));
        }
        Ok(self.push_samples(&samples))
    }

    /// Push one whole message of interleaved samples.
    ///
    /// An empty message is admitted and adds nothing. A message larger than the
    /// whole capacity can never fit: `DropOldest` empties the ring first (the
    /// past is worth less than the present) and `DropNewest` leaves the ring
    /// alone; both then count the incoming message as rejected.
    pub fn push_samples(&mut self, samples: &[i16]) -> PushOutcome {
        self.pushes = self.pushes.saturating_add(1);
        let incoming_len = samples.len();
        if incoming_len == 0 {
            self.accepted = self.accepted.saturating_add(1);
            return PushOutcome::Accepted;
        }

        if incoming_len > self.capacity_samples {
            if self.policy == OverflowPolicy::DropOldest {
                self.evict_all();
            }
            self.reject(incoming_len);
            return PushOutcome::Dropped;
        }

        match self.policy {
            OverflowPolicy::DropOldest => {
                let mut evicted = false;
                while self.samples.saturating_add(incoming_len) > self.capacity_samples {
                    let Some(front) = self.messages.pop_front() else {
                        break;
                    };
                    self.samples = self.samples.saturating_sub(front.len());
                    self.count_eviction(front.len());
                    evicted = true;
                }
                self.admit(samples);
                if evicted {
                    PushOutcome::Replaced
                } else {
                    PushOutcome::Accepted
                }
            }
            OverflowPolicy::DropNewest => {
                if self.samples.saturating_add(incoming_len) > self.capacity_samples {
                    self.reject(incoming_len);
                    PushOutcome::Dropped
                } else {
                    self.admit(samples);
                    PushOutcome::Accepted
                }
            }
        }
    }

    /// Take every retained message as one [`AudioChunk`], oldest first.
    ///
    /// The ring is empty afterwards; cumulative counters are preserved.
    pub fn drain(&mut self) -> AudioChunk {
        let mut pcm = Vec::with_capacity(self.samples);
        for message in self.messages.drain(..) {
            pcm.extend_from_slice(&message);
        }
        self.samples = 0;
        AudioChunk::new(self.sample_rate, self.channels, pcm)
    }

    /// Take the drop counters accumulated since the previous call.
    ///
    /// Cumulative counters in [`IngressRing::stats`] are not reset.
    pub fn take_dropped(&mut self) -> IngressDrops {
        let drops = IngressDrops {
            frames: self.pending_frames,
            ms: self.pending_ms,
        };
        self.pending_frames = 0;
        self.pending_ms = 0;
        drops
    }

    /// Current depth, capacity, and cumulative counters.
    pub fn stats(&self) -> IngressStats {
        IngressStats {
            pushes: self.pushes,
            accepted: self.accepted,
            rejected: self.rejected,
            evicted: self.evicted,
            dropped_frames: self.dropped_frames,
            dropped_ms: self.dropped_ms,
            depth_samples: self.samples,
            capacity_samples: self.capacity_samples,
        }
    }

    /// Whether no message is currently retained.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Number of whole messages currently retained.
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    fn admit(&mut self, samples: &[i16]) {
        self.samples = self.samples.saturating_add(samples.len());
        self.messages.push_back(samples.to_vec());
        self.accepted = self.accepted.saturating_add(1);
    }

    fn evict_all(&mut self) {
        while let Some(front) = self.messages.pop_front() {
            self.samples = self.samples.saturating_sub(front.len());
            self.count_eviction(front.len());
        }
    }

    fn count_eviction(&mut self, len: usize) {
        self.evicted = self.evicted.saturating_add(1);
        self.count_drop(len);
    }

    fn reject(&mut self, len: usize) {
        self.rejected = self.rejected.saturating_add(1);
        self.count_drop(len);
    }

    fn count_drop(&mut self, len: usize) {
        let ms = self.message_ms(len);
        self.dropped_frames = self.dropped_frames.saturating_add(1);
        self.dropped_ms = self.dropped_ms.saturating_add(ms);
        self.pending_frames = self.pending_frames.saturating_add(1);
        self.pending_ms = self.pending_ms.saturating_add(ms);
    }

    /// Whole milliseconds held by a message, floor-rounded.
    fn message_ms(&self, len: usize) -> u64 {
        let per_channel = len / usize::from(self.channels);
        let samples = u64::try_from(per_channel).unwrap_or(u64::MAX);
        samples
            .saturating_mul(1000)
            .checked_div(u64::from(self.sample_rate))
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring(capacity: usize, policy: OverflowPolicy) -> IngressRing {
        IngressRing::new(24_000, 1, capacity, policy).expect("valid ring")
    }

    #[test]
    fn drop_oldest_evicts_whole_messages_and_counts_them() {
        let mut ring = ring(320, OverflowPolicy::DropOldest);
        assert_eq!(ring.push_samples(&vec![1; 160]), PushOutcome::Accepted);
        assert_eq!(ring.push_samples(&vec![2; 160]), PushOutcome::Accepted);
        assert_eq!(ring.push_samples(&vec![3; 160]), PushOutcome::Replaced);
        assert_eq!(ring.push_samples(&vec![4; 160]), PushOutcome::Replaced);

        let stats = ring.stats();
        assert_eq!(stats.pushes, 4);
        assert_eq!(stats.accepted, 4);
        assert_eq!(stats.evicted, 2);
        assert_eq!(stats.dropped_frames, 2);
        assert_eq!(stats.depth_samples, 320);

        let drops = ring.take_dropped();
        assert_eq!(drops.frames, 2);
        assert!(drops.ms > 0);
        assert_eq!(ring.take_dropped(), IngressDrops::default());
        assert_eq!(ring.stats().dropped_frames, 2);

        let chunk = ring.drain();
        assert_eq!(chunk.pcm.len(), 320);
        assert_eq!(chunk.pcm[0], 3);
        assert_eq!(chunk.pcm[160], 4);
        assert!(ring.is_empty());
    }

    #[test]
    fn drop_newest_keeps_the_past_and_rejects_the_incoming() {
        let mut ring = ring(320, OverflowPolicy::DropNewest);
        assert_eq!(ring.push_samples(&vec![1; 160]), PushOutcome::Accepted);
        assert_eq!(ring.push_samples(&vec![2; 160]), PushOutcome::Accepted);
        assert_eq!(ring.push_samples(&vec![3; 160]), PushOutcome::Dropped);

        let stats = ring.stats();
        assert_eq!(stats.pushes, 3);
        assert_eq!(stats.accepted, 2);
        assert_eq!(stats.rejected, 1);
        assert_eq!(stats.evicted, 0);
        assert_eq!(stats.dropped_frames, 1);

        let chunk = ring.drain();
        assert_eq!(chunk.pcm[0], 1);
        assert_eq!(chunk.pcm[160], 2);
    }

    #[test]
    fn oversized_message_never_enters_the_ring() {
        let mut oldest = ring(160, OverflowPolicy::DropOldest);
        oldest.push_samples(&vec![7; 160]);
        assert_eq!(oldest.push_samples(&vec![8; 161]), PushOutcome::Dropped);
        assert!(oldest.is_empty());
        assert_eq!(oldest.stats().evicted, 1);
        assert_eq!(oldest.stats().rejected, 1);

        let mut newest = ring(160, OverflowPolicy::DropNewest);
        newest.push_samples(&vec![7; 160]);
        assert_eq!(newest.push_samples(&vec![8; 161]), PushOutcome::Dropped);
        assert_eq!(newest.len(), 1);
        assert_eq!(newest.stats().rejected, 1);
    }

    #[test]
    fn odd_byte_count_is_a_protocol_error_and_pushes_nothing() {
        let mut ring = ring(320, OverflowPolicy::DropOldest);
        let err = ring.push_bytes(&[1, 2, 3]).expect_err("odd payload");
        assert!(matches!(err, ParlevoxError::Protocol(_)));
        assert_eq!(ring.stats().pushes, 0);
        assert!(ring.is_empty());
    }

    #[test]
    fn bytes_round_trip_as_little_endian_samples() {
        let mut ring = ring(320, OverflowPolicy::DropOldest);
        let bytes = [1i16, -2, 300].map(i16::to_le_bytes).concat();
        assert_eq!(
            ring.push_bytes(&bytes).expect("even payload"),
            PushOutcome::Accepted
        );
        assert_eq!(ring.drain().pcm, vec![1, -2, 300]);
    }
}
