//! Audio frames and the bounded ring every audio path must go through.
//!
//! Real-time audio cannot wait for a slow consumer: a provider that stalls for
//! two seconds must not turn into two seconds of unbounded heap growth. The
//! ring makes the choice explicit — drop the oldest frame (keep the present) or
//! drop the newest (keep the past) — and accounts for every dropped millisecond.

use std::collections::VecDeque;

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
        DurationMs::from_secs_f64(self.samples_per_channel() as f64 / self.sample_rate as f64)
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
}

impl AudioRing {
    /// `capacity` is in frames. A capacity of zero is legal and means "no
    /// buffering": every frame is dropped and counted, which is the honest
    /// configuration for a path that must never delay its consumer.
    pub fn new(capacity: usize, policy: OverflowPolicy) -> Self {
        AudioRing {
            capacity,
            policy,
            queue: VecDeque::with_capacity(capacity.min(4096)),
            accepted: 0,
            dropped: 0,
            dropped_duration: DurationMs::ZERO,
        }
    }

    pub fn push(&mut self, frame: AudioFrame) -> PushOutcome {
        if self.capacity == 0 {
            self.record_drop(&frame);
            return PushOutcome::Dropped;
        }

        if self.queue.len() < self.capacity {
            self.queue.push_back(frame);
            self.accepted += 1;
            return PushOutcome::Accepted;
        }

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
    }

    pub fn pop(&mut self) -> Option<AudioFrame> {
        self.queue.pop_front()
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
}
