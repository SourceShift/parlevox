//! Bounded buffers and the connector counters that make loss observable.
//!
//! A pull-based stream cannot silently accumulate: the caller decides when to
//! poll. Loss therefore only happens where this crate deliberately accepts a
//! burst larger than a configured bound, and every such acceptance is counted.
//! [`BoundedQueue`] is that bound; [`ConnectorStats`] is the counter.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

/// A point-in-time copy of a connector's counters.
///
/// The conservation invariant for a bounded frame queue is
/// `frames_pushed == frames_delivered + frames_dropped + frames_buffered`;
/// [`ConnectorStats::snapshot`] exposes the first three, and the queue length
/// (reported by the stream's owner) supplies the last.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ConnectorStatsSnapshot {
    /// Bytes accepted from the transport body.
    pub bytes_received: u64,
    /// Parsed wire events (SSE events or NDJSON records).
    pub events: u64,
    /// Frames offered to a bounded queue.
    pub frames_pushed: u64,
    /// Frames handed to the caller.
    pub frames_delivered: u64,
    /// Frames discarded because the bounded queue was full.
    pub frames_dropped: u64,
    /// Requests issued, including retries.
    pub attempts: u64,
    /// Retries issued after a transient pre-first-byte failure.
    pub retries: u64,
}

/// Monotonic connector counters shared between a connector and its streams.
///
/// Counters are `Relaxed`: they are telemetry, not synchronisation.
#[derive(Debug, Default)]
pub struct ConnectorStats {
    bytes_received: AtomicU64,
    events: AtomicU64,
    frames_pushed: AtomicU64,
    frames_delivered: AtomicU64,
    frames_dropped: AtomicU64,
    attempts: AtomicU64,
    retries: AtomicU64,
}

impl ConnectorStats {
    /// An empty counter set.
    #[must_use]
    pub fn new() -> Self {
        ConnectorStats::default()
    }

    /// A consistent-enough copy of every counter.
    #[must_use]
    pub fn snapshot(&self) -> ConnectorStatsSnapshot {
        ConnectorStatsSnapshot {
            bytes_received: self.bytes_received.load(Ordering::Relaxed),
            events: self.events.load(Ordering::Relaxed),
            frames_pushed: self.frames_pushed.load(Ordering::Relaxed),
            frames_delivered: self.frames_delivered.load(Ordering::Relaxed),
            frames_dropped: self.frames_dropped.load(Ordering::Relaxed),
            attempts: self.attempts.load(Ordering::Relaxed),
            retries: self.retries.load(Ordering::Relaxed),
        }
    }

    /// Take and reset the dropped-frame counter.
    ///
    /// This is the report a sidecar-backed path exposes through
    /// `take_sidecar_dropped_frames`: take-and-reset so a caller counts each
    /// loss exactly once.
    #[must_use]
    pub fn take_dropped(&self) -> u64 {
        self.frames_dropped.swap(0, Ordering::Relaxed)
    }

    pub(crate) fn record_bytes(&self, bytes: usize) {
        let bytes = u64::try_from(bytes).unwrap_or(u64::MAX);
        self.bytes_received.fetch_add(bytes, Ordering::Relaxed);
    }

    pub(crate) fn record_events(&self, events: usize) {
        let events = u64::try_from(events).unwrap_or(u64::MAX);
        self.events.fetch_add(events, Ordering::Relaxed);
    }

    pub(crate) fn record_pushed(&self, frames: usize) {
        let frames = u64::try_from(frames).unwrap_or(u64::MAX);
        self.frames_pushed.fetch_add(frames, Ordering::Relaxed);
    }

    pub(crate) fn record_delivered(&self, frames: usize) {
        let frames = u64::try_from(frames).unwrap_or(u64::MAX);
        self.frames_delivered.fetch_add(frames, Ordering::Relaxed);
    }

    pub(crate) fn record_dropped(&self, frames: usize) {
        let frames = u64::try_from(frames).unwrap_or(u64::MAX);
        self.frames_dropped.fetch_add(frames, Ordering::Relaxed);
    }

    pub(crate) fn record_attempt(&self) {
        self.attempts.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_retry(&self) {
        self.retries.fetch_add(1, Ordering::Relaxed);
    }
}

/// A bounded FIFO with a declared overflow policy: drop the oldest item.
///
/// A zero capacity is legal and means "drop every item immediately". The
/// number of discarded items is returned from [`BoundedQueue::push`] and
/// [`BoundedQueue::push_all`] so callers can count loss instead of guessing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundedQueue<T> {
    items: VecDeque<T>,
    capacity: usize,
}

impl<T> BoundedQueue<T> {
    /// Create a queue that holds at most `capacity` items.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        BoundedQueue {
            items: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// The configured bound.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of buffered items.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether nothing is buffered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Push one item, dropping the oldest when full.
    ///
    /// Returns the discarded item, if any.
    pub fn push(&mut self, item: T) -> Option<T> {
        if self.capacity == 0 {
            return Some(item);
        }
        if self.items.len() >= self.capacity {
            let dropped = self.items.pop_front();
            self.items.push_back(item);
            return dropped;
        }
        self.items.push_back(item);
        None
    }

    /// Push every item, returning how many were discarded.
    ///
    /// Items past capacity evict the oldest buffered items one for one.
    pub fn push_all(&mut self, items: impl IntoIterator<Item = T>) -> usize {
        let mut dropped = 0usize;
        for item in items {
            if self.push(item).is_some() {
                dropped = dropped.saturating_add(1);
            }
        }
        dropped
    }

    /// Remove and return the oldest item.
    pub fn pop(&mut self) -> Option<T> {
        self.items.pop_front()
    }
}

impl<T> Default for BoundedQueue<T> {
    /// A queue with zero capacity: nothing is buffered.
    fn default() -> Self {
        BoundedQueue::new(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overflow_drops_the_oldest_and_reports_it() {
        let mut queue = BoundedQueue::new(2);
        assert_eq!(queue.push(1), None);
        assert_eq!(queue.push(2), None);
        assert_eq!(queue.push(3), Some(1));
        assert_eq!(queue.len(), 2);
        assert_eq!(queue.pop(), Some(2));
        assert_eq!(queue.pop(), Some(3));
        assert_eq!(queue.pop(), None);
    }

    #[test]
    fn zero_capacity_drops_everything() {
        let mut queue = BoundedQueue::new(0);
        assert_eq!(queue.push(7), Some(7));
        assert!(queue.is_empty());
        assert_eq!(queue.push_all([1, 2, 3]), 3);
    }

    #[test]
    fn counters_take_and_reset() {
        let stats = ConnectorStats::new();
        stats.record_dropped(4);
        assert_eq!(stats.snapshot().frames_dropped, 4);
        assert_eq!(stats.take_dropped(), 4);
        assert_eq!(stats.take_dropped(), 0);
    }
}
