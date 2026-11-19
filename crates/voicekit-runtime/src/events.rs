//! Typed session events.
//!
//! These are the only things a transport (CLI, websocket, test harness) needs
//! to render a turn. Audio is carried as an owned [`AudioFrame`], never as
//! base64 or whole-turn JSON.

use std::collections::VecDeque;

use voicekit_core::{AudioFrame, ErrorClass, OverflowPolicy, VoicekitError};

/// Smallest event capacity a session can operate with.
///
/// Two slots are the minimum contract: one for the terminal `TurnDone` reserve
/// and one for a streaming or control event ahead of it. A smaller queue can
/// hold the terminal event but has no room to carry the turn that led to it,
/// so [`crate::session::Session::new`] rejects it.
pub const MIN_EVENT_CAPACITY: usize = 2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionEvent {
    /// A transcript was committed and a turn index opened.
    TurnStarted { turn: u64 },
    /// The committed user transcript for `turn`.
    TranscriptFinal { turn: u64, text: String },
    /// One streamed assistant delta.
    ReplyDelta { turn: u64, text: String },
    /// One audio frame ready for playout.
    AudioOut { turn: u64, frame: AudioFrame },
    /// The turn finished. `interrupted` is true after a barge-in.
    TurnDone {
        turn: u64,
        audio_frames: u64,
        dropped_frames: u64,
        dropped_ms: u64,
        /// Events this turn's bounded queue had to drop.
        dropped_events: u64,
        /// Frames the sidecar plugin reported dropping for this turn. Kept
        /// separate from `dropped_frames` (Rust playout) so loss at the
        /// process boundary is distinguishable from local backpressure.
        sidecar_dropped_frames: u64,
        interrupted: bool,
    },
    /// The turn failed with a classified error.
    Failed { class: ErrorClass, message: String },
}

impl SessionEvent {
    /// Short label used by the CLI and tests.
    pub fn kind(&self) -> &'static str {
        match self {
            SessionEvent::TurnStarted { .. } => "TurnStarted",
            SessionEvent::TranscriptFinal { .. } => "TranscriptFinal",
            SessionEvent::ReplyDelta { .. } => "ReplyDelta",
            SessionEvent::AudioOut { .. } => "AudioOut",
            SessionEvent::TurnDone { .. } => "TurnDone",
            SessionEvent::Failed { .. } => "Failed",
        }
    }

    /// Control events carry turn structure and are never dropped silently.
    fn is_control(&self) -> bool {
        matches!(
            self,
            SessionEvent::TurnStarted { .. }
                | SessionEvent::TranscriptFinal { .. }
                | SessionEvent::TurnDone { .. }
                | SessionEvent::Failed { .. }
        )
    }

    /// The terminal event reserves its own slot.
    fn is_terminal(&self) -> bool {
        matches!(self, SessionEvent::TurnDone { .. })
    }

    /// Streaming events may be dropped under backpressure; control may not.
    fn is_droppable(&self) -> bool {
        !self.is_control()
    }
}

/// What happened to one [`EventQueue::push`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventPushOutcome {
    /// The event was buffered without evicting anything.
    Accepted,
    /// The event was buffered and an older droppable event was evicted to
    /// make room (counted once as a drop).
    Replaced,
    /// The event was rejected without being buffered (counted once as a drop).
    Dropped,
}

/// Counters for the bounded session event queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EventQueueStats {
    /// Events accepted into the queue.
    pub accepted: u64,
    /// Events lost to overflow (droppable overflow plus control-lane failures).
    pub dropped: u64,
    /// Control events the queue refused, each of which failed its turn loudly.
    pub control_overflow: u64,
    pub depth: usize,
    pub capacity: usize,
}

/// A bounded handoff queue for [`SessionEvent`]s.
///
/// Modeled on [`voicekit_core::AudioRing`]: an explicit capacity, an explicit
/// [`OverflowPolicy`], and a counter for every event that did not reach the
/// queue. Two lanes share the capacity:
///
/// * **Droppable** (`ReplyDelta`, `AudioOut`) may be dropped under
///   backpressure. `DropOldest` evicts the oldest droppable event; `DropNewest`
///   drops the incoming one. Control events are never evicted.
/// * **Control** (`TurnStarted`, `TranscriptFinal`, `Failed`, `TurnDone`) is
///   never dropped: when it cannot be admitted, [`EventQueue::push`] returns
///   [`VoicekitError::Protocol`] so the turn fails loudly and the refusal is
///   counted.
/// * **Terminal** (`TurnDone`) is control *and* owns the otherwise-reserved
///   last slot, so a turn always has room to report how it ended.
pub struct EventQueue {
    capacity: usize,
    policy: OverflowPolicy,
    queue: VecDeque<SessionEvent>,
    accepted: u64,
    dropped: u64,
    control_overflow: u64,
}

impl EventQueue {
    /// Create a bounded queue.
    ///
    /// `capacity` is the total slot count shared by both lanes. Capacities of
    /// zero and one are accepted here because the raw queue is still
    /// well-defined: capacity `0` refuses every event, including the terminal
    /// one, and capacity `1` reserves the single slot for `TurnDone` so no
    /// non-terminal event is ever admitted. Callers that drive a real session
    /// should use at least [`MIN_EVENT_CAPACITY`], which
    /// [`crate::session::Session::new`] enforces.
    pub fn new(capacity: usize, policy: OverflowPolicy) -> Self {
        EventQueue {
            capacity,
            policy,
            queue: VecDeque::with_capacity(capacity.min(4096)),
            accepted: 0,
            dropped: 0,
            control_overflow: 0,
        }
    }

    /// Push one event, applying the overflow policy.
    ///
    /// The producer can branch on the returned [`EventPushOutcome`]: droppable
    /// overflow is counted and reported as `Ok(Replaced | Dropped)`; control
    /// overflow is counted and reported as a protocol error.
    ///
    /// # Errors
    ///
    /// Returns a protocol-class [`VoicekitError`] when a control event cannot
    /// be admitted, so the turn fails loudly instead of silently losing turn
    /// structure.
    pub fn push(&mut self, event: SessionEvent) -> Result<EventPushOutcome, VoicekitError> {
        let reserve = if event.is_terminal() { 0 } else { 1 };
        let limit = self.capacity.saturating_sub(reserve);
        if self.queue.len() < limit {
            self.queue.push_back(event);
            self.accepted += 1;
            return Ok(EventPushOutcome::Accepted);
        }

        if event.is_droppable() {
            match self.policy {
                OverflowPolicy::DropOldest => {
                    if let Some(index) = self.queue.iter().position(SessionEvent::is_droppable) {
                        self.queue.remove(index);
                        self.dropped += 1;
                        self.queue.push_back(event);
                        self.accepted += 1;
                        return Ok(EventPushOutcome::Replaced);
                    }
                }
                OverflowPolicy::DropNewest => {}
            }
            self.dropped += 1;
            return Ok(EventPushOutcome::Dropped);
        }

        // Control lane cannot accept the event: fail the turn loudly rather
        // than losing turn structure silently.
        self.dropped += 1;
        self.control_overflow += 1;
        Err(VoicekitError::Protocol(format!(
            "session event queue control lane full at capacity {}",
            self.capacity
        )))
    }

    /// Drain every buffered event. Cumulative drop counters are preserved.
    pub fn take(&mut self) -> Vec<SessionEvent> {
        self.queue.drain(..).collect()
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Cumulative events dropped by this queue since the session started.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    pub fn stats(&self) -> EventQueueStats {
        EventQueueStats {
            accepted: self.accepted,
            dropped: self.dropped,
            control_overflow: self.control_overflow,
            depth: self.queue.len(),
            capacity: self.capacity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn started() -> SessionEvent {
        SessionEvent::TurnStarted { turn: 1 }
    }

    fn final_transcript() -> SessionEvent {
        SessionEvent::TranscriptFinal {
            turn: 1,
            text: "hello".into(),
        }
    }

    fn delta(text: &str) -> SessionEvent {
        SessionEvent::ReplyDelta {
            turn: 1,
            text: text.into(),
        }
    }

    fn done() -> SessionEvent {
        SessionEvent::TurnDone {
            turn: 1,
            audio_frames: 0,
            dropped_frames: 0,
            dropped_ms: 0,
            dropped_events: 0,
            sidecar_dropped_frames: 0,
            interrupted: false,
        }
    }

    fn kinds(queue: &mut EventQueue) -> Vec<&'static str> {
        queue.take().iter().map(SessionEvent::kind).collect()
    }

    #[test]
    fn capacity_bounds_the_queue_and_counts_every_drop() {
        let mut queue = EventQueue::new(2, OverflowPolicy::DropNewest);
        assert_eq!(queue.push(started()).unwrap(), EventPushOutcome::Accepted);
        // The one non-terminal slot is occupied by a control event, so every
        // droppable is rejected and counted.
        for i in 0..5 {
            assert_eq!(
                queue.push(delta(&i.to_string())).unwrap(),
                EventPushOutcome::Dropped
            );
        }
        assert!(queue.len() <= 2);
        assert_eq!(queue.stats().dropped, 5);
        assert_eq!(queue.stats().control_overflow, 0);
        assert_eq!(kinds(&mut queue), vec!["TurnStarted"]);
    }

    #[test]
    fn drop_oldest_evicts_only_droppable_events() {
        let mut queue = EventQueue::new(3, OverflowPolicy::DropOldest);
        assert_eq!(queue.push(started()).unwrap(), EventPushOutcome::Accepted);
        assert_eq!(queue.push(delta("a")).unwrap(), EventPushOutcome::Accepted);
        // Evicts "a", the oldest droppable, and reports the replacement.
        assert_eq!(queue.push(delta("b")).unwrap(), EventPushOutcome::Replaced);
        assert_eq!(queue.push(done()).unwrap(), EventPushOutcome::Accepted);

        assert_eq!(queue.stats().dropped, 1);
        let events = queue.take();
        assert_eq!(
            events.iter().map(SessionEvent::kind).collect::<Vec<_>>(),
            vec!["TurnStarted", "ReplyDelta", "TurnDone"]
        );
        assert!(matches!(&events[1], SessionEvent::ReplyDelta { text, .. } if text == "b"));
    }

    #[test]
    fn control_events_are_never_dropped_and_overflow_fails_loudly() {
        // Capacity 2 leaves one non-terminal slot plus the terminal reserve.
        let mut queue = EventQueue::new(2, OverflowPolicy::DropOldest);
        assert!(queue.push(started()).is_ok());
        let err = queue.push(final_transcript()).unwrap_err();
        assert!(matches!(err, VoicekitError::Protocol(_)));
        assert_eq!(queue.stats().dropped, 1);
        assert_eq!(queue.stats().control_overflow, 1);
        // The control event that was accepted survived.
        assert_eq!(kinds(&mut queue), vec!["TurnStarted"]);
    }

    #[test]
    fn terminal_event_owns_the_reserved_last_slot() {
        let mut queue = EventQueue::new(1, OverflowPolicy::DropOldest);
        assert!(queue.push(done()).is_ok());
        assert_eq!(queue.len(), 1);
        let err = queue.push(done()).unwrap_err();
        assert!(matches!(err, VoicekitError::Protocol(_)));
        assert_eq!(queue.stats().control_overflow, 1);
    }

    #[test]
    fn capacity_one_reserves_the_terminal_slot() {
        let mut queue = EventQueue::new(1, OverflowPolicy::DropOldest);
        let err = queue.push(started()).unwrap_err();
        assert!(matches!(err, VoicekitError::Protocol(_)));
        assert_eq!(queue.stats().control_overflow, 1);
        assert!(queue.is_empty());

        assert_eq!(queue.push(done()).unwrap(), EventPushOutcome::Accepted);
        assert_eq!(kinds(&mut queue), vec!["TurnDone"]);
    }

    #[test]
    fn zero_capacity_rejects_control_rather_than_growing() {
        let mut queue = EventQueue::new(0, OverflowPolicy::DropOldest);
        let err = queue.push(started()).unwrap_err();
        assert!(matches!(err, VoicekitError::Protocol(_)));
        assert!(queue.is_empty());
        assert_eq!(queue.stats().control_overflow, 1);
    }

    #[test]
    fn take_drains_the_queue_but_keeps_cumulative_drops() {
        let mut queue = EventQueue::new(2, OverflowPolicy::DropNewest);
        queue.push(started()).unwrap();
        for _ in 0..3 {
            queue.push(delta("x")).unwrap();
        }
        assert_eq!(queue.stats().dropped, 3);
        assert_eq!(queue.take().len(), 1);
        assert!(queue.is_empty());
        assert_eq!(queue.stats().dropped, 3);
    }
}
