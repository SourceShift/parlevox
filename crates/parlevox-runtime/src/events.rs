//! Typed session events.
//!
//! These are the only things a transport (CLI, websocket, test harness) needs
//! to render a turn. Audio is carried as an owned [`AudioFrame`], never as
//! base64 or whole-turn JSON.

use std::collections::VecDeque;

use parlevox_core::{
    AudioFrame, ErrorClass, Observation, ObservationKind, ObservationSequencer, OverflowPolicy,
    ParlevoxError, TurnEventKind,
};
use parlevox_providers::{TranscriptChunk, VadEvent, VadTransition};
use serde::{Serialize, Serializer};

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

/// Private externally tagged mirror of [`SessionEvent`] used by its [`Serialize`]
/// impl.
///
/// Variant names are preserved exactly, and [`SessionEvent::AudioOut`] is
/// reduced to its format and sample count: the PCM payload is not part of the
/// observation/trace form. No existing consumer deserializes `SessionEvent`, so
/// this shape is additive rather than a wire change.
#[derive(Serialize)]
enum SessionEventWire<'a> {
    TurnStarted {
        turn: u64,
    },
    TranscriptFinal {
        turn: u64,
        text: &'a str,
    },
    ReplyDelta {
        turn: u64,
        text: &'a str,
    },
    AudioOut {
        turn: u64,
        sample_rate: u32,
        channels: u16,
        pcm_samples: u64,
    },
    TurnDone {
        turn: u64,
        audio_frames: u64,
        dropped_frames: u64,
        dropped_ms: u64,
        dropped_events: u64,
        sidecar_dropped_frames: u64,
        interrupted: bool,
    },
    Failed {
        class: ErrorClass,
        message: &'a str,
    },
}

impl Serialize for SessionEvent {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let wire = match self {
            SessionEvent::TurnStarted { turn } => SessionEventWire::TurnStarted { turn: *turn },
            SessionEvent::TranscriptFinal { turn, text } => {
                SessionEventWire::TranscriptFinal { turn: *turn, text }
            }
            SessionEvent::ReplyDelta { turn, text } => {
                SessionEventWire::ReplyDelta { turn: *turn, text }
            }
            SessionEvent::AudioOut { turn, frame } => SessionEventWire::AudioOut {
                turn: *turn,
                sample_rate: frame.sample_rate,
                channels: frame.channels,
                pcm_samples: u64::try_from(frame.pcm.len()).unwrap_or(u64::MAX),
            },
            SessionEvent::TurnDone {
                turn,
                audio_frames,
                dropped_frames,
                dropped_ms,
                dropped_events,
                sidecar_dropped_frames,
                interrupted,
            } => SessionEventWire::TurnDone {
                turn: *turn,
                audio_frames: *audio_frames,
                dropped_frames: *dropped_frames,
                dropped_ms: *dropped_ms,
                dropped_events: *dropped_events,
                sidecar_dropped_frames: *sidecar_dropped_frames,
                interrupted: *interrupted,
            },
            SessionEvent::Failed { class, message } => SessionEventWire::Failed {
                class: *class,
                message,
            },
        };
        wire.serialize(serializer)
    }
}

/// A [`SessionEvent`] paired with the monotonic millisecond it was observed at.
///
/// `t_ms` is mandatory — never optional and never skipped — because a trace line
/// with a missing timestamp silently corrupts every latency computation built
/// on top of it. The timestamp is a companion field rather than a field on
/// [`SessionEvent`] itself because the transports destructure every session
/// variant exhaustively; adding a field would break them for no behavioural
/// gain.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TimedSessionEvent {
    /// Milliseconds elapsed on the session's monotonic clock.
    pub t_ms: u64,
    /// The event observed at `t_ms`.
    pub event: SessionEvent,
}

/// A provider or session input in the form the trace adapter consumes.
///
/// This adapter normalizes the runtime's existing inputs to
/// [`ObservationKind`] so one code path can record them. It is **inert** in the
/// current slice: the live session does not call it, and wiring it into the
/// turn loop is TS-4. Keeping it here, unused, is deliberate — it fixes the
/// normalization contract before the session depends on it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IngressInput<'a> {
    /// One streaming STT result; final and partial both normalize here.
    Transcript(&'a TranscriptChunk),
    /// One voice-activity transition.
    Vad(VadTransition),
    /// One streamed reply delta.
    ReplyDelta(&'a str),
    /// One audio frame queued for playout.
    AudioFrame(&'a AudioFrame),
    /// The reply finished.
    ReplyFinished,
    /// A barge-in was reported.
    BargeIn,
    /// A cancellation was observed.
    Cancellation,
    /// A classified failure.
    Failure {
        /// Recoverability class of the failure.
        class: ErrorClass,
        /// Human-readable failure message.
        message: &'a str,
    },
    /// A turn-machine event observed at the ingress boundary.
    Control(TurnEventKind),
}

impl IngressInput<'_> {
    /// The observation kind this input records.
    #[must_use]
    pub fn kind(&self) -> ObservationKind {
        match self {
            IngressInput::Transcript(chunk) => {
                if chunk.is_final {
                    ObservationKind::FinalTranscript {
                        text: chunk.text.clone(),
                    }
                } else {
                    ObservationKind::PartialTranscript {
                        text: chunk.text.clone(),
                    }
                }
            }
            IngressInput::Vad(transition) => match transition.event {
                VadEvent::SpeechStart => ObservationKind::VadStarted,
                VadEvent::SpeechEnd => ObservationKind::VadStopped,
            },
            IngressInput::ReplyDelta(text) => ObservationKind::ReplyDelta {
                text: (*text).to_owned(),
            },
            IngressInput::AudioFrame(frame) => ObservationKind::AudioFrameQueued {
                sample_rate: frame.sample_rate,
                channels: frame.channels,
                pcm_samples: u64::try_from(frame.pcm.len()).unwrap_or(u64::MAX),
            },
            IngressInput::ReplyFinished => ObservationKind::ReplyFinished,
            IngressInput::BargeIn => ObservationKind::BargeInReported,
            IngressInput::Cancellation => ObservationKind::CancellationObserved,
            IngressInput::Failure { class, message } => ObservationKind::Failure {
                class: *class,
                message: (*message).to_owned(),
            },
            IngressInput::Control(kind) => ObservationKind::Control { event: *kind },
        }
    }

    /// Normalize this input and stamp it into an [`Observation`].
    #[must_use]
    pub fn observe(
        &self,
        sequencer: &mut ObservationSequencer,
        turn: u64,
        generation: u64,
    ) -> Observation {
        sequencer.stamp(turn, generation, self.kind())
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
/// Modeled on [`parlevox_core::AudioRing`]: an explicit capacity, an explicit
/// [`OverflowPolicy`], and a counter for every event that did not reach the
/// queue. Two lanes share the capacity:
///
/// * **Droppable** (`ReplyDelta`, `AudioOut`) may be dropped under
///   backpressure. `DropOldest` evicts the oldest droppable event; `DropNewest`
///   drops the incoming one. Control events are never evicted.
/// * **Control** (`TurnStarted`, `TranscriptFinal`, `Failed`, `TurnDone`) is
///   never dropped: when it cannot be admitted, [`EventQueue::push`] returns
///   [`ParlevoxError::Protocol`] so the turn fails loudly and the refusal is
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
    /// Returns a protocol-class [`ParlevoxError`] when a control event cannot
    /// be admitted, so the turn fails loudly instead of silently losing turn
    /// structure.
    pub fn push(&mut self, event: SessionEvent) -> Result<EventPushOutcome, ParlevoxError> {
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
        Err(ParlevoxError::Protocol(format!(
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
        assert!(matches!(err, ParlevoxError::Protocol(_)));
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
        assert!(matches!(err, ParlevoxError::Protocol(_)));
        assert_eq!(queue.stats().control_overflow, 1);
    }

    #[test]
    fn capacity_one_reserves_the_terminal_slot() {
        let mut queue = EventQueue::new(1, OverflowPolicy::DropOldest);
        let err = queue.push(started()).unwrap_err();
        assert!(matches!(err, ParlevoxError::Protocol(_)));
        assert_eq!(queue.stats().control_overflow, 1);
        assert!(queue.is_empty());

        assert_eq!(queue.push(done()).unwrap(), EventPushOutcome::Accepted);
        assert_eq!(kinds(&mut queue), vec!["TurnDone"]);
    }

    #[test]
    fn zero_capacity_rejects_control_rather_than_growing() {
        let mut queue = EventQueue::new(0, OverflowPolicy::DropOldest);
        let err = queue.push(started()).unwrap_err();
        assert!(matches!(err, ParlevoxError::Protocol(_)));
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

    #[test]
    fn session_event_is_serializable() {
        fn assert_serialize<T: serde::Serialize>() {}
        assert_serialize::<SessionEvent>();
        assert_serialize::<TimedSessionEvent>();
    }

    #[test]
    fn timed_session_event_requires_a_timestamp() {
        // `t_ms` is a plain `u64`, not an `Option`, and the struct has no
        // `Default`: constructing a timed event without a timestamp is a
        // compile error, so a trace line can never silently lose it.
        let timed = TimedSessionEvent {
            t_ms: 12,
            event: started(),
        };
        assert_eq!(timed.t_ms, 12);
        assert_eq!(timed.event, started());
    }

    #[test]
    fn ingress_inputs_normalize_to_kinds() {
        let final_chunk = TranscriptChunk {
            text: "hello".into(),
            is_final: true,
        };
        let partial_chunk = TranscriptChunk {
            text: "hel".into(),
            is_final: false,
        };
        let frame = AudioFrame::new(
            parlevox_core::Timestamp::from_millis(0),
            16_000,
            1,
            vec![0i16; 160],
        );

        let cases = [
            (
                IngressInput::Transcript(&final_chunk),
                ObservationKind::FinalTranscript {
                    text: "hello".into(),
                },
            ),
            (
                IngressInput::Transcript(&partial_chunk),
                ObservationKind::PartialTranscript { text: "hel".into() },
            ),
            (
                IngressInput::Vad(VadTransition {
                    event: VadEvent::SpeechStart,
                    timestamp_ms: 5,
                }),
                ObservationKind::VadStarted,
            ),
            (
                IngressInput::Vad(VadTransition {
                    event: VadEvent::SpeechEnd,
                    timestamp_ms: 9,
                }),
                ObservationKind::VadStopped,
            ),
            (
                IngressInput::ReplyDelta("hi"),
                ObservationKind::ReplyDelta { text: "hi".into() },
            ),
            (
                IngressInput::AudioFrame(&frame),
                ObservationKind::AudioFrameQueued {
                    sample_rate: 16_000,
                    channels: 1,
                    pcm_samples: 160,
                },
            ),
            (IngressInput::ReplyFinished, ObservationKind::ReplyFinished),
            (IngressInput::BargeIn, ObservationKind::BargeInReported),
            (
                IngressInput::Cancellation,
                ObservationKind::CancellationObserved,
            ),
            (
                IngressInput::Failure {
                    class: ErrorClass::Fatal,
                    message: "boom",
                },
                ObservationKind::Failure {
                    class: ErrorClass::Fatal,
                    message: "boom".into(),
                },
            ),
            (
                IngressInput::Control(TurnEventKind::BargeInConfirmed),
                ObservationKind::Control {
                    event: TurnEventKind::BargeInConfirmed,
                },
            ),
        ];

        for (input, expected) in cases {
            assert_eq!(input.kind(), expected, "normalizing {input:?}");
        }
    }
}
