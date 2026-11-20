//! Optional OpenTelemetry tracing for one turn.
//!
//! Enabled only by the `otel` cargo feature. The runtime never installs a
//! subscriber or exporter itself: it asks the OpenTelemetry *facade* for a
//! tracer (or accepts an injected one through
//! [`crate::Session::with_tracer`]), so a library consumer decides where spans
//! go.
//!
//! Timing authority is the session's [`MonoClock`]. Wall-clock time appears in
//! exactly one place — the UNIX export anchor captured at session construction —
//! and every span start/end is the anchor plus a monotonic offset. That keeps
//! NTP corrections from making stage latencies nonsensical while still giving
//! an exporter usable `SystemTime` values. Spans are always ended with an
//! explicit timestamp; none of the wall-clock convenience methods (`end`,
//! `record_error`, `add_event`) are called on the turn path.
//!
//! The trace is one root `turn` span with `vad`, `stt`, `llm`, `tts`, and
//! `playout` children. Attributes carry only boundary costs, counters, and
//! lengths. Transcript and reply text are never recorded.

use std::sync::Arc;
use std::time::{Duration, SystemTime};

use opentelemetry::global::BoxedTracer;
use opentelemetry::trace::{Span, SpanKind, Status, TraceContextExt, Tracer};
use opentelemetry::{Context, KeyValue};
use parlevox_core::{DurationMs, ErrorClass, MonoClock, ParlevoxError, RingStats, Timestamp};

use crate::events::EventQueueStats;
use crate::session::TurnReport;

/// The single wall-clock capture per session, paired with the monotonic clock
/// it anchors.
///
/// [`TraceClock`] is `Copy`; every copy points at the same anchor and the same
/// [`MonoClock`] epoch, so a guard created early and dropped later measures on
/// one timeline.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TraceClock {
    anchor: SystemTime,
    clock: MonoClock,
}

impl TraceClock {
    /// Capture the export anchor. This is the only `SystemTime::now()` call in
    /// the feature-enabled session path.
    pub(crate) fn new(clock: MonoClock) -> Self {
        TraceClock {
            anchor: SystemTime::now(),
            clock,
        }
    }

    /// The current point on the session's monotonic timeline.
    pub(crate) fn now(&self) -> Timestamp {
        self.clock.now()
    }

    /// Map a monotonic point to the exporter's `SystemTime` timeline.
    pub(crate) fn to_system_time(self, at: Timestamp) -> SystemTime {
        let elapsed = at.since(Timestamp::from_millis(0));
        self.anchor + Duration::from_millis(elapsed.as_millis())
    }
}

/// Root `turn` span plus the monotonic clock used to time its children.
pub(crate) struct TurnTrace {
    tracer: Arc<BoxedTracer>,
    clock: TraceClock,
    cx: Context,
    started: Timestamp,
    ended: bool,
    /// The committed turn index, once known. `None` until the turn reaches
    /// `FinalTranscript` (or the turn ends and its outcome supplies it).
    turn: Option<u64>,
}

impl TurnTrace {
    /// Start the root `turn` span. The turn index is not known yet: a turn can
    /// fail before it commits one, so [`TurnTrace::record_turn`] records it
    /// once the state machine reports it.
    pub(crate) fn start(tracer: &Arc<BoxedTracer>, clock: &TraceClock) -> Self {
        let started = clock.now();
        let span = tracer
            .span_builder("turn")
            .with_kind(SpanKind::Server)
            .with_start_time(clock.to_system_time(started))
            .start_with_context(tracer.as_ref(), &Context::current());
        TurnTrace {
            tracer: Arc::clone(tracer),
            clock: *clock,
            cx: Context::current_with_span(span),
            started,
            ended: false,
            turn: None,
        }
    }

    /// Record the committed turn index on the root span.
    ///
    /// First write wins. The turn body records the index as soon as
    /// `FinalTranscript` commits; the error fallback that runs when a turn
    /// fails earlier only fills in the id the next commit will report, and must
    /// never overwrite an index the turn actually committed.
    pub(crate) fn record_turn(&mut self, turn: u64) {
        if self.turn.is_some() {
            return;
        }
        self.turn = Some(turn);
        self.cx
            .span()
            .set_attribute(KeyValue::new("turn.index", counter(turn)));
    }

    /// Record the depth and capacity of the three bounded handoffs (input,
    /// playout, event queue) as sampled at turn end.
    pub(crate) fn record_boundaries(
        &self,
        input: &RingStats,
        playout: &RingStats,
        events: &EventQueueStats,
    ) {
        self.cx
            .span()
            .set_attributes(boundary_attributes(input, playout, events));
    }

    /// Start a child stage span rooted at the turn span.
    pub(crate) fn stage(&self, name: &'static str, started: Timestamp) -> StageSpan {
        let span = self
            .tracer
            .span_builder(name)
            .with_kind(SpanKind::Internal)
            .with_start_time(self.clock.to_system_time(started))
            .start_with_context(self.tracer.as_ref(), &self.cx);
        StageSpan {
            span,
            clock: self.clock,
            started,
            ended: false,
        }
    }

    /// Set an attribute on the root span, including boundary costs discovered
    /// mid-turn (for example the wait for the first reply delta).
    pub(crate) fn set_attribute(&self, attribute: KeyValue) {
        self.cx.span().set_attribute(attribute);
    }

    /// Close the root span with the turn's outcome and boundary counters.
    ///
    /// Boundary attributes come from the existing [`TurnReport`] counters plus
    /// text lengths; no second source of truth is introduced.
    pub(crate) fn finish(&mut self, result: &Result<TurnReport, ParlevoxError>) {
        if self.ended {
            return;
        }
        self.ended = true;
        // A turn that ended before `FinalTranscript` (a pre-commit barge-in)
        // still has a `TurnReport`; record its id so the span always matches
        // the report. The first-write-wins rule keeps the committed id intact.
        if let Ok(report) = result {
            self.record_turn(report.turn);
        }
        let span = self.cx.span();
        match result {
            Ok(report) => {
                span.set_attributes(report_attributes(report));
                if report.interrupted {
                    span.set_attribute(KeyValue::new("turn.interrupted", true));
                }
                span.set_status(Status::Ok);
            }
            Err(err) => {
                // Only the classified error name is exported; provider error
                // messages may embed user content and are kept out of spans.
                let class = class_name(err);
                span.set_attribute(KeyValue::new("turn.error_class", class));
                span.set_status(Status::error(class));
            }
        }
        let ended = self.clock.now();
        span.set_attribute(KeyValue::new(
            "duration.monotonic_ms",
            millis(ended.since(self.started)),
        ));
        span.end_with_timestamp(self.clock.to_system_time(ended));
    }
}

impl Drop for TurnTrace {
    fn drop(&mut self) {
        if self.ended {
            return;
        }
        self.ended = true;
        let now = self.clock.now();
        self.cx
            .span()
            .end_with_timestamp(self.clock.to_system_time(now));
    }
}

/// RAII guard for one stage span.
///
/// An explicit [`StageSpan::end`] closes the span with a monotonic end
/// timestamp. If the turn returns early (a provider error or a barge-in) the
/// guard's `Drop` closes it the same way, so a failure path never leaks an open
/// span and never falls back to OpenTelemetry's wall clock.
pub(crate) struct StageSpan {
    span: opentelemetry::global::BoxedSpan,
    clock: TraceClock,
    started: Timestamp,
    ended: bool,
}

impl StageSpan {
    /// Record an attribute discovered while the stage runs.
    pub(crate) fn set_attribute(&mut self, attribute: KeyValue) {
        self.span.set_attribute(attribute);
    }

    /// Close the span now, recording the measured duration.
    pub(crate) fn end(mut self) {
        self.finish();
    }

    fn finish(&mut self) {
        if self.ended {
            return;
        }
        self.ended = true;
        let ended = self.clock.now();
        self.span.set_attribute(KeyValue::new(
            "stage.duration_ms",
            millis(ended.since(self.started)),
        ));
        self.span
            .end_with_timestamp(self.clock.to_system_time(ended));
    }
}

impl Drop for StageSpan {
    fn drop(&mut self) {
        self.finish();
    }
}

/// Boundary occupancy at turn end, read from the same stats structs a caller
/// observes ([`parlevox_core::AudioRing::stats`] and
/// [`crate::events::EventQueue::stats`]) so the span never invents a second
/// source of truth.
fn boundary_attributes(
    input: &RingStats,
    playout: &RingStats,
    events: &EventQueueStats,
) -> Vec<KeyValue> {
    // A fixed six-element literal allocates exactly once, so this satisfies
    // the known-size case without a `Vec::with_capacity` + `push` sequence.
    vec![
        KeyValue::new("ring.input.depth", counter(to_u64(input.depth))),
        KeyValue::new("ring.input.capacity", counter(to_u64(input.capacity))),
        KeyValue::new("ring.playout.depth", counter(to_u64(playout.depth))),
        KeyValue::new("ring.playout.capacity", counter(to_u64(playout.capacity))),
        KeyValue::new("events.depth", counter(to_u64(events.depth))),
        KeyValue::new("events.capacity", counter(to_u64(events.capacity))),
    ]
}

fn report_attributes(report: &TurnReport) -> Vec<KeyValue> {
    vec![
        KeyValue::new("turn.audio_frames", counter(report.audio_frames)),
        KeyValue::new("turn.dropped_frames", counter(report.dropped_frames)),
        KeyValue::new("turn.dropped_ms", counter(report.dropped_ms)),
        KeyValue::new("turn.dropped_events", counter(report.dropped_events)),
        KeyValue::new(
            "turn.sidecar_dropped_frames",
            counter(report.sidecar_dropped_frames),
        ),
        KeyValue::new(
            "turn.input_dropped_frames",
            counter(report.input_dropped_frames),
        ),
        KeyValue::new(
            "turn.context_messages",
            counter(to_u64(report.context_messages)),
        ),
        KeyValue::new(
            "turn.context_evicted_messages",
            counter(report.context_evicted_messages),
        ),
        KeyValue::new(
            "transcript.chars",
            counter(to_u64(report.transcript.chars().count())),
        ),
        KeyValue::new("reply.chars", counter(to_u64(report.reply.chars().count()))),
    ]
}

fn class_name(err: &ParlevoxError) -> &'static str {
    match err.class() {
        ErrorClass::Transient => "transient",
        ErrorClass::Fatal => "fatal",
        ErrorClass::Protocol => "protocol",
        ErrorClass::Cancelled => "cancelled",
        ErrorClass::Capacity => "capacity",
    }
}

/// Widening is lossless on every target this runtime is built for; the
/// fallback keeps a hostile 128-bit target from wrapping silently.
pub(crate) fn to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

pub(crate) fn counter(value: u64) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

pub(crate) fn millis(value: DurationMs) -> i64 {
    i64::try_from(value.as_millis()).unwrap_or(i64::MAX)
}
