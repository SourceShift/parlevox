//! The session actor.
//!
//! One session owns exactly one [`TurnMachine`], one [`MonoClock`], bounded
//! input and output [`AudioRing`]s, and the resolved providers. A turn drives
//! VAD → STT → LLM → TTS in sequence, emitting typed events as work completes.
//!
//! Barge-in is deterministic: [`Session::request_barge_in`] cancels the turn's
//! [`CancellationToken`] and increments the generation id. Every provider call
//! in the turn holds that token, so an in-flight stream observes cancellation
//! on its next poll instead of racing a timer. Items from an older generation
//! are never emitted.

use std::sync::Arc;

use futures_util::StreamExt;
use parlevox_core::{
    AudioRing, MonoClock, OverflowPolicy, ParlevoxError, PushOutcome, RingStats, TurnEvent,
    TurnMachine, TurnState,
};
use parlevox_providers::{
    AudioChunk, GenerateRequest, ProviderManifest, SynthesizeRequest, VadEvent,
};
use tokio_util::sync::CancellationToken;

use crate::context::{ChatContext, ContextStats};
use crate::events::{
    EventPushOutcome, EventQueue, EventQueueStats, SessionEvent, MIN_EVENT_CAPACITY,
};
use crate::registry::Providers;
#[cfg(feature = "otel")]
use crate::telemetry::{counter, millis, to_u64, TraceClock, TurnTrace};
#[cfg(feature = "otel")]
use opentelemetry::KeyValue;

/// Tunables for one session.
#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub sample_rate: u32,
    pub channels: u16,
    /// Bounded frame capacity for the input path.
    pub input_ring_capacity: usize,
    /// Bounded frame capacity for the playout path.
    pub output_ring_capacity: usize,
    /// Bounded event capacity for the session-to-transport handoff.
    pub event_capacity: usize,
    /// Explicit overflow policy for droppable events.
    pub event_overflow_policy: OverflowPolicy,
    /// Hard cap on the number of messages retained across turns.
    pub context_messages: usize,
    /// Hard cap on `text.chars().count()` retained across turns. This is a
    /// transport-safe estimate, not a model token budget.
    pub context_chars: usize,
    /// Test hook: trigger a barge-in after this many reply deltas.
    pub barge_in_after_deltas: Option<usize>,
}

impl Default for SessionConfig {
    fn default() -> Self {
        SessionConfig {
            sample_rate: 16_000,
            channels: 1,
            input_ring_capacity: 32,
            output_ring_capacity: 64,
            event_capacity: 256,
            event_overflow_policy: OverflowPolicy::DropOldest,
            context_messages: 16,
            context_chars: 8_192,
            barge_in_after_deltas: None,
        }
    }
}

/// What one turn produced, independent of how it was rendered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnReport {
    pub turn: u64,
    pub transcript: String,
    pub reply: String,
    pub audio_frames: u64,
    pub dropped_frames: u64,
    pub dropped_ms: u64,
    /// Events dropped from the bounded handoff queue during this turn.
    pub dropped_events: u64,
    /// Frames the sidecar plugin reported dropping. Distinct from
    /// `dropped_frames`, which is the Rust playout ring.
    pub sidecar_dropped_frames: u64,
    pub input_dropped_frames: u64,
    pub interrupted: bool,
    /// Message depth of the bounded conversation context at turn end.
    pub context_messages: usize,
    /// Context messages evicted during this turn, as a delta from the depth
    /// captured at turn start.
    pub context_evicted_messages: u64,
}

/// Counters captured when a turn starts, so per-turn drops are deltas from a
/// cumulative baseline rather than session totals.
#[derive(Debug, Clone, Copy)]
struct TurnBaseline {
    output: RingStats,
    input_dropped: u64,
    events_dropped: u64,
    context_evicted_messages: u64,
}

pub struct Session {
    config: SessionConfig,
    machine: TurnMachine,
    clock: MonoClock,
    input: AudioRing,
    output: AudioRing,
    providers: Providers,
    cancel: CancellationToken,
    generation: u64,
    events: EventQueue,
    context: ChatContext,
    /// Facade tracer used by the opt-in `otel` feature. Absent from the
    /// default build, so the default session carries no OpenTelemetry state.
    #[cfg(feature = "otel")]
    tracer: Arc<opentelemetry::global::BoxedTracer>,
    /// One wall-clock export anchor plus the session's monotonic clock.
    #[cfg(feature = "otel")]
    trace_clock: TraceClock,
}

impl Session {
    /// Build a session from a validated provider set.
    ///
    /// # Errors
    ///
    /// Returns a protocol-class error when a provider manifest is invalid, when
    /// `event_capacity` is below [`MIN_EVENT_CAPACITY`] (the minimum needed to
    /// reserve the terminal slot plus streaming headroom), or when
    /// `context_messages` / `context_chars` cannot hold one turn.
    pub fn new(config: SessionConfig, providers: Providers) -> Result<Self, ParlevoxError> {
        #[cfg(feature = "otel")]
        {
            Self::assemble(
                config,
                providers,
                Arc::new(opentelemetry::global::tracer("parlevox-runtime")),
            )
        }
        #[cfg(not(feature = "otel"))]
        {
            Self::assemble(config, providers)
        }
    }

    /// Build a session with an explicitly supplied tracer.
    ///
    /// This is the hermetic hook for tests: pass a tracer backed by an
    /// in-memory exporter and no process-global provider is touched. It is
    /// only available with the `otel` feature.
    ///
    /// # Errors
    ///
    /// Returns the same protocol-class errors as [`Session::new`].
    #[cfg(feature = "otel")]
    pub fn with_tracer<T>(
        config: SessionConfig,
        providers: Providers,
        tracer: T,
    ) -> Result<Self, ParlevoxError>
    where
        T: opentelemetry::trace::Tracer + Send + Sync + 'static,
        T::Span: Send + Sync + 'static,
    {
        let tracer = opentelemetry::global::BoxedTracer::new(Box::new(tracer));
        Self::assemble(config, providers, Arc::new(tracer))
    }

    fn assemble(
        config: SessionConfig,
        providers: Providers,
        #[cfg(feature = "otel")] tracer: Arc<opentelemetry::global::BoxedTracer>,
    ) -> Result<Self, ParlevoxError> {
        for manifest in [
            providers.stt.manifest(),
            providers.llm.manifest(),
            providers.tts.manifest(),
            providers.vad.manifest(),
        ] {
            manifest.validate()?;
        }
        if config.event_capacity < MIN_EVENT_CAPACITY {
            return Err(ParlevoxError::Protocol(format!(
                "event_capacity {} is below MIN_EVENT_CAPACITY ({MIN_EVENT_CAPACITY}): \
                 one slot reserves the terminal event and one carries streaming \
                 or control events",
                config.event_capacity
            )));
        }
        // Every audio, event, and context path is bounded by an explicit
        // capacity; the context's own constructor validates its two bounds.
        let input = AudioRing::new(config.input_ring_capacity, OverflowPolicy::DropOldest);
        let output = AudioRing::new(config.output_ring_capacity, OverflowPolicy::DropOldest);
        let events = EventQueue::new(config.event_capacity, config.event_overflow_policy);
        let context = ChatContext::new(config.context_messages, config.context_chars)?;
        let clock = MonoClock::new();
        Ok(Session {
            config,
            machine: TurnMachine::new(),
            clock,
            input,
            output,
            providers,
            cancel: CancellationToken::new(),
            generation: 0,
            events,
            context,
            #[cfg(feature = "otel")]
            tracer,
            #[cfg(feature = "otel")]
            trace_clock: TraceClock::new(clock),
        })
    }

    pub fn state(&self) -> TurnState {
        self.machine.state()
    }

    pub fn turn_index(&self) -> u64 {
        self.machine.turn_index()
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn input_stats(&self) -> RingStats {
        self.input.stats()
    }

    pub fn output_stats(&self) -> RingStats {
        self.output.stats()
    }

    /// Depth and cumulative loss counters for the bounded conversation context.
    pub fn context_stats(&self) -> ContextStats {
        self.context.stats()
    }

    pub fn provider_manifests(&self) -> Vec<ProviderManifest> {
        vec![
            self.providers.stt.manifest().clone(),
            self.providers.llm.manifest().clone(),
            self.providers.tts.manifest().clone(),
            self.providers.vad.manifest().clone(),
        ]
    }

    /// Apply a turn-state event directly. Illegal events surface as
    /// [`ParlevoxError::Protocol`]; they are never swallowed.
    ///
    /// # Errors
    ///
    /// Returns a protocol-class error for an illegal transition and a
    /// closed-class error once the session is terminal.
    pub fn apply(
        &mut self,
        event: TurnEvent,
    ) -> Result<parlevox_core::turn::Transition, ParlevoxError> {
        self.machine.apply(event)
    }

    /// Open the session (Idle → Listening).
    ///
    /// # Errors
    ///
    /// Returns a closed-class error when the session has already terminated,
    /// or a protocol-class error if the state machine rejects `Start`.
    pub fn start(&mut self) -> Result<(), ParlevoxError> {
        if self.machine.state() == TurnState::Idle {
            self.machine.apply(TurnEvent::Start)?;
        }
        Ok(())
    }

    pub fn take_events(&mut self) -> Vec<SessionEvent> {
        self.events.take()
    }

    /// Counters for the bounded session-to-transport event handoff.
    pub fn event_stats(&self) -> EventQueueStats {
        self.events.stats()
    }

    /// Cancel the in-flight turn. Returns `true` when a turn was interrupted.
    ///
    /// The generation id advances and the token fires, so provider streams
    /// stop on their next poll rather than being killed from outside.
    ///
    /// # Errors
    ///
    /// Returns a protocol-class error if the state machine rejects the
    /// barge-in transition; an idle or listening session simply returns
    /// `Ok(false)`.
    pub fn request_barge_in(&mut self) -> Result<bool, ParlevoxError> {
        let state = self.machine.state();
        if !matches!(state, TurnState::Thinking | TurnState::Speaking) {
            return Ok(false);
        }
        self.machine.apply(TurnEvent::BargeIn)?;
        self.generation += 1;
        self.cancel.cancel();
        self.machine.apply(TurnEvent::ResumeListening)?;
        Ok(true)
    }

    /// Drive one complete turn: VAD, STT, LLM, TTS, playout.
    ///
    /// # Errors
    ///
    /// Returns a protocol-class error when a provider produces unusable output
    /// or a turn transition is illegal, a closed-class error when the session
    /// is terminal, and the classified provider or transport error when a
    /// provider call fails. Control-event overflow is reported the same way.
    pub async fn run_turn(&mut self, utterance: AudioChunk) -> Result<TurnReport, ParlevoxError> {
        #[cfg(feature = "otel")]
        {
            let mut trace = TurnTrace::start(&self.tracer, &self.trace_clock);
            let result = self.drive_turn(utterance, &mut trace).await;
            // A turn that fails before `FinalTranscript` has no commit and so
            // no report id; carry the id the next commit will report instead.
            // An already-recorded committed id is never overwritten.
            if result.is_err() {
                trace.record_turn(self.turn_index().saturating_add(1));
            }
            // Sample the bounded handoffs after the turn body has finished and
            // before the span closes, so the values match what a caller reads
            // from the session immediately after `run_turn`.
            trace.record_boundaries(
                &self.input.stats(),
                &self.output.stats(),
                &self.events.stats(),
            );
            trace.finish(&result);
            result
        }
        #[cfg(not(feature = "otel"))]
        {
            self.drive_turn(utterance).await
        }
    }

    /// The turn body. Kept separate from [`Session::run_turn`] so the optional
    /// trace can be closed with the turn's final outcome, including on the
    /// error and interruption paths.
    async fn drive_turn(
        &mut self,
        utterance: AudioChunk,
        #[cfg(feature = "otel")] trace: &mut TurnTrace,
    ) -> Result<TurnReport, ParlevoxError> {
        self.start()?;
        if self.machine.state() == TurnState::Closed {
            return Err(ParlevoxError::Closed);
        }

        let token = CancellationToken::new();
        self.cancel = token.clone();
        self.generation += 1;
        let generation = self.generation;

        let baseline = TurnBaseline {
            output: self.output.stats(),
            input_dropped: self.input.stats().dropped,
            events_dropped: self.events.dropped(),
            context_evicted_messages: self.context.stats().evicted_messages,
        };

        // Bounded input path: one frame carrying the utterance. The outcome is
        // explicit so a zero-capacity ring reports loss instead of hiding it.
        let input_ts = self.clock.now();
        match self.input.push(utterance.to_frame(input_ts)) {
            PushOutcome::Accepted | PushOutcome::Replaced => {}
            PushOutcome::Dropped => {
                tracing::trace!(
                    target: "parlevox_runtime",
                    "input ring rejected the utterance frame"
                );
            }
        }

        // 1. Voice activity.
        #[cfg(feature = "otel")]
        let vad_span = trace.stage("vad", self.clock.now());
        let vad_provider = Arc::clone(&self.providers.vad);
        let mut vad = match vad_provider.analyze(utterance.clone(), token.clone()).await {
            Ok(stream) => stream,
            Err(err) => return self.fail(err),
        };
        let mut speech_started = false;
        while let Some(item) = vad.next().await {
            if token.is_cancelled() {
                return self.finish_interrupted(0, 0, "", "", baseline);
            }
            match item {
                Ok(transition) => match transition.event {
                    VadEvent::SpeechStart => {
                        if !speech_started {
                            self.apply(TurnEvent::UserStartedSpeaking)?;
                            speech_started = true;
                        }
                    }
                    VadEvent::SpeechEnd => {
                        if speech_started {
                            self.apply(TurnEvent::UserStoppedSpeaking)?;
                            speech_started = false;
                        }
                    }
                },
                Err(ParlevoxError::Cancelled) => {
                    return self.finish_interrupted(0, 0, "", "", baseline);
                }
                Err(err) => return self.fail(err),
            }
        }
        if !speech_started && self.machine.state() == TurnState::Listening {
            self.apply(TurnEvent::UserStartedSpeaking)?;
        }
        #[cfg(feature = "otel")]
        vad_span.end();

        // 2. Speech-to-text, consuming partials until the final.
        #[cfg(feature = "otel")]
        let mut stt_span = trace.stage("stt", self.clock.now());
        let stt_provider = Arc::clone(&self.providers.stt);
        let mut stt = match stt_provider.transcribe(utterance, token.clone()).await {
            Ok(stream) => stream,
            Err(err) => return self.fail(err),
        };
        let mut transcript = String::new();
        let mut saw_final = false;
        while let Some(item) = stt.next().await {
            match item {
                Ok(chunk) => {
                    if chunk.is_final {
                        transcript = chunk.text;
                        saw_final = true;
                    }
                }
                Err(ParlevoxError::Cancelled) => {
                    return self.finish_interrupted(0, 0, "", "", baseline);
                }
                Err(err) => return self.fail(err),
            }
        }
        if !saw_final || transcript.trim().is_empty() {
            return self.fail(ParlevoxError::Protocol(
                "stt produced no final transcript".into(),
            ));
        }
        #[cfg(feature = "otel")]
        {
            stt_span.set_attribute(KeyValue::new(
                "transcript.chars",
                counter(to_u64(transcript.chars().count())),
            ));
            stt_span.end();
        }

        let transition = self.apply(TurnEvent::FinalTranscript {
            text: transcript.clone(),
        })?;
        // The turn is committed here: record its id before any later failure
        // or barge-in can only describe how the committed turn ended.
        #[cfg(feature = "otel")]
        trace.record_turn(transition.turn);
        let turn = transition.turn;
        self.emit(SessionEvent::TurnStarted { turn })?;
        self.emit(SessionEvent::TranscriptFinal {
            turn,
            text: transcript.clone(),
        })?;

        // The committed utterance becomes the next request's prompt. Snapshot
        // the prior turns *before* recording it so the current utterance
        // appears exactly once: as `prompt`, never duplicated inside `history`.
        let history = self.context.messages();
        self.context.push_user(transcript.clone());

        // 3. Streaming completion.
        #[cfg(feature = "otel")]
        let llm_started = self.clock.now();
        #[cfg(feature = "otel")]
        let mut llm_span = trace.stage("llm", llm_started);
        let llm_provider = Arc::clone(&self.providers.llm);
        let mut llm = match llm_provider
            .generate(
                GenerateRequest {
                    prompt: transcript.clone(),
                    history,
                },
                token.clone(),
            )
            .await
        {
            Ok(stream) => stream,
            Err(err) => return self.fail(err),
        };
        let mut reply = String::new();
        let mut reply_started = false;
        let mut delta_count = 0usize;
        let mut interrupted = false;
        loop {
            if token.is_cancelled() || self.generation != generation {
                interrupted = true;
                break;
            }
            match llm.next().await {
                None => break,
                Some(Ok(delta)) => {
                    if !reply_started {
                        self.apply(TurnEvent::ReplyStarted)?;
                        reply_started = true;
                    }
                    #[cfg(feature = "otel")]
                    if delta_count == 0 {
                        trace.set_attribute(KeyValue::new(
                            "llm.first_delta_ms",
                            millis(self.clock.now().since(llm_started)),
                        ));
                    }
                    reply.push_str(&delta);
                    delta_count += 1;
                    self.emit(SessionEvent::ReplyDelta { turn, text: delta })?;
                    if self.config.barge_in_after_deltas == Some(delta_count) {
                        self.request_barge_in()?;
                        interrupted = true;
                        break;
                    }
                }
                Some(Err(ParlevoxError::Cancelled)) => {
                    interrupted = true;
                    break;
                }
                Some(Err(err)) => return self.fail(err),
            }
        }
        #[cfg(feature = "otel")]
        {
            llm_span.set_attribute(KeyValue::new(
                "reply.chars",
                counter(to_u64(reply.chars().count())),
            ));
            llm_span.set_attribute(KeyValue::new("reply.deltas", counter(to_u64(delta_count))));
            llm_span.end();
        }
        if !reply_started && !interrupted {
            return self.fail(ParlevoxError::Protocol(
                "llm produced no reply deltas".into(),
            ));
        }
        if interrupted {
            return self.finish_interrupted(turn, 0, &transcript, &reply, baseline);
        }

        // 4. Streaming synthesis with bounded playout. Producing audio is a
        // bounded operation: overflow drops the oldest frame and is counted.
        #[cfg(feature = "otel")]
        let tts_started = self.clock.now();
        #[cfg(feature = "otel")]
        let mut tts_span = trace.stage("tts", tts_started);
        #[cfg(feature = "otel")]
        let mut playout_span = trace.stage("playout", tts_started);
        let tts_provider = Arc::clone(&self.providers.tts);
        let request = SynthesizeRequest {
            text: reply.clone(),
            sample_rate: self.config.sample_rate,
            channels: self.config.channels,
        };
        let mut tts = match tts_provider.synthesize(request, token.clone()).await {
            Ok(stream) => stream,
            Err(err) => return self.fail(err),
        };
        let mut audio_frames = 0u64;
        loop {
            if token.is_cancelled() || self.generation != generation {
                interrupted = true;
                break;
            }
            match tts.next().await {
                None => break,
                Some(Ok(chunk)) => {
                    #[cfg(feature = "otel")]
                    if audio_frames == 0 {
                        trace.set_attribute(KeyValue::new(
                            "tts.first_audio_ms",
                            millis(self.clock.now().since(tts_started)),
                        ));
                    }
                    let ts = self.clock.now();
                    // Branch on the bounded playout ring's outcome: a frame the
                    // ring rejects is counted inside the ring, but the producer
                    // must still observe that the drop happened.
                    match self.output.push(chunk.to_frame(ts)) {
                        PushOutcome::Accepted | PushOutcome::Replaced => {}
                        PushOutcome::Dropped => {
                            tracing::trace!(
                                target: "parlevox_runtime",
                                "playout ring rejected an audio frame"
                            );
                        }
                    }
                    while let Some(played) = self.output.pop() {
                        audio_frames += 1;
                        self.emit(SessionEvent::AudioOut {
                            turn,
                            frame: played,
                        })?;
                    }
                }
                Some(Err(ParlevoxError::Cancelled)) => {
                    interrupted = true;
                    break;
                }
                Some(Err(err)) => return self.fail(err),
            }
        }
        #[cfg(feature = "otel")]
        {
            let output = self.output.stats();
            let dropped = output.dropped.saturating_sub(baseline.output.dropped);
            playout_span.set_attribute(KeyValue::new("playout.frames", counter(audio_frames)));
            playout_span.set_attribute(KeyValue::new("playout.dropped_frames", counter(dropped)));
            playout_span.end();
            tts_span.set_attribute(KeyValue::new("audio.frames", counter(audio_frames)));
            tts_span.end();
        }
        if interrupted {
            return self.finish_interrupted(turn, audio_frames, &transcript, &reply, baseline);
        }

        // 5. Close the turn.
        self.apply(TurnEvent::ReplyFinished)?;
        // Only a turn that ran to completion records an assistant message. An
        // interrupted or failed turn keeps the already-committed user message
        // but must not teach the model a reply the caller never heard.
        self.context.push_assistant(reply.clone());
        let context_messages = self.context.len();
        let context_evicted_messages = self
            .context
            .stats()
            .evicted_messages
            .saturating_sub(baseline.context_evicted_messages);
        let output_dropped = self.output.stats();
        let dropped_frames = output_dropped
            .dropped
            .saturating_sub(baseline.output.dropped);
        let dropped_ms = output_dropped
            .dropped_duration
            .as_millis()
            .saturating_sub(baseline.output.dropped_duration.as_millis());
        let input_dropped_frames = self
            .input
            .stats()
            .dropped
            .saturating_sub(baseline.input_dropped);
        let dropped_events = self
            .events
            .dropped()
            .saturating_sub(baseline.events_dropped);
        // The adapters accumulate per-request sidecar loss; taking it after the
        // TTS stream ends keeps the value per-turn and never double counts.
        let sidecar_dropped_frames = self.take_sidecar_dropped_frames();

        self.emit(SessionEvent::TurnDone {
            turn,
            audio_frames,
            dropped_frames,
            dropped_ms,
            dropped_events,
            sidecar_dropped_frames,
            interrupted: false,
        })?;

        Ok(TurnReport {
            turn,
            transcript,
            reply,
            audio_frames,
            dropped_frames,
            dropped_ms,
            dropped_events,
            sidecar_dropped_frames,
            input_dropped_frames,
            interrupted: false,
            context_messages,
            context_evicted_messages,
        })
    }

    /// Enqueue one event under the bounded-queue policy. Droppable overflow is
    /// absorbed by the queue; control overflow surfaces as a protocol error.
    fn emit(&mut self, event: SessionEvent) -> Result<(), ParlevoxError> {
        match self.events.push(event)? {
            EventPushOutcome::Accepted => {}
            EventPushOutcome::Replaced => {
                tracing::trace!(
                    target: "parlevox_runtime",
                    "session event queue evicted the oldest droppable event"
                );
            }
            EventPushOutcome::Dropped => {
                tracing::trace!(
                    target: "parlevox_runtime",
                    "session event queue dropped an incoming event"
                );
            }
        }
        Ok(())
    }

    /// Take the sidecar adapters' per-request dropped-frame telemetry. The
    /// transcribe path is the one that buffers audio, so STT is the only
    /// adapter that can legitimately report non-zero; LLM and TTS are included
    /// so a future/other plugin's count is never discarded. Swap semantics
    /// mean the value belongs to exactly one turn.
    fn take_sidecar_dropped_frames(&self) -> u64 {
        self.providers
            .stt
            .take_sidecar_dropped_frames()
            .saturating_add(self.providers.llm.take_sidecar_dropped_frames())
            .saturating_add(self.providers.tts.take_sidecar_dropped_frames())
    }

    fn fail<T>(&mut self, err: ParlevoxError) -> Result<T, ParlevoxError> {
        // A failed turn owns no telemetry: drain whatever the adapters recorded
        // so a partial turn cannot bill its sidecar drops to the next one.
        let _ = self.take_sidecar_dropped_frames();
        // If the control lane cannot even carry `Failed`, the queue error is
        // the louder and more accurate failure to report.
        self.emit(SessionEvent::Failed {
            class: err.class(),
            message: err.to_string(),
        })?;
        Err(err)
    }

    fn ensure_listening_after_interrupt(&mut self) -> Result<(), ParlevoxError> {
        match self.machine.state() {
            TurnState::Thinking | TurnState::Speaking => {
                self.machine.apply(TurnEvent::BargeIn)?;
                self.machine.apply(TurnEvent::ResumeListening)?;
            }
            TurnState::Interrupted => {
                self.machine.apply(TurnEvent::ResumeListening)?;
            }
            TurnState::UserSpeaking => {
                self.machine.apply(TurnEvent::UserStoppedSpeaking)?;
            }
            _ => {}
        }
        Ok(())
    }

    fn finish_interrupted(
        &mut self,
        turn: u64,
        audio_frames: u64,
        transcript: &str,
        reply: &str,
        baseline: TurnBaseline,
    ) -> Result<TurnReport, ParlevoxError> {
        self.ensure_listening_after_interrupt()?;
        // Drain any telemetry the adapters collected before the interruption
        // so it cannot leak into a later turn. A turn that never saw a `Done`
        // reports zero.
        let sidecar_dropped_frames = self.take_sidecar_dropped_frames();
        let output_dropped = self.output.stats();
        let dropped_frames = output_dropped
            .dropped
            .saturating_sub(baseline.output.dropped);
        let dropped_ms = output_dropped
            .dropped_duration
            .as_millis()
            .saturating_sub(baseline.output.dropped_duration.as_millis());
        let input_dropped_frames = self
            .input
            .stats()
            .dropped
            .saturating_sub(baseline.input_dropped);
        let dropped_events = self
            .events
            .dropped()
            .saturating_sub(baseline.events_dropped);
        // An interrupted turn keeps the user message it already committed (if
        // the transcript had been final) and adds no assistant message.
        let context_messages = self.context.len();
        let context_evicted_messages = self
            .context
            .stats()
            .evicted_messages
            .saturating_sub(baseline.context_evicted_messages);

        if turn != 0 {
            self.emit(SessionEvent::TurnDone {
                turn,
                audio_frames,
                dropped_frames,
                dropped_ms,
                dropped_events,
                sidecar_dropped_frames,
                interrupted: true,
            })?;
        }

        Ok(TurnReport {
            turn,
            transcript: transcript.to_string(),
            reply: reply.to_string(),
            audio_frames,
            dropped_frames,
            dropped_ms,
            dropped_events,
            sidecar_dropped_frames,
            input_dropped_frames,
            interrupted: true,
            context_messages,
            context_evicted_messages,
        })
    }
}

/// Convenience: build an [`AudioChunk`] of deterministic speech-like PCM for
/// demos and tests. Not a fake provider — just input.
pub fn demo_utterance(sample_rate: u32, channels: u16, ms: u64) -> AudioChunk {
    let samples = (sample_rate as u64 * ms / 1000) as usize;
    let mut pcm = Vec::with_capacity(samples * channels as usize);
    for i in 0..samples {
        // `(i * 31) % 1001` keeps the same phase if `i` is reduced modulo
        // 1001 first, which keeps the product inside `i32` for any length.
        let value = i32::try_from(i % 1001).unwrap_or(i32::MAX) * 31 % 1001 - 500;
        let value = i16::try_from(value).unwrap_or(0);
        for _ in 0..channels {
            pcm.push(value);
        }
    }
    AudioChunk::new(sample_rate, channels, pcm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    use async_trait::async_trait;

    use crate::events::SessionEvent;
    use crate::registry::{ProviderNames, ProviderRegistry};
    use parlevox_core::ErrorClass;
    use parlevox_providers::{
        cancelable_stream, AudioChunk, ChatMessage, FakeLlm, FakeStt, FakeTts, FakeVad, Modality,
        ProviderStream, Role, SttProvider, SynthesizeRequest, TranscriptChunk, TtsProvider,
        FAKE_REPLY, FAKE_SAMPLE_RATE, FAKE_TRANSCRIPT, FAKE_TTS_FRAMES,
    };

    /// TTS double that reports a fixed sidecar drop count exactly once, proving
    /// the runtime aggregates adapter telemetry rather than inventing it.
    struct ReportingTts {
        manifest: ProviderManifest,
        dropped: Arc<AtomicU64>,
    }

    impl ReportingTts {
        fn new(reported: u64) -> Self {
            ReportingTts {
                manifest: ProviderManifest::new("reporting-tts", Modality::Tts),
                dropped: Arc::new(AtomicU64::new(reported)),
            }
        }
    }

    #[async_trait]
    impl TtsProvider for ReportingTts {
        fn manifest(&self) -> &ProviderManifest {
            &self.manifest
        }

        fn take_sidecar_dropped_frames(&self) -> u64 {
            self.dropped.swap(0, Ordering::SeqCst)
        }

        async fn synthesize(
            &self,
            request: SynthesizeRequest,
            cancel: CancellationToken,
        ) -> Result<ProviderStream<AudioChunk>, ParlevoxError> {
            let items = (0..FAKE_TTS_FRAMES)
                .map(|_| {
                    Ok(AudioChunk::new(
                        request.sample_rate,
                        request.channels,
                        vec![1; 16],
                    ))
                })
                .collect();
            Ok(cancelable_stream(items, cancel))
        }
    }

    /// STT double that reports a fixed sidecar drop count exactly once. This is
    /// the path that actually buffers audio in the plugin, so STT telemetry is
    /// the one that must never be dropped by the aggregation.
    struct ReportingStt {
        manifest: ProviderManifest,
        dropped: Arc<AtomicU64>,
    }

    impl ReportingStt {
        fn new(reported: u64) -> Self {
            ReportingStt {
                manifest: ProviderManifest::new("reporting-stt", Modality::Stt),
                dropped: Arc::new(AtomicU64::new(reported)),
            }
        }
    }

    #[async_trait]
    impl SttProvider for ReportingStt {
        fn manifest(&self) -> &ProviderManifest {
            &self.manifest
        }

        fn take_sidecar_dropped_frames(&self) -> u64 {
            self.dropped.swap(0, Ordering::SeqCst)
        }

        async fn transcribe(
            &self,
            _audio: AudioChunk,
            cancel: CancellationToken,
        ) -> Result<ProviderStream<TranscriptChunk>, ParlevoxError> {
            let items = vec![Ok(TranscriptChunk {
                text: FAKE_TRANSCRIPT.into(),
                is_final: true,
            })];
            Ok(cancelable_stream(items, cancel))
        }
    }

    fn fake_session(config: SessionConfig) -> Session {
        let providers = ProviderRegistry::fakes()
            .unwrap()
            .resolve(&Default::default())
            .unwrap();
        Session::new(config, providers).unwrap()
    }

    async fn run_fake(config: SessionConfig) -> (Session, Result<TurnReport, ParlevoxError>) {
        let mut session = fake_session(config);
        let utterance = demo_utterance(FAKE_SAMPLE_RATE, 1, 40);
        let result = session.run_turn(utterance).await;
        (session, result)
    }

    #[tokio::test]
    async fn fake_turn_emits_the_full_typed_sequence() {
        let (mut session, result) = run_fake(SessionConfig::default()).await;
        let report = result.unwrap();
        assert_eq!(report.transcript, FAKE_TRANSCRIPT);
        assert_eq!(report.reply, FAKE_REPLY);
        assert_eq!(report.audio_frames as usize, FAKE_TTS_FRAMES);
        assert_eq!(report.dropped_events, 0);
        assert_eq!(report.sidecar_dropped_frames, 0);
        assert!(!report.interrupted);
        assert_eq!(session.state(), TurnState::Listening);
        assert_eq!(session.turn_index(), 1);

        let kinds: Vec<&str> = session
            .take_events()
            .iter()
            .map(SessionEvent::kind)
            .collect();
        assert_eq!(kinds.first(), Some(&"TurnStarted"));
        assert_eq!(kinds.last(), Some(&"TurnDone"));
        assert!(kinds.contains(&"TranscriptFinal"));
        assert!(kinds.contains(&"ReplyDelta"));
        assert!(kinds.contains(&"AudioOut"));
        assert!(!kinds.contains(&"Failed"));
    }

    #[tokio::test]
    async fn fake_turn_is_deterministic_across_runs() {
        let (_, first) = run_fake(SessionConfig::default()).await;
        let (_, second) = run_fake(SessionConfig::default()).await;
        assert_eq!(first.unwrap(), second.unwrap());
    }

    fn session_with_llm(config: SessionConfig, llm: Arc<FakeLlm>) -> Session {
        let mut registry = ProviderRegistry::new();
        registry.register_stt(Arc::new(FakeStt::new())).unwrap();
        registry.register_llm(llm).unwrap();
        registry.register_tts(Arc::new(FakeTts::new())).unwrap();
        registry.register_vad(Arc::new(FakeVad::new())).unwrap();
        let names = ProviderNames {
            stt: "fake-stt".into(),
            llm: "fake-llm".into(),
            tts: "fake-tts".into(),
            vad: "fake-vad".into(),
        };
        let providers = registry.resolve(&names).unwrap();
        Session::new(config, providers).unwrap()
    }

    #[tokio::test]
    async fn two_turn_request_carries_prior_history_without_duplicating_the_prompt() {
        let llm = Arc::new(FakeLlm::new());
        let mut session = session_with_llm(SessionConfig::default(), Arc::clone(&llm));
        let utterance = demo_utterance(FAKE_SAMPLE_RATE, 1, 40);

        let first = session.run_turn(utterance.clone()).await.unwrap();
        assert_eq!(first.context_messages, 2);
        let second = session.run_turn(utterance).await.unwrap();
        assert_eq!(second.context_messages, 4);

        let request = llm.last_request().expect("second request was recorded");
        assert_eq!(request.prompt, FAKE_TRANSCRIPT);
        // Exactly the prior user + assistant turn. The current utterance is
        // the prompt, so it must not appear as a third history entry.
        assert_eq!(
            request.history,
            vec![
                ChatMessage {
                    role: Role::User,
                    text: FAKE_TRANSCRIPT.into(),
                },
                ChatMessage {
                    role: Role::Assistant,
                    text: FAKE_REPLY.into(),
                },
            ]
        );
    }

    #[tokio::test]
    async fn interrupted_turn_keeps_the_user_message_but_records_no_assistant() {
        let llm = Arc::new(FakeLlm::new());
        let config = SessionConfig {
            barge_in_after_deltas: Some(2),
            ..SessionConfig::default()
        };
        let mut session = session_with_llm(config, Arc::clone(&llm));
        let utterance = demo_utterance(FAKE_SAMPLE_RATE, 1, 40);

        let report = session.run_turn(utterance.clone()).await.unwrap();
        assert!(report.interrupted);
        assert_eq!(report.context_messages, 1);
        assert_eq!(session.context_stats().messages, 1);

        // The next request sees the committed user turn and no assistant.
        let _ = session.run_turn(utterance).await.unwrap();
        let request = llm.last_request().expect("request was recorded");
        assert_eq!(
            request.history,
            vec![ChatMessage {
                role: Role::User,
                text: FAKE_TRANSCRIPT.into(),
            }]
        );
    }

    #[tokio::test]
    async fn report_counts_context_depth_and_evictions_per_turn() {
        let config = SessionConfig {
            context_messages: 2,
            ..SessionConfig::default()
        };
        let (mut session, first) = run_fake(config).await;
        let first = first.unwrap();
        assert_eq!(first.context_messages, 2);
        assert_eq!(first.context_evicted_messages, 0);

        let second = session
            .run_turn(demo_utterance(FAKE_SAMPLE_RATE, 1, 40))
            .await
            .unwrap();
        // Turn two evicts the prior user and assistant to keep the depth at 2.
        assert_eq!(second.context_messages, 2);
        assert_eq!(second.context_evicted_messages, 2);
    }

    #[tokio::test]
    async fn zero_capacity_output_ring_drops_every_frame_and_counts_it() {
        let config = SessionConfig {
            output_ring_capacity: 0,
            ..SessionConfig::default()
        };
        let (mut session, result) = run_fake(config).await;
        let report = result.unwrap();
        assert_eq!(report.audio_frames, 0);
        assert_eq!(report.dropped_frames as usize, FAKE_TTS_FRAMES);
        assert!(report.dropped_ms > 0);
        let kinds: Vec<&str> = session
            .take_events()
            .iter()
            .map(SessionEvent::kind)
            .collect();
        assert!(!kinds.contains(&"AudioOut"));
        assert!(kinds.contains(&"TurnDone"));
    }

    #[tokio::test]
    async fn zero_capacity_input_ring_takes_the_explicit_dropped_branch() {
        let config = SessionConfig {
            input_ring_capacity: 0,
            ..SessionConfig::default()
        };
        let (session, result) = run_fake(config).await;
        let report = result.unwrap();
        // The utterance frame was rejected by the bounded input ring and
        // counted, but the turn itself still completes.
        assert_eq!(report.input_dropped_frames, 1);
        assert_eq!(session.input_stats().dropped, 1);
        assert_eq!(report.sidecar_dropped_frames, 0);
        assert_eq!(report.audio_frames as usize, FAKE_TTS_FRAMES);
    }

    #[tokio::test]
    async fn sidecar_dropped_frames_is_reported_and_taken_per_turn() {
        let tts = Arc::new(ReportingTts::new(5));
        let mut registry = ProviderRegistry::new();
        registry.register_stt(Arc::new(FakeStt::new())).unwrap();
        registry.register_llm(Arc::new(FakeLlm::new())).unwrap();
        registry.register_tts(tts).unwrap();
        registry.register_vad(Arc::new(FakeVad::new())).unwrap();
        let names = ProviderNames {
            stt: "fake-stt".into(),
            llm: "fake-llm".into(),
            tts: "reporting-tts".into(),
            vad: "fake-vad".into(),
        };
        let providers = registry.resolve(&names).unwrap();
        let mut session = Session::new(SessionConfig::default(), providers).unwrap();

        let first = session
            .run_turn(demo_utterance(FAKE_SAMPLE_RATE, 1, 40))
            .await
            .unwrap();
        assert_eq!(first.sidecar_dropped_frames, 5);
        // Playout loss stays a separate, distinguishable counter.
        assert_eq!(first.dropped_frames, 0);

        let events = session.take_events();
        match events.last() {
            Some(SessionEvent::TurnDone {
                sidecar_dropped_frames,
                dropped_frames,
                ..
            }) => {
                assert_eq!(*sidecar_dropped_frames, 5);
                assert_eq!(*dropped_frames, 0);
            }
            other => panic!("expected TurnDone, got {other:?}"),
        }

        // Swap semantics: the same telemetry is never counted twice.
        let second = session
            .run_turn(demo_utterance(FAKE_SAMPLE_RATE, 1, 40))
            .await
            .unwrap();
        assert_eq!(second.sidecar_dropped_frames, 0);
    }

    #[tokio::test]
    async fn sidecar_dropped_frames_includes_the_stt_adapter() {
        // The transcribe path is the only one that buffers audio in the plugin,
        // so its telemetry must reach the turn report even when LLM and TTS
        // report nothing.
        let stt = Arc::new(ReportingStt::new(4));
        let mut registry = ProviderRegistry::new();
        registry.register_stt(stt).unwrap();
        registry.register_llm(Arc::new(FakeLlm::new())).unwrap();
        registry.register_tts(Arc::new(FakeTts::new())).unwrap();
        registry.register_vad(Arc::new(FakeVad::new())).unwrap();
        let names = ProviderNames {
            stt: "reporting-stt".into(),
            llm: "fake-llm".into(),
            tts: "fake-tts".into(),
            vad: "fake-vad".into(),
        };
        let providers = registry.resolve(&names).unwrap();
        let mut session = Session::new(SessionConfig::default(), providers).unwrap();

        let report = session
            .run_turn(demo_utterance(FAKE_SAMPLE_RATE, 1, 40))
            .await
            .unwrap();
        assert_eq!(report.sidecar_dropped_frames, 4);
        assert_eq!(report.dropped_frames, 0);
    }

    #[tokio::test]
    async fn barge_in_cancels_in_flight_reply_deterministically() {
        let config = SessionConfig {
            barge_in_after_deltas: Some(2),
            ..SessionConfig::default()
        };
        let mut session = fake_session(config);
        let before_generation = session.generation();
        let report = session
            .run_turn(demo_utterance(FAKE_SAMPLE_RATE, 1, 40))
            .await
            .unwrap();

        assert!(report.interrupted);
        assert_eq!(report.audio_frames, 0);
        assert!(report.reply.len() < FAKE_REPLY.len());
        assert_eq!(session.state(), TurnState::Listening);
        assert!(session.generation() > before_generation + 1);

        let events = session.take_events();
        let deltas = events
            .iter()
            .filter(|e| matches!(e, SessionEvent::ReplyDelta { .. }))
            .count();
        assert_eq!(deltas, 2);
        assert!(!events
            .iter()
            .any(|e| matches!(e, SessionEvent::AudioOut { .. })));
        assert!(matches!(
            events.last(),
            Some(SessionEvent::TurnDone {
                interrupted: true,
                ..
            })
        ));
    }

    #[tokio::test]
    async fn bounded_event_queue_drops_streaming_events_and_reports_the_count() {
        let config = SessionConfig {
            event_capacity: 4,
            ..SessionConfig::default()
        };
        let (mut session, result) = run_fake(config).await;
        let report = result.unwrap();

        // The reply and audio still complete: backpressure only loses events.
        assert_eq!(report.audio_frames as usize, FAKE_TTS_FRAMES);
        assert!(report.dropped_events > 0);
        let stats = session.event_stats();
        assert_eq!(stats.dropped, report.dropped_events);
        assert_eq!(stats.capacity, 4);
        assert!(stats.depth <= stats.capacity);
        assert_eq!(stats.control_overflow, 0);

        let events = session.take_events();
        let kinds: Vec<&str> = events.iter().map(SessionEvent::kind).collect();
        // Control structure survives even when streaming events are dropped.
        assert_eq!(kinds.first(), Some(&"TurnStarted"));
        assert!(kinds.contains(&"TranscriptFinal"));
        assert_eq!(kinds.last(), Some(&"TurnDone"));
        match events.last() {
            Some(SessionEvent::TurnDone {
                dropped_events,
                interrupted,
                ..
            }) => {
                assert_eq!(*dropped_events, report.dropped_events);
                assert!(!interrupted);
            }
            other => panic!("expected TurnDone, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn control_lane_overflow_fails_the_turn_loudly() {
        let config = SessionConfig {
            event_capacity: 2,
            ..SessionConfig::default()
        };
        let (session, result) = run_fake(config).await;

        let err = result.unwrap_err();
        assert!(matches!(err, ParlevoxError::Protocol(_)));
        let stats = session.event_stats();
        assert_eq!(stats.control_overflow, 1);
        assert_eq!(stats.dropped, 1);
    }

    #[test]
    fn zero_event_capacity_is_rejected_at_construction() {
        let providers = ProviderRegistry::fakes()
            .unwrap()
            .resolve(&Default::default())
            .unwrap();
        let config = SessionConfig {
            event_capacity: 0,
            ..SessionConfig::default()
        };
        let err = match Session::new(config, providers) {
            Err(err) => err,
            Ok(_) => panic!("expected zero event capacity to be rejected"),
        };
        assert!(matches!(err, ParlevoxError::Protocol(_)));
    }

    #[test]
    fn event_capacity_below_minimum_is_rejected_at_construction() {
        let providers = ProviderRegistry::fakes()
            .unwrap()
            .resolve(&Default::default())
            .unwrap();
        let config = SessionConfig {
            event_capacity: MIN_EVENT_CAPACITY - 1,
            ..SessionConfig::default()
        };
        let err = match Session::new(config, providers) {
            Err(err) => err,
            Ok(_) => panic!("expected capacity below MIN_EVENT_CAPACITY to be rejected"),
        };
        assert!(matches!(err, ParlevoxError::Protocol(_)));
        assert!(err.to_string().contains("MIN_EVENT_CAPACITY"));
    }

    #[tokio::test]
    async fn illegal_transition_is_still_a_protocol_error() {
        let mut session = fake_session(SessionConfig::default());
        session.start().unwrap();
        let err = session.apply(TurnEvent::ReplyStarted).unwrap_err();
        assert!(matches!(err, ParlevoxError::Protocol(_)));
        assert_eq!(err.class(), ErrorClass::Protocol);
        assert_eq!(session.state(), TurnState::Listening);
    }

    #[test]
    fn demo_utterance_is_deterministic_and_non_empty() {
        let a = demo_utterance(16_000, 1, 20);
        let b = demo_utterance(16_000, 1, 20);
        assert_eq!(a, b);
        assert_eq!(a.pcm.len(), 320);
    }
}
