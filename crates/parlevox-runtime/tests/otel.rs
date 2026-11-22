//! Hermetic OpenTelemetry span-shape test.
//!
//! Enabled only by the `otel` feature. Nothing here touches the network, a
//! collector, or a process-global provider: the session is handed a tracer
//! backed by an in-memory span exporter, so the test observes exactly the
//! spans one fake turn produced.

#![cfg(feature = "otel")]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry::Value;
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::trace::{SdkTracerProvider, SimpleSpanProcessor, SpanData, SpanExporter};
use parlevox_core::{ParlevoxError, TurnState};
use parlevox_providers::{
    cancelable_stream, AudioChunk, FakeLlm, FakeTts, FakeVad, Modality, ProviderManifest,
    ProviderStream, SttProvider, TranscriptChunk, FAKE_PARTIAL_TRANSCRIPT, FAKE_TRANSCRIPT,
};
use parlevox_runtime::{
    demo_utterance, ProviderNames, ProviderRegistry, Providers, Session, SessionConfig,
};
use tokio_util::sync::CancellationToken;

/// In-memory exporter collecting every finished span.
///
/// The SDK's `InMemorySpanExporter` sits behind its `testing` feature, which
/// also pulls metrics and logs and would put an OpenTelemetry edge in the
/// default dev tree. This tiny exporter keeps the same contract with only the
/// trace feature enabled.
#[derive(Clone, Debug, Default)]
struct CollectingExporter {
    spans: Arc<Mutex<Vec<SpanData>>>,
}

impl CollectingExporter {
    fn finished(&self) -> Vec<SpanData> {
        self.spans.lock().expect("exporter mutex poisoned").clone()
    }
}

impl SpanExporter for CollectingExporter {
    async fn export(&self, batch: Vec<SpanData>) -> OTelSdkResult {
        self.spans
            .lock()
            .expect("exporter mutex poisoned")
            .extend(batch);
        Ok(())
    }
}

fn i64_attribute(span: &SpanData, key: &str) -> Option<i64> {
    span.attributes
        .iter()
        .find(|attribute| attribute.key.as_str() == key)
        .and_then(|attribute| match attribute.value {
            Value::I64(value) => Some(value),
            _ => None,
        })
}

fn string_value(attribute: &Value) -> Option<&str> {
    match attribute {
        Value::String(value) => Some(value.as_str()),
        _ => None,
    }
}

fn string_attribute<'a>(span: &'a SpanData, key: &str) -> Option<&'a str> {
    span.attributes
        .iter()
        .find(|attribute| attribute.key.as_str() == key)
        .and_then(|attribute| string_value(&attribute.value))
}

/// STT double with a one-shot failure switch.
///
/// A failing attempt returns an error from `transcribe`, which is a failure
/// *before* the final transcript commits: the turn returns `Err` and never
/// produces a `TurnReport`.
struct ScriptedStt {
    manifest: ProviderManifest,
    fail_next: AtomicBool,
}

impl ScriptedStt {
    fn new() -> Self {
        ScriptedStt {
            manifest: ProviderManifest::new("scripted-stt", Modality::Stt),
            fail_next: AtomicBool::new(false),
        }
    }

    /// Make the next `transcribe` call fail. Cleared by that call.
    fn fail_this_attempt(&self) {
        self.fail_next.store(true, Ordering::SeqCst);
    }
}

#[async_trait]
impl SttProvider for ScriptedStt {
    fn manifest(&self) -> &ProviderManifest {
        &self.manifest
    }

    async fn transcribe(
        &self,
        _audio: AudioChunk,
        cancel: CancellationToken,
    ) -> Result<ProviderStream<TranscriptChunk>, ParlevoxError> {
        if self.fail_next.swap(false, Ordering::SeqCst) {
            return Err(ParlevoxError::provider(
                "scripted-stt",
                "scripted pre-final failure",
            ));
        }
        let items = vec![
            Ok(TranscriptChunk {
                text: FAKE_PARTIAL_TRANSCRIPT.into(),
                is_final: false,
            }),
            Ok(TranscriptChunk {
                text: FAKE_TRANSCRIPT.into(),
                is_final: true,
            }),
        ];
        Ok(cancelable_stream(items, cancel))
    }
}

fn scripted_providers(stt: Arc<ScriptedStt>) -> Providers {
    Providers {
        stt: Arc::clone(&stt) as Arc<dyn SttProvider>,
        llm: Arc::new(FakeLlm::new()),
        tts: Arc::new(FakeTts::new()),
        vad: Arc::new(FakeVad::new()),
    }
}

/// A failed attempt now closes the turn itself (`Session::fail` calls
/// `TurnMachine::abort`), even when the failure happens before `FinalTranscript`
/// while the speaker is still active. The next attempt must therefore start
/// from `Listening` with no manual repair.
fn assert_listening_after_failure(session: &Session) {
    assert_eq!(
        session.state(),
        TurnState::Listening,
        "a failed turn must already be back in Listening"
    );
}

#[tokio::test]
async fn turn_emits_root_span_with_stage_children() {
    // Arrange: one session whose tracer exports into memory.
    let exporter = CollectingExporter::default();
    let provider = SdkTracerProvider::builder()
        .with_span_processor(SimpleSpanProcessor::new(exporter.clone()))
        .build();
    let tracer = provider.tracer("parlevox-runtime-test");
    let providers = ProviderRegistry::fakes()
        .expect("built-in fakes resolve")
        .resolve(&ProviderNames::default())
        .expect("fake providers resolve");
    let mut session =
        Session::with_tracer(SessionConfig::default(), providers, tracer).expect("session builds");

    // Act: one complete fake turn.
    let report = session
        .run_turn(demo_utterance(16_000, 1, 40))
        .await
        .expect("fake turn completes");

    // Assert: exactly one root and one of each named child, all in one trace.
    let spans = exporter.finished();
    let roots: Vec<&SpanData> = spans.iter().filter(|span| span.name == "turn").collect();
    assert_eq!(roots.len(), 1, "expected exactly one turn root span");
    let root = roots[0];
    let trace_id = root.span_context.trace_id();
    let root_span_id = root.span_context.span_id();
    assert_ne!(trace_id, opentelemetry::TraceId::INVALID);

    for name in ["vad", "stt", "llm", "tts", "playout"] {
        let child = spans
            .iter()
            .find(|span| span.name == name)
            .unwrap_or_else(|| panic!("missing {name} child span"));
        assert_eq!(
            child.span_context.trace_id(),
            trace_id,
            "{name} must share the turn trace id"
        );
        assert_eq!(
            child.parent_span_id, root_span_id,
            "{name} must be a direct child of the turn span"
        );
        assert!(
            i64_attribute(child, "stage.duration_ms").is_some_and(|ms| ms >= 0),
            "{name} must carry a MonoClock stage duration"
        );
        assert!(
            child.start_time <= child.end_time,
            "{name} start/end must be ordered"
        );
    }

    // Boundary costs come from the turn report and MonoClock waits.
    assert_eq!(i64_attribute(root, "turn.index"), Some(1));
    assert_eq!(
        i64_attribute(root, "turn.audio_frames"),
        Some(i64::try_from(report.audio_frames).unwrap())
    );
    assert_eq!(
        i64_attribute(root, "turn.dropped_events"),
        Some(i64::try_from(report.dropped_events).unwrap())
    );
    assert!(i64_attribute(root, "duration.monotonic_ms").is_some_and(|ms| ms >= 0));
    assert!(i64_attribute(root, "llm.first_delta_ms").is_some_and(|ms| ms >= 0));
    assert!(i64_attribute(root, "tts.first_audio_ms").is_some_and(|ms| ms >= 0));
    assert_eq!(
        i64_attribute(root, "transcript.chars"),
        Some(i64::try_from(report.transcript.chars().count()).unwrap())
    );
    assert_eq!(
        i64_attribute(root, "reply.chars"),
        Some(i64::try_from(report.reply.chars().count()).unwrap())
    );
    assert!(
        i64_attribute(root, "turn.interrupted").is_none() || report.interrupted,
        "an uninterrupted turn must not claim to be interrupted"
    );
    assert!(
        root.start_time <= root.end_time,
        "turn start/end must be ordered"
    );

    // No user content is exported: only lengths and counters.
    for span in &spans {
        for attribute in &span.attributes {
            let value = string_value(&attribute.value);
            assert_ne!(value, Some(report.transcript.as_str()));
            assert_ne!(value, Some(report.reply.as_str()));
        }
    }

    provider.shutdown().expect("provider shuts down cleanly");
}

#[tokio::test]
async fn failed_pre_final_turn_shares_committed_turn_index() {
    // Arrange: one session whose scripted STT can fail before the final.
    let exporter = CollectingExporter::default();
    let provider = SdkTracerProvider::builder()
        .with_span_processor(SimpleSpanProcessor::new(exporter.clone()))
        .build();
    let tracer = provider.tracer("parlevox-runtime-test");
    let stt = Arc::new(ScriptedStt::new());
    let mut session = Session::with_tracer(
        SessionConfig::default(),
        scripted_providers(Arc::clone(&stt)),
        tracer,
    )
    .expect("session builds");
    let utterance = demo_utterance(16_000, 1, 40);

    // Act: fail, succeed (turn 1), fail, succeed (turn 2).
    stt.fail_this_attempt();
    assert!(
        session.run_turn(utterance.clone()).await.is_err(),
        "the scripted STT failure must fail the turn"
    );
    assert_listening_after_failure(&session);
    let first_success = session
        .run_turn(utterance.clone())
        .await
        .expect("first retry completes");
    assert_eq!(first_success.turn, 1);

    stt.fail_this_attempt();
    assert!(
        session.run_turn(utterance.clone()).await.is_err(),
        "the second scripted STT failure must fail the turn"
    );
    assert_listening_after_failure(&session);
    let second_success = session
        .run_turn(utterance)
        .await
        .expect("second retry completes");
    assert_eq!(second_success.turn, 2);

    // Assert: two failed spans each carry the index their retry committed.
    let spans = exporter.finished();
    let roots: Vec<&SpanData> = spans.iter().filter(|span| span.name == "turn").collect();
    assert_eq!(roots.len(), 4, "expected four turn root spans");

    let failed: Vec<&SpanData> = roots
        .iter()
        .copied()
        .filter(|span| string_attribute(span, "turn.error_class").is_some())
        .collect();
    let succeeded: Vec<&SpanData> = roots
        .iter()
        .copied()
        .filter(|span| string_attribute(span, "turn.error_class").is_none())
        .collect();
    assert_eq!(failed.len(), 2, "two attempts fail before FinalTranscript");
    assert_eq!(succeeded.len(), 2, "two retries commit a turn");

    let failed_index = |span: &SpanData| {
        i64_attribute(span, "turn.index").expect("a failed turn span still records its index")
    };
    let first_turn = i64::try_from(first_success.turn).expect("turn index fits i64");
    let second_turn = i64::try_from(second_success.turn).expect("turn index fits i64");
    assert_eq!(failed_index(failed[0]), first_turn);
    assert_eq!(failed_index(failed[1]), second_turn);
    assert_ne!(
        failed_index(failed[0]),
        failed_index(failed[1]),
        "two failed attempts must not share one turn index"
    );

    assert_eq!(i64_attribute(succeeded[0], "turn.index"), Some(first_turn));
    assert_eq!(i64_attribute(succeeded[1], "turn.index"), Some(second_turn));

    provider.shutdown().expect("provider shuts down cleanly");
}

#[tokio::test]
async fn turn_span_records_boundary_depth_and_capacity() {
    // Arrange: one session whose tracer exports into memory.
    let exporter = CollectingExporter::default();
    let provider = SdkTracerProvider::builder()
        .with_span_processor(SimpleSpanProcessor::new(exporter.clone()))
        .build();
    let tracer = provider.tracer("parlevox-runtime-test");
    let providers = ProviderRegistry::fakes()
        .expect("built-in fakes resolve")
        .resolve(&ProviderNames::default())
        .expect("fake providers resolve");
    let mut session =
        Session::with_tracer(SessionConfig::default(), providers, tracer).expect("session builds");

    // Act: one complete fake turn.
    session
        .run_turn(demo_utterance(16_000, 1, 40))
        .await
        .expect("fake turn completes");

    // Sample the boundaries before `take_events`: draining the queue would
    // change `events.depth` from the value the span recorded at turn end.
    let input = session.input_stats();
    let playout = session.output_stats();
    let events = session.event_stats();

    // Assert: the root span carries exactly the session's boundary occupancy.
    let spans = exporter.finished();
    let root = spans
        .iter()
        .find(|span| span.name == "turn")
        .expect("one turn root span");
    let as_i64 = |value: usize| i64::try_from(value).expect("boundary value fits i64");

    assert_eq!(
        i64_attribute(root, "ring.input.depth"),
        Some(as_i64(input.depth))
    );
    assert_eq!(
        i64_attribute(root, "ring.input.capacity"),
        Some(as_i64(input.capacity))
    );
    assert_eq!(
        i64_attribute(root, "ring.playout.depth"),
        Some(as_i64(playout.depth))
    );
    assert_eq!(
        i64_attribute(root, "ring.playout.capacity"),
        Some(as_i64(playout.capacity))
    );
    assert_eq!(
        i64_attribute(root, "events.depth"),
        Some(as_i64(events.depth))
    );
    assert_eq!(
        i64_attribute(root, "events.capacity"),
        Some(as_i64(events.capacity))
    );

    // The comparison must not be vacuous: the turn buffered input and its
    // events are still queued at sampling time.
    assert_eq!(
        input.depth, 1,
        "the utterance frame is buffered at turn end"
    );
    assert!(events.depth > 0, "the turn's events are still queued");

    provider.shutdown().expect("provider shuts down cleanly");
}
