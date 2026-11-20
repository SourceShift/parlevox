//! Hermetic OpenTelemetry span-shape test.
//!
//! Enabled only by the `otel` feature. Nothing here touches the network, a
//! collector, or a process-global provider: the session is handed a tracer
//! backed by an in-memory span exporter, so the test observes exactly the
//! spans one fake turn produced.

#![cfg(feature = "otel")]

use std::sync::{Arc, Mutex};

use opentelemetry::trace::TracerProvider as _;
use opentelemetry::Value;
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::trace::{SdkTracerProvider, SimpleSpanProcessor, SpanData, SpanExporter};
use parlevox_runtime::{demo_utterance, ProviderNames, ProviderRegistry, Session, SessionConfig};

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
