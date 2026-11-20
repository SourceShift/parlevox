//! Hermetic tests for the opt-in pipelined-TTS mode.
//!
//! Everything here runs against in-process providers: no network and no Python.
//! Ordering is proven from the fakes' call log rather than from timing, and
//! cancellation is proven from the TTS stream's drop record rather than from a
//! timeout. The span test uses a deliberate 2 ms pause between stream items
//! because the session's monotonic clock and exported span timestamps have
//! millisecond resolution; the pause makes two overlapping spans
//! distinguishable and never asserts how long anything takes.

use std::sync::{Arc, Mutex};
#[cfg(feature = "otel")]
use std::time::Duration;

use async_trait::async_trait;
#[cfg(feature = "otel")]
use futures_util::stream::unfold;
#[cfg(feature = "otel")]
use opentelemetry::trace::TracerProvider as _;
#[cfg(feature = "otel")]
use opentelemetry::Value;
#[cfg(feature = "otel")]
use opentelemetry_sdk::error::OTelSdkResult;
#[cfg(feature = "otel")]
use opentelemetry_sdk::trace::{SdkTracerProvider, SimpleSpanProcessor, SpanData, SpanExporter};
use parlevox_core::ParlevoxError;
#[cfg(feature = "otel")]
use parlevox_providers::GenerateRequest;
use parlevox_providers::{
    cancelable_stream, AudioChunk, FakeLlm, FakeStt, FakeTts, FakeVad, LlmProvider, Modality,
    ProviderCall, ProviderCallLog, ProviderManifest, ProviderStream, SynthesizeRequest,
    TtsProvider,
};
use parlevox_runtime::{
    demo_utterance, ProviderNames, ProviderRegistry, Providers, Session, SessionConfig,
    SessionEvent, TurnReport,
};
use tokio_util::sync::CancellationToken;

/// TTS double with compositional audio: exactly one frame per character of the
/// requested text, where each frame's PCM depends only on the character.
///
/// That makes `concat(audio("a")) ++ audio("b") == audio("ab")` as sample
/// sequences, so the pipelined path's per-sentence requests and the sequential
/// path's whole-reply request must produce identical concatenated PCM. The real
/// [`FakeTts`] deliberately emits a fixed two frames per request, which cannot
/// prove that property.
struct CharTts {
    manifest: ProviderManifest,
}

impl CharTts {
    fn new() -> Self {
        CharTts {
            manifest: ProviderManifest::new("char-tts", Modality::Tts)
                .with_capability("streaming")
                .with_capability("cancel"),
        }
    }
}

fn char_pcm(ch: char) -> Vec<i16> {
    // Bounded and deterministic: `ch as u32 % 1000` is in `0..1000`, so the
    // shifted value fits in `i16` without an unchecked cast.
    let base = i32::try_from(ch as u32 % 1000).unwrap_or(0);
    let value = i16::try_from(base - 500).unwrap_or(0);
    vec![value; 8]
}

#[async_trait]
impl TtsProvider for CharTts {
    fn manifest(&self) -> &ProviderManifest {
        &self.manifest
    }

    async fn synthesize(
        &self,
        request: SynthesizeRequest,
        cancel: CancellationToken,
    ) -> Result<ProviderStream<AudioChunk>, ParlevoxError> {
        if request.text.trim().is_empty() {
            return Err(ParlevoxError::Protocol("text must not be empty".into()));
        }
        let items = request
            .text
            .chars()
            .map(|ch| {
                Ok(AudioChunk::new(
                    request.sample_rate,
                    request.channels,
                    char_pcm(ch),
                ))
            })
            .collect();
        Ok(cancelable_stream(items, cancel))
    }
}

fn providers_for(llm: Arc<dyn LlmProvider>, tts: Arc<dyn TtsProvider>) -> Providers {
    let names = ProviderNames {
        stt: "fake-stt".into(),
        llm: llm.manifest().name.clone(),
        tts: tts.manifest().name.clone(),
        vad: "fake-vad".into(),
    };
    let mut registry = ProviderRegistry::new();
    registry.register_stt(Arc::new(FakeStt::new())).unwrap();
    registry.register_llm(llm).unwrap();
    registry.register_tts(tts).unwrap();
    registry.register_vad(Arc::new(FakeVad::new())).unwrap();
    registry.resolve(&names).unwrap()
}

fn scripted_llm(deltas: &[&str]) -> FakeLlm {
    FakeLlm::new().with_deltas(deltas.iter().map(|delta| (*delta).to_string()).collect())
}

/// LLM double that pauses before every item, including end-of-stream.
///
/// The span test needs the LLM to still be live after the first synthesis
/// starts. [`FakeLlm`] is instantaneous, so both events land in the same
/// millisecond and the exported trace cannot show the overlap. The pause makes
/// the overlap observable without asserting any latency.
#[cfg(feature = "otel")]
struct PacedLlm {
    manifest: ProviderManifest,
    deltas: Vec<String>,
}

#[cfg(feature = "otel")]
impl PacedLlm {
    fn new(deltas: &[&str]) -> Self {
        PacedLlm {
            manifest: ProviderManifest::new("paced-llm", Modality::Llm)
                .with_capability("streaming")
                .with_capability("cancel"),
            deltas: deltas.iter().map(|delta| (*delta).to_string()).collect(),
        }
    }
}

#[cfg(feature = "otel")]
#[async_trait]
impl LlmProvider for PacedLlm {
    fn manifest(&self) -> &ProviderManifest {
        &self.manifest
    }

    async fn generate(
        &self,
        _request: GenerateRequest,
        _cancel: CancellationToken,
    ) -> Result<ProviderStream<String>, ParlevoxError> {
        let items: Vec<Result<String, ParlevoxError>> =
            self.deltas.iter().cloned().map(Ok).collect();
        let stream = unfold(items.into_iter(), |mut items| async move {
            tokio::time::sleep(Duration::from_millis(2)).await;
            items.next().map(|item| (item, items))
        });
        Ok(Box::pin(stream))
    }
}

async fn run(config: SessionConfig, providers: Providers) -> (Session, TurnReport) {
    let mut session = Session::new(config, providers).unwrap();
    let report = session
        .run_turn(demo_utterance(16_000, 1, 40))
        .await
        .unwrap();
    (session, report)
}

fn concatenated_pcm(events: &[SessionEvent]) -> Vec<i16> {
    let mut pcm = Vec::new();
    for event in events {
        if let SessionEvent::AudioOut { frame, .. } = event {
            pcm.extend_from_slice(&frame.pcm);
        }
    }
    pcm
}

#[tokio::test]
async fn pipelined_tts_first_synthesis_precedes_the_llm_end_of_stream() {
    let log: ProviderCallLog = Arc::new(Mutex::new(Vec::new()));
    let llm = Arc::new(scripted_llm(&["One. ", "Two. ", "Three."]).with_call_log(Arc::clone(&log)));
    let tts = Arc::new(FakeTts::new().with_call_log(Arc::clone(&log)));

    let (mut session, report) = run(
        SessionConfig {
            pipelined_tts: true,
            ..SessionConfig::default()
        },
        providers_for(llm, tts),
    )
    .await;
    let events = session.take_events();

    let calls = log.lock().unwrap().clone();
    let first_synthesis = calls
        .iter()
        .position(|call| matches!(call, ProviderCall::TtsSynthesize(_)))
        .expect("the pipelined path issued a synthesis request");
    let llm_end = calls
        .iter()
        .position(|call| matches!(call, ProviderCall::LlmEnded))
        .expect("the fake LLM recorded end of stream");
    assert!(
        first_synthesis < llm_end,
        "first synthesis at {first_synthesis} was not before LLM end-of-stream at {llm_end}: {calls:?}"
    );

    // Three sentences, two frames each, in reply order.
    assert_eq!(report.reply, "One. Two. Three.");
    assert_eq!(report.audio_frames, 6);
    assert!(matches!(
        events.last(),
        Some(SessionEvent::TurnDone {
            interrupted: false,
            ..
        })
    ));
}

#[tokio::test]
async fn pipelined_tts_audio_bytes_match_the_sequential_path() {
    let deltas = ["Hello there. ", "How are you? ", "Fine."];
    let (mut sequential, sequential_report) = {
        let providers = providers_for(Arc::new(scripted_llm(&deltas)), Arc::new(CharTts::new()));
        run(SessionConfig::default(), providers).await
    };
    let (mut pipelined, pipelined_report) = {
        let providers = providers_for(Arc::new(scripted_llm(&deltas)), Arc::new(CharTts::new()));
        run(
            SessionConfig {
                pipelined_tts: true,
                ..SessionConfig::default()
            },
            providers,
        )
        .await
    };

    let sequential_events = sequential.take_events();
    let pipelined_events = pipelined.take_events();

    // Identical reply, identical frame count, identical sample sequence: the
    // per-sentence split must partition the reply byte for byte.
    assert_eq!(
        concatenated_pcm(&pipelined_events),
        concatenated_pcm(&sequential_events),
        "pipelined synthesis changed the audio, not just its scheduling"
    );
    assert_eq!(pipelined_report.reply, sequential_report.reply);
    assert_eq!(
        pipelined_report.audio_frames,
        sequential_report.audio_frames
    );
}

#[tokio::test]
async fn pipelined_tts_releases_a_sentence_whose_terminator_ends_a_delta() {
    // F1 regression under the binding hold ruling: a terminator that ends a
    // delta is held for exactly one delta, then released as soon as the next
    // delta supplies text — which is still before the LLM stream is exhausted.
    let log: ProviderCallLog = Arc::new(Mutex::new(Vec::new()));
    let llm = Arc::new(scripted_llm(&["One.", " Two."]).with_call_log(Arc::clone(&log)));
    let tts = Arc::new(FakeTts::new().with_call_log(Arc::clone(&log)));

    let (mut session, report) = run(
        SessionConfig {
            pipelined_tts: true,
            ..SessionConfig::default()
        },
        providers_for(llm, tts),
    )
    .await;
    let _ = session.take_events();

    let calls = log.lock().unwrap().clone();
    let first_synthesis = calls
        .iter()
        .position(|call| matches!(call, ProviderCall::TtsSynthesize(_)))
        .expect("the pipelined path issued a synthesis request");
    let llm_end = calls
        .iter()
        .position(|call| matches!(call, ProviderCall::LlmEnded))
        .expect("the fake LLM recorded end of stream");
    assert!(
        first_synthesis < llm_end,
        "first synthesis at {first_synthesis} was not before LLM end-of-stream \
         at {llm_end}: {calls:?}"
    );

    let synthesized: Vec<String> = calls
        .iter()
        .filter_map(|call| match call {
            ProviderCall::TtsSynthesize(text) => Some(text.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        synthesized,
        vec!["One. ".to_string(), "Two.".to_string()],
        "the split must release the first sentence and preserve every byte"
    );
    assert!(
        synthesized.iter().all(|text| !text.trim().is_empty()),
        "a whitespace-only request reached the provider: {synthesized:?}"
    );
    assert_eq!(
        synthesized.concat(),
        "One. Two.",
        "the split must preserve every byte of the reply"
    );
    assert_eq!(report.reply, "One. Two.");
    assert!(!report.interrupted);
}

#[tokio::test]
async fn pipelined_tts_terminator_at_delta_end_covers_every_byte() {
    // P11: a terminator that ends the first delta followed by a
    // whitespace-only delta must produce exactly one request, "Hello. " —
    // never a whitespace-only chunk and never "Hello." alone — and the
    // playout must match the sequential path.
    let deltas = ["Hello.", " "];

    let log: ProviderCallLog = Arc::new(Mutex::new(Vec::new()));
    let llm = Arc::new(scripted_llm(&deltas).with_call_log(Arc::clone(&log)));
    let tts = Arc::new(FakeTts::new().with_call_log(Arc::clone(&log)));
    let (mut pipelined, pipelined_report) = run(
        SessionConfig {
            pipelined_tts: true,
            ..SessionConfig::default()
        },
        providers_for(llm, tts),
    )
    .await;
    let pipelined_events = pipelined.take_events();

    let requests: Vec<String> = log
        .lock()
        .unwrap()
        .iter()
        .filter_map(|call| match call {
            ProviderCall::TtsSynthesize(text) => Some(text.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        requests,
        vec!["Hello. ".to_string()],
        "expected exactly one request covering every byte: {requests:?}"
    );
    assert_eq!(requests.concat(), "Hello. ");
    assert_eq!(pipelined_report.reply, "Hello. ");
    assert!(!pipelined_report.interrupted);

    let (mut sequential, sequential_report) = {
        let providers = providers_for(Arc::new(scripted_llm(&deltas)), Arc::new(FakeTts::new()));
        run(SessionConfig::default(), providers).await
    };
    let sequential_events = sequential.take_events();
    assert_eq!(
        concatenated_pcm(&pipelined_events),
        concatenated_pcm(&sequential_events),
        "pipelined synthesis changed the audio for a terminator at a delta boundary"
    );
    assert_eq!(
        pipelined_report.audio_frames,
        sequential_report.audio_frames
    );
}

#[tokio::test]
async fn pipelined_tts_never_synthesizes_whitespace_only_text() {
    // F2/P7 regression: trailing whitespace arriving in its own delta must ride
    // with the sentence it follows, and the turn must still complete.
    let deltas = ["Hello. ", " "];

    let (mut sequential, sequential_report) = {
        let providers = providers_for(Arc::new(scripted_llm(&deltas)), Arc::new(FakeTts::new()));
        run(SessionConfig::default(), providers).await
    };

    let log: ProviderCallLog = Arc::new(Mutex::new(Vec::new()));
    let llm = Arc::new(scripted_llm(&deltas));
    let tts = Arc::new(FakeTts::new().with_call_log(Arc::clone(&log)));
    let (mut pipelined, pipelined_report) = run(
        SessionConfig {
            pipelined_tts: true,
            ..SessionConfig::default()
        },
        providers_for(llm, tts),
    )
    .await;

    let sequential_events = sequential.take_events();
    let pipelined_events = pipelined.take_events();

    assert!(!pipelined_report.interrupted);
    assert_eq!(pipelined_report.reply, "Hello.  ");

    let synthesized: Vec<String> = log
        .lock()
        .unwrap()
        .iter()
        .filter_map(|call| match call {
            ProviderCall::TtsSynthesize(text) => Some(text.clone()),
            _ => None,
        })
        .collect();
    assert!(
        synthesized.iter().all(|text| !text.trim().is_empty()),
        "a whitespace-only request reached the provider: {synthesized:?}"
    );
    assert_eq!(synthesized.concat(), "Hello.  ");

    // The pipelined requests must produce the same playout as one whole-reply
    // request: the same reply, partitioned without losing bytes.
    assert_eq!(
        concatenated_pcm(&pipelined_events),
        concatenated_pcm(&sequential_events),
        "pipelined synthesis changed the audio for a whitespace-only delta"
    );
    assert_eq!(
        pipelined_report.audio_frames,
        sequential_report.audio_frames
    );

    // The gate must also be load-bearing on its own: this stream reaches the
    // pre-synthesis check directly. An all-whitespace reply cannot be both
    // covered byte-for-byte and handed to a provider that rejects empty text,
    // so it is outside the byte-coverage guarantee; the gate must refuse it
    // rather than let it become a hard turn failure. Deleting the gate would
    // pass "  " to the fake TTS and make `run` panic here.
    let blank_log: ProviderCallLog = Arc::new(Mutex::new(Vec::new()));
    let blank_llm = Arc::new(scripted_llm(&[" ", " "]).with_call_log(Arc::clone(&blank_log)));
    let blank_tts = Arc::new(FakeTts::new().with_call_log(Arc::clone(&blank_log)));
    let (mut blank_session, blank_report) = run(
        SessionConfig {
            pipelined_tts: true,
            ..SessionConfig::default()
        },
        providers_for(blank_llm, blank_tts),
    )
    .await;
    let _ = blank_session.take_events();

    assert!(
        !blank_report.interrupted,
        "the all-whitespace reply must complete, not fail the turn"
    );
    assert_eq!(blank_report.reply, "  ");
    let blank_requests: Vec<String> = blank_log
        .lock()
        .unwrap()
        .iter()
        .filter_map(|call| match call {
            ProviderCall::TtsSynthesize(text) => Some(text.clone()),
            _ => None,
        })
        .collect();
    assert!(
        blank_requests.is_empty(),
        "the gate let a whitespace-only unit reach the provider: {blank_requests:?}"
    );
}

#[tokio::test]
async fn pipelined_tts_ellipsis_is_one_sentence_unit() {
    // F2/P8 regression: the `..` that opens the second delta is part of the
    // sentence, not its own synthesis request.
    let log: ProviderCallLog = Arc::new(Mutex::new(Vec::new()));
    let llm = Arc::new(scripted_llm(&["Wait.", ".. Next."]).with_call_log(Arc::clone(&log)));
    let tts = Arc::new(FakeTts::new().with_call_log(Arc::clone(&log)));

    let (mut session, report) = run(
        SessionConfig {
            pipelined_tts: true,
            ..SessionConfig::default()
        },
        providers_for(llm, tts),
    )
    .await;
    let _ = session.take_events();

    let units: Vec<String> = log
        .lock()
        .unwrap()
        .iter()
        .filter_map(|call| match call {
            ProviderCall::TtsSynthesize(text) => Some(text.clone()),
            _ => None,
        })
        .collect();

    assert_eq!(
        units,
        vec!["Wait... ".to_string(), "Next.".to_string()],
        "ellipsis was split into fragments: {units:?}"
    );
    assert!(units.iter().all(|unit| !unit.trim().is_empty()));
    assert!(
        units
            .iter()
            .all(|unit| unit.trim() != "." && unit.trim() != ". "),
        "a lone punctuation fragment reached the provider: {units:?}"
    );
    assert_eq!(units.concat(), "Wait... Next.");
    assert_eq!(report.reply, "Wait... Next.");
    assert!(!report.interrupted);
}

#[tokio::test]
async fn pipelined_tts_barge_in_leaves_no_synthesizer_alive() {
    let log: ProviderCallLog = Arc::new(Mutex::new(Vec::new()));
    let llm =
        Arc::new(scripted_llm(&["First. ", "Second. ", "Third."]).with_call_log(Arc::clone(&log)));
    let tts = Arc::new(FakeTts::new().with_call_log(Arc::clone(&log)));

    let (mut session, report) = run(
        SessionConfig {
            pipelined_tts: true,
            barge_in_after_deltas: Some(3),
            ..SessionConfig::default()
        },
        providers_for(llm, tts),
    )
    .await;
    let events = session.take_events();

    // The terminal event is the interrupted TurnDone, and nothing follows it.
    match events.last() {
        Some(SessionEvent::TurnDone {
            interrupted: true, ..
        }) => {}
        other => panic!("expected an interrupted TurnDone last, got {other:?}"),
    }

    // Every synthesis that started was abandoned, never completed: the drop
    // guard is the "no synthesizer task alive" assertion, and it does not
    // depend on timing.
    let calls = log.lock().unwrap().clone();
    let started = calls
        .iter()
        .filter(|call| matches!(call, ProviderCall::TtsSynthesize(_)))
        .count();
    let dropped = calls
        .iter()
        .filter(|call| matches!(call, ProviderCall::TtsStreamDropped))
        .count();
    assert_eq!(
        started, 1,
        "expected exactly one in-flight synthesis: {calls:?}"
    );
    assert_eq!(
        dropped, started,
        "barge-in left a synthesizer alive: {calls:?}"
    );

    assert!(report.interrupted);
}

/// In-memory exporter collecting every finished span.
///
/// `tests/otel.rs` defines the same minimal exporter; this module keeps its own
/// copy so the two integration-test crates stay independent and neither has to
/// share a `tests/common` module (out of scope for this change).
#[cfg(feature = "otel")]
#[derive(Clone, Debug, Default)]
struct CollectingExporter {
    spans: Arc<Mutex<Vec<SpanData>>>,
}

#[cfg(feature = "otel")]
impl CollectingExporter {
    fn finished(&self) -> Vec<SpanData> {
        self.spans.lock().expect("exporter mutex poisoned").clone()
    }
}

#[cfg(feature = "otel")]
impl SpanExporter for CollectingExporter {
    async fn export(&self, batch: Vec<SpanData>) -> OTelSdkResult {
        self.spans
            .lock()
            .expect("exporter mutex poisoned")
            .extend(batch);
        Ok(())
    }
}

#[cfg(feature = "otel")]
fn i64_attribute(span: &SpanData, key: &str) -> Option<i64> {
    span.attributes
        .iter()
        .find(|attribute| attribute.key.as_str() == key)
        .and_then(|attribute| match attribute.value {
            Value::I64(value) => Some(value),
            _ => None,
        })
}

#[cfg(feature = "otel")]
#[tokio::test]
async fn pipelined_tts_spans_are_children_of_the_turn() {
    // F3: a green feature build is not evidence of overlapping spans. Register
    // an in-memory exporter, run one pipelined turn, and inspect the exported
    // spans themselves.
    let exporter = CollectingExporter::default();
    let provider = SdkTracerProvider::builder()
        .with_span_processor(SimpleSpanProcessor::new(exporter.clone()))
        .build();
    let tracer = provider.tracer("parlevox-runtime-pipelined-test");

    let log: ProviderCallLog = Arc::new(Mutex::new(Vec::new()));
    let llm = Arc::new(PacedLlm::new(&["One. ", "Two. ", "Three."]));
    let tts = Arc::new(FakeTts::new().with_call_log(Arc::clone(&log)));

    let mut session = Session::with_tracer(
        SessionConfig {
            pipelined_tts: true,
            ..SessionConfig::default()
        },
        providers_for(llm, tts),
        tracer,
    )
    .expect("session builds");
    let report = session
        .run_turn(demo_utterance(16_000, 1, 40))
        .await
        .expect("pipelined turn completes");
    let _ = session.take_events();

    // Flush the simple span processor before reading: an exporter that is not
    // shut down may still hold the spans in flight.
    provider.shutdown().expect("provider shuts down cleanly");

    let requests: Vec<String> = log
        .lock()
        .unwrap()
        .iter()
        .filter_map(|call| match call {
            ProviderCall::TtsSynthesize(text) => Some(text.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        requests.len(),
        3,
        "expected one synthesis request per sentence: {requests:?}"
    );

    let spans = exporter.finished();
    let turn_spans: Vec<&SpanData> = spans.iter().filter(|span| span.name == "turn").collect();
    assert_eq!(turn_spans.len(), 1, "expected exactly one turn span");
    let turn = turn_spans[0];
    let turn_id = turn.span_context.span_id();
    let trace_id = turn.span_context.trace_id();

    let mut tts_spans: Vec<&SpanData> = spans.iter().filter(|span| span.name == "tts").collect();
    assert_eq!(
        tts_spans.len(),
        requests.len(),
        "expected exactly one tts span per synthesized sentence"
    );
    // Spans finish in synthesis order; sorting by start makes the index
    // assertion independent of exporter batching.
    tts_spans.sort_by_key(|span| span.start_time);

    for (index, span) in tts_spans.iter().enumerate() {
        assert_eq!(
            span.span_context.trace_id(),
            trace_id,
            "tts span {index} must share the turn trace id"
        );
        assert_eq!(
            span.parent_span_id, turn_id,
            "tts span {index} must be a direct child of the turn span"
        );
        let expected_index = i64::try_from(index).expect("index fits i64");
        assert_eq!(
            i64_attribute(span, "tts.sentence_index"),
            Some(expected_index),
            "tts spans must carry sequential sentence indexes"
        );
        let expected_chars =
            i64::try_from(requests[index].chars().count()).expect("char count fits i64");
        assert_eq!(
            i64_attribute(span, "tts.sentence_chars"),
            Some(expected_chars),
            "tts.sentence_chars must describe the recorded request"
        );
    }

    let llm_span = spans
        .iter()
        .find(|span| span.name == "llm")
        .expect("the pipelined phase opens an llm span");
    assert!(
        tts_spans[0].start_time < llm_span.end_time,
        "the first tts span must start before the llm span ends (tts starts {:?}, llm ends {:?})",
        tts_spans[0].start_time,
        llm_span.end_time
    );

    assert_eq!(report.reply, "One. Two. Three.");
    assert!(!report.interrupted);
}
