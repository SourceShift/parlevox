//! Hermetic acceptance tests for the connector family.
//!
//! Every response is a recorded fixture replayed through
//! [`ReplayTransport`]; no test opens a socket, reads a real environment
//! variable, or needs a key.

#![cfg(feature = "http")]

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::Path;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use bytes::Bytes;
use futures_core::Stream;
use futures_util::StreamExt;
use parlevox_connectors::{
    build_provider, bytes_stream, Auth, BuiltProvider, ConnectorConfig, ConnectorConfigBuilder,
    ConnectorError, Credentials, ReplayTransport, Secret, Transport, TransportResponse,
};
use parlevox_core::{DurationMs, ErrorClass, ParlevoxError};
use parlevox_providers::{
    AdapterRef, AudioChunk, GenerateRequest, LlmProvider, Modality, ModelEntry, ModelLimits,
    ProviderDescriptor, ProviderLimits, ProviderRegistry, RetryPolicy, SynthesizeRequest,
};
use tokio_util::sync::CancellationToken;

// ---------------------------------------------------------------------------
// Fixtures and builders
// ---------------------------------------------------------------------------

fn descriptor(
    name: &str,
    modality: Modality,
    kind: &str,
    capabilities: &[&str],
) -> ProviderDescriptor {
    ProviderDescriptor {
        name: name.to_owned(),
        modality,
        adapter: AdapterRef {
            kind: kind.to_owned(),
            endpoint: Some(format!("https://example.invalid/{kind}")),
        },
        api_key_env: None,
        models: vec![ModelEntry {
            id: "test-model".to_owned(),
            display_name: String::new(),
            capabilities: Vec::new(),
            limits: ModelLimits::default(),
            modality: None,
        }],
        capabilities: capabilities.iter().map(|cap| (*cap).to_owned()).collect(),
        provider_limits: ProviderLimits::default(),
        config_schema: serde_json::json!({"type": "object", "properties": {}}),
        protocol_version: 2,
    }
}

fn llm_descriptor() -> ProviderDescriptor {
    descriptor(
        "test-llm",
        Modality::Llm,
        "openai-compatible-chat",
        &["streaming"],
    )
}

fn response(status: u16, content_type: &str, body: impl Into<Vec<u8>>) -> TransportResponse {
    TransportResponse::new(status, bytes_stream([Bytes::from(body.into())]))
        .with_header("content-type", content_type)
        .expect("content type is a valid header")
}

fn build(
    provider: &ProviderDescriptor,
    transport: Arc<dyn Transport>,
    configure: impl FnOnce(ConnectorConfigBuilder) -> ConnectorConfigBuilder,
) -> BuiltProvider {
    let config = configure(ConnectorConfig::builder(transport)).build();
    build_provider(provider, Credentials::new("sk-test-unit-key"), config).expect("provider builds")
}

fn build_llm(
    transport: Arc<dyn Transport>,
    configure: impl FnOnce(ConnectorConfigBuilder) -> ConnectorConfigBuilder,
) -> Box<dyn LlmProvider> {
    match build(&llm_descriptor(), transport, configure) {
        BuiltProvider::Llm(provider) => provider,
        other => panic!("expected an llm provider, got {other:?}"),
    }
}

async fn drain_llm(provider: &dyn LlmProvider) -> Vec<String> {
    let mut stream = provider
        .generate(GenerateRequest::new("hello"), CancellationToken::new())
        .await
        .expect("generate returns a stream");
    let mut deltas = Vec::new();
    while let Some(item) = stream.next().await {
        deltas.push(item.expect("delta"));
    }
    deltas
}

// ---------------------------------------------------------------------------
// Required acceptance tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn streaming_partials_then_one_final_from_recorded_sse() {
    let transport: Arc<dyn Transport> = Arc::new(ReplayTransport::new(vec![response(
        200,
        "text/event-stream",
        include_str!("fixtures/stt_stream.sse"),
    )]));
    let provider = match build(
        &descriptor(
            "test-stt",
            Modality::Stt,
            "openai-audio-stt",
            &["streaming"],
        ),
        transport,
        std::convert::identity,
    ) {
        BuiltProvider::Stt(provider) => provider,
        other => panic!("expected an stt provider, got {other:?}"),
    };
    let audio = AudioChunk::new(16_000, 1, vec![0i16; 160]);
    let mut stream = provider
        .transcribe(audio, CancellationToken::new())
        .await
        .expect("transcribe returns a stream");
    let mut chunks = Vec::new();
    while let Some(item) = stream.next().await {
        chunks.push(item.expect("transcript chunk"));
    }
    let partials: Vec<_> = chunks.iter().filter(|chunk| !chunk.is_final).collect();
    let finals: Vec<_> = chunks.iter().filter(|chunk| chunk.is_final).collect();
    assert_eq!(partials.len(), 2, "two partials: {chunks:?}");
    assert_eq!(partials[0].text, "Hello");
    assert_eq!(partials[1].text, " world");
    assert_eq!(finals.len(), 1, "exactly one final: {chunks:?}");
    assert_eq!(finals[0].text, "Hello world");
    assert!(
        chunks.last().expect("stream was not empty").is_final,
        "the final chunk is last"
    );
}

#[tokio::test]
async fn rate_limited_429_is_classified_and_retried_per_policy() {
    let first = response(429, "application/json", r#"{"error":"slow down"}"#)
        .with_header("retry-after", "0")
        .expect("retry-after is a valid header");
    let second = response(
        200,
        "text/event-stream",
        include_str!("fixtures/chat_stream.sse"),
    );
    let replay = Arc::new(ReplayTransport::new(vec![first, second]));
    let transport: Arc<dyn Transport> = replay.clone();
    let provider = build_llm(transport, |builder| {
        builder.retry_policy(RetryPolicy {
            max_attempts: 3,
            base_delay: Duration::ZERO,
            max_delay: Duration::ZERO,
            jitter: false,
        })
    });
    let deltas = drain_llm(provider.as_ref()).await;
    assert_eq!(deltas, vec!["Hel".to_owned(), "lo".to_owned()]);
    assert_eq!(
        replay.request_count(),
        2,
        "the 429 attempt and one retry were both issued"
    );

    let classified = ConnectorError::from_status(429, Some(DurationMs::from_millis(0)))
        .expect("429 is an error");
    match &classified {
        ConnectorError::RateLimited { retry_after } => {
            assert_eq!(*retry_after, Some(DurationMs::from_millis(0)));
        }
        other => panic!("expected a rate limit classification, got {other:?}"),
    }
    // Documented core gap: ErrorClass has no RateLimit variant, so the public
    // class is Transient with the retry hint preserved.
    let mapped = classified.into_provider("test-llm");
    assert_eq!(mapped.class(), ErrorClass::Transient);
    match mapped {
        ParlevoxError::Provider { retry_after, .. } => {
            assert_eq!(retry_after, Some(DurationMs::from_millis(0)));
        }
        other => panic!("expected a provider error, got {other:?}"),
    }
}

struct CountingStream {
    chunks: VecDeque<Bytes>,
    pulls: Arc<AtomicUsize>,
    dropped: Arc<AtomicBool>,
}

impl Stream for CountingStream {
    type Item = Result<Bytes, ConnectorError>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        match this.chunks.pop_front() {
            Some(chunk) => {
                this.pulls.fetch_add(1, Ordering::SeqCst);
                Poll::Ready(Some(Ok(chunk)))
            }
            None => Poll::Ready(None),
        }
    }
}

impl Drop for CountingStream {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::SeqCst);
    }
}

#[tokio::test]
async fn cancellation_observed_within_one_chunk() {
    let pulls = Arc::new(AtomicUsize::new(0));
    let dropped = Arc::new(AtomicBool::new(false));
    let first = Bytes::from_static(
        b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"first\"}}]}\n\n",
    );
    let second = Bytes::from_static(
        b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"second\"}}]}\n\n",
    );
    let body = CountingStream {
        chunks: VecDeque::from(vec![first, second]),
        pulls: Arc::clone(&pulls),
        dropped: Arc::clone(&dropped),
    };
    let response = TransportResponse::new(200, Box::pin(body))
        .with_header("content-type", "text/event-stream")
        .expect("content type is a valid header");
    let transport: Arc<dyn Transport> = Arc::new(ReplayTransport::new(vec![response]));
    let provider = build_llm(transport, std::convert::identity);
    let cancel = CancellationToken::new();
    let mut stream = provider
        .generate(GenerateRequest::new("hello"), cancel.clone())
        .await
        .expect("generate returns a stream");

    assert_eq!(
        stream.next().await.expect("first item").expect("delta"),
        "first"
    );
    assert_eq!(pulls.load(Ordering::SeqCst), 1, "only one chunk so far");

    cancel.cancel();
    let error = stream
        .next()
        .await
        .expect("cancellation is reported")
        .expect_err("cancelled stream yields an error");
    assert!(matches!(error, ParlevoxError::Cancelled), "{error:?}");
    assert_eq!(
        pulls.load(Ordering::SeqCst),
        1,
        "the second fixture chunk must not be pulled"
    );
    assert!(
        dropped.load(Ordering::SeqCst),
        "the body stream must be dropped so the request aborts"
    );
    assert!(
        stream.next().await.is_none(),
        "the stream ends after cancellation"
    );
}

#[tokio::test]
async fn dropped_frames_under_backpressure_are_counted() {
    let pcm = include_bytes!("fixtures/tts.pcm").as_slice();
    let transport: Arc<dyn Transport> =
        Arc::new(ReplayTransport::new(vec![response(200, "audio/pcm", pcm)]));
    let provider = match build(
        &descriptor("test-tts", Modality::Tts, "openai-audio-tts", &[]),
        transport,
        |builder| builder.max_buffered_frames(1),
    ) {
        BuiltProvider::Tts(provider) => provider,
        other => panic!("expected a tts provider, got {other:?}"),
    };
    let mut stream = provider
        .synthesize(
            SynthesizeRequest {
                text: "hello".to_owned(),
                sample_rate: 24_000,
                channels: 1,
            },
            CancellationToken::new(),
        )
        .await
        .expect("synthesize returns a stream");
    let first = stream.next().await.expect("one frame").expect("frame");
    assert_eq!(first.pcm.len(), 480, "one 20 ms frame at 24 kHz");
    // The fixture is three frames; the queue held one, dropping the oldest
    // two, so the surviving frame starts at sample 960.
    assert_eq!(first.pcm[0], 960);

    let delivered = 1u64;
    let pushed = 3u64;
    let dropped = provider.take_sidecar_dropped_frames();
    assert_eq!(dropped, 2, "two frames were discarded");
    assert_eq!(pushed, delivered + dropped, "pushed == delivered + dropped");
    assert_eq!(
        provider.take_sidecar_dropped_frames(),
        0,
        "the report is take-and-reset"
    );
    assert!(stream.next().await.is_none(), "the queue is drained");
}

#[tokio::test]
async fn auth_injection_covers_bearer_header_and_query_template() {
    struct Case {
        auth: Auth,
        header: Option<(&'static str, &'static str)>,
        query: Option<(&'static str, &'static str)>,
    }
    let cases = vec![
        Case {
            auth: Auth::bearer(Secret::new("sk-test-bearer")),
            header: Some(("authorization", "Bearer sk-test-bearer")),
            query: None,
        },
        Case {
            auth: Auth::custom_header("x-api-key", Secret::new("sk-test-header"))
                .expect("valid custom header"),
            header: Some(("x-api-key", "sk-test-header")),
            query: None,
        },
        Case {
            auth: Auth::query_template("api_key", "key={key}", Secret::new("sk-test-query"))
                .expect("valid query template"),
            header: None,
            query: Some(("api_key", "key=sk-test-query")),
        },
    ];

    for case in cases {
        let replay = Arc::new(ReplayTransport::new(vec![response(
            200,
            "application/json",
            include_str!("fixtures/chat_single.json"),
        )]));
        let transport: Arc<dyn Transport> = replay.clone();
        let provider = build_llm(transport, |builder| builder.auth(case.auth));
        let deltas = drain_llm(provider.as_ref()).await;
        assert_eq!(deltas, vec!["Hello from JSON".to_owned()]);

        let requests = replay.requests();
        let request = requests.first().expect("one request was recorded");
        if let Some((name, value)) = case.header {
            let header = request
                .headers
                .iter()
                .find(|header| header.name() == name)
                .unwrap_or_else(|| panic!("missing header {name}"));
            assert_eq!(header.value(), value);
        }
        if let Some((name, value)) = case.query {
            let parameter = request
                .query
                .iter()
                .find(|(key, _)| key == name)
                .unwrap_or_else(|| panic!("missing query parameter {name}"));
            assert_eq!(parameter.1, value);
        }
    }
}

struct SharedBuffer(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for SharedBuffer {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        self.0
            .lock()
            .map_err(|_| std::io::Error::other("log buffer poisoned"))?
            .extend_from_slice(data);
        Ok(data.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn planted_fake_key_never_appears_in_logs_or_errors() {
    const PLANTED: &str = "sk-test-planted-connectors-key";
    let buffer = Arc::new(Mutex::new(Vec::new()));
    let writer = Arc::clone(&buffer);
    let subscriber = tracing_subscriber::fmt()
        .with_writer(move || SharedBuffer(Arc::clone(&writer)))
        .with_ansi(false)
        .with_max_level(tracing::Level::TRACE)
        .finish();

    tracing::subscriber::with_default(subscriber, || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime builds");
        runtime.block_on(async {
            let replay = Arc::new(ReplayTransport::new(vec![response(
                401,
                "application/json",
                r#"{"error":"unauthorized"}"#,
            )]));
            let transport: Arc<dyn Transport> = replay.clone();
            let auth = Auth::query_template("api_key", "{key}", Secret::new(PLANTED))
                .expect("valid query template");
            let provider = build_llm(transport, move |builder| builder.auth(auth));
            let error = match provider
                .generate(GenerateRequest::new("hello"), CancellationToken::new())
                .await
            {
                Ok(_) => panic!("401 must be an error"),
                Err(error) => error,
            };
            let rendered = format!("{error} {error:?}");
            assert!(
                !rendered.contains(PLANTED),
                "error leaked the key: {rendered}"
            );
            for request in replay.requests() {
                let debug = format!("{request:?}");
                assert!(
                    !debug.contains(PLANTED),
                    "request debug leaked the key: {debug}"
                );
            }
        });
    });

    let logs = String::from_utf8(buffer.lock().expect("log buffer is not poisoned").clone())
        .expect("logs are UTF-8");
    assert!(
        !logs.contains(PLANTED),
        "a log line leaked the planted key: {logs}"
    );
}

#[tokio::test]
async fn groq_style_descriptor_builds_without_rust_changes() {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/descriptors");
    let registry = ProviderRegistry::load(&[directory]).expect("registry loads");
    let descriptor = registry
        .get(Modality::Llm, "groq-llm")
        .expect("groq-llm is registered");
    let transport: Arc<dyn Transport> = Arc::new(ReplayTransport::new(vec![response(
        200,
        "text/event-stream",
        include_str!("fixtures/chat_stream.sse"),
    )]));
    let provider = match build_provider(
        descriptor,
        Credentials::new("sk-test-groq-key"),
        ConnectorConfig::builder(transport).build(),
    )
    .expect("groq descriptor builds a provider")
    {
        BuiltProvider::Llm(provider) => provider,
        other => panic!("expected an llm provider, got {other:?}"),
    };
    assert_eq!(
        provider.manifest().adapter.kind,
        "openai-compatible-chat",
        "the Groq descriptor reuses the same connector"
    );
    assert_eq!(provider.manifest().name, "groq-llm");
    let mut stream = provider
        .generate(GenerateRequest::new("hi"), CancellationToken::new())
        .await
        .expect("generate returns a stream");
    let mut text = String::new();
    while let Some(item) = stream.next().await {
        text.push_str(&item.expect("delta"));
    }
    assert_eq!(text, "Hello");
}

/// Guards against the connector layer growing a dependency on the runtime
/// crates.
///
/// An earlier version of this test shelled out to `git status` and failed on
/// any working-tree change under a list of protected paths. That measured the
/// caller's checkout rather than this crate, so it failed on any concurrent
/// edit anywhere in the workspace and proved nothing about linking. This is a
/// static dependency and import check instead; do not replace it with a
/// working-tree tripwire again.
#[test]
fn connectors_do_not_link_runtime_crates() {
    // Arrange: the dependencies this crate may legitimately declare.
    const ALLOWED: [&str; 12] = [
        "async-trait",
        "bytes",
        "futures-core",
        "futures-util",
        "parlevox-core",
        "parlevox-providers",
        "reqwest",
        "serde_json",
        "thiserror",
        "tokio",
        "tokio-util",
        "tracing",
    ];
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let manifest_path = manifest_dir.join("Cargo.toml");

    // Act: parse the `[dependencies]` table only, so `[features]`,
    // `[dev-dependencies]` and their `dep:` entries never leak into the set.
    let manifest = std::fs::read_to_string(&manifest_path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", manifest_path.display()));
    let mut in_dependencies = false;
    let mut declared = BTreeMap::new();
    for (index, line) in manifest.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_dependencies = trimmed == "[dependencies]";
            continue;
        }
        if !in_dependencies || trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let (key, _) = trimmed
            .split_once('=')
            .unwrap_or_else(|| panic!("malformed dependency at Cargo.toml:{}", index + 1));
        // `dep.workspace = true` is a dotted key naming the dependency `dep`.
        let name = key
            .trim()
            .split_once('.')
            .map_or_else(|| key.trim(), |(name, _)| name);
        declared.insert(name.to_owned(), index + 1);
    }

    let allowed: BTreeSet<&str> = ALLOWED.into_iter().collect();
    let unexpected: Vec<String> = declared
        .iter()
        .filter(|(name, _)| !allowed.contains(name.as_str()))
        .map(|(name, line)| format!("{name} at Cargo.toml:{line}"))
        .collect();
    let missing: Vec<&str> = allowed
        .iter()
        .filter(|name| !declared.contains_key(**name))
        .copied()
        .collect();

    // Assert: the manifest declares exactly the allow-list, nothing more.
    assert!(
        unexpected.is_empty() && missing.is_empty(),
        "connector dependencies drifted from the allow-list; \
         unexpected: {unexpected:?}, missing: {missing:?}"
    );

    // Act: scan the crate sources for reach-ins into runtime crates.
    let forbidden = [
        "parlevox_runtime",
        "parlevox_ipc",
        "parlevox_sidecar",
        "parlevox_cli",
        "parlevox_transport_ws",
    ];
    let mut hits = BTreeSet::new();
    let mut pending = vec![manifest_dir.join("src")];
    while let Some(directory) = pending.pop() {
        let entries = std::fs::read_dir(&directory)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", directory.display()));
        for entry in entries {
            let path = entry.expect("directory entry is readable").path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                let source = std::fs::read_to_string(&path)
                    .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
                let relative = path.strip_prefix(manifest_dir).unwrap_or(&path);
                for (index, line) in source.lines().enumerate() {
                    if forbidden.iter().any(|name| line.contains(*name)) {
                        hits.insert(format!("{}:{}", relative.display(), index + 1));
                    }
                }
            }
        }
    }

    // Assert: no source file names a runtime crate, and only paths/lines are
    // reported, never the matched text.
    assert!(
        hits.is_empty(),
        "connector sources reach into runtime crates: {hits:?}"
    );
}

// ---------------------------------------------------------------------------
// Additional wire-shape and failure-mode tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ndjson_chat_response_is_streamed() {
    let transport: Arc<dyn Transport> = Arc::new(ReplayTransport::new(vec![response(
        200,
        "application/x-ndjson",
        include_str!("fixtures/chat_ndjson.ndjson"),
    )]));
    let provider = build_llm(transport, std::convert::identity);
    assert_eq!(
        drain_llm(provider.as_ref()).await,
        vec!["Hel".to_owned(), "lo".to_owned()]
    );
}

#[tokio::test]
async fn single_json_transcript_is_one_final() {
    let transport: Arc<dyn Transport> = Arc::new(ReplayTransport::new(vec![response(
        200,
        "application/json",
        include_str!("fixtures/stt_single.json"),
    )]));
    let provider = match build(
        &descriptor("test-stt", Modality::Stt, "openai-audio-stt", &[]),
        transport,
        std::convert::identity,
    ) {
        BuiltProvider::Stt(provider) => provider,
        other => panic!("expected an stt provider, got {other:?}"),
    };
    let audio = AudioChunk::new(16_000, 1, vec![0i16; 160]);
    let mut stream = provider
        .transcribe(audio, CancellationToken::new())
        .await
        .expect("transcribe returns a stream");
    let chunk = stream
        .next()
        .await
        .expect("one transcript chunk")
        .expect("transcript");
    assert!(chunk.is_final);
    assert_eq!(chunk.text, "Hello from whisper");
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn malformed_provider_output_is_protocol_not_panic() {
    let bodies: Vec<Vec<u8>> = vec![
        b"data: not-json\n\n".to_vec(),
        vec![0xff, 0xfe, b'\n', b'\n'],
        b"data: {\"choices\":\"not-an-array\"}\n\n".to_vec(),
    ];
    for body in bodies {
        let transport: Arc<dyn Transport> = Arc::new(ReplayTransport::new(vec![response(
            200,
            "text/event-stream",
            body,
        )]));
        let provider = build_llm(transport, std::convert::identity);
        let mut stream = provider
            .generate(GenerateRequest::new("hello"), CancellationToken::new())
            .await
            .expect("generate returns a stream");
        let error = match stream.next().await {
            Some(item) => item.expect_err("malformed output is an error"),
            None => panic!("expected a protocol error before end of stream"),
        };
        assert!(matches!(error, ParlevoxError::Protocol(_)), "{error:?}");
    }
}

struct FailingStream {
    sent: bool,
}

impl Stream for FailingStream {
    type Item = Result<Bytes, ConnectorError>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.sent {
            return Poll::Ready(Some(Err(ConnectorError::Transport(
                "connection reset".to_owned(),
            ))));
        }
        this.sent = true;
        Poll::Ready(Some(Ok(Bytes::from_static(
            b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"one\"}}]}\n\n",
        ))))
    }
}

#[tokio::test]
async fn mid_stream_failure_is_not_retried() {
    let first = TransportResponse::new(200, Box::pin(FailingStream { sent: false }))
        .with_header("content-type", "text/event-stream")
        .expect("content type is a valid header");
    let second = response(
        200,
        "text/event-stream",
        include_str!("fixtures/chat_stream.sse"),
    );
    let replay = Arc::new(ReplayTransport::new(vec![first, second]));
    let transport: Arc<dyn Transport> = replay.clone();
    let provider = build_llm(transport, |builder| {
        builder.retry_policy(RetryPolicy {
            max_attempts: 3,
            base_delay: Duration::ZERO,
            max_delay: Duration::ZERO,
            jitter: false,
        })
    });
    let mut stream = provider
        .generate(GenerateRequest::new("hello"), CancellationToken::new())
        .await
        .expect("generate returns a stream");
    assert_eq!(stream.next().await.expect("first").expect("delta"), "one");
    let error = stream
        .next()
        .await
        .expect("failure is reported")
        .expect_err("mid-stream failure");
    assert!(matches!(error, ParlevoxError::Transport(_)), "{error:?}");
    assert_eq!(
        replay.request_count(),
        1,
        "a failure after the first body byte must not be retried"
    );
}
