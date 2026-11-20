//! Integration tests for the provider fallback chains.
//!
//! The doubles below implement the real [`LlmProvider`] / [`TtsProvider`]
//! traits, record their name in a shared call log, and script their failures
//! so every test can prove *which* provider ran — not merely which error came
//! back.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures_util::StreamExt;
use parlevox_core::{ErrorClass, ParlevoxError};
use parlevox_providers::{
    AudioChunk, FallbackLlm, FallbackTts, GenerateRequest, LlmProvider, Modality, ProviderManifest,
    ProviderStream, SynthesizeRequest, TtsProvider,
};
use tokio_util::sync::CancellationToken;

type ErrFactory = fn() -> ParlevoxError;
type CallLog = Arc<Mutex<Vec<&'static str>>>;

fn transient_error() -> ParlevoxError {
    ParlevoxError::provider("scripted", "temporary failure")
}

fn capacity_error() -> ParlevoxError {
    ParlevoxError::Capacity("queue full".into())
}

fn fatal_error() -> ParlevoxError {
    ParlevoxError::fatal("scripted", "bad credentials")
}

fn protocol_error() -> ParlevoxError {
    ParlevoxError::Protocol("malformed frame".into())
}

fn cancelled_error() -> ParlevoxError {
    ParlevoxError::Cancelled
}

/// What a double does when it is called.
enum Script<T> {
    /// The `generate`/`synthesize` call itself fails before a stream exists.
    FailStart(ErrFactory),
    /// A stream of pre-scripted results is returned.
    Items(Vec<Result<T, ErrFactory>>),
}

fn recorded_calls(calls: &CallLog) -> Vec<&'static str> {
    calls
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}

fn record(calls: &CallLog, name: &'static str) {
    calls
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(name);
}

struct ScriptedLlm {
    name: &'static str,
    manifest: ProviderManifest,
    script: Script<String>,
    calls: CallLog,
    dropped: AtomicU64,
}

impl ScriptedLlm {
    fn scripted(
        name: &'static str,
        script: Script<String>,
        calls: &CallLog,
        dropped: u64,
    ) -> Arc<dyn LlmProvider> {
        Arc::new(ScriptedLlm {
            name,
            manifest: ProviderManifest::new(name, Modality::Llm),
            script,
            calls: Arc::clone(calls),
            dropped: AtomicU64::new(dropped),
        })
    }
}

#[async_trait]
impl LlmProvider for ScriptedLlm {
    fn manifest(&self) -> &ProviderManifest {
        &self.manifest
    }

    async fn generate(
        &self,
        _request: GenerateRequest,
        _cancel: CancellationToken,
    ) -> Result<ProviderStream<String>, ParlevoxError> {
        record(&self.calls, self.name);
        match &self.script {
            Script::FailStart(make) => Err(make()),
            Script::Items(items) => {
                let items = items
                    .iter()
                    .map(|result| match result {
                        Ok(item) => Ok(item.clone()),
                        Err(make) => Err(make()),
                    })
                    .collect::<Vec<_>>();
                Ok(Box::pin(futures_util::stream::iter(items)))
            }
        }
    }

    fn take_sidecar_dropped_frames(&self) -> u64 {
        self.dropped.swap(0, Ordering::Relaxed)
    }
}

struct ScriptedTts {
    name: &'static str,
    manifest: ProviderManifest,
    script: Script<AudioChunk>,
    calls: CallLog,
    dropped: AtomicU64,
}

impl ScriptedTts {
    fn scripted(
        name: &'static str,
        script: Script<AudioChunk>,
        calls: &CallLog,
        dropped: u64,
    ) -> Arc<dyn TtsProvider> {
        Arc::new(ScriptedTts {
            name,
            manifest: ProviderManifest::new(name, Modality::Tts),
            script,
            calls: Arc::clone(calls),
            dropped: AtomicU64::new(dropped),
        })
    }
}

#[async_trait]
impl TtsProvider for ScriptedTts {
    fn manifest(&self) -> &ProviderManifest {
        &self.manifest
    }

    async fn synthesize(
        &self,
        _request: SynthesizeRequest,
        _cancel: CancellationToken,
    ) -> Result<ProviderStream<AudioChunk>, ParlevoxError> {
        record(&self.calls, self.name);
        match &self.script {
            Script::FailStart(make) => Err(make()),
            Script::Items(items) => {
                let items = items
                    .iter()
                    .map(|result| match result {
                        Ok(item) => Ok(item.clone()),
                        Err(make) => Err(make()),
                    })
                    .collect::<Vec<_>>();
                Ok(Box::pin(futures_util::stream::iter(items)))
            }
        }
    }

    fn take_sidecar_dropped_frames(&self) -> u64 {
        self.dropped.swap(0, Ordering::Relaxed)
    }
}

async fn drain_llm(mut stream: ProviderStream<String>) -> (Vec<String>, Option<ParlevoxError>) {
    let mut items = Vec::new();
    while let Some(result) = stream.next().await {
        match result {
            Ok(item) => items.push(item),
            Err(error) => return (items, Some(error)),
        }
    }
    (items, None)
}

async fn drain_tts(
    mut stream: ProviderStream<AudioChunk>,
) -> (Vec<AudioChunk>, Option<ParlevoxError>) {
    let mut items = Vec::new();
    while let Some(result) = stream.next().await {
        match result {
            Ok(item) => items.push(item),
            Err(error) => return (items, Some(error)),
        }
    }
    (items, None)
}

fn generate_request() -> GenerateRequest {
    GenerateRequest::new("hello")
}

fn synthesize_request() -> SynthesizeRequest {
    SynthesizeRequest {
        text: "hello".to_owned(),
        sample_rate: 16_000,
        channels: 1,
    }
}

fn chunk(sample: i16) -> AudioChunk {
    AudioChunk::new(16_000, 1, vec![sample])
}

#[tokio::test]
async fn transient_start_error_switches_to_next_provider() {
    // Arrange
    let calls: CallLog = Arc::new(Mutex::new(Vec::new()));
    let primary = ScriptedLlm::scripted("primary", Script::FailStart(transient_error), &calls, 0);
    let backup = ScriptedLlm::scripted(
        "backup",
        Script::Items(vec![Ok("second".to_owned())]),
        &calls,
        0,
    );
    let chain = FallbackLlm::new(vec![primary, backup]).expect("chain is non-empty");

    // Act
    let stream = chain
        .generate(generate_request(), CancellationToken::new())
        .await
        .expect("chain returns a stream");
    let (items, error) = drain_llm(stream).await;

    // Assert
    assert_eq!(items, vec!["second".to_owned()]);
    assert!(error.is_none());
    assert_eq!(recorded_calls(&calls), vec!["primary", "backup"]);
    let stats = chain.stats();
    assert_eq!(stats.switched, 1);
    assert_eq!(stats.exhausted, 0);
}

#[tokio::test]
async fn capacity_mid_stream_before_commit_switches_to_next_provider() {
    // Arrange
    let calls: CallLog = Arc::new(Mutex::new(Vec::new()));
    let primary = ScriptedLlm::scripted(
        "primary",
        Script::Items(vec![Err(capacity_error)]),
        &calls,
        0,
    );
    let backup = ScriptedLlm::scripted(
        "backup",
        Script::Items(vec![Ok("second".to_owned())]),
        &calls,
        0,
    );
    let chain = FallbackLlm::new(vec![primary, backup]).expect("chain is non-empty");

    // Act
    let stream = chain
        .generate(generate_request(), CancellationToken::new())
        .await
        .expect("chain returns a stream");
    let (items, error) = drain_llm(stream).await;

    // Assert
    assert_eq!(items, vec!["second".to_owned()]);
    assert!(error.is_none());
    assert_eq!(recorded_calls(&calls), vec!["primary", "backup"]);
    assert_eq!(chain.stats().switched, 1);
}

#[tokio::test]
async fn fatal_error_does_not_switch_providers() {
    // Arrange
    let calls: CallLog = Arc::new(Mutex::new(Vec::new()));
    let primary = ScriptedLlm::scripted("primary", Script::FailStart(fatal_error), &calls, 0);
    let backup = ScriptedLlm::scripted(
        "backup",
        Script::Items(vec![Ok("second".to_owned())]),
        &calls,
        0,
    );
    let chain = FallbackLlm::new(vec![primary, backup]).expect("chain is non-empty");

    // Act
    let stream = chain
        .generate(generate_request(), CancellationToken::new())
        .await
        .expect("chain returns a stream");
    let (items, error) = drain_llm(stream).await;

    // Assert
    assert!(items.is_empty());
    let error = error.expect("the fatal error is surfaced");
    assert_eq!(error.class(), ErrorClass::Fatal);
    assert_eq!(recorded_calls(&calls), vec!["primary"]);
    assert_eq!(chain.stats().switched, 0);
    assert_eq!(chain.stats().exhausted, 0);
}

#[tokio::test]
async fn protocol_error_does_not_switch_providers() {
    // Arrange
    let calls: CallLog = Arc::new(Mutex::new(Vec::new()));
    let primary = ScriptedLlm::scripted(
        "primary",
        Script::Items(vec![Err(protocol_error)]),
        &calls,
        0,
    );
    let backup = ScriptedLlm::scripted(
        "backup",
        Script::Items(vec![Ok("second".to_owned())]),
        &calls,
        0,
    );
    let chain = FallbackLlm::new(vec![primary, backup]).expect("chain is non-empty");

    // Act
    let stream = chain
        .generate(generate_request(), CancellationToken::new())
        .await
        .expect("chain returns a stream");
    let (items, error) = drain_llm(stream).await;

    // Assert
    assert!(items.is_empty());
    let error = error.expect("the protocol error is surfaced");
    assert_eq!(error.class(), ErrorClass::Protocol);
    assert_eq!(recorded_calls(&calls), vec!["primary"]);
}

#[tokio::test]
async fn cancelled_error_does_not_switch_and_never_calls_the_next_provider() {
    // Arrange
    let calls: CallLog = Arc::new(Mutex::new(Vec::new()));
    let primary = ScriptedLlm::scripted("primary", Script::FailStart(cancelled_error), &calls, 0);
    let backup = ScriptedLlm::scripted(
        "backup",
        Script::Items(vec![Ok("second".to_owned())]),
        &calls,
        0,
    );
    let chain = FallbackLlm::new(vec![primary, backup]).expect("chain is non-empty");

    // Act
    let stream = chain
        .generate(generate_request(), CancellationToken::new())
        .await
        .expect("chain returns a stream");
    let (items, error) = drain_llm(stream).await;

    // Assert: the returned error alone is not enough; the call log proves the
    // next provider was never started.
    assert!(items.is_empty());
    let error = error.expect("cancellation is surfaced");
    assert_eq!(error.class(), ErrorClass::Cancelled);
    assert_eq!(recorded_calls(&calls), vec!["primary"]);
    assert_eq!(chain.stats().switched, 0);
}

#[tokio::test]
async fn error_after_committed_item_is_returned_unchanged_and_does_not_switch() {
    // Arrange
    let calls: CallLog = Arc::new(Mutex::new(Vec::new()));
    let primary = ScriptedLlm::scripted(
        "primary",
        Script::Items(vec![Ok("first".to_owned()), Err(transient_error)]),
        &calls,
        0,
    );
    let backup = ScriptedLlm::scripted(
        "backup",
        Script::Items(vec![Ok("second".to_owned())]),
        &calls,
        0,
    );
    let chain = FallbackLlm::new(vec![primary, backup]).expect("chain is non-empty");

    // Act
    let mut stream = chain
        .generate(generate_request(), CancellationToken::new())
        .await
        .expect("chain returns a stream");
    let first = stream.next().await.expect("first item").expect("ok item");
    let error = stream
        .next()
        .await
        .expect("terminal error item")
        .expect_err("the committed provider's error is surfaced");

    // Assert
    assert_eq!(first, "first");
    assert_eq!(error.class(), ErrorClass::Transient);
    // The stream ends after the terminal error, never before it.
    assert!(stream.next().await.is_none());
    assert_eq!(recorded_calls(&calls), vec!["primary"]);
    let stats = chain.stats();
    assert_eq!(stats.switched, 0);
    assert_eq!(stats.exhausted, 0);
}

#[tokio::test]
async fn exhausted_chain_returns_the_last_error_with_its_class_preserved() {
    // Arrange
    let calls: CallLog = Arc::new(Mutex::new(Vec::new()));
    let primary = ScriptedLlm::scripted("primary", Script::FailStart(transient_error), &calls, 0);
    let backup = ScriptedLlm::scripted("backup", Script::FailStart(capacity_error), &calls, 0);
    let chain = FallbackLlm::new(vec![primary, backup]).expect("chain is non-empty");

    // Act
    let stream = chain
        .generate(generate_request(), CancellationToken::new())
        .await
        .expect("chain returns a stream");
    let (items, error) = drain_llm(stream).await;

    // Assert
    assert!(items.is_empty());
    let error = error.expect("the last error is surfaced");
    assert_eq!(error.class(), ErrorClass::Capacity);
    assert_eq!(recorded_calls(&calls), vec!["primary", "backup"]);
    let stats = chain.stats();
    assert_eq!(stats.switched, 1);
    assert_eq!(stats.exhausted, 1);
}

#[tokio::test]
async fn single_provider_chain_behaves_like_that_provider() {
    // Arrange
    let calls: CallLog = Arc::new(Mutex::new(Vec::new()));
    let only = ScriptedLlm::scripted(
        "only",
        Script::Items(vec![Ok("a".to_owned()), Ok("b".to_owned())]),
        &calls,
        0,
    );
    let chain = FallbackLlm::new(vec![only]).expect("chain is non-empty");

    // Act
    assert_eq!(chain.manifest().name, "only");
    let stream = chain
        .generate(generate_request(), CancellationToken::new())
        .await
        .expect("chain returns a stream");
    let (items, error) = drain_llm(stream).await;

    // Assert
    assert_eq!(items, vec!["a".to_owned(), "b".to_owned()]);
    assert!(error.is_none());
    assert_eq!(recorded_calls(&calls), vec!["only"]);
    let stats = chain.stats();
    assert_eq!(stats.switched, 0);
    assert_eq!(stats.exhausted, 0);
}

#[tokio::test]
async fn single_provider_switchable_failure_counts_exhaustion_not_switch() {
    // Arrange
    let calls: CallLog = Arc::new(Mutex::new(Vec::new()));
    let only = ScriptedLlm::scripted("only", Script::FailStart(transient_error), &calls, 0);
    let chain = FallbackLlm::new(vec![only]).expect("chain is non-empty");

    // Act
    let stream = chain
        .generate(generate_request(), CancellationToken::new())
        .await
        .expect("chain returns a stream");
    let (items, error) = drain_llm(stream).await;

    // Assert
    assert!(items.is_empty());
    let error = error.expect("the only provider's error is surfaced unchanged");
    assert_eq!(error.class(), ErrorClass::Transient);
    assert_eq!(recorded_calls(&calls), vec!["only"]);
    let stats = chain.stats();
    assert_eq!(stats.switched, 0);
    assert_eq!(stats.exhausted, 1);
}

#[tokio::test]
async fn take_sidecar_dropped_frames_sums_every_provider_in_the_chain() {
    // Arrange
    let calls: CallLog = Arc::new(Mutex::new(Vec::new()));
    let primary = ScriptedLlm::scripted("primary", Script::Items(Vec::new()), &calls, 3);
    let backup = ScriptedLlm::scripted("backup", Script::Items(Vec::new()), &calls, 4);
    let chain = FallbackLlm::new(vec![primary, backup]).expect("chain is non-empty");

    // Act and assert: every provider is drained once, so the second take is 0.
    assert_eq!(chain.take_sidecar_dropped_frames(), 7);
    assert_eq!(chain.take_sidecar_dropped_frames(), 0);
}

#[tokio::test]
async fn empty_llm_chain_is_rejected_at_construction() {
    // Act
    let result = FallbackLlm::new(Vec::new());

    // Assert: an Err, never a panic.
    let error = match result {
        Ok(_) => panic!("an empty chain must be rejected"),
        Err(error) => error,
    };
    assert_eq!(error.class(), ErrorClass::Protocol);
}

#[tokio::test]
async fn manifest_delegates_to_the_first_provider() {
    // Arrange
    let calls: CallLog = Arc::new(Mutex::new(Vec::new()));
    let primary = ScriptedLlm::scripted("primary", Script::Items(Vec::new()), &calls, 0);
    let backup = ScriptedLlm::scripted("backup", Script::Items(Vec::new()), &calls, 0);

    // Act
    let chain = FallbackLlm::new(vec![primary, backup]).expect("chain is non-empty");

    // Assert
    assert_eq!(chain.manifest().name, "primary");
}

#[tokio::test]
async fn healthy_chain_leaves_both_counters_at_zero() {
    // Arrange
    let calls: CallLog = Arc::new(Mutex::new(Vec::new()));
    let primary = ScriptedLlm::scripted(
        "primary",
        Script::Items(vec![Ok("a".to_owned())]),
        &calls,
        0,
    );
    let backup = ScriptedLlm::scripted(
        "backup",
        Script::Items(vec![Ok("never".to_owned())]),
        &calls,
        0,
    );
    let chain = FallbackLlm::new(vec![primary, backup]).expect("chain is non-empty");

    // Act
    let stream = chain
        .generate(generate_request(), CancellationToken::new())
        .await
        .expect("chain returns a stream");
    let (items, error) = drain_llm(stream).await;

    // Assert
    assert_eq!(items, vec!["a".to_owned()]);
    assert!(error.is_none());
    assert_eq!(recorded_calls(&calls), vec!["primary"]);
    let stats = chain.stats();
    assert_eq!(stats.switched, 0);
    assert_eq!(stats.exhausted, 0);
}

#[tokio::test]
async fn tts_transient_before_commit_switches_to_next_provider() {
    // Arrange
    let calls: CallLog = Arc::new(Mutex::new(Vec::new()));
    let primary = ScriptedTts::scripted("primary", Script::FailStart(transient_error), &calls, 0);
    let backup = ScriptedTts::scripted("backup", Script::Items(vec![Ok(chunk(2))]), &calls, 0);
    let chain = FallbackTts::new(vec![primary, backup]).expect("chain is non-empty");

    // Act
    let stream = chain
        .synthesize(synthesize_request(), CancellationToken::new())
        .await
        .expect("chain returns a stream");
    let (items, error) = drain_tts(stream).await;

    // Assert
    assert_eq!(items, vec![chunk(2)]);
    assert!(error.is_none());
    assert_eq!(recorded_calls(&calls), vec!["primary", "backup"]);
    assert_eq!(chain.stats().switched, 1);
}

#[tokio::test]
async fn tts_cancelled_error_does_not_switch() {
    // Arrange
    let calls: CallLog = Arc::new(Mutex::new(Vec::new()));
    let primary = ScriptedTts::scripted("primary", Script::FailStart(cancelled_error), &calls, 0);
    let backup = ScriptedTts::scripted("backup", Script::Items(vec![Ok(chunk(2))]), &calls, 0);
    let chain = FallbackTts::new(vec![primary, backup]).expect("chain is non-empty");

    // Act
    let stream = chain
        .synthesize(synthesize_request(), CancellationToken::new())
        .await
        .expect("chain returns a stream");
    let (items, error) = drain_tts(stream).await;

    // Assert
    assert!(items.is_empty());
    let error = error.expect("cancellation is surfaced");
    assert_eq!(error.class(), ErrorClass::Cancelled);
    assert_eq!(recorded_calls(&calls), vec!["primary"]);
}

#[tokio::test]
async fn tts_error_after_committed_item_does_not_switch() {
    // Arrange
    let calls: CallLog = Arc::new(Mutex::new(Vec::new()));
    let primary = ScriptedTts::scripted(
        "primary",
        Script::Items(vec![Ok(chunk(1)), Err(transient_error)]),
        &calls,
        0,
    );
    let backup = ScriptedTts::scripted("backup", Script::Items(vec![Ok(chunk(2))]), &calls, 0);
    let chain = FallbackTts::new(vec![primary, backup]).expect("chain is non-empty");

    // Act
    let stream = chain
        .synthesize(synthesize_request(), CancellationToken::new())
        .await
        .expect("chain returns a stream");
    let (items, error) = drain_tts(stream).await;

    // Assert
    assert_eq!(items, vec![chunk(1)]);
    let error = error.expect("the committed provider's error is surfaced");
    assert_eq!(error.class(), ErrorClass::Transient);
    assert_eq!(recorded_calls(&calls), vec!["primary"]);
    assert_eq!(chain.stats().switched, 0);
}

#[tokio::test]
async fn tts_exhaustion_is_counted() {
    // Arrange
    let calls: CallLog = Arc::new(Mutex::new(Vec::new()));
    let primary = ScriptedTts::scripted("primary", Script::FailStart(transient_error), &calls, 0);
    let backup = ScriptedTts::scripted("backup", Script::FailStart(capacity_error), &calls, 0);
    let chain = FallbackTts::new(vec![primary, backup]).expect("chain is non-empty");

    // Act
    let stream = chain
        .synthesize(synthesize_request(), CancellationToken::new())
        .await
        .expect("chain returns a stream");
    let (items, error) = drain_tts(stream).await;

    // Assert
    assert!(items.is_empty());
    let error = error.expect("the last error is surfaced");
    assert_eq!(error.class(), ErrorClass::Capacity);
    let stats = chain.stats();
    assert_eq!(stats.switched, 1);
    assert_eq!(stats.exhausted, 1);
}

#[tokio::test]
async fn empty_tts_chain_is_rejected_at_construction() {
    // Act
    let result = FallbackTts::new(Vec::new());

    // Assert
    let error = match result {
        Ok(_) => panic!("an empty chain must be rejected"),
        Err(error) => error,
    };
    assert_eq!(error.class(), ErrorClass::Protocol);
}
