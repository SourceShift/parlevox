//! Integration tests for the opt-in reply-side retry decorator.
//!
//! The [`ScriptedLlm`] double below implements the real [`LlmProvider`] trait,
//! records every `generate` call, and serves one scripted attempt per call. A
//! *stateful* script is what lets these tests prove the decorator called the
//! same provider again — a single canned provider could not distinguish one
//! attempt from three.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use async_trait::async_trait;
use futures_util::StreamExt;
use parlevox_core::{ErrorClass, ParlevoxError};
use parlevox_providers::{
    GenerateRequest, LlmProvider, Modality, ProviderManifest, ProviderStream, RetryPolicy,
    RetryingLlm,
};
use tokio_util::sync::CancellationToken;

type MakeErr = fn() -> ParlevoxError;

fn transient() -> ParlevoxError {
    ParlevoxError::provider("scripted", "temporary failure")
}

fn capacity() -> ParlevoxError {
    ParlevoxError::Capacity("queue full".into())
}

fn fatal() -> ParlevoxError {
    ParlevoxError::fatal("scripted", "bad credentials")
}

fn cancelled() -> ParlevoxError {
    ParlevoxError::Cancelled
}

fn first_failure() -> ParlevoxError {
    ParlevoxError::provider("scripted", "first")
}

fn second_failure() -> ParlevoxError {
    ParlevoxError::provider("scripted", "second")
}

fn third_failure() -> ParlevoxError {
    ParlevoxError::provider("scripted", "third")
}

fn fourth_failure() -> ParlevoxError {
    ParlevoxError::provider("scripted", "fourth")
}

/// One stream item: literal text, or a scripted error made fresh per attempt.
enum Item {
    Text(&'static str),
    Error(MakeErr),
}

/// What one `generate` call does.
enum Script {
    /// The call itself fails before a stream exists.
    Fail(MakeErr),
    /// A stream of scripted items is returned.
    Items(Vec<Item>),
}

struct ScriptedLlm {
    manifest: ProviderManifest,
    scripts: Mutex<VecDeque<Script>>,
    calls: Arc<AtomicU64>,
    dropped: AtomicU64,
    dropped_per_call: u64,
}

impl ScriptedLlm {
    /// Build a provider plus a handle that counts `generate` calls.
    fn scripted(
        scripts: Vec<Script>,
        dropped_per_call: u64,
    ) -> (Arc<dyn LlmProvider>, Arc<AtomicU64>) {
        let calls = Arc::new(AtomicU64::new(0));
        let provider = Arc::new(ScriptedLlm {
            manifest: ProviderManifest::new("inner-provider", Modality::Llm),
            scripts: Mutex::new(scripts.into()),
            calls: Arc::clone(&calls),
            dropped: AtomicU64::new(0),
            dropped_per_call,
        });
        (provider, calls)
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
        self.calls.fetch_add(1, Ordering::Relaxed);
        if self.dropped_per_call != 0 {
            self.dropped
                .fetch_add(self.dropped_per_call, Ordering::Relaxed);
        }
        let script = self
            .scripts
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop_front();
        match script {
            Some(Script::Fail(make)) => Err(make()),
            Some(Script::Items(items)) => {
                let items = items
                    .into_iter()
                    .map(|item| match item {
                        Item::Text(text) => Ok(text.to_owned()),
                        Item::Error(make) => Err(make()),
                    })
                    .collect::<Vec<Result<String, ParlevoxError>>>();
                Ok(Box::pin(futures_util::stream::iter(items)))
            }
            None => {
                let empty: Vec<Result<String, ParlevoxError>> = Vec::new();
                Ok(Box::pin(futures_util::stream::iter(empty)))
            }
        }
    }

    fn take_sidecar_dropped_frames(&self) -> u64 {
        self.dropped.swap(0, Ordering::Relaxed)
    }
}

/// A policy with zero delays so retry tests never touch a timer.
fn retry_policy(max_attempts: u32) -> RetryPolicy {
    RetryPolicy {
        max_attempts,
        base_delay: Duration::ZERO,
        max_delay: Duration::ZERO,
        jitter: false,
    }
}

/// Drain a stream, keeping the deltas seen before any terminal error.
async fn drain(mut stream: ProviderStream<String>) -> (Vec<String>, Option<ParlevoxError>) {
    let mut items = Vec::new();
    while let Some(result) = stream.next().await {
        match result {
            Ok(item) => items.push(item),
            Err(error) => return (items, Some(error)),
        }
    }
    (items, None)
}

async fn generate(retrying: &RetryingLlm, cancel: CancellationToken) -> ProviderStream<String> {
    retrying
        .generate(GenerateRequest::new("hello"), cancel)
        .await
        .expect("generate returns a stream without calling the provider")
}

#[tokio::test]
async fn transient_then_transient_then_success_retries_the_same_provider() {
    let (inner, calls) = ScriptedLlm::scripted(
        vec![
            Script::Fail(transient),
            Script::Fail(transient),
            Script::Items(vec![Item::Text("hello"), Item::Text(" world")]),
        ],
        0,
    );
    let retrying = RetryingLlm::new(inner, retry_policy(3));

    let (items, error) = drain(generate(&retrying, CancellationToken::new()).await).await;

    assert!(
        error.is_none(),
        "attempt 3 succeeds, so the stream ends cleanly"
    );
    assert_eq!(
        items,
        vec!["hello".to_owned(), " world".to_owned()],
        "the successful attempt's deltas are delivered"
    );
    assert_eq!(
        calls.load(Ordering::Relaxed),
        3,
        "the same provider was called three times"
    );
    let stats = retrying.stats();
    assert_eq!(stats.attempts, 3, "attempts == 3");
    assert_eq!(stats.retried, 2, "retried == 2");
}

#[tokio::test]
async fn error_after_emitted_delta_is_not_retried() {
    let (inner, calls) = ScriptedLlm::scripted(
        vec![
            Script::Items(vec![Item::Text("partial"), Item::Error(transient)]),
            Script::Items(vec![Item::Text("must not run")]),
        ],
        0,
    );
    let retrying = RetryingLlm::new(inner, retry_policy(3));

    let (items, error) = drain(generate(&retrying, CancellationToken::new()).await).await;

    assert_eq!(items, vec!["partial".to_owned()]);
    let error = error.expect("the post-commit error surfaces");
    assert_eq!(
        error.class(),
        ErrorClass::Transient,
        "the provider error is returned unchanged"
    );
    assert_eq!(
        calls.load(Ordering::Relaxed),
        1,
        "no second call after a committed delta"
    );
    assert_eq!(retrying.stats().attempts, 1, "attempts == 1");
}

#[tokio::test]
async fn cancelled_is_never_retried() {
    let (inner, calls) = ScriptedLlm::scripted(
        vec![
            Script::Fail(cancelled),
            Script::Items(vec![Item::Text("must not run")]),
        ],
        0,
    );
    let retrying = RetryingLlm::new(inner, retry_policy(3));

    let (items, error) = drain(generate(&retrying, CancellationToken::new()).await).await;

    assert!(items.is_empty(), "a cancelled attempt emits no delta");
    let error = error.expect("cancellation surfaces");
    assert!(
        matches!(error, ParlevoxError::Cancelled),
        "Cancelled is not retried"
    );
    assert_eq!(calls.load(Ordering::Relaxed), 1, "no second call");
    assert_eq!(retrying.stats().attempts, 1, "attempts == 1");
}

#[tokio::test]
async fn exhaustion_returns_the_last_error() {
    let (inner, calls) = ScriptedLlm::scripted(
        vec![
            Script::Fail(first_failure),
            Script::Fail(second_failure),
            Script::Fail(third_failure),
            Script::Fail(fourth_failure),
        ],
        0,
    );
    let policy = retry_policy(3);
    let retrying = RetryingLlm::new(inner, policy);

    let (items, error) = drain(generate(&retrying, CancellationToken::new()).await).await;

    assert!(items.is_empty());
    match error.expect("exhaustion returns an error") {
        ParlevoxError::Provider { message, class, .. } => {
            assert_eq!(message, "third", "the last error is returned unchanged");
            assert_eq!(class, ErrorClass::Transient, "the class is preserved");
        }
        other => panic!("expected a provider error, got {other:?}"),
    }
    assert_eq!(
        calls.load(Ordering::Relaxed),
        3,
        "attempts stop at max_attempts"
    );
    let stats = retrying.stats();
    assert_eq!(
        stats.attempts,
        u64::from(policy.max_attempts),
        "attempts == max_attempts"
    );
    assert_eq!(stats.retried, 2, "retried == 2");
}

#[tokio::test]
async fn pre_cancelled_token_makes_no_provider_call() {
    let (inner, calls) =
        ScriptedLlm::scripted(vec![Script::Items(vec![Item::Text("must not run")])], 0);
    let retrying = RetryingLlm::new(inner, retry_policy(3));

    let cancel = CancellationToken::new();
    cancel.cancel();
    let mut stream = generate(&retrying, cancel).await;

    let error = stream
        .next()
        .await
        .expect("one error is yielded")
        .expect_err("a pre-cancelled token yields Err(Cancelled)");
    assert!(matches!(error, ParlevoxError::Cancelled));
    assert!(stream.next().await.is_none(), "the stream then ends");
    assert_eq!(calls.load(Ordering::Relaxed), 0, "calls == 0");
    let stats = retrying.stats();
    assert_eq!(stats.attempts, 0, "attempts == 0");
    assert_eq!(stats.retried, 0, "retried == 0");
}

#[tokio::test]
async fn capacity_is_retried() {
    let (inner, calls) = ScriptedLlm::scripted(
        vec![
            Script::Fail(capacity),
            Script::Items(vec![Item::Text("ok")]),
        ],
        0,
    );
    let retrying = RetryingLlm::new(inner, retry_policy(2));

    let (items, error) = drain(generate(&retrying, CancellationToken::new()).await).await;

    assert!(error.is_none());
    assert_eq!(items, vec!["ok".to_owned()]);
    assert_eq!(calls.load(Ordering::Relaxed), 2, "Capacity is retryable");
    let stats = retrying.stats();
    assert_eq!(stats.attempts, 2);
    assert_eq!(stats.retried, 1);
}

#[tokio::test]
async fn fatal_is_not_retried() {
    let (inner, calls) = ScriptedLlm::scripted(
        vec![
            Script::Fail(fatal),
            Script::Items(vec![Item::Text("must not run")]),
        ],
        0,
    );
    let retrying = RetryingLlm::new(inner, retry_policy(3));

    let (items, error) = drain(generate(&retrying, CancellationToken::new()).await).await;

    assert!(items.is_empty());
    let error = error.expect("a fatal error surfaces");
    assert_eq!(error.class(), ErrorClass::Fatal, "Fatal is not retryable");
    assert_eq!(calls.load(Ordering::Relaxed), 1, "no second call");
    assert_eq!(retrying.stats().attempts, 1, "attempts == 1");
}

#[tokio::test]
async fn default_policy_does_not_retry() {
    let (inner, calls) = ScriptedLlm::scripted(
        vec![
            Script::Fail(transient),
            Script::Items(vec![Item::Text("must not run")]),
        ],
        0,
    );
    let retrying = RetryingLlm::new(inner, RetryPolicy::default());

    let (_items, error) = drain(generate(&retrying, CancellationToken::new()).await).await;

    assert!(error.is_some(), "the transient failure surfaces");
    assert_eq!(
        calls.load(Ordering::Relaxed),
        1,
        "the default policy makes one attempt"
    );
    assert_eq!(retrying.stats().attempts, 1, "attempts == 1");
    assert_eq!(retrying.stats().retried, 0, "retried == 0");
}

#[tokio::test]
async fn take_sidecar_dropped_frames_sums_every_attempt() {
    let (inner, _calls) = ScriptedLlm::scripted(
        vec![
            Script::Fail(transient),
            Script::Items(vec![Item::Text("done")]),
        ],
        7,
    );
    let retrying = RetryingLlm::new(inner, retry_policy(2));

    let (items, error) = drain(generate(&retrying, CancellationToken::new()).await).await;

    assert!(error.is_none());
    assert_eq!(items, vec!["done".to_owned()]);
    assert_eq!(
        retrying.take_sidecar_dropped_frames(),
        14,
        "both attempts' loss reports are summed"
    );
    assert_eq!(
        retrying.take_sidecar_dropped_frames(),
        0,
        "the report is take-and-reset"
    );
}

#[tokio::test]
async fn manifest_delegates_to_the_inner_provider() {
    let (inner, _calls) = ScriptedLlm::scripted(vec![], 0);
    let expected = inner.manifest().clone();
    let retrying = RetryingLlm::new(inner, RetryPolicy::default());

    assert_eq!(
        retrying.manifest(),
        &expected,
        "the wrapper's manifest is the inner provider's"
    );
}
