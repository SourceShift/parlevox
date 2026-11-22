//! Integration tests for the opt-in reply-side circuit breaker.
//!
//! The [`ScriptedLlm`] double below implements the real [`LlmProvider`] trait,
//! records every `generate` call, and serves one scripted attempt per call. The
//! stateful script is what lets these tests prove that a refusal made *no*
//! provider call and that a probe reached the provider exactly once.
//!
//! Every cooldown is driven by paused time: `#[tokio::test(start_paused =
//! true)]` plus `tokio::time::advance`, never a wall-clock sleep.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use async_trait::async_trait;
use futures_util::StreamExt;
use parlevox_core::{ErrorClass, ParlevoxError};
use parlevox_providers::{
    BreakerPolicy, CircuitBreakingLlm, GenerateRequest, LlmProvider, Modality, ProviderManifest,
    ProviderStream,
};
use tokio_util::sync::CancellationToken;

type MakeErr = fn() -> ParlevoxError;

fn transient() -> ParlevoxError {
    ParlevoxError::provider("scripted", "temporary failure")
}

fn cancelled() -> ParlevoxError {
    ParlevoxError::Cancelled
}

fn first_failure() -> ParlevoxError {
    ParlevoxError::provider("scripted", "first failure")
}

fn second_failure() -> ParlevoxError {
    ParlevoxError::provider("scripted", "second failure")
}

fn third_failure() -> ParlevoxError {
    ParlevoxError::provider("scripted", "third failure")
}

/// One stream item: literal text, or a scripted error made fresh per call.
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
}

impl ScriptedLlm {
    /// Build a provider plus a handle that counts `generate` calls.
    fn scripted(scripts: Vec<Script>) -> (Arc<dyn LlmProvider>, Arc<AtomicU64>) {
        let calls = Arc::new(AtomicU64::new(0));
        let provider = Arc::new(ScriptedLlm {
            manifest: ProviderManifest::new("inner-provider", Modality::Llm),
            scripts: Mutex::new(scripts.into()),
            calls: Arc::clone(&calls),
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
            None => panic!("scripted provider was called more often than scripted"),
        }
    }
}

/// A policy that opens after `failure_threshold` failures and waits `open_for`.
fn policy(failure_threshold: u32, open_for: Duration) -> BreakerPolicy {
    BreakerPolicy {
        failure_threshold,
        open_for,
    }
}

/// Start one request. `generate` itself must not call the provider or decide
/// admission; the returned stream does both on its first poll.
async fn start(breaker: &CircuitBreakingLlm) -> ProviderStream<String> {
    breaker
        .generate(GenerateRequest::new("hello"), CancellationToken::new())
        .await
        .expect("generate builds a stream without calling the provider")
}

/// Drain a stream, keeping the items seen before any terminal error.
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

/// Start and drain one request in one step.
async fn run(breaker: &CircuitBreakingLlm) -> (Vec<String>, Option<ParlevoxError>) {
    drain(start(breaker).await).await
}

fn call_count(calls: &Arc<AtomicU64>) -> u64 {
    calls.load(Ordering::Relaxed)
}

#[tokio::test(start_paused = true)]
async fn open_after_two_pre_commit_failures_then_third_request_is_refused() {
    let policy = policy(2, Duration::from_secs(30));
    let (inner, calls) = ScriptedLlm::scripted(vec![
        Script::Fail(transient),
        Script::Fail(transient),
        Script::Items(vec![Item::Text("unused")]),
    ]);
    let breaker = CircuitBreakingLlm::new(inner, policy);

    let (_, error) = run(&breaker).await;
    assert!(error.is_some(), "the first failure surfaces to the caller");
    let (_, error) = run(&breaker).await;
    assert!(error.is_some(), "the second failure surfaces to the caller");
    assert_eq!(
        call_count(&calls),
        2,
        "two failures made two provider calls"
    );
    assert_eq!(
        breaker.stats().opened,
        1,
        "the breaker opened at the threshold"
    );

    let (items, error) = run(&breaker).await;
    let error = error.expect("the third request is refused");
    assert!(items.is_empty(), "a refusal yields no provider item");
    assert_eq!(error.class(), ErrorClass::Transient);
    assert!(
        error.to_string().contains("circuit breaker"),
        "a refusal message names the breaker: {error}"
    );
    assert_eq!(
        call_count(&calls),
        2,
        "a refused request makes no provider call"
    );
    assert_eq!(breaker.stats().opened, 1);
    assert_eq!(breaker.stats().refused, 1);
}

#[tokio::test(start_paused = true)]
async fn half_open_probe_success_closes_breaker() {
    let policy = policy(2, Duration::from_secs(30));
    let (inner, calls) = ScriptedLlm::scripted(vec![
        Script::Fail(transient),
        Script::Fail(transient),
        Script::Items(vec![Item::Text("probe ok")]),
        Script::Items(vec![Item::Text("after close")]),
    ]);
    let breaker = CircuitBreakingLlm::new(inner, policy);
    let _ = run(&breaker).await;
    let _ = run(&breaker).await;
    assert_eq!(call_count(&calls), 2);

    tokio::time::advance(policy.open_for + Duration::from_nanos(1)).await;

    let (items, error) = run(&breaker).await;
    assert!(error.is_none(), "the probe succeeds: {error:?}");
    assert_eq!(items, vec!["probe ok".to_string()]);
    assert_eq!(
        call_count(&calls),
        3,
        "exactly one probe reached the provider"
    );
    assert_eq!(breaker.stats().probes, 1);
    assert_eq!(
        breaker.stats().opened,
        1,
        "a successful probe opens nothing new"
    );

    let (items, error) = run(&breaker).await;
    assert!(error.is_none(), "the closed breaker answers: {error:?}");
    assert_eq!(items, vec!["after close".to_string()]);
    assert_eq!(
        call_count(&calls),
        4,
        "the next request calls the provider again"
    );
}

#[tokio::test(start_paused = true)]
async fn half_open_probe_failure_reopens() {
    let policy = policy(2, Duration::from_secs(30));
    let (inner, calls) = ScriptedLlm::scripted(vec![
        Script::Fail(transient),
        Script::Fail(transient),
        Script::Fail(transient),
        Script::Items(vec![Item::Text("unused")]),
    ]);
    let breaker = CircuitBreakingLlm::new(inner, policy);
    let _ = run(&breaker).await;
    let _ = run(&breaker).await;

    tokio::time::advance(policy.open_for + Duration::from_nanos(1)).await;

    let (items, error) = run(&breaker).await;
    assert!(items.is_empty());
    let error = error.expect("a failing probe surfaces its error");
    assert_eq!(error.class(), ErrorClass::Transient);
    assert_eq!(call_count(&calls), 3, "the probe reached the provider");
    assert_eq!(breaker.stats().opened, 2, "a failed probe re-opens");
    assert_eq!(breaker.stats().probes, 1);

    let (_, refused) = run(&breaker).await;
    let refused = refused.expect("the re-opened breaker refuses");
    assert_eq!(refused.class(), ErrorClass::Transient);
    assert_eq!(
        call_count(&calls),
        3,
        "no extra provider call before the next cool-down"
    );
    assert_eq!(breaker.stats().refused, 1);
}

#[tokio::test(start_paused = true)]
async fn success_before_threshold_resets_failure_count() {
    let policy = policy(2, Duration::from_secs(30));
    // fail (count 1), success (reset to 0), fail (count 1), then a stream that
    // commits and *then* fails. The last failure is post-commit, so it is
    // neutral by the no-replay-after-commit law and never reaches the
    // threshold.
    let (inner, calls) = ScriptedLlm::scripted(vec![
        Script::Fail(transient),
        Script::Items(vec![Item::Text("ok")]),
        Script::Fail(transient),
        Script::Items(vec![Item::Text("committed"), Item::Error(transient)]),
    ]);
    let breaker = CircuitBreakingLlm::new(inner, policy);

    let _ = run(&breaker).await;
    let _ = run(&breaker).await;
    let _ = run(&breaker).await;
    let (items, error) = run(&breaker).await;
    assert_eq!(items, vec!["committed".to_string()]);
    assert!(error.is_some(), "the post-commit failure is surfaced");

    assert_eq!(
        breaker.stats().opened,
        0,
        "the success reset the count before the threshold"
    );
    assert_eq!(breaker.stats().refused, 0);
    assert_eq!(call_count(&calls), 4);
}

#[tokio::test(start_paused = true)]
async fn cancelled_and_post_commit_failures_do_not_open_breaker() {
    let policy = policy(2, Duration::from_secs(30));
    let (inner, calls) = ScriptedLlm::scripted(vec![
        Script::Fail(cancelled),
        Script::Items(vec![Item::Text("one"), Item::Error(transient)]),
        Script::Items(vec![Item::Text("two"), Item::Error(transient)]),
        Script::Items(vec![Item::Text("three")]),
    ]);
    let breaker = CircuitBreakingLlm::new(inner, policy);

    let (_, error) = run(&breaker).await;
    assert!(matches!(error, Some(ParlevoxError::Cancelled)));
    assert_eq!(
        breaker.stats().opened,
        0,
        "a cancellation is not a provider fault"
    );

    let (items, error) = run(&breaker).await;
    assert_eq!(items, vec!["one".to_string()]);
    assert!(error.is_some());
    assert_eq!(
        breaker.stats().opened,
        0,
        "a post-commit failure does not count"
    );

    let (items, error) = run(&breaker).await;
    assert_eq!(items, vec!["two".to_string()]);
    assert!(error.is_some());
    assert_eq!(breaker.stats().opened, 0);

    let (items, error) = run(&breaker).await;
    assert_eq!(items, vec!["three".to_string()]);
    assert!(error.is_none(), "later requests still reach the provider");
    assert_eq!(call_count(&calls), 4);
    assert_eq!(breaker.stats().refused, 0);
}

#[tokio::test(start_paused = true)]
async fn default_policy_never_refuses() {
    let (inner, calls) = ScriptedLlm::scripted(vec![
        Script::Fail(first_failure),
        Script::Fail(second_failure),
        Script::Fail(third_failure),
    ]);
    let breaker = CircuitBreakingLlm::new(inner, BreakerPolicy::default());

    for expected in ["first failure", "second failure", "third failure"] {
        let (_, error) = run(&breaker).await;
        let error = error.expect("the inner failure is surfaced unchanged");
        let ParlevoxError::Provider { message, class, .. } = &error else {
            panic!("expected a provider error, got {error}");
        };
        assert_eq!(*class, ErrorClass::Transient);
        assert_eq!(message.as_str(), expected);
    }

    assert_eq!(call_count(&calls), 3, "every request reached the provider");
    assert_eq!(breaker.stats().refused, 0);
    assert_eq!(breaker.stats().opened, 0);
}

#[tokio::test(start_paused = true)]
async fn state_is_shared_across_clones_and_independent_across_wrappers() {
    let policy = policy(2, Duration::from_secs(30));
    let (inner, calls) = ScriptedLlm::scripted(vec![
        Script::Fail(transient),
        Script::Fail(transient),
        Script::Items(vec![Item::Text("fresh wrapper")]),
    ]);
    let a = CircuitBreakingLlm::new(Arc::clone(&inner), policy);
    let _ = run(&a).await;
    let _ = run(&a).await;
    assert_eq!(a.stats().opened, 1);
    assert_eq!(call_count(&calls), 2);

    let b = a.clone();
    let (items, error) = run(&b).await;
    let error = error.expect("a cloned wrapper shares the open breaker");
    assert!(items.is_empty());
    assert_eq!(error.class(), ErrorClass::Transient);
    assert_eq!(
        call_count(&calls),
        2,
        "the clone's refused request made no provider call"
    );
    assert_eq!(b.stats().refused, 1);
    assert_eq!(a.stats().opened, 1);

    let fresh = CircuitBreakingLlm::new(Arc::clone(&inner), policy);
    let (items, error) = run(&fresh).await;
    assert!(
        error.is_none(),
        "a separately constructed wrapper has independent state: {error:?}"
    );
    assert_eq!(items, vec!["fresh wrapper".to_string()]);
    assert_eq!(
        call_count(&calls),
        3,
        "the fresh wrapper reached the provider"
    );
    assert_eq!(fresh.stats().refused, 0);
}
