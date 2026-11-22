//! Opt-in reply-side circuit breaker.
//!
//! [`CircuitBreakingLlm`] wraps one [`LlmProvider`] and stops sending requests
//! once that provider has failed repeatedly, then admits a single probe after a
//! cool-down. It composes from outside and is deliberately not wired into the
//! runtime or the registry.
//!
//! # States
//!
//! The breaker is a three-state machine, held in an [`Arc`] shared by every
//! clone of the wrapper so state is per-provider rather than per-turn:
//!
//! - **Closed** — requests flow through. A pre-commit [`ErrorClass::Transient`]
//!   or [`ErrorClass::Capacity`] failure increments the consecutive-failure
//!   count; the first yielded `Ok` item resets it to zero.
//! - **Open** — set when the count reaches
//!   [`BreakerPolicy::failure_threshold`]. Requests are refused *without
//!   calling the provider* until [`BreakerPolicy::open_for`] has elapsed.
//! - **Half-open** — one request at or after the cool-down becomes a probe. Its
//!   first `Ok` item closes the breaker; a non-cancelled failure re-opens it;
//!   a cancellation returns to *Open* without counting a provider fault.
//!   Any other request is refused while the probe is in flight.
//!
//! A `failure_threshold` of `0` (the [`Default`] policy) never opens: the
//! decorator then behaves exactly like its inner provider.
//!
//! # What counts
//!
//! "Committed" means the attempt's stream has already yielded at least one
//! `Ok` item. Only pre-commit `Transient`/`Capacity` failures count toward the
//! threshold; [`ErrorClass::Fatal`], [`ErrorClass::Protocol`],
//! [`ErrorClass::Cancelled`] and every post-commit failure are neutral. An
//! empty successful stream yields no item and is likewise neutral: it does not
//! reset the count and does not increment [`BreakerStats::opened`]. For a probe
//! it only releases the single-probe slot so the next request may probe again.
//!
//! # Counters
//!
//! [`BreakerStats::opened`] counts transitions into *Open*, including a
//! re-open after a failed probe. [`BreakerStats::refused`] counts requests that
//! never reached the provider. [`BreakerStats::probes`] counts probes admitted
//! after a cool-down. All three are cumulative and read with
//! [`CircuitBreakingLlm::stats`].
//!
//! # Time and tasks
//!
//! Deadlines use [`tokio::time::Instant`], so tests can drive them with paused
//! time. The decorator never spawns a task and never sleeps: every decision is
//! taken synchronously in [`Stream::poll_next`], and the state mutex is never
//! held across an `await` or across a poll of the inner stream.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::task::{Context, Poll};
use std::time::Duration;

use async_trait::async_trait;
use futures_core::Stream;
use parlevox_core::{ErrorClass, ParlevoxError};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::manifest::ProviderManifest;
use crate::traits::{GenerateRequest, LlmProvider, ProviderStream};

/// When to open the breaker and how long to keep it open.
///
/// The [`Default`] policy is a permanently-closed breaker: `failure_threshold`
/// is `0`, which means "never open", and the decorator is a transparent
/// wrapper around its inner provider.
///
/// ```
/// use std::time::Duration;
/// use parlevox_providers::BreakerPolicy;
///
/// let policy = BreakerPolicy {
///     failure_threshold: 3,
///     open_for: Duration::from_secs(5),
/// };
/// assert_eq!(BreakerPolicy::default().failure_threshold, 0);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BreakerPolicy {
    /// Consecutive pre-commit failures that open the breaker. `0` never opens.
    pub failure_threshold: u32,
    /// How long the breaker stays open before admitting one probe.
    pub open_for: Duration,
}

/// A point-in-time copy of the decorator's cumulative counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BreakerStats {
    /// Transitions into *Open*, including a re-open after a failed probe.
    pub opened: u64,
    /// Requests refused without reaching the provider.
    pub refused: u64,
    /// Probes admitted after a cool-down.
    pub probes: u64,
}

/// The three states of the breaker. All variants are cheap to copy so the
/// decision helpers can work on a value without holding a borrow of the guard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BreakerState {
    /// Requests flow; `failures` counts consecutive pre-commit failures.
    Closed { failures: u32 },
    /// Requests are refused until `until`.
    Open { until: Instant },
    /// One probe is in flight; every other request is refused.
    HalfOpen,
}

/// State and counters shared by every clone of one wrapper.
#[derive(Debug)]
struct BreakerShared {
    state: Mutex<BreakerState>,
    opened: AtomicU64,
    refused: AtomicU64,
    probes: AtomicU64,
}

impl BreakerShared {
    fn new() -> Self {
        BreakerShared {
            state: Mutex::new(BreakerState::Closed { failures: 0 }),
            opened: AtomicU64::new(0),
            refused: AtomicU64::new(0),
            probes: AtomicU64::new(0),
        }
    }

    fn snapshot(&self) -> BreakerStats {
        BreakerStats {
            opened: self.opened.load(Ordering::Relaxed),
            refused: self.refused.load(Ordering::Relaxed),
            probes: self.probes.load(Ordering::Relaxed),
        }
    }
}

/// Whether one request was admitted as a normal call or as a half-open probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttemptKind {
    Normal,
    Probe,
}

/// The admission decision for one request.
enum Admission {
    Refused(ParlevoxError),
    Allowed(AttemptKind),
}

/// An [`LlmProvider`] that refuses requests while its inner provider is
/// failing, then probes it again after a cool-down.
///
/// Build one with [`CircuitBreakingLlm::new`]. Cloning the wrapper shares the
/// breaker: a clone observes the same state and the same counters. A separately
/// constructed wrapper around the same inner provider has independent state.
/// See the [module docs](self) for the state machine and counting rules.
#[derive(Clone)]
pub struct CircuitBreakingLlm {
    inner: Arc<dyn LlmProvider>,
    policy: BreakerPolicy,
    shared: Arc<BreakerShared>,
}

impl CircuitBreakingLlm {
    /// Wrap `inner` with `policy`.
    ///
    /// No provider call happens here or in [`LlmProvider::generate`]; the first
    /// call runs only when the returned stream is first polled.
    #[must_use]
    pub fn new(inner: Arc<dyn LlmProvider>, policy: BreakerPolicy) -> Self {
        CircuitBreakingLlm {
            inner,
            policy,
            shared: Arc::new(BreakerShared::new()),
        }
    }

    /// A point-in-time copy of the cumulative counters.
    #[must_use]
    pub fn stats(&self) -> BreakerStats {
        self.shared.snapshot()
    }
}

#[async_trait]
impl LlmProvider for CircuitBreakingLlm {
    fn manifest(&self) -> &ProviderManifest {
        self.inner.manifest()
    }

    async fn generate(
        &self,
        request: GenerateRequest,
        cancel: CancellationToken,
    ) -> Result<ProviderStream<String>, ParlevoxError> {
        Ok(Box::pin(BreakerStream {
            inner: Arc::clone(&self.inner),
            policy: self.policy,
            shared: Arc::clone(&self.shared),
            provider: self.inner.manifest().name.clone(),
            request: Some(request),
            cancel,
            kind: None,
            committed: false,
            phase: Phase::Idle,
        }))
    }

    fn take_sidecar_dropped_frames(&self) -> u64 {
        self.inner.take_sidecar_dropped_frames()
    }
}

/// A boxed `generate` future held by the stream's state machine.
type StartFuture =
    Pin<Box<dyn Future<Output = Result<ProviderStream<String>, ParlevoxError>> + Send>>;

/// What the stream is currently waiting on.
enum Phase {
    /// Not started; the first poll decides admission.
    Idle,
    /// The admitted attempt's `generate` call is in flight.
    Starting(StartFuture),
    /// The admitted attempt returned a stream that is being polled.
    Streaming(ProviderStream<String>),
    /// The stream is finished; repeated polls return end-of-stream.
    Ended,
}

/// The stream returned by [`CircuitBreakingLlm::generate`].
struct BreakerStream {
    inner: Arc<dyn LlmProvider>,
    policy: BreakerPolicy,
    shared: Arc<BreakerShared>,
    /// The inner provider's name, used in a refusal message.
    provider: String,
    /// Taken when the admitted attempt starts; the decorator tries only once.
    request: Option<GenerateRequest>,
    cancel: CancellationToken,
    /// Set once admission succeeds; `None` while idle or refused.
    kind: Option<AttemptKind>,
    /// Whether the admitted attempt has yielded its first `Ok` item.
    committed: bool,
    phase: Phase,
}

impl BreakerStream {
    /// Issue the single admitted attempt.
    fn issue_attempt(&mut self) -> Option<StartFuture> {
        let request = self.request.take()?;
        let inner = Arc::clone(&self.inner);
        let cancel = self.cancel.clone();
        Some(Box::pin(
            async move { inner.generate(request, cancel).await },
        ))
    }

    /// Apply a failure observed before the attempt committed.
    fn record_precommit_failure(&self, error: &ParlevoxError) {
        let Some(kind) = self.kind else {
            return;
        };
        record_failure(&self.shared, self.policy, kind, error.class());
    }

    /// Close or reset the breaker when the attempt yields its first `Ok` item.
    fn record_success(&self) {
        let Some(kind) = self.kind else {
            return;
        };
        record_success(&self.shared, kind);
    }

    /// Reconcile a stream that ended without yielding any item or error.
    fn record_empty_end(&self) {
        let Some(kind) = self.kind else {
            return;
        };
        if kind == AttemptKind::Probe {
            // Neutral: the probe neither succeeded nor failed. Release the
            // single-probe slot (the cool-down has already elapsed) without
            // counting an open.
            let mut state = self
                .shared
                .state
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            if matches!(*state, BreakerState::HalfOpen) {
                *state = BreakerState::Open {
                    until: Instant::now(),
                };
            }
        }
    }
}

impl Stream for BreakerStream {
    type Item = Result<String, ParlevoxError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            match std::mem::replace(&mut this.phase, Phase::Ended) {
                Phase::Ended => return Poll::Ready(None),
                Phase::Idle => {
                    if this.cancel.is_cancelled() {
                        return Poll::Ready(Some(Err(ParlevoxError::Cancelled)));
                    }
                    match admit(&this.shared, &this.provider) {
                        Admission::Refused(error) => return Poll::Ready(Some(Err(error))),
                        Admission::Allowed(kind) => {
                            this.kind = Some(kind);
                            match this.issue_attempt() {
                                Some(start) => this.phase = Phase::Starting(start),
                                None => return Poll::Ready(None),
                            }
                        }
                    }
                }
                Phase::Starting(mut start) => match start.as_mut().poll(cx) {
                    Poll::Pending => {
                        this.phase = Phase::Starting(start);
                        return Poll::Pending;
                    }
                    Poll::Ready(Ok(stream)) => {
                        this.phase = Phase::Streaming(stream);
                    }
                    Poll::Ready(Err(error)) => {
                        this.record_precommit_failure(&error);
                        this.phase = Phase::Ended;
                        return Poll::Ready(Some(Err(error)));
                    }
                },
                Phase::Streaming(mut stream) => match stream.as_mut().poll_next(cx) {
                    Poll::Pending => {
                        this.phase = Phase::Streaming(stream);
                        return Poll::Pending;
                    }
                    Poll::Ready(Some(Ok(item))) => {
                        if !this.committed {
                            this.committed = true;
                            this.record_success();
                        }
                        this.phase = Phase::Streaming(stream);
                        return Poll::Ready(Some(Ok(item)));
                    }
                    Poll::Ready(Some(Err(error))) => {
                        if !this.committed {
                            this.record_precommit_failure(&error);
                        }
                        this.phase = Phase::Ended;
                        return Poll::Ready(Some(Err(error)));
                    }
                    Poll::Ready(None) => {
                        this.record_empty_end();
                        this.phase = Phase::Ended;
                        return Poll::Ready(None);
                    }
                },
            }
        }
    }
}

/// Decide whether `shared` admits one request, transitioning `Open` to
/// `HalfOpen` when the cool-down has elapsed.
///
/// The mutex is taken and released synchronously; no await or inner poll runs
/// while it is held.
fn admit(shared: &BreakerShared, provider: &str) -> Admission {
    let mut state = shared.state.lock().unwrap_or_else(PoisonError::into_inner);
    match *state {
        BreakerState::Closed { .. } => Admission::Allowed(AttemptKind::Normal),
        BreakerState::HalfOpen => {
            shared.refused.fetch_add(1, Ordering::Relaxed);
            Admission::Refused(refusal(provider, "half-open"))
        }
        BreakerState::Open { until } => {
            if Instant::now() < until {
                shared.refused.fetch_add(1, Ordering::Relaxed);
                Admission::Refused(refusal(provider, "open"))
            } else {
                *state = BreakerState::HalfOpen;
                shared.probes.fetch_add(1, Ordering::Relaxed);
                Admission::Allowed(AttemptKind::Probe)
            }
        }
    }
}

/// Apply one pre-commit failure to the shared state.
fn record_failure(
    shared: &BreakerShared,
    policy: BreakerPolicy,
    kind: AttemptKind,
    class: ErrorClass,
) {
    let mut state = shared.state.lock().unwrap_or_else(PoisonError::into_inner);
    match kind {
        AttemptKind::Normal => {
            if !matches!(class, ErrorClass::Transient | ErrorClass::Capacity) {
                return;
            }
            if policy.failure_threshold == 0 {
                return;
            }
            let BreakerState::Closed { failures } = *state else {
                return;
            };
            let failures = failures.saturating_add(1);
            if failures >= policy.failure_threshold {
                *state = BreakerState::Open {
                    until: open_deadline(policy.open_for),
                };
                shared.opened.fetch_add(1, Ordering::Relaxed);
            } else {
                *state = BreakerState::Closed { failures };
            }
        }
        AttemptKind::Probe => {
            // A cancellation is not a provider fault: re-open without counting
            // an open. Every other failure (including Fatal/Protocol) re-opens.
            let counts = !matches!(class, ErrorClass::Cancelled);
            *state = BreakerState::Open {
                until: open_deadline(policy.open_for),
            };
            if counts {
                shared.opened.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// Apply a committed attempt's first `Ok` item to the shared state.
fn record_success(shared: &BreakerShared, kind: AttemptKind) {
    let mut state = shared.state.lock().unwrap_or_else(PoisonError::into_inner);
    match kind {
        AttemptKind::Normal => {
            if matches!(*state, BreakerState::Closed { .. }) {
                *state = BreakerState::Closed { failures: 0 };
            }
        }
        AttemptKind::Probe => {
            if matches!(*state, BreakerState::HalfOpen) {
                *state = BreakerState::Closed { failures: 0 };
            }
        }
    }
}

/// `now + open_for`, saturating to `now` on overflow instead of panicking.
fn open_deadline(open_for: Duration) -> Instant {
    let now = Instant::now();
    now.checked_add(open_for).unwrap_or(now)
}

/// A transient refusal that names the breaker in its message.
fn refusal(provider: &str, reason: &str) -> ParlevoxError {
    ParlevoxError::provider(
        provider,
        format!("circuit breaker is {reason}; request refused without calling the provider"),
    )
}
