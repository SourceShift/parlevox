//! Opt-in reply-side retry with exponential backoff.
//!
//! [`RetryingLlm`] decorates one [`LlmProvider`] and retries the *same*
//! provider when an attempt fails before committing and the failure is classed
//! [`ErrorClass::Transient`] or [`ErrorClass::Capacity`]. It is composed from
//! outside and is deliberately not wired into the runtime or the registry.
//!
//! # The no-replay-after-commit law
//!
//! "Committed" means the current attempt's stream has already yielded at least
//! one `Ok` delta. Retrying after a committed delta would replay the prefix a
//! user already heard, so once an attempt commits, a later error is returned to
//! the caller unchanged and no further attempt is made. [`ErrorClass::Cancelled`]
//! is likewise never retried: the caller asked to stop.
//!
//! # Counters
//!
//! [`RetryStats::attempts`] counts the `generate` calls actually issued by the
//! decorator; [`RetryStats::retried`] counts only the attempts after the first.
//! A cancellation observed before an attempt, or during a backoff delay, issues
//! no call and increments neither counter.
//!
//! # Backoff
//!
//! [`RetryPolicy::delay_for`] is pure and 1-based: attempt `0` and `1` wait
//! zero, attempt `2` waits `base_delay`, and later attempts double the wait,
//! capped at `max_delay` and with optional full jitter. Delays are slept with a
//! [`tokio::time::sleep`] raced against the caller's cancellation token, so a
//! barge-in wakes the stream instead of waiting out the timer. A non-zero
//! `base_delay` therefore requires the stream to be polled inside a Tokio
//! runtime with the `time` feature; the default policy never uses a timer.

use std::collections::hash_map::RandomState;
use std::future::Future;
use std::hash::{BuildHasher, Hasher};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use async_trait::async_trait;
use futures_core::Stream;
use parlevox_core::{ErrorClass, ParlevoxError};
use tokio_util::sync::CancellationToken;

use crate::manifest::ProviderManifest;
use crate::traits::{GenerateRequest, LlmProvider, ProviderStream};

const NANOS_PER_SEC: u128 = 1_000_000_000;

/// How many times to retry, how long to wait, and whether to jitter.
///
/// The default retries nothing: `max_attempts` is `1` and every delay is
/// [`Duration::ZERO`]. Build a policy with struct-update syntax:
///
/// ```
/// use std::time::Duration;
/// use parlevox_providers::RetryPolicy;
///
/// let policy = RetryPolicy {
///     max_attempts: 3,
///     base_delay: Duration::from_millis(100),
///     max_delay: Duration::from_secs(2),
///     jitter: true,
/// };
/// // With jitter the second attempt's wait is sampled from the base delay.
/// assert!(policy.delay_for(2) <= Duration::from_millis(100));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryPolicy {
    /// Total attempts allowed, including the first. Values below `1` are
    /// clamped to `1` when the decorator runs so a provider always tries once.
    pub max_attempts: u32,
    /// Delay before the second attempt; each later attempt doubles it.
    pub base_delay: Duration,
    /// Upper bound applied to the exponential delay before jitter.
    pub max_delay: Duration,
    /// Sample the delay uniformly from `[0, capped]` instead of using it whole.
    pub jitter: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        RetryPolicy {
            max_attempts: 1,
            base_delay: Duration::ZERO,
            max_delay: Duration::ZERO,
            jitter: false,
        }
    }
}

impl RetryPolicy {
    /// The wait before the 1-based `attempt` about to run.
    ///
    /// Attempt `0` and `1` return [`Duration::ZERO`]; attempt `2` returns
    /// `base_delay`; later attempts double until `max_delay` caps them. With
    /// `jitter` the result is sampled uniformly from `[Duration::ZERO,
    /// capped]`. Large attempts saturate instead of overflowing.
    #[must_use]
    pub fn delay_for(&self, attempt: u32) -> Duration {
        if attempt <= 1 {
            return Duration::ZERO;
        }
        // `attempt - 2` is in range because `attempt > 1`; `saturating_pow`
        // turns an astronomically large attempt into `u32::MAX` rather than
        // overflowing, and `saturating_mul` clamps the duration.
        let factor = 2u32.saturating_pow(attempt - 2);
        let raw = self.base_delay.saturating_mul(factor);
        let capped = raw.min(self.max_delay);
        if self.jitter {
            sample_below(capped, attempt)
        } else {
            capped
        }
    }
}

/// A uniform sample in `[Duration::ZERO, upper]`.
///
/// Uses `RandomState` for entropy and integer-only arithmetic so no `rand`
/// crate is needed and no cast can wrap. Nothing here can panic: the division
/// is by a non-zero constant.
fn sample_below(upper: Duration, attempt: u32) -> Duration {
    if upper.is_zero() {
        return Duration::ZERO;
    }
    let span = upper.as_nanos();
    let mut hasher = RandomState::new().build_hasher();
    hasher.write_u32(attempt);
    let sample = u128::from(hasher.finish());
    // `span / divisor * sample` and `(span % divisor) * sample` both stay below
    // `u128::MAX`, and their sum stays below `span`.
    let divisor = u128::from(u64::MAX);
    let scaled = span / divisor * sample + (span % divisor) * sample / divisor;
    duration_from_nanos(scaled)
}

/// Rebuild a [`Duration`] from `u128` nanoseconds without a narrowing cast.
fn duration_from_nanos(nanos: u128) -> Duration {
    let secs = u64::try_from(nanos / NANOS_PER_SEC).unwrap_or(u64::MAX);
    let subsec = u32::try_from(nanos % NANOS_PER_SEC).unwrap_or(0);
    Duration::new(secs, subsec)
}

/// A point-in-time copy of the decorator's counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RetryStats {
    /// `generate` calls actually issued across every stream.
    pub attempts: u64,
    /// Attempts after the first actually issued across every stream.
    pub retried: u64,
}

/// Monotonic counters shared between the decorator and each stream it returns.
#[derive(Debug, Default)]
struct RetryCounters {
    attempts: AtomicU64,
    retried: AtomicU64,
}

/// An [`LlmProvider`] that retries the same provider on pre-commit transient
/// failures.
///
/// Build one with [`RetryingLlm::new`]. See the [module docs](self) for the
/// no-replay-after-commit law, the counter semantics and the backoff contract.
pub struct RetryingLlm {
    inner: Arc<dyn LlmProvider>,
    policy: RetryPolicy,
    counters: Arc<RetryCounters>,
    dropped: Arc<AtomicU64>,
}

impl RetryingLlm {
    /// Wrap `inner` with `policy`.
    ///
    /// No provider call happens here or in [`LlmProvider::generate`]; the first
    /// attempt runs only when the returned stream is first polled.
    #[must_use]
    pub fn new(inner: Arc<dyn LlmProvider>, policy: RetryPolicy) -> Self {
        RetryingLlm {
            inner,
            policy,
            counters: Arc::new(RetryCounters::default()),
            dropped: Arc::new(AtomicU64::new(0)),
        }
    }

    /// A point-in-time copy of the decorator's counters.
    #[must_use]
    pub fn stats(&self) -> RetryStats {
        RetryStats {
            attempts: self.counters.attempts.load(Ordering::Relaxed),
            retried: self.counters.retried.load(Ordering::Relaxed),
        }
    }
}

#[async_trait]
impl LlmProvider for RetryingLlm {
    fn manifest(&self) -> &ProviderManifest {
        self.inner.manifest()
    }

    async fn generate(
        &self,
        request: GenerateRequest,
        cancel: CancellationToken,
    ) -> Result<ProviderStream<String>, ParlevoxError> {
        Ok(Box::pin(RetryStream::new(
            Arc::clone(&self.inner),
            self.policy,
            request,
            cancel,
            Arc::clone(&self.counters),
            Arc::clone(&self.dropped),
        )))
    }

    fn take_sidecar_dropped_frames(&self) -> u64 {
        let accumulated = self.dropped.swap(0, Ordering::Relaxed);
        accumulated.saturating_add(self.inner.take_sidecar_dropped_frames())
    }
}

/// A boxed `generate` future held by the stream's state machine.
type StartFuture =
    Pin<Box<dyn Future<Output = Result<ProviderStream<String>, ParlevoxError>> + Send>>;

/// The outcome of a backoff delay: either the timer elapsed or cancellation won.
enum DelayOutcome {
    Elapsed,
    Cancelled,
}

/// A boxed sleep/cancel race held by the stream's state machine.
type DelayFuture = Pin<Box<dyn Future<Output = DelayOutcome> + Send>>;

/// One `poll_next` result, aliased to keep helper signatures readable.
type PollItem = Poll<Option<Result<String, ParlevoxError>>>;

/// What the stream is currently waiting on.
enum State {
    /// Not started; the first poll issues attempt 1.
    Idle,
    /// The current attempt's `generate` call has been issued.
    Starting(StartFuture),
    /// The current attempt accepted the request; its stream is being polled.
    Streaming(ProviderStream<String>),
    /// Waiting out a backoff delay, raced against cancellation.
    Delaying(DelayFuture),
    /// The stream is finished; repeated polls return end-of-stream.
    Ended,
}

/// The stream returned by [`RetryingLlm::generate`]: one attempt at a time.
struct RetryStream {
    inner: Arc<dyn LlmProvider>,
    policy: RetryPolicy,
    max_attempts: u32,
    request: GenerateRequest,
    cancel: CancellationToken,
    counters: Arc<RetryCounters>,
    dropped: Arc<AtomicU64>,
    attempt: u32,
    committed: bool,
    state: State,
}

impl RetryStream {
    fn new(
        inner: Arc<dyn LlmProvider>,
        policy: RetryPolicy,
        request: GenerateRequest,
        cancel: CancellationToken,
        counters: Arc<RetryCounters>,
        dropped: Arc<AtomicU64>,
    ) -> Self {
        RetryStream {
            inner,
            policy,
            max_attempts: policy.max_attempts.max(1),
            request,
            cancel,
            counters,
            dropped,
            attempt: 0,
            committed: false,
            state: State::Idle,
        }
    }

    /// Issue the next attempt, counting it as soon as the call is made.
    fn issue_attempt(&mut self) -> StartFuture {
        self.attempt = self.attempt.saturating_add(1);
        if self.attempt > 1 {
            self.counters.retried.fetch_add(1, Ordering::Relaxed);
        }
        self.counters.attempts.fetch_add(1, Ordering::Relaxed);
        self.committed = false;
        let inner = Arc::clone(&self.inner);
        let request = self.request.clone();
        let cancel = self.cancel.clone();
        Box::pin(async move { inner.generate(request, cancel).await })
    }

    /// Take the just-finished attempt's loss report into the shared total.
    ///
    /// The trait method is take-and-reset, so each attempt is counted once.
    fn absorb_dropped(&self) {
        let dropped = self.inner.take_sidecar_dropped_frames();
        if dropped != 0 {
            self.dropped.fetch_add(dropped, Ordering::Relaxed);
        }
    }

    /// Decide what follows an attempt error.
    ///
    /// Returns `Some(poll)` when the stream ends with `error`; returns `None`
    /// when a retry was scheduled, in which case the state machine loops again.
    fn after_attempt_error(&mut self, error: ParlevoxError) -> Option<PollItem> {
        let retryable = !self.committed
            && matches!(error.class(), ErrorClass::Transient | ErrorClass::Capacity);
        if !retryable || self.attempt >= self.max_attempts {
            self.state = State::Ended;
            return Some(Poll::Ready(Some(Err(error))));
        }
        if self.cancel.is_cancelled() {
            // The caller asked to stop between attempts; issue no further call.
            self.state = State::Ended;
            return Some(Poll::Ready(Some(Err(ParlevoxError::Cancelled))));
        }
        let delay = self.policy.delay_for(self.attempt.saturating_add(1));
        if delay.is_zero() {
            // No timer for the default policy, and for tests that use it.
            self.state = State::Starting(self.issue_attempt());
        } else {
            self.state = State::Delaying(delay_future(delay, self.cancel.clone()));
        }
        None
    }
}

/// Race a backoff sleep against cancellation without spawning a task.
fn delay_future(delay: Duration, cancel: CancellationToken) -> DelayFuture {
    Box::pin(async move {
        tokio::select! {
            () = tokio::time::sleep(delay) => DelayOutcome::Elapsed,
            () = cancel.cancelled() => DelayOutcome::Cancelled,
        }
    })
}

impl Stream for RetryStream {
    type Item = Result<String, ParlevoxError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            match std::mem::replace(&mut this.state, State::Ended) {
                State::Ended => return Poll::Ready(None),
                State::Idle => {
                    if this.cancel.is_cancelled() {
                        this.state = State::Ended;
                        return Poll::Ready(Some(Err(ParlevoxError::Cancelled)));
                    }
                    this.state = State::Starting(this.issue_attempt());
                }
                State::Starting(mut start) => match start.as_mut().poll(cx) {
                    Poll::Pending => {
                        this.state = State::Starting(start);
                        return Poll::Pending;
                    }
                    Poll::Ready(Ok(stream)) => {
                        this.state = State::Streaming(stream);
                    }
                    Poll::Ready(Err(error)) => {
                        this.absorb_dropped();
                        if let Some(poll) = this.after_attempt_error(error) {
                            return poll;
                        }
                    }
                },
                State::Streaming(mut stream) => match stream.as_mut().poll_next(cx) {
                    Poll::Pending => {
                        this.state = State::Streaming(stream);
                        return Poll::Pending;
                    }
                    Poll::Ready(Some(Ok(item))) => {
                        this.committed = true;
                        this.state = State::Streaming(stream);
                        return Poll::Ready(Some(Ok(item)));
                    }
                    Poll::Ready(Some(Err(error))) => {
                        this.absorb_dropped();
                        if let Some(poll) = this.after_attempt_error(error) {
                            return poll;
                        }
                    }
                    Poll::Ready(None) => {
                        this.absorb_dropped();
                        this.state = State::Ended;
                        return Poll::Ready(None);
                    }
                },
                State::Delaying(mut delay) => match delay.as_mut().poll(cx) {
                    Poll::Pending => {
                        this.state = State::Delaying(delay);
                        return Poll::Pending;
                    }
                    Poll::Ready(DelayOutcome::Cancelled) => {
                        this.state = State::Ended;
                        return Poll::Ready(Some(Err(ParlevoxError::Cancelled)));
                    }
                    Poll::Ready(DelayOutcome::Elapsed) => {
                        if this.cancel.is_cancelled() {
                            this.state = State::Ended;
                            return Poll::Ready(Some(Err(ParlevoxError::Cancelled)));
                        }
                        this.state = State::Starting(this.issue_attempt());
                    }
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(base_delay: Duration, max_delay: Duration, jitter: bool) -> RetryPolicy {
        RetryPolicy {
            max_attempts: 3,
            base_delay,
            max_delay,
            jitter,
        }
    }

    #[test]
    fn default_policy_retries_nothing_without_delay() {
        let default = RetryPolicy::default();
        assert_eq!(default.max_attempts, 1, "default retries nothing");
        assert_eq!(default.base_delay, Duration::ZERO);
        assert_eq!(default.delay_for(2), Duration::ZERO);
    }

    #[test]
    fn attempt_zero_and_one_are_zero() {
        let policy = policy(Duration::from_millis(100), Duration::from_secs(10), false);
        assert_eq!(policy.delay_for(0), Duration::ZERO);
        assert_eq!(policy.delay_for(1), Duration::ZERO);
    }

    #[test]
    fn second_attempt_waits_base_delay() {
        let base = Duration::from_millis(100);
        let policy = policy(base, Duration::from_secs(10), false);
        assert_eq!(policy.delay_for(2), base);
    }

    #[test]
    fn third_attempt_doubles_base_delay() {
        let base = Duration::from_millis(100);
        let policy = policy(base, Duration::from_secs(10), false);
        assert_eq!(policy.delay_for(3), base * 2);
        assert_eq!(policy.delay_for(4), base * 4);
    }

    #[test]
    fn delay_is_capped_at_max_delay() {
        let base = Duration::from_millis(100);
        let max = Duration::from_millis(250);
        let policy = policy(base, max, false);
        assert_eq!(policy.delay_for(3), base * 2);
        assert_eq!(policy.delay_for(4), max);
        assert_eq!(policy.delay_for(20), max);
    }

    #[test]
    fn huge_attempts_saturate_without_panicking() {
        let policy = policy(Duration::from_millis(100), Duration::from_secs(10), false);
        assert_eq!(policy.delay_for(u32::MAX), Duration::from_secs(10));
    }

    #[test]
    fn jitter_stays_within_bounds() {
        let base = Duration::from_millis(100);
        let max = Duration::from_secs(10);
        let policy = policy(base, max, true);
        // The uncapped delay for attempt 5 is 800ms, below the 10s cap.
        let uncapped = base * 8;
        for _ in 0..1_000 {
            let sample = policy.delay_for(5);
            assert!(
                sample <= uncapped,
                "jittered delay {sample:?} escaped [0, {uncapped:?}]"
            );
        }
    }

    #[test]
    fn jitter_respects_the_cap() {
        let base = Duration::from_millis(100);
        let max = Duration::from_millis(250);
        let policy = policy(base, max, true);
        for _ in 0..1_000 {
            let sample = policy.delay_for(20);
            assert!(
                sample <= max,
                "jittered delay {sample:?} escaped [0, {max:?}]"
            );
        }
    }

    #[test]
    fn saturation_holds_with_jitter_and_huge_attempts() {
        let policy = policy(Duration::from_millis(100), Duration::MAX, true);
        let sample = policy.delay_for(u32::MAX);
        assert!(
            sample <= Duration::MAX,
            "jittered delay {sample:?} escaped [0, Duration::MAX]"
        );
    }
}
