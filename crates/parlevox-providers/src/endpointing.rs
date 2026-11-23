//! Opt-in adaptive endpointing at the VAD boundary.
//!
//! [`AdaptiveEndpointingVad`] decorates one [`VadProvider`] and decides, per
//! [`VadEvent::SpeechEnd`], whether the analysed [`AudioChunk`] still holds
//! enough trailing silence for the endpoint to be real. It is composed from
//! outside and is deliberately not wired into the runtime or the registry.
//!
//! # The trailing-silence law
//!
//! For each [`VadEvent::SpeechStart`] at `t0` followed by a
//! [`VadEvent::SpeechEnd`] at `t1` over audio of length `audio_duration_ms`:
//!
//! - `speech_ms = t1.saturating_sub(t0)`, or `0` when no start was seen;
//! - `effective = min(min_trailing_silence_ms +
//!   speech_ms.saturating_mul(adapt_ms_per_speech_second).saturating_div(1000),
//!   max_trailing_silence_ms)`;
//! - `trailing_ms = audio_duration_ms.saturating_sub(t1)`;
//! - `trailing_ms >= effective` accepts the end and forwards it unchanged;
//! - `trailing_ms < effective` drops it and keeps the speech open.
//!
//! A later [`VadEvent::SpeechStart`] while the speech is open is forwarded
//! unchanged and does not reset `t0`; only an accepted end closes the
//! utterance. Every arithmetic step is saturating, so no length can overflow
//! or panic.
//!
//! # End-of-stream integrity
//!
//! If the inner stream ends while speech is still open, the wrapper emits one
//! final [`VadEvent::SpeechEnd`] at `audio_duration_ms` and counts it accepted.
//! Without it the turn machine would stay in `UserSpeaking`.
//!
//! # Errors, cancellation and tasks
//!
//! An inner [`Err`] is forwarded unchanged and permanently suppresses the
//! synthesized end; [`ParlevoxError::Cancelled`] is forwarded like any other
//! error and ends the wrapper's stream immediately. The decorator never spawns
//! a task: the returned stream pulls the inner stream lazily on poll and holds
//! only its own per-stream speech state plus an [`Arc`] share of the wrapper's
//! counters.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

use async_trait::async_trait;
use futures_core::Stream;
use parlevox_core::ParlevoxError;
use tokio_util::sync::CancellationToken;

use crate::manifest::ProviderManifest;
use crate::traits::{AudioChunk, ProviderStream, VadEvent, VadProvider, VadTransition};

/// How much trailing silence a [`VadEvent::SpeechEnd`] must leave in the
/// analysed audio before the endpoint is accepted.
///
/// The [`Default`] policy is all zeros, which makes the effective allowance
/// zero: every end is accepted and the decorator is a byte-for-byte
/// pass-through of the inner stream. Build a policy with struct-update syntax:
///
/// ```
/// use parlevox_providers::EndpointingPolicy;
///
/// let policy = EndpointingPolicy {
///     min_trailing_silence_ms: 200,
///     adapt_ms_per_speech_second: 100,
///     max_trailing_silence_ms: 1_000,
/// };
/// assert_eq!(EndpointingPolicy::default().min_trailing_silence_ms, 0);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EndpointingPolicy {
    /// Trailing silence, in ms, required before a [`SpeechEnd`](VadEvent::SpeechEnd)
    /// is accepted.
    pub min_trailing_silence_ms: u64,
    /// Additional allowance per second of observed speech, in ms.
    pub adapt_ms_per_speech_second: u64,
    /// Hard cap on the effective trailing-silence allowance, in ms.
    pub max_trailing_silence_ms: u64,
}

impl EndpointingPolicy {
    /// The allowance for speech lasting `speech_ms`, capped at the policy's
    /// maximum. Every step saturates instead of overflowing.
    fn effective_allowance(self, speech_ms: u64) -> u64 {
        let adaptive = speech_ms
            .saturating_mul(self.adapt_ms_per_speech_second)
            .saturating_div(1000);
        self.min_trailing_silence_ms
            .saturating_add(adaptive)
            .min(self.max_trailing_silence_ms)
    }
}

/// A point-in-time copy of the decorator's cumulative counters.
///
/// Both counters describe [`VadEvent::SpeechEnd`] decisions only: every accepted
/// end, including the synthesized end-of-stream one, moves `accepted`, and
/// every dropped one moves `rejected`. [`VadEvent::SpeechStart`] transitions are
/// forwarded and never counted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EndpointingStats {
    /// `SpeechEnd` transitions the wrapper forwarded, plus synthesized ends.
    pub accepted: u64,
    /// `SpeechEnd` transitions the wrapper dropped.
    pub rejected: u64,
}

/// Monotonic counters shared between the decorator and each stream it returns.
#[derive(Debug, Default)]
struct EndpointingCounters {
    accepted: AtomicU64,
    rejected: AtomicU64,
}

impl EndpointingCounters {
    fn snapshot(&self) -> EndpointingStats {
        EndpointingStats {
            accepted: self.accepted.load(Ordering::Relaxed),
            rejected: self.rejected.load(Ordering::Relaxed),
        }
    }
}

/// A [`VadProvider`] that accepts or drops [`VadEvent::SpeechEnd`] transitions
/// according to an adaptive trailing-silence policy.
///
/// Build one with [`AdaptiveEndpointingVad::new`]. It wraps any
/// [`Arc<dyn VadProvider>`](VadProvider), including a
/// [`FallbackVad`](crate::FallbackVad), and inherits the inner provider's
/// manifest. See the [module docs](self) for the trailing-silence law and the
/// end-of-stream guarantee.
#[derive(Clone)]
pub struct AdaptiveEndpointingVad {
    inner: Arc<dyn VadProvider>,
    policy: EndpointingPolicy,
    counters: Arc<EndpointingCounters>,
}

impl AdaptiveEndpointingVad {
    /// Wrap `inner` with `policy`.
    ///
    /// No inner [`VadProvider::analyze`] call happens here or in the wrapper's
    /// `analyze`; the inner stream is first polled when the returned stream is
    /// first polled.
    #[must_use]
    pub fn new(inner: Arc<dyn VadProvider>, policy: EndpointingPolicy) -> Self {
        AdaptiveEndpointingVad {
            inner,
            policy,
            counters: Arc::new(EndpointingCounters::default()),
        }
    }

    /// A point-in-time copy of the decorator's cumulative counters.
    #[must_use]
    pub fn stats(&self) -> EndpointingStats {
        self.counters.snapshot()
    }
}

impl std::fmt::Debug for AdaptiveEndpointingVad {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdaptiveEndpointingVad")
            .field("inner", &self.inner.manifest().name)
            .field("policy", &self.policy)
            .field("stats", &self.stats())
            .finish()
    }
}

#[async_trait]
impl VadProvider for AdaptiveEndpointingVad {
    fn manifest(&self) -> &ProviderManifest {
        self.inner.manifest()
    }

    async fn analyze(
        &self,
        audio: AudioChunk,
        cancel: CancellationToken,
    ) -> Result<ProviderStream<VadTransition>, ParlevoxError> {
        Ok(Box::pin(EndpointingStream {
            inner: Arc::clone(&self.inner),
            policy: self.policy,
            counters: Arc::clone(&self.counters),
            audio_duration_ms: audio_duration_ms(&audio),
            audio: Some(audio),
            cancel,
            phase: Phase::Idle,
            speech_open: false,
            t0: None,
        }))
    }
}

/// A boxed `analyze` future held by the stream's state machine.
type StartFuture =
    Pin<Box<dyn Future<Output = Result<ProviderStream<VadTransition>, ParlevoxError>> + Send>>;

/// What the stream is currently waiting on.
enum Phase {
    /// Not started; the first poll issues the inner `analyze` call.
    Idle,
    /// The inner provider's `analyze` call is in flight.
    Starting(StartFuture),
    /// The inner provider accepted the request; its stream is being polled.
    Streaming(ProviderStream<VadTransition>),
    /// The stream is finished; repeated polls return end-of-stream.
    Ended,
}

/// The stream returned by [`AdaptiveEndpointingVad::analyze`].
struct EndpointingStream {
    inner: Arc<dyn VadProvider>,
    policy: EndpointingPolicy,
    counters: Arc<EndpointingCounters>,
    audio_duration_ms: u64,
    /// The chunk to hand the inner provider on the first poll.
    audio: Option<AudioChunk>,
    cancel: CancellationToken,
    phase: Phase,
    /// Whether the current utterance is open and awaiting an accepted end.
    speech_open: bool,
    /// The accepted [`VadEvent::SpeechStart`] timestamp, if one was seen.
    t0: Option<u64>,
}

impl EndpointingStream {
    /// Apply one inner transition and decide what to yield.
    ///
    /// Returns `Some(transition)` when the transition is forwarded and `None`
    /// when a [`SpeechEnd`](VadEvent::SpeechEnd) is dropped.
    fn apply(&mut self, transition: VadTransition) -> Option<VadTransition> {
        match transition.event {
            VadEvent::SpeechStart => {
                if !self.speech_open {
                    self.speech_open = true;
                    self.t0 = Some(transition.timestamp_ms);
                }
                Some(transition)
            }
            VadEvent::SpeechEnd => {
                let speech_ms = self
                    .t0
                    .map_or(0, |t0| transition.timestamp_ms.saturating_sub(t0));
                let effective = self.policy.effective_allowance(speech_ms);
                let trailing_ms = self
                    .audio_duration_ms
                    .saturating_sub(transition.timestamp_ms);
                if trailing_ms >= effective {
                    self.counters.accepted.fetch_add(1, Ordering::Relaxed);
                    self.speech_open = false;
                    self.t0 = None;
                    Some(transition)
                } else {
                    self.counters.rejected.fetch_add(1, Ordering::Relaxed);
                    // Keep the utterance open even when no start was seen, so a
                    // clean end of stream still synthesizes a final end.
                    self.speech_open = true;
                    None
                }
            }
        }
    }

    /// The synthesized end emitted when the inner stream ends with speech open.
    fn synthesized_end(&mut self) -> VadTransition {
        self.counters.accepted.fetch_add(1, Ordering::Relaxed);
        self.speech_open = false;
        self.t0 = None;
        VadTransition {
            event: VadEvent::SpeechEnd,
            timestamp_ms: self.audio_duration_ms,
        }
    }
}

impl Stream for EndpointingStream {
    type Item = Result<VadTransition, ParlevoxError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            match std::mem::replace(&mut this.phase, Phase::Ended) {
                Phase::Ended => return Poll::Ready(None),
                Phase::Idle => {
                    let Some(audio) = this.audio.take() else {
                        return Poll::Ready(None);
                    };
                    let inner = Arc::clone(&this.inner);
                    let cancel = this.cancel.clone();
                    this.phase =
                        Phase::Starting(Box::pin(
                            async move { inner.analyze(audio, cancel).await },
                        ));
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
                        return Poll::Ready(Some(Err(error)));
                    }
                },
                Phase::Streaming(mut stream) => match stream.as_mut().poll_next(cx) {
                    Poll::Pending => {
                        this.phase = Phase::Streaming(stream);
                        return Poll::Pending;
                    }
                    Poll::Ready(Some(Ok(transition))) => {
                        this.phase = Phase::Streaming(stream);
                        if let Some(forwarded) = this.apply(transition) {
                            return Poll::Ready(Some(Ok(forwarded)));
                        }
                    }
                    Poll::Ready(Some(Err(error))) => {
                        // A failed analysis is terminal and suppresses the
                        // synthesized end; the next poll returns end-of-stream.
                        return Poll::Ready(Some(Err(error)));
                    }
                    Poll::Ready(None) => {
                        if this.speech_open {
                            let end = this.synthesized_end();
                            return Poll::Ready(Some(Ok(end)));
                        }
                        return Poll::Ready(None);
                    }
                },
            }
        }
    }
}

/// Derive the analysed audio's duration in milliseconds.
///
/// Integer samples per channel times `1000` divided by the sample rate, using
/// [`try_from`](u64::try_from) and saturating arithmetic so a hostile length
/// cannot overflow or wrap. A zero channel count or sample rate yields `0`.
fn audio_duration_ms(audio: &AudioChunk) -> u64 {
    if audio.channels == 0 || audio.sample_rate == 0 {
        return 0;
    }
    let channels = usize::from(audio.channels);
    let samples_per_channel = audio.pcm.len() / channels;
    let samples = u64::try_from(samples_per_channel).unwrap_or(u64::MAX);
    samples.saturating_mul(1000) / u64::from(audio.sample_rate)
}
