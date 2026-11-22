//! Ordered provider fallback chains that never replay after commit.
//!
//! [`FallbackLlm`], [`FallbackTts`] and [`FallbackStt`] compose an ordered list
//! of providers into one provider of the same modality. A request goes to the
//! first provider; the chain moves to the next only when the current provider
//! fails *before* it has yielded a single item and the failure is classed
//! [`ErrorClass::Transient`] or [`ErrorClass::Capacity`].
//!
//! For [`FallbackStt`] the request is the utterance [`AudioChunk`]. It is
//! cloned per attempt, so a provider that fails before committing never
//! consumes the audio: a switched attempt transcribes byte-identical input.
//!
//! # The no-fallback-after-commit law
//!
//! "Committed" means the current provider's stream has already yielded at
//! least one `Ok` item. The runtime cannot un-say a spoken sentence or
//! un-stream half a reply, so once a provider has committed, a later error is
//! returned to the caller unchanged and the chain ends. No other provider is
//! started after a commit; every fall-back decision is guarded through the
//! `committed` helper.
//!
//! # Identity
//!
//! A chain is identified by its primary, the first provider. The chain's
//! `manifest()` delegates to it. `take_sidecar_dropped_frames()` instead sums
//! every provider in the chain, so a failed provider's per-turn loss report is
//! not dropped by the adapter.
//!
//! # Counters
//!
//! `FallbackStats::switched` counts providers abandoned before committing and
//! replaced by the next provider. `FallbackStats::exhausted` counts switchable
//! failures that had no next provider to try; a non-switchable stop is not an
//! exhaustion, and a chain whose providers never fail leaves both at zero.
//!
//! # Cancellation
//!
//! The caller's [`CancellationToken`] is handed to every provider the chain
//! starts, and no provider is started after the token fires.
//! [`ErrorClass::Cancelled`] is deliberately *not* switchable: the caller
//! asked to stop, and starting another provider would speak after a barge-in.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

use async_trait::async_trait;
use futures_core::Stream;
use parlevox_core::{ErrorClass, ParlevoxError};
use tokio_util::sync::CancellationToken;

use crate::manifest::ProviderManifest;
use crate::traits::{
    AudioChunk, GenerateRequest, LlmProvider, ProviderStream, SttProvider, SynthesizeRequest,
    TranscriptChunk, TtsProvider,
};

/// Monotonic counters accumulated by a fallback chain.
///
/// Read a point-in-time copy with [`FallbackStats::snapshot`].
#[derive(Debug, Default)]
pub struct FallbackStats {
    switched: AtomicU64,
    exhausted: AtomicU64,
}

/// A point-in-time copy of [`FallbackStats`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FallbackStatsSnapshot {
    /// Providers abandoned before committing, each followed by starting the
    /// next provider in the chain.
    pub switched: u64,
    /// Switchable failures that found no next provider to try.
    pub exhausted: u64,
}

impl FallbackStats {
    /// A point-in-time copy of the counters.
    #[must_use]
    pub fn snapshot(&self) -> FallbackStatsSnapshot {
        FallbackStatsSnapshot {
            switched: self.switched.load(Ordering::Relaxed),
            exhausted: self.exhausted.load(Ordering::Relaxed),
        }
    }

    fn bump(counter: &AtomicU64) {
        counter.fetch_add(1, Ordering::Relaxed);
    }
}

/// An LLM provider that tries each provider in turn until one commits.
///
/// Build one with [`FallbackLlm::new`]. See the [module docs](self) for the
/// no-fallback-after-commit law and the counter semantics.
pub struct FallbackLlm {
    primary: LlmLink,
    links: Arc<[LlmLink]>,
    stats: Arc<FallbackStats>,
}

impl FallbackLlm {
    /// Build a chain from `providers`, tried in order.
    ///
    /// The first provider is the chain's primary and defines its manifest.
    ///
    /// # Errors
    ///
    /// Returns [`ParlevoxError::Protocol`] when `providers` is empty: a chain
    /// with no provider can never answer a request.
    pub fn new(providers: Vec<Arc<dyn LlmProvider>>) -> Result<Self, ParlevoxError> {
        let mut remaining = providers.into_iter();
        let Some(first) = remaining.next() else {
            return Err(ParlevoxError::Protocol(
                "fallback chain has no providers".into(),
            ));
        };
        let primary = LlmLink(first);
        let mut links = Vec::with_capacity(1 + remaining.len());
        links.push(primary.clone());
        links.extend(remaining.map(LlmLink));
        Ok(FallbackLlm {
            primary,
            links: Arc::from(links),
            stats: Arc::new(FallbackStats::default()),
        })
    }

    /// A point-in-time copy of the chain's counters.
    #[must_use]
    pub fn stats(&self) -> FallbackStatsSnapshot {
        self.stats.snapshot()
    }
}

#[async_trait]
impl LlmProvider for FallbackLlm {
    fn manifest(&self) -> &ProviderManifest {
        self.primary.manifest()
    }

    async fn generate(
        &self,
        request: GenerateRequest,
        cancel: CancellationToken,
    ) -> Result<ProviderStream<String>, ParlevoxError> {
        Ok(Box::pin(ChainStream::new(
            Arc::clone(&self.links),
            request,
            cancel,
            Arc::clone(&self.stats),
        )))
    }

    fn take_sidecar_dropped_frames(&self) -> u64 {
        self.links
            .iter()
            .map(|link| link.take_sidecar_dropped_frames())
            .sum()
    }
}

/// A TTS provider that tries each provider in turn until one commits.
///
/// Build one with [`FallbackTts::new`]. See the [module docs](self) for the
/// no-fallback-after-commit law and the counter semantics.
pub struct FallbackTts {
    primary: TtsLink,
    links: Arc<[TtsLink]>,
    stats: Arc<FallbackStats>,
}

impl FallbackTts {
    /// Build a chain from `providers`, tried in order.
    ///
    /// The first provider is the chain's primary and defines its manifest.
    ///
    /// # Errors
    ///
    /// Returns [`ParlevoxError::Protocol`] when `providers` is empty: a chain
    /// with no provider can never answer a request.
    pub fn new(providers: Vec<Arc<dyn TtsProvider>>) -> Result<Self, ParlevoxError> {
        let mut remaining = providers.into_iter();
        let Some(first) = remaining.next() else {
            return Err(ParlevoxError::Protocol(
                "fallback chain has no providers".into(),
            ));
        };
        let primary = TtsLink(first);
        let mut links = Vec::with_capacity(1 + remaining.len());
        links.push(primary.clone());
        links.extend(remaining.map(TtsLink));
        Ok(FallbackTts {
            primary,
            links: Arc::from(links),
            stats: Arc::new(FallbackStats::default()),
        })
    }

    /// A point-in-time copy of the chain's counters.
    #[must_use]
    pub fn stats(&self) -> FallbackStatsSnapshot {
        self.stats.snapshot()
    }
}

#[async_trait]
impl TtsProvider for FallbackTts {
    fn manifest(&self) -> &ProviderManifest {
        self.primary.manifest()
    }

    async fn synthesize(
        &self,
        request: SynthesizeRequest,
        cancel: CancellationToken,
    ) -> Result<ProviderStream<AudioChunk>, ParlevoxError> {
        Ok(Box::pin(ChainStream::new(
            Arc::clone(&self.links),
            request,
            cancel,
            Arc::clone(&self.stats),
        )))
    }

    fn take_sidecar_dropped_frames(&self) -> u64 {
        self.links
            .iter()
            .map(|link| link.take_sidecar_dropped_frames())
            .sum()
    }
}

/// An STT provider that tries each provider in turn until one commits.
///
/// Build one with [`FallbackStt::new`]. See the [module docs](self) for the
/// no-fallback-after-commit law and the counter semantics.
pub struct FallbackStt {
    primary: SttLink,
    links: Arc<[SttLink]>,
    stats: Arc<FallbackStats>,
}

impl FallbackStt {
    /// Build a chain from `providers`, tried in order.
    ///
    /// The first provider is the chain's primary and defines its manifest.
    ///
    /// # Errors
    ///
    /// Returns [`ParlevoxError::Protocol`] when `providers` is empty: a chain
    /// with no provider can never transcribe an utterance.
    pub fn new(providers: Vec<Arc<dyn SttProvider>>) -> Result<Self, ParlevoxError> {
        let mut remaining = providers.into_iter();
        let Some(first) = remaining.next() else {
            return Err(ParlevoxError::Protocol(
                "fallback chain has no providers".into(),
            ));
        };
        let primary = SttLink(first);
        let mut links = Vec::with_capacity(1 + remaining.len());
        links.push(primary.clone());
        links.extend(remaining.map(SttLink));
        Ok(FallbackStt {
            primary,
            links: Arc::from(links),
            stats: Arc::new(FallbackStats::default()),
        })
    }

    /// A point-in-time copy of the chain's counters.
    #[must_use]
    pub fn stats(&self) -> FallbackStatsSnapshot {
        self.stats.snapshot()
    }
}

#[async_trait]
impl SttProvider for FallbackStt {
    fn manifest(&self) -> &ProviderManifest {
        self.primary.manifest()
    }

    async fn transcribe(
        &self,
        audio: AudioChunk,
        cancel: CancellationToken,
    ) -> Result<ProviderStream<TranscriptChunk>, ParlevoxError> {
        Ok(Box::pin(ChainStream::new(
            Arc::clone(&self.links),
            audio,
            cancel,
            Arc::clone(&self.stats),
        )))
    }

    fn take_sidecar_dropped_frames(&self) -> u64 {
        self.links
            .iter()
            .map(|link| link.take_sidecar_dropped_frames())
            .sum()
    }
}

/// Guard for the no-fallback-after-commit law: once a provider has yielded at
/// least one item, the chain must never fall back to another provider.
fn committed(committed: bool) -> bool {
    committed
}

/// A boxed start future held by the chain's state machine.
type StartFuture<T> =
    Pin<Box<dyn Future<Output = Result<ProviderStream<T>, ParlevoxError>> + Send>>;

/// One modality-erased link in a fallback chain.
///
/// This is the single private abstraction that lets one state machine serve
/// [`FallbackLlm`], [`FallbackTts`] and [`FallbackStt`]; it has exactly three
/// implementations, [`LlmLink`], [`TtsLink`] and [`SttLink`].
trait ChainLink: Send + Sync {
    type Item: Send + 'static;
    type Request: Clone + Send + Unpin + 'static;
    type Start: Future<Output = Result<ProviderStream<Self::Item>, ParlevoxError>>
        + Send
        + Unpin
        + 'static;

    fn manifest(&self) -> &ProviderManifest;

    fn take_sidecar_dropped_frames(&self) -> u64;

    fn start(&self, request: Self::Request, cancel: CancellationToken) -> Self::Start;
}

/// [`ChainLink`] adapter over an [`LlmProvider`].
#[derive(Clone)]
struct LlmLink(Arc<dyn LlmProvider>);

impl ChainLink for LlmLink {
    type Item = String;
    type Request = GenerateRequest;
    type Start = StartFuture<String>;

    fn manifest(&self) -> &ProviderManifest {
        self.0.manifest()
    }

    fn take_sidecar_dropped_frames(&self) -> u64 {
        self.0.take_sidecar_dropped_frames()
    }

    fn start(&self, request: Self::Request, cancel: CancellationToken) -> Self::Start {
        let provider = Arc::clone(&self.0);
        Box::pin(async move { provider.generate(request, cancel).await })
    }
}

/// [`ChainLink`] adapter over a [`TtsProvider`].
#[derive(Clone)]
struct TtsLink(Arc<dyn TtsProvider>);

impl ChainLink for TtsLink {
    type Item = AudioChunk;
    type Request = SynthesizeRequest;
    type Start = StartFuture<AudioChunk>;

    fn manifest(&self) -> &ProviderManifest {
        self.0.manifest()
    }

    fn take_sidecar_dropped_frames(&self) -> u64 {
        self.0.take_sidecar_dropped_frames()
    }

    fn start(&self, request: Self::Request, cancel: CancellationToken) -> Self::Start {
        let provider = Arc::clone(&self.0);
        Box::pin(async move { provider.synthesize(request, cancel).await })
    }
}

/// [`ChainLink`] adapter over an [`SttProvider`].
#[derive(Clone)]
struct SttLink(Arc<dyn SttProvider>);

impl ChainLink for SttLink {
    type Item = TranscriptChunk;
    type Request = AudioChunk;
    type Start = StartFuture<TranscriptChunk>;

    fn manifest(&self) -> &ProviderManifest {
        self.0.manifest()
    }

    fn take_sidecar_dropped_frames(&self) -> u64 {
        self.0.take_sidecar_dropped_frames()
    }

    fn start(&self, request: Self::Request, cancel: CancellationToken) -> Self::Start {
        let provider = Arc::clone(&self.0);
        Box::pin(async move { provider.transcribe(request, cancel).await })
    }
}

/// What the chain is currently waiting on.
enum Current<L: ChainLink> {
    /// The next provider's `start` call has been issued and is being polled.
    Starting(L::Start),
    /// The current provider accepted the request; its stream is being polled.
    Streaming(ProviderStream<L::Item>),
    /// The chain is finished; repeated polls return end-of-stream.
    Ended,
}

/// The stream returned by a chain: a provider-by-provider state machine.
///
/// It never spins: each `poll_next` either makes observable progress or
/// returns [`Poll::Pending`]. It never returns end-of-stream before delivering
/// a terminal error.
struct ChainStream<L: ChainLink> {
    links: Arc<[L]>,
    request: L::Request,
    cancel: CancellationToken,
    stats: Arc<FallbackStats>,
    index: usize,
    committed: bool,
    current: Current<L>,
}

impl<L: ChainLink> ChainStream<L> {
    fn new(
        links: Arc<[L]>,
        request: L::Request,
        cancel: CancellationToken,
        stats: Arc<FallbackStats>,
    ) -> Self {
        let current = match links.first() {
            Some(first) => Current::Starting(first.start(request.clone(), cancel.clone())),
            None => Current::Ended,
        };
        ChainStream {
            links,
            request,
            cancel,
            stats,
            index: 0,
            committed: false,
            current,
        }
    }

    /// Decide whether this failure moves the chain to the next provider.
    ///
    /// Returns `true` when a next provider was started, in which case
    /// `self.current` holds its pending start. Consults [`committed`] so a
    /// failure after a provider has yielded an item never switches.
    fn try_switch(&mut self, error: &ParlevoxError) -> bool {
        if committed(self.committed) {
            return false;
        }
        let switchable = matches!(error.class(), ErrorClass::Transient | ErrorClass::Capacity);
        if !switchable {
            return false;
        }
        let next = self.index + 1;
        if next >= self.links.len() {
            FallbackStats::bump(&self.stats.exhausted);
            return false;
        }
        if self.cancel.is_cancelled() {
            // The caller asked to stop; never start a provider after barge-in.
            return false;
        }
        self.index = next;
        self.committed = false;
        FallbackStats::bump(&self.stats.switched);
        tracing::warn!(
            next = next,
            error = %error,
            "provider failed before commit; falling back"
        );
        let request = self.request.clone();
        let cancel = self.cancel.clone();
        let start = {
            let link = &self.links[next];
            link.start(request, cancel)
        };
        self.current = Current::Starting(start);
        true
    }
}

impl<L: ChainLink> Stream for ChainStream<L> {
    type Item = Result<L::Item, ParlevoxError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            match std::mem::replace(&mut this.current, Current::Ended) {
                Current::Ended => return Poll::Ready(None),
                Current::Starting(mut start) => match Pin::new(&mut start).poll(cx) {
                    Poll::Pending => {
                        this.current = Current::Starting(start);
                        return Poll::Pending;
                    }
                    Poll::Ready(Ok(stream)) => {
                        this.current = Current::Streaming(stream);
                    }
                    Poll::Ready(Err(error)) => {
                        if this.try_switch(&error) {
                            continue;
                        }
                        return Poll::Ready(Some(Err(error)));
                    }
                },
                Current::Streaming(mut stream) => match stream.as_mut().poll_next(cx) {
                    Poll::Pending => {
                        this.current = Current::Streaming(stream);
                        return Poll::Pending;
                    }
                    Poll::Ready(Some(Ok(item))) => {
                        this.committed = true;
                        this.current = Current::Streaming(stream);
                        return Poll::Ready(Some(Ok(item)));
                    }
                    Poll::Ready(Some(Err(error))) => {
                        if this.try_switch(&error) {
                            continue;
                        }
                        return Poll::Ready(Some(Err(error)));
                    }
                    Poll::Ready(None) => {
                        // A clean end is not a failure: the chain is done.
                        return Poll::Ready(None);
                    }
                },
            }
        }
    }
}
