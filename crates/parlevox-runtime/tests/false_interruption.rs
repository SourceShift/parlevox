//! Opt-in false-interruption recovery by whole-reply replay.
//!
//! These tests drive the public session API with test-local provider wrappers
//! only: a scripted STT that yields one final per turn, a counting LLM whose
//! reply is split into several deltas, and a TTS that can block after its first
//! frame until it is released or cancelled. Coordination is channel-based with
//! no sleeps; the blocking TTS announces itself so a second task can run a real
//! [`BargeInHandle`] at a deterministic point.

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures_core::Stream;
use parlevox_core::{AudioFrame, ParlevoxError, Timestamp};
use parlevox_providers::{
    cancelable_stream, AudioChunk, FakeVad, GenerateRequest, LlmProvider, Modality,
    ProviderManifest, ProviderStream, SttProvider, SynthesizeRequest, TranscriptChunk, TtsProvider,
};
use parlevox_runtime::{
    demo_utterance, BargeInHandle, InterruptionRecovery, ProviderNames, ProviderRegistry, Session,
    SessionConfig, SessionEvent,
};
use tokio::sync::oneshot;
use tokio_util::sync::{CancellationToken, WaitForCancellationFutureOwned};

const STT_NAME: &str = "scripted-stt";
const LLM_NAME: &str = "counting-llm";
const TTS_NAME: &str = "test-tts";
const VAD_NAME: &str = "fake-vad";

/// A scripted STT that hands out one final transcript per `transcribe` call.
struct ScriptedStt {
    manifest: ProviderManifest,
    finals: Mutex<VecDeque<String>>,
}

impl ScriptedStt {
    fn new(finals: &[&str]) -> Self {
        ScriptedStt {
            manifest: ProviderManifest::new(STT_NAME, Modality::Stt),
            finals: Mutex::new(finals.iter().map(|text| (*text).to_string()).collect()),
        }
    }
}

#[async_trait]
impl SttProvider for ScriptedStt {
    fn manifest(&self) -> &ProviderManifest {
        &self.manifest
    }

    async fn transcribe(
        &self,
        _audio: AudioChunk,
        cancel: CancellationToken,
    ) -> Result<ProviderStream<TranscriptChunk>, ParlevoxError> {
        let text = self
            .finals
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop_front();
        let Some(text) = text else {
            return Err(ParlevoxError::Protocol(
                "scripted stt ran out of finals".into(),
            ));
        };
        Ok(cancelable_stream(
            vec![Ok(TranscriptChunk {
                text,
                is_final: true,
            })],
            cancel,
        ))
    }
}

/// A counting LLM that records every prompt and streams one scripted reply,
/// split into space-preserving deltas so a barge-in can land mid-stream.
struct CountingLlm {
    manifest: ProviderManifest,
    replies: Mutex<VecDeque<String>>,
    prompts: Mutex<Vec<String>>,
    calls: AtomicUsize,
    /// Optional one-shot signal fired when the reply stream ends, so a test can
    /// barge in only after the LLM has delivered its last delta.
    end_signal: Mutex<Option<oneshot::Sender<()>>>,
}

impl CountingLlm {
    fn new(replies: &[&str]) -> Self {
        CountingLlm::with_end_signal(replies, None)
    }

    fn with_end_signal(replies: &[&str], end_signal: Option<oneshot::Sender<()>>) -> Self {
        CountingLlm {
            manifest: ProviderManifest::new(LLM_NAME, Modality::Llm),
            replies: Mutex::new(replies.iter().map(|text| (*text).to_string()).collect()),
            prompts: Mutex::new(Vec::new()),
            calls: AtomicUsize::new(0),
            end_signal: Mutex::new(end_signal),
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    fn prompts(&self) -> Vec<String> {
        self.prompts
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

#[async_trait]
impl LlmProvider for CountingLlm {
    fn manifest(&self) -> &ProviderManifest {
        &self.manifest
    }

    async fn generate(
        &self,
        request: GenerateRequest,
        cancel: CancellationToken,
    ) -> Result<ProviderStream<String>, ParlevoxError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.prompts
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(request.prompt);
        let reply = self
            .replies
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop_front();
        let Some(reply) = reply else {
            return Err(ParlevoxError::Protocol(
                "counting llm ran out of replies".into(),
            ));
        };
        let items = reply
            .split_inclusive(' ')
            .map(|delta| Ok(delta.to_string()))
            .collect();
        let stream = cancelable_stream(items, cancel);
        let end_signal = self
            .end_signal
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        match end_signal {
            Some(ended) => Ok(Box::pin(LlmEndStream {
                inner: stream,
                ended: Some(ended),
            })),
            None => Ok(stream),
        }
    }
}

/// Wraps a provider stream and fires a one-shot signal once the underlying
/// stream reports its end. The signal is sent from the poll that observes
/// `None`, so the LLM is guaranteed to have delivered every delta first.
struct LlmEndStream {
    inner: ProviderStream<String>,
    ended: Option<oneshot::Sender<()>>,
}

impl Stream for LlmEndStream {
    type Item = Result<String, ParlevoxError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let polled = self.inner.as_mut().poll_next(cx);
        match polled {
            Poll::Ready(None) => {
                if let Some(ended) = self.ended.take() {
                    let _ = ended.send(());
                }
                Poll::Ready(None)
            }
            other => other,
        }
    }
}

/// How the TTS behaves on its very first `synthesize` call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FirstCall {
    /// Yield one frame, announce itself, then wait for release or cancellation.
    Block,
    /// Yield one frame, then report an upstream `Cancelled` error.
    Cancel,
    /// Stream two frames and finish normally.
    Normal,
}

/// A TTS double that records the text it was asked to synthesize.
struct TestTts {
    manifest: ProviderManifest,
    first_call: FirstCall,
    synthesized: Mutex<Vec<String>>,
    calls: AtomicUsize,
    producing: Mutex<Option<oneshot::Sender<()>>>,
    permission: Mutex<Option<oneshot::Receiver<()>>>,
}

impl TestTts {
    fn new(
        first_call: FirstCall,
        producing: Option<oneshot::Sender<()>>,
        permission: Option<oneshot::Receiver<()>>,
    ) -> Self {
        TestTts {
            manifest: ProviderManifest::new(TTS_NAME, Modality::Tts),
            first_call,
            synthesized: Mutex::new(Vec::new()),
            calls: AtomicUsize::new(0),
            producing: Mutex::new(producing),
            permission: Mutex::new(permission),
        }
    }

    fn synthesized(&self) -> Vec<String> {
        self.synthesized
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

#[async_trait]
impl TtsProvider for TestTts {
    fn manifest(&self) -> &ProviderManifest {
        &self.manifest
    }

    async fn synthesize(
        &self,
        request: SynthesizeRequest,
        cancel: CancellationToken,
    ) -> Result<ProviderStream<AudioChunk>, ParlevoxError> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        self.synthesized
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(request.text);
        let chunk = || AudioChunk::new(request.sample_rate, request.channels, vec![1; 16]);
        if call != 0 {
            return Ok(cancelable_stream(vec![Ok(chunk()), Ok(chunk())], cancel));
        }
        match self.first_call {
            FirstCall::Normal => Ok(cancelable_stream(vec![Ok(chunk()), Ok(chunk())], cancel)),
            FirstCall::Cancel => Ok(cancelable_stream(
                vec![Ok(chunk()), Err(ParlevoxError::Cancelled)],
                cancel,
            )),
            FirstCall::Block => {
                let producing = self
                    .producing
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .take();
                let permission = self
                    .permission
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .take();
                let permission = permission.ok_or_else(|| {
                    ParlevoxError::Protocol("blocking tts had no permission channel".into())
                })?;
                Ok(Box::pin(BlockingTtsStream {
                    cancel_wait: Box::pin(cancel.cancelled_owned()),
                    frame: Some(chunk()),
                    producing,
                    permission,
                    announced: false,
                    done: false,
                }))
            }
        }
    }
}

/// A stream that yields one frame, announces itself, then waits until it is
/// released or cancelled. A cancelled wait reports `ParlevoxError::Cancelled`.
///
/// The cancellation future is built once and polled in place. Rebuilding a
/// `cancelled()` waiter on every poll and dropping it would lose the registered
/// waker, leaving this stream parked forever when the release is withheld.
struct BlockingTtsStream {
    cancel_wait: Pin<Box<WaitForCancellationFutureOwned>>,
    frame: Option<AudioChunk>,
    producing: Option<oneshot::Sender<()>>,
    permission: oneshot::Receiver<()>,
    announced: bool,
    done: bool,
}

impl Stream for BlockingTtsStream {
    type Item = Result<AudioChunk, ParlevoxError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.done {
            return Poll::Ready(None);
        }
        if let Some(frame) = self.frame.take() {
            return Poll::Ready(Some(Ok(frame)));
        }
        if !self.announced {
            self.announced = true;
            if let Some(producing) = self.producing.take() {
                let _ = producing.send(());
            }
        }
        // Poll the stored cancellation future and register its waker on every
        // poll, so a barge-in alone can wake a stream whose release is held.
        if self.cancel_wait.as_mut().poll(cx).is_ready() {
            self.done = true;
            return Poll::Ready(Some(Err(ParlevoxError::Cancelled)));
        }
        match Pin::new(&mut self.permission).poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(_) => {
                self.done = true;
                Poll::Ready(None)
            }
        }
    }
}

fn build_session(
    stt: Arc<dyn SttProvider>,
    llm: Arc<dyn LlmProvider>,
    tts: Arc<dyn TtsProvider>,
    config: SessionConfig,
    recovery: Option<InterruptionRecovery>,
) -> Session {
    let mut registry = ProviderRegistry::new();
    registry.register_stt(stt).expect("scripted stt is valid");
    registry.register_llm(llm).expect("counting llm is valid");
    registry.register_tts(tts).expect("test tts is valid");
    registry
        .register_vad(Arc::new(FakeVad::new()))
        .expect("fake vad is valid");
    let names = ProviderNames {
        stt: STT_NAME.into(),
        llm: LLM_NAME.into(),
        tts: TTS_NAME.into(),
        vad: VAD_NAME.into(),
    };
    let providers = registry
        .resolve(&names)
        .expect("the scripted names resolve");
    let session = Session::new(config, providers).expect("valid session config");
    match recovery {
        Some(policy) => session.with_interruption_recovery(policy),
        None => session,
    }
}

fn utterance() -> AudioChunk {
    demo_utterance(16_000, 1, 40)
}

/// Strip the per-session monotonic timestamp from audio events so two sessions
/// can be compared event-for-event.
fn comparable(events: Vec<SessionEvent>) -> Vec<SessionEvent> {
    events
        .into_iter()
        .map(|event| match event {
            SessionEvent::AudioOut { turn, frame } => SessionEvent::AudioOut {
                turn,
                frame: AudioFrame::new(
                    Timestamp::from_millis(0),
                    frame.sample_rate,
                    frame.channels,
                    frame.pcm,
                ),
            },
            other => other,
        })
        .collect()
}

const REPLY_ONE: &str = "RETAINED REPLY ONE.";
const REPLY_TWO: &str = "FRESH REPLY TWO.";

#[tokio::test]
async fn resume_short_backchannel_replays_retained_reply() {
    // Arrange: turn 1's TTS blocks after one frame; a second task barges in
    // through a real handle as soon as the provider is parked.
    let (producing_tx, producing_rx) = oneshot::channel();
    let (permission_tx, permission_rx) = oneshot::channel();
    let stt = Arc::new(ScriptedStt::new(&["hello there", "mm-hmm"]));
    let llm = Arc::new(CountingLlm::new(&[REPLY_ONE, REPLY_TWO]));
    let tts = Arc::new(TestTts::new(
        FirstCall::Block,
        Some(producing_tx),
        Some(permission_rx),
    ));
    let mut session = build_session(
        stt.clone(),
        llm.clone(),
        tts.clone(),
        SessionConfig::default(),
        Some(InterruptionRecovery::default()),
    );
    let handle: BargeInHandle = session.barge_in_handle();
    let spawned = tokio::spawn(async move {
        producing_rx
            .await
            .expect("the tts announces it is producing");
        handle.barge_in();
        let _ = permission_tx.send(());
    });

    // Act: turn 1 is interrupted mid-TTS, turn 2's final is a backchannel.
    let first = session.run_turn(utterance()).await.expect("turn 1 ends");
    spawned.await.expect("the barge-in task completes");
    // Drain turn 1's events so the replay assertion below sees turn 2 alone.
    let _turn_one_events = session.take_events();
    let second = session
        .run_turn(utterance())
        .await
        .expect("the resumed turn completes");

    // Assert: turn 1 was interrupted with the reply text completely known.
    assert!(first.interrupted, "turn 1 is interrupted mid-TTS");
    assert_eq!(first.reply, REPLY_ONE, "turn 1 produced the complete reply");
    assert_eq!(
        session.recovery_stats().retained,
        1,
        "the complete reply is retained"
    );

    // Assert: turn 2 replayed from memory and never called the LLM.
    assert!(!second.interrupted, "the resumed turn completes");
    assert_eq!(second.reply, REPLY_ONE, "the retained reply is replayed");
    assert_eq!(llm.calls(), 1, "the llm is not called for a backchannel");
    assert_eq!(
        tts.synthesized(),
        vec![REPLY_ONE.to_string(), REPLY_ONE.to_string()],
        "the tts receives the retained reply text"
    );
    let stats = session.recovery_stats();
    assert_eq!(
        (stats.retained, stats.resumed, stats.discarded),
        (1, 1, 0),
        "retained 1, resumed 1, discarded 0"
    );
    let events = session.take_events();
    let replayed: Vec<&str> = events
        .iter()
        .filter_map(|event| match event {
            SessionEvent::ReplyDelta { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(replayed, vec![REPLY_ONE], "reply comes from memory");
}

#[tokio::test]
async fn long_transcript_discards_retained_reply() {
    let (producing_tx, producing_rx) = oneshot::channel();
    let (permission_tx, permission_rx) = oneshot::channel();
    let stt = Arc::new(ScriptedStt::new(&[
        "hello there",
        "please tell me a much longer story about something else",
    ]));
    let llm = Arc::new(CountingLlm::new(&[REPLY_ONE, REPLY_TWO]));
    let tts = Arc::new(TestTts::new(
        FirstCall::Block,
        Some(producing_tx),
        Some(permission_rx),
    ));
    let mut session = build_session(
        stt.clone(),
        llm.clone(),
        tts.clone(),
        SessionConfig::default(),
        Some(InterruptionRecovery::default()),
    );
    let handle = session.barge_in_handle();
    let spawned = tokio::spawn(async move {
        producing_rx
            .await
            .expect("the tts announces it is producing");
        handle.barge_in();
        let _ = permission_tx.send(());
    });

    let _first = session.run_turn(utterance()).await.expect("turn 1 ends");
    spawned.await.expect("the barge-in task completes");
    let second = session.run_turn(utterance()).await.expect("turn 2 ends");

    // The long transcript is not a backchannel: the LLM is called and the
    // retained reply is dropped, never replayed.
    assert!(!second.interrupted, "turn 2 is an ordinary turn");
    assert_eq!(second.reply, REPLY_TWO, "the fresh reply is used");
    assert_eq!(llm.calls(), 2, "a long transcript calls the llm");
    assert_eq!(
        tts.synthesized(),
        vec![REPLY_ONE.to_string(), REPLY_TWO.to_string()],
        "the retained reply is not replayed"
    );
    let stats = session.recovery_stats();
    assert_eq!(
        (stats.retained, stats.resumed, stats.discarded),
        (1, 0, 1),
        "retained 1, resumed 0, discarded 1"
    );
}

#[tokio::test]
async fn policy_off_calls_llm_and_reports_zero_stats() {
    let (producing_tx, producing_rx) = oneshot::channel();
    let (permission_tx, permission_rx) = oneshot::channel();
    let stt = Arc::new(ScriptedStt::new(&["hello there", "mm-hmm"]));
    let llm = Arc::new(CountingLlm::new(&[REPLY_ONE, REPLY_TWO]));
    let tts = Arc::new(TestTts::new(
        FirstCall::Block,
        Some(producing_tx),
        Some(permission_rx),
    ));
    let mut session = build_session(
        stt.clone(),
        llm.clone(),
        tts.clone(),
        SessionConfig::default(),
        None,
    );
    let handle = session.barge_in_handle();
    let spawned = tokio::spawn(async move {
        producing_rx
            .await
            .expect("the tts announces it is producing");
        handle.barge_in();
        let _ = permission_tx.send(());
    });

    let first = session.run_turn(utterance()).await.expect("turn 1 ends");
    spawned.await.expect("the barge-in task completes");
    let second = session.run_turn(utterance()).await.expect("turn 2 ends");

    assert!(first.interrupted, "turn 1 is interrupted as before");
    assert_eq!(
        second.reply, REPLY_TWO,
        "the same two turns behave as today"
    );
    assert_eq!(llm.calls(), 2, "the llm is called for turn 2");
    assert_eq!(
        session.recovery_stats(),
        parlevox_runtime::RecoveryStats::default(),
        "recovery stats are all zeros"
    );
}

#[tokio::test]
async fn upstream_cancel_retains_nothing() {
    let stt = Arc::new(ScriptedStt::new(&["hello there", "mm-hmm"]));
    let llm = Arc::new(CountingLlm::new(&[REPLY_ONE, REPLY_TWO]));
    let tts = Arc::new(TestTts::new(FirstCall::Cancel, None, None));
    let mut session = build_session(
        stt.clone(),
        llm.clone(),
        tts.clone(),
        SessionConfig::default(),
        Some(InterruptionRecovery::default()),
    );

    let first = session.run_turn(utterance()).await.expect("turn 1 ends");
    assert!(first.interrupted, "an upstream cancel stops the turn");
    assert_eq!(
        session.recovery_stats().retained,
        0,
        "an upstream cancel retains nothing"
    );

    let second = session.run_turn(utterance()).await.expect("turn 2 ends");
    assert_eq!(second.reply, REPLY_TWO, "the next short turn is ordinary");
    assert_eq!(llm.calls(), 2, "the next short turn calls the llm");
}

#[tokio::test]
async fn incomplete_reply_retains_nothing() {
    let stt = Arc::new(ScriptedStt::new(&["hello there", "mm-hmm"]));
    let llm = Arc::new(CountingLlm::new(&[REPLY_ONE, REPLY_TWO]));
    let tts = Arc::new(TestTts::new(FirstCall::Normal, None, None));
    let config = SessionConfig {
        barge_in_after_deltas: Some(1),
        ..SessionConfig::default()
    };
    let mut session = build_session(
        stt.clone(),
        llm.clone(),
        tts.clone(),
        config,
        Some(InterruptionRecovery::default()),
    );

    let first = session.run_turn(utterance()).await.expect("turn 1 ends");
    assert!(first.interrupted, "the llm stream is interrupted");
    assert!(
        first.reply.len() < REPLY_ONE.len(),
        "the reply is partial, not complete"
    );
    assert_eq!(
        session.recovery_stats().retained,
        0,
        "an incomplete reply retains nothing"
    );

    // `barge_in_after_deltas` is a session-wide test hook, so turn 2 is cut
    // short too. The point is that it reached the LLM with the short final
    // instead of replaying anything retained.
    let _second = session.run_turn(utterance()).await.expect("turn 2 ends");
    assert_eq!(llm.calls(), 2, "the next short turn calls the llm");
    assert_eq!(
        llm.prompts()[1],
        "mm-hmm",
        "the next turn used the final transcript, not a replay"
    );
    assert_eq!(
        session.recovery_stats().retained,
        0,
        "an incomplete reply retains nothing"
    );
}

#[tokio::test]
async fn pending_reply_consumed_once() {
    let (producing_tx, producing_rx) = oneshot::channel();
    let (permission_tx, permission_rx) = oneshot::channel();
    let stt = Arc::new(ScriptedStt::new(&["hello there", "mm-hmm", "mm-hmm"]));
    let llm = Arc::new(CountingLlm::new(&[REPLY_ONE, REPLY_TWO]));
    let tts = Arc::new(TestTts::new(
        FirstCall::Block,
        Some(producing_tx),
        Some(permission_rx),
    ));
    let mut session = build_session(
        stt.clone(),
        llm.clone(),
        tts.clone(),
        SessionConfig::default(),
        Some(InterruptionRecovery::default()),
    );
    let handle = session.barge_in_handle();
    let spawned = tokio::spawn(async move {
        producing_rx
            .await
            .expect("the tts announces it is producing");
        handle.barge_in();
        let _ = permission_tx.send(());
    });

    let _first = session.run_turn(utterance()).await.expect("turn 1 ends");
    spawned.await.expect("the barge-in task completes");
    let second = session.run_turn(utterance()).await.expect("turn 2 ends");
    let third = session.run_turn(utterance()).await.expect("turn 3 ends");

    assert_eq!(second.reply, REPLY_ONE, "turn 2 replays the retained reply");
    assert_eq!(third.reply, REPLY_TWO, "turn 3 calls the llm again");
    assert_eq!(llm.calls(), 2, "the pending reply is not replayed twice");
    let stats = session.recovery_stats();
    assert_eq!(
        (stats.retained, stats.resumed, stats.discarded),
        (1, 1, 0),
        "the pending reply is consumed exactly once"
    );
}

#[tokio::test]
async fn uninterrupted_turn_matches_policy_off_path() {
    // Two identical sessions, one with the policy enabled and one without, run
    // one ordinary turn. The opt-in branch must be inert: same provider inputs.
    let enabled_stt = Arc::new(ScriptedStt::new(&["hello there"]));
    let enabled_llm = Arc::new(CountingLlm::new(&[REPLY_ONE]));
    let enabled_tts = Arc::new(TestTts::new(FirstCall::Normal, None, None));
    let mut enabled = build_session(
        enabled_stt,
        enabled_llm.clone(),
        enabled_tts.clone(),
        SessionConfig::default(),
        Some(InterruptionRecovery::default()),
    );
    let plain_stt = Arc::new(ScriptedStt::new(&["hello there"]));
    let plain_llm = Arc::new(CountingLlm::new(&[REPLY_ONE]));
    let plain_tts = Arc::new(TestTts::new(FirstCall::Normal, None, None));
    let mut plain = build_session(
        plain_stt.clone(),
        plain_llm.clone(),
        plain_tts.clone(),
        SessionConfig::default(),
        None,
    );

    let enabled_report = enabled.run_turn(utterance()).await.expect("turn 1 ends");
    let plain_report = plain.run_turn(utterance()).await.expect("turn 2 ends");

    assert!(!enabled_report.interrupted);
    assert_eq!(
        enabled_report, plain_report,
        "the same report is produced with and without the policy"
    );
    assert_eq!(
        comparable(enabled.take_events()),
        comparable(plain.take_events()),
        "the event sequences are identical"
    );
    assert_eq!(
        enabled.recovery_stats(),
        parlevox_runtime::RecoveryStats::default(),
        "an uninterrupted turn retains and resumes nothing"
    );
    assert_eq!(plain_llm.prompts(), vec!["hello there".to_string()]);
    assert_eq!(plain_tts.synthesized(), vec![REPLY_ONE.to_string()]);
    assert_eq!(
        enabled_llm.prompts(),
        plain_llm.prompts(),
        "the enabled session sends the llm the same prompt"
    );
    assert_eq!(
        enabled_tts.synthesized(),
        plain_tts.synthesized(),
        "the enabled session sends the tts the same text"
    );
}

#[tokio::test]
async fn cancel_without_release_returns_interrupted_outcome() {
    // Arrange: the first TTS call parks after one frame and is never released,
    // so only the barge-in's cancellation can end the turn.
    let (producing_tx, producing_rx) = oneshot::channel();
    let (permission_tx, permission_rx) = oneshot::channel();
    let stt = Arc::new(ScriptedStt::new(&["hello there"]));
    let llm = Arc::new(CountingLlm::new(&[REPLY_ONE]));
    let tts = Arc::new(TestTts::new(
        FirstCall::Block,
        Some(producing_tx),
        Some(permission_rx),
    ));
    let mut session = build_session(
        stt,
        llm,
        tts,
        SessionConfig::default(),
        Some(InterruptionRecovery::default()),
    );
    let handle = session.barge_in_handle();
    let spawned = tokio::spawn(async move {
        producing_rx
            .await
            .expect("the tts announces it is producing");
        handle.barge_in();
    });

    // Act: the hang guard fails the test if the blocked stream never observes
    // the cancellation token instead of parking on the withheld release.
    let started = Instant::now();
    let first = tokio::time::timeout(Duration::from_secs(5), session.run_turn(utterance()))
        .await
        .expect("turn 1 ends before the hang guard")
        .expect("turn 1 returns a report");
    let elapsed = started.elapsed();
    spawned.await.expect("the barge-in task completes");

    // Assert: cancellation alone stopped the turn and retained the reply.
    assert!(first.interrupted, "cancellation interrupts turn 1");
    // The timeout alone cannot discriminate: on its deadline wake tokio polls
    // the inner future once more, and a stream that only checks
    // `is_cancelled()` at poll entry would then see the cancelled token and
    // complete after the full guard. A stream that registers the cancellation
    // waker returns as soon as the barge-in fires, so the elapsed budget is
    // what fails this test against a non-cancellation-aware stream.
    assert!(
        elapsed < Duration::from_secs(1),
        "the turn ends promptly on cancellation ({elapsed:?}), not on the 5s guard"
    );
    assert_eq!(first.reply, REPLY_ONE, "the whole reply was produced");
    assert_eq!(
        session.recovery_stats().retained,
        1,
        "the complete reply is retained without a release"
    );
    assert!(
        permission_tx.send(()).is_err(),
        "the blocked tts was torn down by cancellation, not released"
    );
}

#[tokio::test]
async fn pipelined_barge_in_resumes_retained_reply() {
    // Arrange: pipelined synthesis parks after its first frame, the LLM end is
    // signalled explicitly, and the barge-in lands only after both, so the
    // reply is known to be complete when the turn unwinds.
    let (producing_tx, producing_rx) = oneshot::channel();
    let (permission_tx, permission_rx) = oneshot::channel();
    let (llm_end_tx, llm_end_rx) = oneshot::channel();
    let stt = Arc::new(ScriptedStt::new(&["hello there", "mm-hmm"]));
    let llm = Arc::new(CountingLlm::with_end_signal(&[REPLY_ONE], Some(llm_end_tx)));
    let tts = Arc::new(TestTts::new(
        FirstCall::Block,
        Some(producing_tx),
        Some(permission_rx),
    ));
    let config = SessionConfig {
        pipelined_tts: true,
        ..SessionConfig::default()
    };
    let mut session = build_session(
        stt,
        llm.clone(),
        tts.clone(),
        config,
        Some(InterruptionRecovery::default()),
    );
    let handle = session.barge_in_handle();
    let spawned = tokio::spawn(async move {
        let (producing, ended) = tokio::join!(producing_rx, llm_end_rx);
        producing.expect("the tts announces it is producing");
        ended.expect("the llm announces the end of its stream");
        handle.barge_in();
    });

    // Act: turn 1 is interrupted with the release withheld, then turn 2 runs
    // the same resume reasoning as the ordinary backchannel path.
    let started = Instant::now();
    let first = tokio::time::timeout(Duration::from_secs(5), session.run_turn(utterance()))
        .await
        .expect("turn 1 ends before the hang guard")
        .expect("turn 1 returns a report");
    let elapsed = started.elapsed();
    spawned.await.expect("the barge-in task completes");
    // Drain turn 1's events so the assertions below see turn 2 alone.
    let _turn_one_events = session.take_events();
    let second = tokio::time::timeout(Duration::from_secs(5), session.run_turn(utterance()))
        .await
        .expect("the resumed turn ends before the hang guard")
        .expect("the resumed turn returns a report");

    // Assert: the pipelined turn retained the whole reply and the following
    // backchannel replayed it without calling the LLM again.
    assert!(
        first.interrupted,
        "turn 1 is interrupted after the llm ended"
    );
    // Same discriminator as the sequential test: a stream without a registered
    // cancellation wait would only finish when the guard's deadline re-poll
    // observes the already-cancelled token.
    assert!(
        elapsed < Duration::from_secs(1),
        "the pipelined turn ends promptly on cancellation ({elapsed:?}), not on the 5s guard"
    );
    assert_eq!(first.reply, REPLY_ONE, "the whole reply was produced");
    assert!(
        permission_tx.send(()).is_err(),
        "the blocked tts was torn down by cancellation, not released"
    );
    assert!(!second.interrupted, "the resumed turn completes");
    assert_eq!(second.reply, REPLY_ONE, "the retained reply is replayed");
    assert_eq!(llm.calls(), 1, "the follow-up turn does not call the llm");
    assert_eq!(
        tts.synthesized(),
        vec![REPLY_ONE.to_string(), REPLY_ONE.to_string()],
        "the retained text is synthesized again"
    );
    let stats = session.recovery_stats();
    assert_eq!(
        (stats.retained, stats.resumed, stats.discarded),
        (1, 1, 0),
        "retained 1, resumed 1, discarded 0"
    );
}
