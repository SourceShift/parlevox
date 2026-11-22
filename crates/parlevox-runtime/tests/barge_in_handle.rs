//! Cross-task barge-in and the observable `Interrupted` state.
//!
//! `Session::run_turn(&mut self)` owns the session for the whole turn, so the
//! only way another task can barge in is a [`BargeInHandle`]. These tests prove
//! the handle crosses a task boundary, that the cross-task signal ends the turn
//! in `Listening`, that `Interrupted` is observable between the signal and the
//! cleanup, and that a barge-in with no live turn does not poison the next one.
//!
//! The only test double is [`GatedTts`]; every other role uses the existing
//! deterministic fakes. Coordination is channel-based with no sleeps.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, PoisonError};
use std::task::{Context, Poll};

use async_trait::async_trait;
use futures_core::Stream;
use parlevox_core::{ParlevoxError, TurnEvent, TurnState};
use parlevox_providers::{
    AudioChunk, FakeLlm, FakeStt, FakeVad, Modality, ProviderManifest, ProviderStream,
    SynthesizeRequest, TtsProvider, FAKE_SAMPLE_RATE, FAKE_TRANSCRIPT,
};
use parlevox_runtime::{
    demo_utterance, BargeInHandle, ProviderNames, ProviderRegistry, Session, SessionConfig,
};
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

/// A TTS stream that announces "I am producing" on its first poll and then
/// waits for permission before yielding one chunk.
///
/// This is the deterministic handshake the cross-task test needs: the turn
/// parks inside a genuine provider stream, a spawned task runs a real
/// [`BargeInHandle`], and only then is the stream released.
struct GatedTtsStream {
    cancel: CancellationToken,
    chunk: Option<AudioChunk>,
    producing: Option<oneshot::Sender<()>>,
    permission: oneshot::Receiver<()>,
    announced: bool,
    done: bool,
}

impl Stream for GatedTtsStream {
    type Item = Result<AudioChunk, ParlevoxError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.done {
            return Poll::Ready(None);
        }
        if !self.announced {
            self.announced = true;
            if let Some(producing) = self.producing.take() {
                let _ = producing.send(());
            }
        }
        if self.cancel.is_cancelled() {
            self.done = true;
            return Poll::Ready(Some(Err(ParlevoxError::Cancelled)));
        }
        match Pin::new(&mut self.permission).poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Ok(())) => {
                self.done = true;
                match self.chunk.take() {
                    Some(chunk) => Poll::Ready(Some(Ok(chunk))),
                    None => Poll::Ready(None),
                }
            }
            Poll::Ready(Err(_)) => {
                self.done = true;
                Poll::Ready(None)
            }
        }
    }
}

/// TTS provider that hands out exactly one [`GatedTtsStream`].
struct GatedTts {
    manifest: ProviderManifest,
    producing: Mutex<Option<oneshot::Sender<()>>>,
    permission: Mutex<Option<oneshot::Receiver<()>>>,
}

impl GatedTts {
    fn new(producing: oneshot::Sender<()>, permission: oneshot::Receiver<()>) -> Self {
        GatedTts {
            manifest: ProviderManifest::new("gated-tts", Modality::Tts),
            producing: Mutex::new(Some(producing)),
            permission: Mutex::new(Some(permission)),
        }
    }
}

#[async_trait]
impl TtsProvider for GatedTts {
    fn manifest(&self) -> &ProviderManifest {
        &self.manifest
    }

    async fn synthesize(
        &self,
        request: SynthesizeRequest,
        cancel: CancellationToken,
    ) -> Result<ProviderStream<AudioChunk>, ParlevoxError> {
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
        let Some(permission) = permission else {
            return Err(ParlevoxError::Protocol(
                "gated tts synthesized more than once".into(),
            ));
        };
        let chunk = AudioChunk::new(request.sample_rate, request.channels, vec![1; 16]);
        Ok(Box::pin(GatedTtsStream {
            cancel,
            chunk: Some(chunk),
            producing,
            permission,
            announced: false,
            done: false,
        }))
    }
}

/// A session backed by the four deterministic fakes.
fn fake_session(config: SessionConfig) -> Session {
    let providers = ProviderRegistry::fakes()
        .expect("the built-in fakes are valid")
        .resolve(&ProviderNames::default())
        .expect("the default names resolve to the fakes");
    Session::new(config, providers).expect("valid session config")
}

/// A session whose TTS role is the gated double.
fn gated_session(tts: Arc<dyn TtsProvider>) -> Session {
    let mut registry = ProviderRegistry::new();
    registry
        .register_stt(Arc::new(FakeStt::new()))
        .expect("fake stt is valid");
    registry
        .register_llm(Arc::new(FakeLlm::new()))
        .expect("fake llm is valid");
    registry.register_tts(tts).expect("the tts is valid");
    registry
        .register_vad(Arc::new(FakeVad::new()))
        .expect("fake vad is valid");
    let names = ProviderNames {
        stt: "fake-stt".into(),
        llm: "fake-llm".into(),
        tts: "gated-tts".into(),
        vad: "fake-vad".into(),
    };
    let providers = registry.resolve(&names).expect("the names resolve");
    Session::new(SessionConfig::default(), providers).expect("valid session config")
}

#[tokio::test]
async fn barge_in_from_spawned_task_interrupts_turn() {
    // Arrange: a TTS stream that announces itself and then waits for
    // permission, and a handle cloned onto another task.
    let (producing_tx, producing_rx) = oneshot::channel();
    let (permission_tx, permission_rx) = oneshot::channel();
    let tts = Arc::new(GatedTts::new(producing_tx, permission_rx));
    let mut session = gated_session(tts);
    let handle = session.barge_in_handle();
    let spawned = tokio::spawn(async move {
        // Wait until the turn is genuinely parked inside the TTS stream.
        producing_rx
            .await
            .expect("the gated tts announces that it is producing");
        handle.barge_in();
        // Release the stream so the turn body can observe the cancel.
        let _ = permission_tx.send(());
    });

    // Act
    let report = session
        .run_turn(demo_utterance(FAKE_SAMPLE_RATE, 1, 40))
        .await
        .expect("a barge-in ends the turn without an error");
    spawned.await.expect("the barge-in task completes");

    // Assert
    assert!(report.interrupted);
    assert_eq!(session.state(), TurnState::Listening);
}

#[test]
fn interrupted_is_observable_between_barge_in_and_resume() {
    // Arrange: a session parked between a final transcript and audio.
    let mut session = fake_session(SessionConfig::default());
    session.start().expect("the session starts");
    session
        .apply(TurnEvent::FinalTranscript {
            text: FAKE_TRANSCRIPT.into(),
        })
        .expect("a final transcript opens a turn");
    session
        .apply(TurnEvent::ReplyStarted)
        .expect("reply start is legal while thinking");
    assert_eq!(session.state(), TurnState::Speaking);

    // Act: the barge-in signal lands.
    assert!(session.request_barge_in().expect("barge-in is legal"));

    // Assert: the state dwells in Interrupted until cleanup is applied.
    assert_eq!(session.state(), TurnState::Interrupted);
    session
        .apply(TurnEvent::ResumeListening)
        .expect("cleanup is legal while interrupted");
    assert_eq!(session.state(), TurnState::Listening);
}

#[tokio::test]
async fn barge_in_between_turns_does_not_poison_next_turn() {
    // Arrange: a listening session with no turn in flight.
    let mut session = fake_session(SessionConfig::default());
    session.start().expect("the session starts");
    assert_eq!(session.state(), TurnState::Listening);

    // Act: a barge-in with nothing to interrupt reports false and leaves no
    // pending request behind.
    assert!(!session.request_barge_in().expect("barge-in is a no-op"));

    // Assert: the next turn runs to completion normally.
    let report = session
        .run_turn(demo_utterance(FAKE_SAMPLE_RATE, 1, 40))
        .await
        .expect("the turn completes");
    assert!(report.audio_frames > 0);
    assert!(!report.interrupted);
    assert_eq!(session.state(), TurnState::Listening);
}

#[test]
fn barge_in_handle_is_send_sync_static() {
    fn assert_send_sync_static<T: Send + Sync + 'static>() {}

    assert_send_sync_static::<BargeInHandle>();
}
