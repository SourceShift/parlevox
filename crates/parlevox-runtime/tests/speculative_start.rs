//! Opt-in speculative LLM start: adoption, divergence, barge-in, and flag-off.
//!
//! The provider crate is out of scope, so every assertion about *what the LLM
//! was asked* goes through a local recording double rather than the shared
//! fakes. The doubles are deterministic and channel-gated: no assertion
//! depends on wall-clock timing.
//!
//! The adoption rule is exact: the trimmed partial requested speculatively must
//! be a prefix of the trimmed final. These tests prove that an extension is
//! adopted, that divergence abandons the speculative stream and asks again with
//! the final text, that a barge-in during the speculation interrupts the turn,
//! and that `speculative_start: None` leaves the old discard-the-partial path
//! byte-for-byte intact.

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, PoisonError};
use std::task::{Context, Poll};

use async_trait::async_trait;
use futures_core::Stream;
use parlevox_core::{ParlevoxError, TurnState};
use parlevox_providers::{
    cancelable_stream, AudioChunk, FakeTts, FakeVad, GenerateRequest, LlmProvider, Modality,
    ProviderManifest, ProviderStream, SttProvider, TranscriptChunk, FAKE_SAMPLE_RATE,
};
use parlevox_runtime::{
    demo_utterance, Providers, Session, SessionConfig, SpeculationStats, SpeculativeStart,
};
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

fn speculative_config(min_partial_chars: usize) -> SessionConfig {
    SessionConfig {
        speculative_start: Some(SpeculativeStart { min_partial_chars }),
        ..SessionConfig::default()
    }
}

/// A session over the supplied STT and LLM doubles, with the deterministic
/// fake TTS and VAD filling the other two roles.
fn session_with(
    stt: Arc<dyn SttProvider>,
    llm: Arc<dyn LlmProvider>,
    config: SessionConfig,
) -> Session {
    let providers = Providers {
        stt,
        llm,
        tts: Arc::new(FakeTts::new()),
        vad: Arc::new(FakeVad::new()),
    };
    Session::new(config, providers).expect("a valid session config")
}

fn prompt_log(log: &Arc<Mutex<Vec<String>>>) -> Vec<String> {
    log.lock().unwrap_or_else(PoisonError::into_inner).clone()
}

/// STT double that emits exactly one partial then one final, as the
/// `SttProvider` contract allows. The stream is cancellation-aware.
struct ScriptedStt {
    manifest: ProviderManifest,
    partial: String,
    final_text: String,
}

impl ScriptedStt {
    fn new(partial: impl Into<String>, final_text: impl Into<String>) -> Self {
        ScriptedStt {
            manifest: ProviderManifest::new("scripted-stt", Modality::Stt)
                .with_capability("streaming")
                .with_capability("partials"),
            partial: partial.into(),
            final_text: final_text.into(),
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
        audio: AudioChunk,
        cancel: CancellationToken,
    ) -> Result<ProviderStream<TranscriptChunk>, ParlevoxError> {
        if audio.pcm.is_empty() {
            return Err(ParlevoxError::Protocol("audio must not be empty".into()));
        }
        let items = vec![
            Ok(TranscriptChunk {
                text: self.partial.clone(),
                is_final: false,
            }),
            Ok(TranscriptChunk {
                text: self.final_text.clone(),
                is_final: true,
            }),
        ];
        Ok(cancelable_stream(items, cancel))
    }
}

/// LLM double that records every request prompt and hands back the next
/// scripted reply. Distinct `replies` per call make it observable which stream
/// the turn actually used.
struct RecordingLlm {
    manifest: ProviderManifest,
    prompts: Arc<Mutex<Vec<String>>>,
    replies: Arc<Mutex<VecDeque<Vec<String>>>>,
}

impl RecordingLlm {
    fn new(replies: Vec<Vec<String>>) -> Self {
        RecordingLlm {
            manifest: ProviderManifest::new("recording-llm", Modality::Llm)
                .with_capability("streaming")
                .with_capability("cancel"),
            prompts: Arc::new(Mutex::new(Vec::new())),
            replies: Arc::new(Mutex::new(replies.into())),
        }
    }

    /// Shared prompt log, so the session can own the provider while the test
    /// keeps reading what was asked.
    fn prompts(&self) -> Arc<Mutex<Vec<String>>> {
        Arc::clone(&self.prompts)
    }
}

#[async_trait]
impl LlmProvider for RecordingLlm {
    fn manifest(&self) -> &ProviderManifest {
        &self.manifest
    }

    async fn generate(
        &self,
        request: GenerateRequest,
        cancel: CancellationToken,
    ) -> Result<ProviderStream<String>, ParlevoxError> {
        self.prompts
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(request.prompt);
        let deltas = self
            .replies
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop_front()
            .unwrap_or_default();
        let items: Vec<Result<String, ParlevoxError>> = deltas.into_iter().map(Ok).collect();
        Ok(cancelable_stream(items, cancel))
    }
}

/// STT stream that emits the partial, announces that it has parked, and then
/// waits for permission before either yielding the final or reporting the
/// cancellation.
///
/// This is the deterministic handshake the barge-in test needs: the turn is
/// genuinely parked after the speculative request started, a spawned task runs
/// a real [`parlevox_runtime::BargeInHandle`], and only then is the stream
/// released.
struct GatedSttStream {
    cancel: CancellationToken,
    partial: String,
    final_text: String,
    parked: Option<oneshot::Sender<()>>,
    permission: oneshot::Receiver<()>,
    emitted_partial: bool,
    announced: bool,
    done: bool,
}

impl Stream for GatedSttStream {
    type Item = Result<TranscriptChunk, ParlevoxError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.done {
            return Poll::Ready(None);
        }
        if !self.emitted_partial {
            self.emitted_partial = true;
            return Poll::Ready(Some(Ok(TranscriptChunk {
                text: self.partial.clone(),
                is_final: false,
            })));
        }
        if !self.announced {
            self.announced = true;
            if let Some(parked) = self.parked.take() {
                let _ = parked.send(());
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
                // The barge-in cancels before it releases the gate, so this is
                // a cancellation, never a final transcript.
                if self.cancel.is_cancelled() {
                    Poll::Ready(Some(Err(ParlevoxError::Cancelled)))
                } else {
                    Poll::Ready(Some(Ok(TranscriptChunk {
                        text: self.final_text.clone(),
                        is_final: true,
                    })))
                }
            }
            Poll::Ready(Err(_)) => {
                self.done = true;
                Poll::Ready(Some(Err(ParlevoxError::Cancelled)))
            }
        }
    }
}

/// STT provider that hands out exactly one [`GatedSttStream`].
struct GatedStt {
    manifest: ProviderManifest,
    partial: String,
    final_text: String,
    parked: Mutex<Option<oneshot::Sender<()>>>,
    permission: Mutex<Option<oneshot::Receiver<()>>>,
}

impl GatedStt {
    fn new(
        partial: impl Into<String>,
        final_text: impl Into<String>,
        parked: oneshot::Sender<()>,
        permission: oneshot::Receiver<()>,
    ) -> Self {
        GatedStt {
            manifest: ProviderManifest::new("gated-stt", Modality::Stt)
                .with_capability("streaming")
                .with_capability("partials")
                .with_capability("cancel"),
            partial: partial.into(),
            final_text: final_text.into(),
            parked: Mutex::new(Some(parked)),
            permission: Mutex::new(Some(permission)),
        }
    }
}

#[async_trait]
impl SttProvider for GatedStt {
    fn manifest(&self) -> &ProviderManifest {
        &self.manifest
    }

    async fn transcribe(
        &self,
        audio: AudioChunk,
        cancel: CancellationToken,
    ) -> Result<ProviderStream<TranscriptChunk>, ParlevoxError> {
        if audio.pcm.is_empty() {
            return Err(ParlevoxError::Protocol("audio must not be empty".into()));
        }
        let parked = self
            .parked
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
                "gated stt transcribed more than once".into(),
            ));
        };
        Ok(Box::pin(GatedSttStream {
            cancel,
            partial: self.partial.clone(),
            final_text: self.final_text.clone(),
            parked,
            permission,
            emitted_partial: false,
            announced: false,
            done: false,
        }))
    }
}

#[tokio::test]
async fn adopt_extends_partial_into_final_reply() {
    // Arrange: the final extends the partial, and each scripted call would
    // produce visibly different text.
    let stt = Arc::new(ScriptedStt::new(
        "what is the wea",
        "what is the weather in Paris",
    ));
    let llm = RecordingLlm::new(vec![
        vec!["Sunny".into(), " in Paris.".into()],
        vec!["WRONG".into()],
    ]);
    let prompts = llm.prompts();
    let mut session = session_with(stt, Arc::new(llm), speculative_config(1));

    // Act: one turn whose speculation must be adopted.
    let report = session
        .run_turn(demo_utterance(FAKE_SAMPLE_RATE, 1, 40))
        .await
        .expect("the turn completes");

    // Assert: exactly one request, for the partial; the reply is its stream.
    assert_eq!(
        prompt_log(&prompts),
        vec!["what is the wea".to_string()],
        "adoption must suppress the second request"
    );
    assert_eq!(report.transcript, "what is the weather in Paris");
    assert_eq!(report.reply, "Sunny in Paris.");
    assert!(!report.interrupted);
    let stats = session.speculation_stats();
    assert_eq!(stats.started, 1);
    assert_eq!(stats.adopted, 1);
    assert_eq!(stats.abandoned, 0);
    assert_eq!(stats.turn_started, 1);
    assert_eq!(stats.turn_adopted, 1);
    assert_eq!(stats.turn_abandoned, 0);
    assert_eq!(session.state(), TurnState::Listening);
}

#[tokio::test]
async fn discard_divergent_partial_runs_final_path() {
    // Arrange: the final does not extend the partial.
    let stt = Arc::new(ScriptedStt::new(
        "what is the weather in Paris",
        "what is the capital of Peru",
    ));
    let llm = RecordingLlm::new(vec![
        vec!["It".into(), " rains.".into()],
        vec!["Lima".into(), " is the capital.".into()],
    ]);
    let prompts = llm.prompts();
    let mut session = session_with(stt, Arc::new(llm), speculative_config(1));

    // Act
    let report = session
        .run_turn(demo_utterance(FAKE_SAMPLE_RATE, 1, 40))
        .await
        .expect("the turn completes on the final path");

    // Assert: the speculative request happened first, then a fresh request
    // against the authoritative final, and only the latter's deltas are in the
    // report.
    assert_eq!(
        prompt_log(&prompts),
        vec![
            "what is the weather in Paris".to_string(),
            "what is the capital of Peru".to_string(),
        ]
    );
    assert_eq!(report.reply, "Lima is the capital.");
    assert!(!report.interrupted, "divergence is not a user interrupt");
    let stats = session.speculation_stats();
    assert_eq!(stats.started, 1);
    assert_eq!(stats.adopted, 0);
    assert_eq!(stats.abandoned, 1);
    assert_eq!(stats.turn_abandoned, 1);
    assert_eq!(session.state(), TurnState::Listening);
}

#[tokio::test]
async fn barge_in_during_speculation_interrupts() {
    // Arrange: an STT stream that parks after the partial, and a real
    // cross-task barge-in handle.
    let (parked_tx, parked_rx) = oneshot::channel();
    let (release_tx, release_rx) = oneshot::channel();
    let stt = Arc::new(GatedStt::new(
        "what is the wea",
        "what is the weather in Paris",
        parked_tx,
        release_rx,
    ));
    let llm = RecordingLlm::new(vec![vec!["Sunny".into()]]);
    let prompts = llm.prompts();
    let mut session = session_with(stt, Arc::new(llm), speculative_config(1));
    let handle = session.barge_in_handle();
    let spawned = tokio::spawn(async move {
        // Wait until the turn has consumed the partial (and therefore started
        // the speculative request) and is parked inside the STT stream.
        parked_rx.await.expect("stt parks after the partial");
        handle.barge_in();
        // Release the stream so the turn body observes the cancel.
        let _ = release_tx.send(());
    });

    // Act
    let report = session
        .run_turn(demo_utterance(FAKE_SAMPLE_RATE, 1, 40))
        .await
        .expect("a barge-in ends the turn without an error");
    spawned.await.expect("the barge-in task completes");

    // Assert: the turn unwinds before committing, so the report carries turn
    // zero and no transcript or reply, and the session rests in Listening.
    assert!(report.interrupted);
    assert_eq!(report.turn, 0);
    assert!(report.transcript.is_empty());
    assert!(report.reply.is_empty(), "no speculative reply may leak");
    assert_eq!(session.state(), TurnState::Listening);
    // The speculation was started but killed by the user, so it is counted
    // only in `started`: a barge-in is not a divergence abandonment.
    let stats = session.speculation_stats();
    assert_eq!(stats.started, 1);
    assert_eq!(stats.adopted, 0);
    assert_eq!(stats.abandoned, 0);
    assert_eq!(
        prompt_log(&prompts),
        vec!["what is the wea".to_string()],
        "the speculative request started before the interrupt"
    );
}

#[tokio::test]
async fn flag_off_discards_partial_sends_final_only() {
    // Arrange: the same partial/final pair, but speculation is disabled.
    let stt = Arc::new(ScriptedStt::new(
        "what is the wea",
        "what is the weather in Paris",
    ));
    let llm = RecordingLlm::new(vec![vec!["Sunny".into()]]);
    let prompts = llm.prompts();
    let mut session = session_with(stt, Arc::new(llm), SessionConfig::default());

    // Act
    let report = session
        .run_turn(demo_utterance(FAKE_SAMPLE_RATE, 1, 40))
        .await
        .expect("the turn completes");

    // Assert: exactly one request and it carries the final transcript.
    assert_eq!(
        prompt_log(&prompts),
        vec!["what is the weather in Paris".to_string()],
        "with the flag off the partial is discarded and only the final is asked"
    );
    assert_eq!(report.reply, "Sunny");
    assert_eq!(
        session.speculation_stats(),
        SpeculationStats::default(),
        "the flag-off path must not touch the counters"
    );
    assert_eq!(session.state(), TurnState::Listening);
}
