//! Opt-in partial-audio resume from an estimated playback position.
//!
//! These tests drive the public session API with test-local provider wrappers
//! only: a scripted STT that yields one final per turn, a counting LLM whose
//! reply is split into several deltas, and a TTS that yields a fixed, distinct
//! frame sequence and can park after it or end normally. Coordination is
//! channel-based with no sleeps; the parking TTS announces itself so a second
//! task can run a real [`BargeInHandle`] at a deterministic point.
//!
//! The playback position is estimated from wall-clock elapsed time and the
//! frames' *declared* durations, not from ring occupancy, so the tests control
//! it declaratively. [`LONG_FRAMES`] declares eight seconds per frame, which no
//! test can approach, so the estimate stays at zero; [`ZERO_DURATION_FRAMES`]
//! declares zero, which reaches the end of the reply immediately. Intermediate
//! cursor values are covered precisely by [`playback_cursor_unit_cases`].

use std::collections::VecDeque;
use std::future::Future;
use std::ops::RangeInclusive;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::task::{Context, Poll};

use async_trait::async_trait;
use futures_core::Stream;
use parlevox_core::{AudioFrame, DurationMs, ParlevoxError, Timestamp};
use parlevox_providers::{
    cancelable_stream, AudioChunk, FakeVad, GenerateRequest, LlmProvider, Modality,
    ProviderManifest, ProviderStream, SttProvider, SynthesizeRequest, TranscriptChunk, TtsProvider,
};
use parlevox_runtime::session::playback_cursor;
use parlevox_runtime::{
    demo_utterance, InterruptionRecovery, PartialResume, ProviderNames, ProviderRegistry,
    RecoveryStats, Session, SessionConfig, SessionEvent,
};
use tokio::sync::oneshot;
use tokio_util::sync::{CancellationToken, WaitForCancellationFutureOwned};

const STT_NAME: &str = "scripted-stt";
const LLM_NAME: &str = "counting-llm";
const TTS_NAME: &str = "frame-tts";
const VAD_NAME: &str = "fake-vad";

const REPLY_ONE: &str = "RETAINED REPLY ONE.";
const REPLY_TWO: &str = "FRESH REPLY TWO.";

/// Explicit frame shape for the TTS double, independent of
/// [`SessionConfig::sample_rate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FrameFormat {
    sample_rate: u32,
    channels: u16,
    samples_per_channel: usize,
}

impl FrameFormat {
    /// Build one distinct chunk per frame index.
    fn chunk(&self, index: usize) -> AudioChunk {
        let value = i16::try_from(index).unwrap_or(i16::MAX).saturating_add(1);
        let samples = self
            .samples_per_channel
            .saturating_mul(usize::from(self.channels));
        AudioChunk::new(self.sample_rate, self.channels, vec![value; samples])
    }
}

/// Eight seconds per frame: far longer than any test's wall-clock runtime, so
/// the clock-based estimate stays at zero.
const LONG_FRAMES: FrameFormat = FrameFormat {
    sample_rate: 1,
    channels: 1,
    samples_per_channel: 8,
};

/// Zero declared duration: every produced frame is estimated heard at once.
const ZERO_DURATION_FRAMES: FrameFormat = FrameFormat {
    sample_rate: 0,
    channels: 1,
    samples_per_channel: 8,
};

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
/// split into space-preserving deltas.
struct CountingLlm {
    manifest: ProviderManifest,
    replies: Mutex<VecDeque<String>>,
    prompts: Mutex<Vec<String>>,
    calls: AtomicUsize,
}

impl CountingLlm {
    fn new(replies: &[&str]) -> Self {
        CountingLlm {
            manifest: ProviderManifest::new(LLM_NAME, Modality::Llm),
            replies: Mutex::new(replies.iter().map(|text| (*text).to_string()).collect()),
            prompts: Mutex::new(Vec::new()),
            calls: AtomicUsize::new(0),
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
        Ok(cancelable_stream(items, cancel))
    }
}

/// How the first `synthesize` call ends after it yields its frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FirstCallMode {
    /// Yield every frame, announce itself, then park until released or
    /// cancelled.
    Park,
    /// Yield every frame and end the stream normally.
    End,
}

/// A TTS double that yields a fixed number of distinct frames per call and
/// records the text it was asked to synthesize.
///
/// Frame `i` carries PCM value `i + 1`, so an emitted frame identifies exactly
/// which retained frame was resumed.
struct FrameTts {
    manifest: ProviderManifest,
    format: FrameFormat,
    frames: usize,
    first_call: FirstCallMode,
    synthesized: Mutex<Vec<String>>,
    calls: AtomicUsize,
    producing: Mutex<Option<oneshot::Sender<()>>>,
    permission: Mutex<Option<oneshot::Receiver<()>>>,
}

impl FrameTts {
    fn new(
        format: FrameFormat,
        frames: usize,
        first_call: FirstCallMode,
        producing: Option<oneshot::Sender<()>>,
        permission: Option<oneshot::Receiver<()>>,
    ) -> Self {
        FrameTts {
            manifest: ProviderManifest::new(TTS_NAME, Modality::Tts),
            format,
            frames,
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

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl TtsProvider for FrameTts {
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
        let frames: VecDeque<AudioChunk> = (0..self.frames)
            .map(|index| self.format.chunk(index))
            .collect();
        if call == 0 && self.first_call == FirstCallMode::Park {
            let permission = self
                .permission
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .take()
                .ok_or_else(|| {
                    ParlevoxError::Protocol("parking tts had no permission channel".into())
                })?;
            let producing = self
                .producing
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .take();
            return Ok(Box::pin(FramesThenPark {
                cancel_wait: Box::pin(cancel.cancelled_owned()),
                frames,
                producing,
                permission,
                announced: false,
                done: false,
            }));
        }
        Ok(Box::pin(FramesThenEnd {
            frames,
            done: false,
        }))
    }
}

/// Yields every scripted frame, then ends.
struct FramesThenEnd {
    frames: VecDeque<AudioChunk>,
    done: bool,
}

impl Stream for FramesThenEnd {
    type Item = Result<AudioChunk, ParlevoxError>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.done {
            return Poll::Ready(None);
        }
        if let Some(frame) = self.frames.pop_front() {
            return Poll::Ready(Some(Ok(frame)));
        }
        self.done = true;
        Poll::Ready(None)
    }
}

/// Yields every scripted frame, announces itself, then waits until it is
/// released or cancelled. A cancelled wait reports `ParlevoxError::Cancelled`.
///
/// The cancellation future is built once and polled in place, so a barge-in
/// alone can wake a stream whose release is withheld.
struct FramesThenPark {
    cancel_wait: Pin<Box<WaitForCancellationFutureOwned>>,
    frames: VecDeque<AudioChunk>,
    producing: Option<oneshot::Sender<()>>,
    permission: oneshot::Receiver<()>,
    announced: bool,
    done: bool,
}

impl Stream for FramesThenPark {
    type Item = Result<AudioChunk, ParlevoxError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.done {
            return Poll::Ready(None);
        }
        if let Some(frame) = self.frames.pop_front() {
            return Poll::Ready(Some(Ok(frame)));
        }
        if !self.announced {
            self.announced = true;
            if let Some(producing) = self.producing.take() {
                let _ = producing.send(());
            }
        }
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
    partial: Option<PartialResume>,
) -> Session {
    let mut registry = ProviderRegistry::new();
    registry.register_stt(stt).expect("scripted stt is valid");
    registry.register_llm(llm).expect("counting llm is valid");
    registry.register_tts(tts).expect("frame tts is valid");
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
    let mut session = Session::new(config, providers).expect("valid session config");
    if let Some(policy) = recovery {
        session = session.with_interruption_recovery(policy);
    }
    if let Some(policy) = partial {
        session = session.with_partial_audio_resume(policy);
    }
    session
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

fn audio_out(events: &[SessionEvent]) -> Vec<AudioFrame> {
    events
        .iter()
        .filter_map(|event| match event {
            SessionEvent::AudioOut { frame, .. } => Some(frame.clone()),
            _ => None,
        })
        .collect()
}

fn pcm(frames: &[AudioFrame]) -> Vec<Vec<i16>> {
    frames.iter().map(|frame| frame.pcm.clone()).collect()
}

/// The PCM payload each frame of [`FrameFormat::chunk`] carries, for a range of
/// indices.
fn expected_pcm(values: RangeInclusive<i16>) -> Vec<Vec<i16>> {
    values.map(|value| vec![value; 8]).collect()
}

#[tokio::test]
async fn resume_emits_only_the_unheard_frames() {
    // Arrange: the TTS produces six long-duration frames and parks before the
    // stream ends. The wall-clock estimate cannot reach even the first frame's
    // declared eight seconds, so none is estimated heard, while the barge-in
    // still interrupts the turn.
    let (producing_tx, producing_rx) = oneshot::channel();
    let (permission_tx, permission_rx) = oneshot::channel();
    let stt = Arc::new(ScriptedStt::new(&["hello there", "okay"]));
    let llm = Arc::new(CountingLlm::new(&[REPLY_ONE, REPLY_TWO]));
    let tts = Arc::new(FrameTts::new(
        LONG_FRAMES,
        6,
        FirstCallMode::Park,
        Some(producing_tx),
        Some(permission_rx),
    ));
    let mut session = build_session(
        stt,
        llm.clone(),
        tts.clone(),
        SessionConfig::default(),
        Some(InterruptionRecovery::default()),
        Some(PartialResume::default()),
    );
    let handle = session.barge_in_handle();
    let spawned = tokio::spawn(async move {
        producing_rx
            .await
            .expect("the tts announces it is producing");
        handle.barge_in();
        let _ = permission_tx.send(());
    });

    // Act: turn 1 is interrupted mid-synthesis; turn 2's final is a
    // backchannel, so it resumes instead of calling either provider.
    let first = session.run_turn(utterance()).await.expect("turn 1 ends");
    spawned.await.expect("the barge-in task completes");
    let turn_one = session.take_events();
    let second = session
        .run_turn(utterance())
        .await
        .expect("the resumed turn completes");
    let turn_two = session.take_events();

    // Assert: turn 1 emitted every produced frame (emission is immediate) and
    // retained the whole reply's audio because the estimate is zero.
    assert!(first.interrupted, "turn 1 is interrupted mid-synthesis");
    assert_eq!(first.reply, REPLY_ONE, "turn 1 produced the complete reply");
    let played_first = audio_out(&turn_one);
    assert_eq!(
        pcm(&played_first),
        expected_pcm(1..=6),
        "turn 1 emitted every produced frame at the policy-off cadence"
    );
    assert_eq!(
        session.recovery_stats().retained,
        1,
        "the complete reply's audio is retained for a zero estimate"
    );

    // Assert: turn 2 resumed at the zero estimate and emitted the produced
    // prefix from its first frame.
    assert!(!second.interrupted, "the resumed turn completes");
    assert_eq!(
        second.reply, REPLY_ONE,
        "turn 2 replies with the retained text"
    );
    assert_eq!(
        second.audio_frames, 6,
        "turn 2 reports only the frames it emitted"
    );
    let resumed = audio_out(&turn_two);
    assert_eq!(
        pcm(&resumed),
        expected_pcm(1..=6),
        "turn 2 emits the produced prefix from frame 1 in order"
    );
    // The clock has millisecond resolution, so freshness is asserted as the
    // monotonic invariant: a frame re-stamped at emit time is never older than
    // the last frame the interrupted turn emitted.
    let last_turn_one = played_first
        .iter()
        .map(|frame| frame.timestamp)
        .max()
        .expect("turn 1 emitted frames");
    assert!(
        resumed.iter().all(|frame| frame.timestamp >= last_turn_one),
        "resumed frames are re-stamped at emit time, not replayed with retained timestamps: {:?}",
        resumed
            .iter()
            .map(|frame| frame.timestamp)
            .collect::<Vec<_>>()
    );
    assert_eq!(llm.calls(), 1, "the llm is not called for a resume");
    assert_eq!(tts.calls(), 1, "the tts is not called for a resume");
    assert!(
        !turn_two
            .iter()
            .any(|event| matches!(event, SessionEvent::ReplyDelta { .. })),
        "the resumed turn emits no ReplyDelta"
    );
    let stats = session.recovery_stats();
    assert_eq!(
        (stats.retained, stats.resumed, stats.discarded),
        (1, 1, 0),
        "retained 1, resumed 1, discarded 0"
    );
    println!(
        "turn 1 pcm={:?} turn 2 pcm={:?} llm_calls={} tts_calls={} resumed={}",
        pcm(&played_first),
        pcm(&resumed),
        llm.calls(),
        tts.calls(),
        stats.resumed
    );
}

#[tokio::test]
async fn fully_played_reply_is_not_retained() {
    // Arrange: every frame declares zero duration, so the estimate reaches the
    // end of the reply as soon as the first frame is emitted. The barge-in
    // still stops the turn while the provider is parked.
    let (producing_tx, producing_rx) = oneshot::channel();
    let (permission_tx, permission_rx) = oneshot::channel();
    let stt = Arc::new(ScriptedStt::new(&["hello there", "okay"]));
    let llm = Arc::new(CountingLlm::new(&[REPLY_ONE, REPLY_TWO]));
    let tts = Arc::new(FrameTts::new(
        ZERO_DURATION_FRAMES,
        6,
        FirstCallMode::Park,
        Some(producing_tx),
        Some(permission_rx),
    ));
    let mut session = build_session(
        stt,
        llm.clone(),
        tts.clone(),
        SessionConfig::default(),
        Some(InterruptionRecovery::default()),
        Some(PartialResume::default()),
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
    let turn_one = session.take_events();
    let second = session.run_turn(utterance()).await.expect("turn 2 ends");

    // Assert: the turn still unwinds as an interruption, but with the estimate
    // at the end of the reply nothing is retained.
    assert!(first.interrupted, "the barge-in still stops the turn");
    assert_eq!(first.reply, REPLY_ONE, "the complete reply was produced");
    assert_eq!(
        pcm(&audio_out(&turn_one)),
        expected_pcm(1..=6),
        "all six frames were emitted before the turn stopped"
    );
    assert_eq!(
        session.recovery_stats().retained,
        0,
        "a fully estimated-heard reply is not retained: played == frames.len()"
    );
    assert_eq!(
        session.recovery_stats(),
        RecoveryStats::default(),
        "nothing is retained, resumed or discarded"
    );

    // Assert: the next backchannel turn is ordinary and replays no audio.
    assert!(!second.interrupted, "turn 2 completes");
    assert_eq!(second.reply, REPLY_TWO, "turn 2 replies with a fresh reply");
    assert_eq!(llm.calls(), 2, "the backchannel turn calls the llm");
    assert_eq!(
        llm.prompts(),
        vec!["hello there".to_string(), "okay".to_string()],
        "turn 2 used its final transcript"
    );
    assert_eq!(
        tts.synthesized(),
        vec![REPLY_ONE.to_string(), REPLY_TWO.to_string()],
        "no retained audio is replayed"
    );
    println!(
        "turn 1 pcm={:?} retained={} resumed={} llm_calls={} tts_texts={:?}",
        pcm(&audio_out(&turn_one)),
        session.recovery_stats().retained,
        session.recovery_stats().resumed,
        llm.calls(),
        tts.synthesized()
    );
}

#[tokio::test]
async fn overflow_falls_back_to_text_replay() {
    // Arrange: six frames are produced but only two fit the retention cap, so
    // the buffer overflows and the reply keeps only its text.
    let (producing_tx, producing_rx) = oneshot::channel();
    let (permission_tx, permission_rx) = oneshot::channel();
    let stt = Arc::new(ScriptedStt::new(&["hello there", "okay"]));
    let llm = Arc::new(CountingLlm::new(&[REPLY_ONE, REPLY_TWO]));
    let tts = Arc::new(FrameTts::new(
        LONG_FRAMES,
        6,
        FirstCallMode::Park,
        Some(producing_tx),
        Some(permission_rx),
    ));
    let mut session = build_session(
        stt,
        llm.clone(),
        tts.clone(),
        SessionConfig::default(),
        Some(InterruptionRecovery::default()),
        Some(PartialResume {
            max_retained_frames: 2,
        }),
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
    let turn_one = session.take_events();
    let second = session
        .run_turn(utterance())
        .await
        .expect("the replay turn completes");
    let turn_two = session.take_events();

    // Assert: the over-long reply was not retained as audio, only as text.
    assert!(first.interrupted, "turn 1 is interrupted mid-synthesis");
    assert_eq!(
        pcm(&audio_out(&turn_one)),
        expected_pcm(1..=6),
        "turn 1 emitted every produced frame before the barge-in"
    );
    assert_eq!(
        session.recovery_stats().retained,
        1,
        "the text is still retained"
    );

    // Assert: turn 2 replays the text through TTS and re-emits every frame.
    assert!(!second.interrupted, "the replay turn completes");
    assert_eq!(
        second.reply, REPLY_ONE,
        "turn 2 replies with the retained text"
    );
    assert_eq!(llm.calls(), 1, "the text replay does not call the llm");
    assert_eq!(
        tts.synthesized(),
        vec![REPLY_ONE.to_string(), REPLY_ONE.to_string()],
        "the overflowed reply is re-synthesized from its full text"
    );
    assert_eq!(
        pcm(&audio_out(&turn_two)),
        expected_pcm(1..=6),
        "turn 2 re-emits all six synthesized frames"
    );
    assert_eq!(
        session.recovery_stats().resumed,
        1,
        "the text replay counts as resumed"
    );
    println!(
        "turn 1 pcm={:?} turn 2 pcm={:?} llm_calls={} tts_calls={} tts_texts={:?} resumed={}",
        pcm(&audio_out(&turn_one)),
        pcm(&audio_out(&turn_two)),
        llm.calls(),
        tts.calls(),
        tts.synthesized(),
        session.recovery_stats().resumed
    );
}

#[tokio::test]
async fn inert_without_recovery() {
    // Arrange: partial resume is installed alone. Without the recovery
    // opt-in it is inert: no retention, no resume, no regression.
    let (producing_tx, producing_rx) = oneshot::channel();
    let (permission_tx, permission_rx) = oneshot::channel();
    let stt = Arc::new(ScriptedStt::new(&["hello there", "okay"]));
    let llm = Arc::new(CountingLlm::new(&[REPLY_ONE, REPLY_TWO]));
    let tts = Arc::new(FrameTts::new(
        LONG_FRAMES,
        6,
        FirstCallMode::Park,
        Some(producing_tx),
        Some(permission_rx),
    ));
    let mut session = build_session(
        stt,
        llm.clone(),
        tts.clone(),
        SessionConfig::default(),
        None,
        Some(PartialResume::default()),
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
    let _turn_one = session.take_events();
    let second = session.run_turn(utterance()).await.expect("turn 2 ends");

    assert!(first.interrupted, "turn 1 is interrupted as before");
    assert_eq!(
        session.recovery_stats(),
        RecoveryStats::default(),
        "partial resume alone retains nothing"
    );
    assert_eq!(
        second.reply, REPLY_TWO,
        "the backchannel turn calls the llm"
    );
    assert_eq!(llm.calls(), 2, "the llm is called without recovery");
    assert_eq!(
        tts.synthesized(),
        vec![REPLY_ONE.to_string(), REPLY_TWO.to_string()],
        "no audio is replayed"
    );
    println!(
        "retained={} resumed={} discarded={} llm_calls={} tts_texts={:?}",
        session.recovery_stats().retained,
        session.recovery_stats().resumed,
        session.recovery_stats().discarded,
        llm.calls(),
        tts.synthesized()
    );
}

#[tokio::test]
async fn emission_matches_the_policy_off_path() {
    // Two identical sessions — one with recovery and partial resume installed
    // and one with neither — run one ordinary turn at the default playout-ring
    // capacity. The opted-in path must emit at exactly the same cadence: same
    // events, same provider inputs, same report.
    let enabled_llm = Arc::new(CountingLlm::new(&[REPLY_ONE]));
    let enabled_tts = Arc::new(FrameTts::new(
        LONG_FRAMES,
        6,
        FirstCallMode::End,
        None,
        None,
    ));
    let mut enabled = build_session(
        Arc::new(ScriptedStt::new(&["hello there"])),
        enabled_llm.clone(),
        enabled_tts.clone(),
        SessionConfig::default(),
        Some(InterruptionRecovery::default()),
        Some(PartialResume::default()),
    );
    let plain_llm = Arc::new(CountingLlm::new(&[REPLY_ONE]));
    let plain_tts = Arc::new(FrameTts::new(
        LONG_FRAMES,
        6,
        FirstCallMode::End,
        None,
        None,
    ));
    let mut plain = build_session(
        Arc::new(ScriptedStt::new(&["hello there"])),
        plain_llm.clone(),
        plain_tts.clone(),
        SessionConfig::default(),
        None,
        None,
    );

    let enabled_report = enabled.run_turn(utterance()).await.expect("turn ends");
    let plain_report = plain.run_turn(utterance()).await.expect("turn ends");

    assert!(!enabled_report.interrupted);
    assert_eq!(
        enabled_report, plain_report,
        "the same report is produced with and without partial resume"
    );
    assert_eq!(
        comparable(enabled.take_events()),
        comparable(plain.take_events()),
        "the event sequences are identical"
    );
    assert_eq!(
        enabled.recovery_stats(),
        RecoveryStats::default(),
        "an uninterrupted turn retains and resumes nothing"
    );
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
    println!(
        "enabled_frames={} plain_frames={} llm_prompts={:?} tts_texts={:?}",
        enabled_report.audio_frames,
        plain_report.audio_frames,
        enabled_llm.prompts(),
        enabled_tts.synthesized()
    );
}

#[test]
fn playback_cursor_unit_cases() {
    let ms = DurationMs::from_millis;
    let durations = |values: &[u64]| values.iter().copied().map(ms).collect::<Vec<_>>();

    // An empty reply has no frame to hear.
    assert_eq!(playback_cursor(durations(&[]).into_iter(), ms(10)), 0);

    // Zero elapsed hears no positive-duration frame.
    assert_eq!(
        playback_cursor(durations(&[10, 10, 10]).into_iter(), ms(0)),
        0
    );

    // An exact prefix-sum boundary includes the frame whose duration completes
    // it; one millisecond less excludes it.
    assert_eq!(
        playback_cursor(durations(&[10, 10, 10]).into_iter(), ms(10)),
        1,
        "elapsed == the first frame's duration includes it"
    );
    assert_eq!(
        playback_cursor(durations(&[10, 10, 10]).into_iter(), ms(20)),
        2,
        "elapsed == the first two frames' sum includes both"
    );
    assert_eq!(
        playback_cursor(durations(&[10, 10, 10]).into_iter(), ms(19)),
        1,
        "one millisecond before the boundary excludes the second frame"
    );

    // Elapsed between frame boundaries counts only the completed prefix.
    assert_eq!(
        playback_cursor(durations(&[10, 10, 10]).into_iter(), ms(25)),
        2,
        "the third frame's declared duration has not elapsed"
    );

    // Elapsed beyond the total is bounded by the iterator length.
    assert_eq!(
        playback_cursor(durations(&[10, 10, 10]).into_iter(), ms(u64::MAX)),
        3,
        "the cursor never exceeds the number of frames"
    );

    // Zero-duration frames are all heard at zero elapsed: the sample_rate 0 law.
    assert_eq!(
        playback_cursor(durations(&[0, 0, 0]).into_iter(), ms(0)),
        3,
        "zero-duration frames count immediately"
    );

    // Saturating arithmetic: `u64::MAX` durations neither panic nor exceed the
    // iterator length.
    assert_eq!(
        playback_cursor(durations(&[u64::MAX, 1]).into_iter(), ms(u64::MAX)),
        2,
        "saturating sums never wrap and stay bounded by the frame count"
    );
}

#[tokio::test]
async fn truncated_synthesis_resumes_only_produced_frames() {
    // Arrange: the LLM finishes the full reply text, but synthesis is cut short
    // after four of the reply's frames exist. The reply's remaining audio is
    // never produced, so it can never be retained; the honest outcome is a
    // resume of the produced prefix only, even though the reported reply text
    // is the complete LLM reply.
    let (producing_tx, producing_rx) = oneshot::channel();
    let (permission_tx, permission_rx) = oneshot::channel();
    let stt = Arc::new(ScriptedStt::new(&["hello there", "okay"]));
    let llm = Arc::new(CountingLlm::new(&[REPLY_ONE, REPLY_TWO]));
    let tts = Arc::new(FrameTts::new(
        LONG_FRAMES,
        4,
        FirstCallMode::Park,
        Some(producing_tx),
        Some(permission_rx),
    ));
    let mut session = build_session(
        stt,
        llm.clone(),
        tts.clone(),
        SessionConfig::default(),
        Some(InterruptionRecovery::default()),
        Some(PartialResume::default()),
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
    let turn_one = session.take_events();
    let second = session
        .run_turn(utterance())
        .await
        .expect("the truncated resume completes");
    let turn_two = session.take_events();

    // Assert: the reply text is complete, all produced frames were emitted, and
    // the produced prefix (four frames) is retained for a zero estimate.
    assert!(first.interrupted, "turn 1 is interrupted mid-synthesis");
    assert_eq!(
        first.reply, REPLY_ONE,
        "the llm finished the full reply text before synthesis stopped"
    );
    assert_eq!(
        pcm(&audio_out(&turn_one)),
        expected_pcm(1..=4),
        "turn 1 emitted only the four produced frames"
    );
    assert_eq!(
        session.recovery_stats().retained,
        1,
        "the produced prefix is retained"
    );

    // Assert: turn 2 resumes the produced prefix and stops at its end. The
    // unproduced remainder of the intended audio never existed and is not
    // fabricated.
    assert!(!second.interrupted, "the resume turn completes");
    assert_eq!(
        second.reply, REPLY_ONE,
        "the reply text stays the full llm reply"
    );
    assert_eq!(
        second.audio_frames, 4,
        "the resumed turn emits only the produced prefix"
    );
    assert_eq!(
        pcm(&audio_out(&turn_two)),
        expected_pcm(1..=4),
        "the resume emits frames 1-4 and stops: the rest was never synthesized"
    );
    assert_eq!(llm.calls(), 1, "the llm is not called for the resume");
    assert_eq!(tts.calls(), 1, "the tts is not called for the resume");
    assert!(
        !turn_two
            .iter()
            .any(|event| matches!(event, SessionEvent::ReplyDelta { .. })),
        "the resumed turn emits no ReplyDelta"
    );
    println!(
        "turn 1 pcm={:?} turn 2 pcm={:?} reply_chars={} llm_calls={} tts_calls={} retained={} resumed={}",
        pcm(&audio_out(&turn_one)),
        pcm(&audio_out(&turn_two)),
        second.reply.chars().count(),
        llm.calls(),
        tts.calls(),
        session.recovery_stats().retained,
        session.recovery_stats().resumed
    );
}
