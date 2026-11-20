//! Deterministic in-process fakes for all four modalities.
//!
//! Fixed transcripts and PCM mean CLI demos and tests assert exact values
//! instead of racing timers. The fakes never touch the network, the
//! filesystem, or a Python interpreter.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use parlevox_core::ParlevoxError;
use tokio_util::sync::CancellationToken;

use crate::manifest::{Modality, ProviderManifest};
use crate::stream::cancelable_stream;
use crate::traits::{
    AudioChunk, GenerateRequest, LlmProvider, ProviderStream, SttProvider, SynthesizeRequest,
    TranscriptChunk, TtsProvider, VadEvent, VadProvider, VadTransition,
};

/// The transcript every [`FakeStt`] commits.
pub const FAKE_TRANSCRIPT: &str = "hello world";
/// The partial [`FakeStt`] emits before its final.
pub const FAKE_PARTIAL_TRANSCRIPT: &str = "hello";

/// The deltas every [`FakeLlm`] streams, in order. Concatenation is
/// [`FAKE_REPLY`].
pub const FAKE_LLM_DELTAS: [&str; 4] = ["Hello", ", ", "world", "!"];
/// Concatenation of [`FAKE_LLM_DELTAS`].
pub const FAKE_REPLY: &str = "Hello, world!";

/// Sample rate used by the fake TTS and fake VAD.
pub const FAKE_SAMPLE_RATE: u32 = 16_000;
/// Channels used by the fake TTS and fake VAD.
pub const FAKE_CHANNELS: u16 = 1;
/// Duration of each fake TTS audio frame.
pub const FAKE_FRAME_MS: u64 = 10;
/// Number of audio frames the fake TTS emits for non-empty text.
pub const FAKE_TTS_FRAMES: usize = 2;
/// Samples per channel in each fake TTS frame.
pub const FAKE_FRAME_SAMPLES: usize = (FAKE_SAMPLE_RATE as u64 * FAKE_FRAME_MS / 1000) as usize;

fn require_non_empty(field: &str, value: &str) -> Result<(), ParlevoxError> {
    if value.trim().is_empty() {
        return Err(ParlevoxError::Protocol(format!(
            "{field} must not be empty"
        )));
    }
    Ok(())
}

fn require_pcm(pcm: &[i16]) -> Result<(), ParlevoxError> {
    if pcm.is_empty() {
        return Err(ParlevoxError::Protocol("audio must not be empty".into()));
    }
    Ok(())
}

/// Deterministic PCM: a bounded sawtooth, identical on every run.
///
/// `(base * 7) % 201` repeats every 201 samples, so the running index is
/// reduced modulo the period before it is widened. That keeps the arithmetic
/// in `i32` (and the result in `i16`) for any `frame_index`, including values
/// large enough that `frame_index * FAKE_FRAME_SAMPLES` would overflow
/// `usize`.
fn fake_pcm(frame_index: usize) -> Vec<i16> {
    (0..FAKE_FRAME_SAMPLES)
        .map(|i| {
            let phase = (frame_index % 201) * (FAKE_FRAME_SAMPLES % 201) % 201;
            let base = i32::try_from((i + phase) % 201).unwrap_or(i32::MAX);
            let value = (base * 7) % 201 - 100;
            i16::try_from(value).unwrap_or(0)
        })
        .collect()
}

#[derive(Debug, Clone)]
pub struct FakeStt {
    manifest: ProviderManifest,
}

impl Default for FakeStt {
    fn default() -> Self {
        Self::new()
    }
}

impl FakeStt {
    pub fn new() -> Self {
        FakeStt {
            manifest: ProviderManifest::new("fake-stt", Modality::Stt)
                .with_capability("streaming")
                .with_capability("partials")
                .with_capability("cancel"),
        }
    }
}

#[async_trait]
impl SttProvider for FakeStt {
    fn manifest(&self) -> &ProviderManifest {
        &self.manifest
    }

    async fn transcribe(
        &self,
        audio: AudioChunk,
        cancel: CancellationToken,
    ) -> Result<ProviderStream<TranscriptChunk>, ParlevoxError> {
        require_pcm(&audio.pcm)?;
        let items = vec![
            Ok(TranscriptChunk {
                text: FAKE_PARTIAL_TRANSCRIPT.to_string(),
                is_final: false,
            }),
            Ok(TranscriptChunk {
                text: FAKE_TRANSCRIPT.to_string(),
                is_final: true,
            }),
        ];
        Ok(cancelable_stream(items, cancel))
    }
}

#[derive(Debug, Clone)]
pub struct FakeLlm {
    manifest: ProviderManifest,
    /// The last request this fake served, for assertions. `Arc` so clones of
    /// the fake (the runtime holds one behind `Arc<dyn LlmProvider>`) share the
    /// same cell. The lock is taken and released without an `await` in between.
    last_request: Arc<Mutex<Option<GenerateRequest>>>,
}

impl Default for FakeLlm {
    fn default() -> Self {
        Self::new()
    }
}

impl FakeLlm {
    pub fn new() -> Self {
        FakeLlm {
            manifest: ProviderManifest::new("fake-llm", Modality::Llm)
                .with_capability("streaming")
                .with_capability("cancel"),
            last_request: Arc::new(Mutex::new(None)),
        }
    }

    /// The most recent request this fake served, if any.
    ///
    /// Tests use this to assert what history the runtime passed without
    /// changing the fake's deterministic delta output.
    pub fn last_request(&self) -> Option<GenerateRequest> {
        self.last_request
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

#[async_trait]
impl LlmProvider for FakeLlm {
    fn manifest(&self) -> &ProviderManifest {
        &self.manifest
    }

    async fn generate(
        &self,
        request: GenerateRequest,
        cancel: CancellationToken,
    ) -> Result<ProviderStream<String>, ParlevoxError> {
        require_non_empty("prompt", &request.prompt)?;
        {
            let mut slot = self
                .last_request
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *slot = Some(request.clone());
        }
        let items = FAKE_LLM_DELTAS
            .iter()
            .map(|delta| Ok((*delta).to_string()))
            .collect();
        Ok(cancelable_stream(items, cancel))
    }
}

#[derive(Debug, Clone)]
pub struct FakeTts {
    manifest: ProviderManifest,
}

impl Default for FakeTts {
    fn default() -> Self {
        Self::new()
    }
}

impl FakeTts {
    pub fn new() -> Self {
        FakeTts {
            manifest: ProviderManifest::new("fake-tts", Modality::Tts)
                .with_capability("streaming")
                .with_capability("binary_pcm")
                .with_capability("cancel"),
        }
    }
}

#[async_trait]
impl TtsProvider for FakeTts {
    fn manifest(&self) -> &ProviderManifest {
        &self.manifest
    }

    async fn synthesize(
        &self,
        request: SynthesizeRequest,
        cancel: CancellationToken,
    ) -> Result<ProviderStream<AudioChunk>, ParlevoxError> {
        require_non_empty("text", &request.text)?;
        if request.sample_rate == 0 || request.channels == 0 {
            return Err(ParlevoxError::Protocol(
                "sample_rate and channels must be non-zero".into(),
            ));
        }
        let items = (0..FAKE_TTS_FRAMES)
            .map(|frame_index| {
                Ok(AudioChunk::new(
                    request.sample_rate,
                    request.channels,
                    fake_pcm(frame_index),
                ))
            })
            .collect();
        Ok(cancelable_stream(items, cancel))
    }
}

#[derive(Debug, Clone)]
pub struct FakeVad {
    manifest: ProviderManifest,
}

impl Default for FakeVad {
    fn default() -> Self {
        Self::new()
    }
}

impl FakeVad {
    pub fn new() -> Self {
        FakeVad {
            manifest: ProviderManifest::new("fake-vad", Modality::Vad)
                .with_capability("streaming")
                .with_capability("cancel"),
        }
    }
}

#[async_trait]
impl VadProvider for FakeVad {
    fn manifest(&self) -> &ProviderManifest {
        &self.manifest
    }

    async fn analyze(
        &self,
        audio: AudioChunk,
        cancel: CancellationToken,
    ) -> Result<ProviderStream<VadTransition>, ParlevoxError> {
        require_pcm(&audio.pcm)?;
        let samples_per_channel = if audio.channels == 0 {
            0
        } else {
            audio.pcm.len() / audio.channels as usize
        };
        let duration_ms = if audio.sample_rate == 0 {
            0
        } else {
            (samples_per_channel as u64 * 1000) / audio.sample_rate as u64
        };
        let items = vec![
            Ok(VadTransition {
                event: VadEvent::SpeechStart,
                timestamp_ms: 0,
            }),
            Ok(VadTransition {
                event: VadEvent::SpeechEnd,
                timestamp_ms: duration_ms,
            }),
        ];
        Ok(cancelable_stream(items, cancel))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;

    async fn collect<T>(mut stream: ProviderStream<T>) -> Vec<T> {
        let mut out = Vec::new();
        while let Some(item) = stream.next().await {
            out.push(item.unwrap());
        }
        out
    }

    #[tokio::test]
    async fn fake_stt_streams_partial_then_final() {
        let stt = FakeStt::new();
        let audio = AudioChunk::new(FAKE_SAMPLE_RATE, FAKE_CHANNELS, vec![0; 160]);
        let chunks = collect(
            stt.transcribe(audio, CancellationToken::new())
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].text, FAKE_PARTIAL_TRANSCRIPT);
        assert!(!chunks[0].is_final);
        assert_eq!(chunks[1].text, FAKE_TRANSCRIPT);
        assert!(chunks[1].is_final);
    }

    #[tokio::test]
    async fn fake_llm_deltas_concatenate_to_reply() {
        let llm = FakeLlm::new();
        let deltas = collect(
            llm.generate(
                GenerateRequest::new(FAKE_TRANSCRIPT),
                CancellationToken::new(),
            )
            .await
            .unwrap(),
        )
        .await;
        assert_eq!(deltas, FAKE_LLM_DELTAS.to_vec());
        assert_eq!(deltas.concat(), FAKE_REPLY);
        assert_eq!(
            llm.last_request(),
            Some(GenerateRequest::new(FAKE_TRANSCRIPT))
        );
    }

    #[tokio::test]
    async fn fake_llm_records_history_for_assertions() {
        let llm = FakeLlm::new();
        let request = GenerateRequest {
            prompt: "next".into(),
            history: vec![crate::ChatMessage {
                role: crate::Role::User,
                text: "prior".into(),
            }],
        };
        // Drain the stream so the request is recorded exactly as sent.
        let deltas = collect(
            llm.generate(request.clone(), CancellationToken::new())
                .await
                .unwrap(),
        )
        .await;
        assert_eq!(deltas, FAKE_LLM_DELTAS.to_vec());
        assert_eq!(llm.last_request(), Some(request));
    }

    #[test]
    fn fake_pcm_stays_in_range_for_large_frame_indices() {
        for frame_index in [0, 1, 200, 201, usize::MAX / 2, usize::MAX] {
            let pcm = fake_pcm(frame_index);
            assert_eq!(pcm.len(), FAKE_FRAME_SAMPLES);
            assert!(pcm.iter().all(|sample| (-100..=100).contains(sample)));
        }
        // The waveform is periodic with period 201 frames' worth of samples.
        assert_eq!(fake_pcm(0), fake_pcm(201));
    }

    #[test]
    fn fake_pcm_reduction_preserves_the_original_samples() {
        for frame_index in 0..4 {
            let pcm = fake_pcm(frame_index);
            for (i, sample) in pcm.iter().enumerate() {
                let base = i + frame_index * FAKE_FRAME_SAMPLES;
                let expected = i32::try_from(base).unwrap() * 7 % 201 - 100;
                assert_eq!(*sample, i16::try_from(expected).unwrap());
            }
        }
    }

    #[tokio::test]
    async fn fake_tts_pcm_is_stable_across_runs() {
        let tts = FakeTts::new();
        let req = SynthesizeRequest {
            text: FAKE_REPLY.into(),
            sample_rate: FAKE_SAMPLE_RATE,
            channels: FAKE_CHANNELS,
        };
        let first = collect(
            tts.synthesize(req.clone(), CancellationToken::new())
                .await
                .unwrap(),
        )
        .await;
        let second = collect(tts.synthesize(req, CancellationToken::new()).await.unwrap()).await;
        assert_eq!(first, second);
        assert_eq!(first.len(), FAKE_TTS_FRAMES);
        assert!(first
            .iter()
            .all(|chunk| chunk.pcm.len() == FAKE_FRAME_SAMPLES));
        assert_ne!(first[0].pcm, first[1].pcm);
    }

    #[tokio::test]
    async fn fakes_reject_empty_input_as_protocol_errors() {
        let stt = FakeStt::new();
        let err = match stt
            .transcribe(
                AudioChunk::new(FAKE_SAMPLE_RATE, FAKE_CHANNELS, vec![]),
                CancellationToken::new(),
            )
            .await
        {
            Err(err) => err,
            Ok(_) => panic!("expected empty audio to be rejected"),
        };
        assert!(matches!(err, ParlevoxError::Protocol(_)));

        let tts = FakeTts::new();
        let err = match tts
            .synthesize(
                SynthesizeRequest {
                    text: "   ".into(),
                    sample_rate: FAKE_SAMPLE_RATE,
                    channels: FAKE_CHANNELS,
                },
                CancellationToken::new(),
            )
            .await
        {
            Err(err) => err,
            Ok(_) => panic!("expected empty text to be rejected"),
        };
        assert!(matches!(err, ParlevoxError::Protocol(_)));
    }

    #[tokio::test]
    async fn fakes_stop_when_cancelled() {
        let llm = FakeLlm::new();
        let cancel = CancellationToken::new();
        let mut stream = llm
            .generate(GenerateRequest::new("hi"), cancel.clone())
            .await
            .unwrap();
        assert_eq!(stream.next().await.unwrap().unwrap(), FAKE_LLM_DELTAS[0]);
        cancel.cancel();
        assert!(matches!(
            stream.next().await.unwrap().unwrap_err(),
            ParlevoxError::Cancelled
        ));
        assert!(stream.next().await.is_none());
    }
}
