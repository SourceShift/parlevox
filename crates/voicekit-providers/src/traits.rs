//! Streaming provider traits.
//!
//! The shape is deliberately the same for every modality: an `async fn` that
//! validates the request and returns a [`ProviderStream`]. Cancellation is a
//! parameter, not an afterthought, and the stream yields partial results as
//! they become available.

use std::pin::Pin;

use async_trait::async_trait;
use futures_core::Stream;
use tokio_util::sync::CancellationToken;
use voicekit_core::{AudioFrame, Timestamp, VoicekitError};

use crate::manifest::ProviderManifest;

/// A boxed, sendable stream of provider results.
pub type ProviderStream<T> = Pin<Box<dyn Stream<Item = Result<T, VoicekitError>> + Send>>;

/// Owned PCM handed to a provider. Format travels with the samples.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioChunk {
    pub sample_rate: u32,
    pub channels: u16,
    pub pcm: Vec<i16>,
}

impl AudioChunk {
    pub fn new(sample_rate: u32, channels: u16, pcm: Vec<i16>) -> Self {
        AudioChunk {
            sample_rate,
            channels,
            pcm,
        }
    }

    pub fn from_frame(frame: &AudioFrame) -> Self {
        AudioChunk {
            sample_rate: frame.sample_rate,
            channels: frame.channels,
            pcm: frame.pcm.clone(),
        }
    }

    pub fn to_frame(&self, timestamp: Timestamp) -> AudioFrame {
        AudioFrame::new(timestamp, self.sample_rate, self.channels, self.pcm.clone())
    }

    pub fn is_empty(&self) -> bool {
        self.pcm.is_empty()
    }
}

/// One streaming STT result. `is_final` marks the committed transcript.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptChunk {
    pub text: String,
    pub is_final: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerateRequest {
    pub prompt: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SynthesizeRequest {
    pub text: String,
    pub sample_rate: u32,
    pub channels: u16,
}

/// Voice-activity transition emitted by a VAD stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadEvent {
    SpeechStart,
    SpeechEnd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VadTransition {
    pub event: VadEvent,
    /// Offset from the start of the analysed audio.
    pub timestamp_ms: u64,
}

#[async_trait]
pub trait SttProvider: Send + Sync {
    fn manifest(&self) -> &ProviderManifest;

    /// Transcribe one utterance. The stream emits partials then one final.
    async fn transcribe(
        &self,
        audio: AudioChunk,
        cancel: CancellationToken,
    ) -> Result<ProviderStream<TranscriptChunk>, VoicekitError>;
}

#[async_trait]
pub trait LlmProvider: Send + Sync {
    fn manifest(&self) -> &ProviderManifest;

    /// Stream completion deltas for `request`.
    async fn generate(
        &self,
        request: GenerateRequest,
        cancel: CancellationToken,
    ) -> Result<ProviderStream<String>, VoicekitError>;
}

#[async_trait]
pub trait TtsProvider: Send + Sync {
    fn manifest(&self) -> &ProviderManifest;

    /// Stream audio chunks for `request`. Empty text is an error so a caller
    /// cannot accidentally synthesize silence for a failed turn.
    async fn synthesize(
        &self,
        request: SynthesizeRequest,
        cancel: CancellationToken,
    ) -> Result<ProviderStream<AudioChunk>, VoicekitError>;
}

#[async_trait]
pub trait VadProvider: Send + Sync {
    fn manifest(&self) -> &ProviderManifest;

    /// Stream voice-activity transitions over one audio chunk.
    async fn analyze(
        &self,
        audio: AudioChunk,
        cancel: CancellationToken,
    ) -> Result<ProviderStream<VadTransition>, VoicekitError>;
}
