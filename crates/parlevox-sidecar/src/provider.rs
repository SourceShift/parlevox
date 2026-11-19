//! Adapters that make a sidecar plugin indistinguishable from an in-process
//! provider.
//!
//! The runtime holds `Arc<dyn TtsProvider>` etc., so the turn path does not
//! know whether a provider lives in this process or behind a pipe.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::stream;
use parlevox_core::{AudioFrame, ErrorClass, ParlevoxError, Timestamp};
use parlevox_ipc::{ErrorClassWire, PluginEvent, PluginRequest};
use parlevox_providers::{
    AudioChunk, GenerateRequest, LlmProvider, Modality, ProviderManifest, ProviderStream,
    SttProvider, SynthesizeRequest, TranscriptChunk, TtsProvider,
};
use tokio_util::sync::CancellationToken;

use crate::client::{Incoming, SidecarClient};

fn classify(class: ErrorClassWire, message: String) -> ParlevoxError {
    match class {
        ErrorClassWire::Transient => ParlevoxError::Provider {
            provider: "sidecar".into(),
            message,
            class: ErrorClass::Transient,
            retry_after: None,
        },
        ErrorClassWire::Fatal => ParlevoxError::Provider {
            provider: "sidecar".into(),
            message,
            class: ErrorClass::Fatal,
            retry_after: None,
        },
        ErrorClassWire::Protocol => ParlevoxError::Protocol(message),
        ErrorClassWire::Cancelled => ParlevoxError::Cancelled,
        ErrorClassWire::Capacity => ParlevoxError::Capacity(message),
    }
}

fn adapter_manifest(
    client: &SidecarClient,
    modality: Modality,
    fallback_name: &str,
) -> ProviderManifest {
    if let Some(manifest) = client.plugin_manifest() {
        if manifest.modality == modality {
            return manifest.clone();
        }
    }
    ProviderManifest::new(fallback_name, modality)
        .with_capability("streaming")
        .with_capability("cancel")
}

/// LLM provider backed by a plugin `generate` request.
pub struct SidecarLlm {
    client: Arc<SidecarClient>,
    manifest: ProviderManifest,
    /// Frames the plugin reported dropping on the last terminal `done`.
    dropped_frames: Arc<AtomicU64>,
}

impl SidecarLlm {
    pub fn new(client: Arc<SidecarClient>) -> Self {
        let manifest = adapter_manifest(&client, Modality::Llm, "sidecar-llm");
        SidecarLlm {
            client,
            manifest,
            dropped_frames: Arc::new(AtomicU64::new(0)),
        }
    }
}

#[async_trait]
impl LlmProvider for SidecarLlm {
    fn manifest(&self) -> &ProviderManifest {
        &self.manifest
    }

    fn take_sidecar_dropped_frames(&self) -> u64 {
        self.dropped_frames.swap(0, Ordering::SeqCst)
    }

    async fn generate(
        &self,
        request: GenerateRequest,
        cancel: CancellationToken,
    ) -> Result<ProviderStream<String>, ParlevoxError> {
        if request.prompt.trim().is_empty() {
            return Err(ParlevoxError::Protocol("prompt must not be empty".into()));
        }
        let request_id = self.client.new_request_id("llm");
        let handle = self
            .client
            .open_request(
                PluginRequest::Generate {
                    request_id,
                    prompt: request.prompt,
                },
                Vec::new(),
                Some(cancel),
            )
            .await?;
        let dropped = Arc::clone(&self.dropped_frames);
        let stream = stream::unfold(Some(handle), move |state| {
            let dropped = Arc::clone(&dropped);
            async move {
                let mut handle = state?;
                loop {
                    match handle.recv().await {
                        None => {
                            if handle.is_cancelled() {
                                return Some((Err(ParlevoxError::Cancelled), None));
                            }
                            return None;
                        }
                        Some(Incoming::Event(PluginEvent::Delta { text, .. })) => {
                            return Some((Ok(text), Some(handle)));
                        }
                        Some(Incoming::Event(PluginEvent::Done { dropped_frames, .. })) => {
                            dropped.store(dropped_frames, Ordering::SeqCst);
                            return None;
                        }
                        Some(Incoming::Event(PluginEvent::Error { class, message, .. })) => {
                            return Some((Err(classify(class, message)), None))
                        }
                        Some(Incoming::Event(_)) | Some(Incoming::Audio(_)) => continue,
                    }
                }
            }
        });
        Ok(Box::pin(stream))
    }
}

/// TTS provider backed by a plugin `synthesize` request. Audio arrives as
/// binary frames; nothing is base64-encoded or buffered whole.
pub struct SidecarTts {
    client: Arc<SidecarClient>,
    manifest: ProviderManifest,
    /// Frames the plugin reported dropping on the last terminal `done`.
    dropped_frames: Arc<AtomicU64>,
}

impl SidecarTts {
    pub fn new(client: Arc<SidecarClient>) -> Self {
        let manifest = adapter_manifest(&client, Modality::Tts, "sidecar-tts");
        SidecarTts {
            client,
            manifest,
            dropped_frames: Arc::new(AtomicU64::new(0)),
        }
    }
}

#[async_trait]
impl TtsProvider for SidecarTts {
    fn manifest(&self) -> &ProviderManifest {
        &self.manifest
    }

    fn take_sidecar_dropped_frames(&self) -> u64 {
        self.dropped_frames.swap(0, Ordering::SeqCst)
    }

    async fn synthesize(
        &self,
        request: SynthesizeRequest,
        cancel: CancellationToken,
    ) -> Result<ProviderStream<AudioChunk>, ParlevoxError> {
        if request.text.trim().is_empty() {
            return Err(ParlevoxError::Protocol("text must not be empty".into()));
        }
        let request_id = self.client.new_request_id("tts");
        let handle = self
            .client
            .open_request(
                PluginRequest::Synthesize {
                    request_id,
                    text: request.text,
                    sample_rate: request.sample_rate,
                    channels: request.channels,
                },
                Vec::new(),
                Some(cancel),
            )
            .await?;
        let dropped = Arc::clone(&self.dropped_frames);
        let stream = stream::unfold(Some(handle), move |state| {
            let dropped = Arc::clone(&dropped);
            async move {
                let mut handle = state?;
                loop {
                    match handle.recv().await {
                        None => {
                            if handle.is_cancelled() {
                                return Some((Err(ParlevoxError::Cancelled), None));
                            }
                            return None;
                        }
                        Some(Incoming::Audio(frame)) => {
                            return Some((Ok(AudioChunk::from_frame(&frame)), Some(handle)));
                        }
                        Some(Incoming::Event(PluginEvent::Done { dropped_frames, .. })) => {
                            dropped.store(dropped_frames, Ordering::SeqCst);
                            return None;
                        }
                        Some(Incoming::Event(PluginEvent::Error { class, message, .. })) => {
                            return Some((Err(classify(class, message)), None))
                        }
                        Some(Incoming::Event(_)) => continue,
                    }
                }
            }
        });
        Ok(Box::pin(stream))
    }
}

/// Duration of each audio frame the STT adapter streams to the plugin. The
/// plugin's per-request buffer is bounded to `MAX_PENDING_AUDIO_FRAMES`; sending
/// the utterance in frames (rather than one giant frame) makes that bound
/// reachable and its drop telemetry meaningful.
const STT_FRAME_MS: u64 = 10;

/// STT provider backed by a plugin `transcribe` request.
pub struct SidecarStt {
    client: Arc<SidecarClient>,
    manifest: ProviderManifest,
    /// Frames the plugin reported dropping on the last terminal `done`.
    dropped_frames: Arc<AtomicU64>,
}

impl SidecarStt {
    pub fn new(client: Arc<SidecarClient>) -> Self {
        let manifest = adapter_manifest(&client, Modality::Stt, "sidecar-stt");
        SidecarStt {
            client,
            manifest,
            dropped_frames: Arc::new(AtomicU64::new(0)),
        }
    }
}

#[async_trait]
impl SttProvider for SidecarStt {
    fn manifest(&self) -> &ProviderManifest {
        &self.manifest
    }

    fn take_sidecar_dropped_frames(&self) -> u64 {
        self.dropped_frames.swap(0, Ordering::SeqCst)
    }

    async fn transcribe(
        &self,
        audio: AudioChunk,
        cancel: CancellationToken,
    ) -> Result<ProviderStream<TranscriptChunk>, ParlevoxError> {
        if audio.is_empty() {
            return Err(ParlevoxError::Protocol("audio must not be empty".into()));
        }
        let request_id = self.client.new_request_id("stt");
        // Interleaved samples per 10 ms frame; fall back to one frame for a
        // degenerate sample rate or channel count. Conversion failures and
        // multiplication overflow saturate rather than wrap.
        let samples_per_channel =
            usize::try_from(u64::from(audio.sample_rate) * STT_FRAME_MS / 1000)
                .unwrap_or(usize::MAX);
        let samples_per_frame =
            samples_per_channel.saturating_mul(usize::from(audio.channels.max(1)));
        let frames: Vec<AudioFrame> = if samples_per_frame == 0 {
            vec![AudioFrame::new(
                Timestamp::from_millis(0),
                audio.sample_rate,
                audio.channels,
                audio.pcm,
            )]
        } else {
            audio
                .pcm
                .chunks(samples_per_frame)
                .enumerate()
                .map(|(index, pcm)| {
                    let timestamp_ms = u64::try_from(index)
                        .unwrap_or(u64::MAX)
                        .saturating_mul(STT_FRAME_MS);
                    AudioFrame::new(
                        Timestamp::from_millis(timestamp_ms),
                        audio.sample_rate,
                        audio.channels,
                        pcm.to_vec(),
                    )
                })
                .collect()
        };
        let handle = self
            .client
            .open_request(
                PluginRequest::Transcribe {
                    request_id,
                    sample_rate: audio.sample_rate,
                    channels: audio.channels,
                },
                frames,
                Some(cancel),
            )
            .await?;
        let dropped = Arc::clone(&self.dropped_frames);
        let stream = stream::unfold(Some(handle), move |state| {
            let dropped = Arc::clone(&dropped);
            async move {
                let mut handle = state?;
                loop {
                    match handle.recv().await {
                        None => {
                            if handle.is_cancelled() {
                                return Some((Err(ParlevoxError::Cancelled), None));
                            }
                            return None;
                        }
                        Some(Incoming::Event(PluginEvent::Partial { text, .. })) => {
                            return Some((
                                Ok(TranscriptChunk {
                                    text,
                                    is_final: false,
                                }),
                                Some(handle),
                            ));
                        }
                        Some(Incoming::Event(PluginEvent::Final { text, .. })) => {
                            return Some((
                                Ok(TranscriptChunk {
                                    text,
                                    is_final: true,
                                }),
                                Some(handle),
                            ));
                        }
                        Some(Incoming::Event(PluginEvent::Done { dropped_frames, .. })) => {
                            dropped.store(dropped_frames, Ordering::SeqCst);
                            return None;
                        }
                        Some(Incoming::Event(PluginEvent::Error { class, message, .. })) => {
                            return Some((Err(classify(class, message)), None))
                        }
                        Some(Incoming::Event(_)) | Some(Incoming::Audio(_)) => continue,
                    }
                }
            }
        });
        Ok(Box::pin(stream))
    }
}
