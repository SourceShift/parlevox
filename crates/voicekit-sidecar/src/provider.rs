//! Adapters that make a sidecar plugin indistinguishable from an in-process
//! provider.
//!
//! The runtime holds `Arc<dyn TtsProvider>` etc., so the turn path does not
//! know whether a provider lives in this process or behind a pipe.

use std::sync::Arc;

use async_trait::async_trait;
use futures_util::stream;
use tokio_util::sync::CancellationToken;
use voicekit_core::{AudioFrame, ErrorClass, Timestamp, VoicekitError};
use voicekit_ipc::{ErrorClassWire, PluginEvent, PluginRequest};
use voicekit_providers::{
    AudioChunk, GenerateRequest, LlmProvider, Modality, ProviderManifest, ProviderStream,
    SttProvider, SynthesizeRequest, TranscriptChunk, TtsProvider,
};

use crate::client::{Incoming, SidecarClient};

fn classify(class: ErrorClassWire, message: String) -> VoicekitError {
    match class {
        ErrorClassWire::Transient => VoicekitError::Provider {
            provider: "sidecar".into(),
            message,
            class: ErrorClass::Transient,
            retry_after: None,
        },
        ErrorClassWire::Fatal => VoicekitError::Provider {
            provider: "sidecar".into(),
            message,
            class: ErrorClass::Fatal,
            retry_after: None,
        },
        ErrorClassWire::Protocol => VoicekitError::Protocol(message),
        ErrorClassWire::Cancelled => VoicekitError::Cancelled,
        ErrorClassWire::Capacity => VoicekitError::Capacity(message),
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
}

impl SidecarLlm {
    pub fn new(client: Arc<SidecarClient>) -> Self {
        let manifest = adapter_manifest(&client, Modality::Llm, "sidecar-llm");
        SidecarLlm { client, manifest }
    }
}

#[async_trait]
impl LlmProvider for SidecarLlm {
    fn manifest(&self) -> &ProviderManifest {
        &self.manifest
    }

    async fn generate(
        &self,
        request: GenerateRequest,
        cancel: CancellationToken,
    ) -> Result<ProviderStream<String>, VoicekitError> {
        if request.prompt.trim().is_empty() {
            return Err(VoicekitError::Protocol("prompt must not be empty".into()));
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
        let stream = stream::unfold(Some(handle), |state| async move {
            let mut handle = state?;
            loop {
                match handle.recv().await {
                    None => {
                        if handle.is_cancelled() {
                            return Some((Err(VoicekitError::Cancelled), None));
                        }
                        return None;
                    }
                    Some(Incoming::Event(PluginEvent::Delta { text, .. })) => {
                        return Some((Ok(text), Some(handle)));
                    }
                    Some(Incoming::Event(PluginEvent::Done { .. })) => return None,
                    Some(Incoming::Event(PluginEvent::Error { class, message, .. })) => {
                        return Some((Err(classify(class, message)), None))
                    }
                    Some(Incoming::Event(_)) | Some(Incoming::Audio(_)) => continue,
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
}

impl SidecarTts {
    pub fn new(client: Arc<SidecarClient>) -> Self {
        let manifest = adapter_manifest(&client, Modality::Tts, "sidecar-tts");
        SidecarTts { client, manifest }
    }
}

#[async_trait]
impl TtsProvider for SidecarTts {
    fn manifest(&self) -> &ProviderManifest {
        &self.manifest
    }

    async fn synthesize(
        &self,
        request: SynthesizeRequest,
        cancel: CancellationToken,
    ) -> Result<ProviderStream<AudioChunk>, VoicekitError> {
        if request.text.trim().is_empty() {
            return Err(VoicekitError::Protocol("text must not be empty".into()));
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
        let stream = stream::unfold(Some(handle), |state| async move {
            let mut handle = state?;
            loop {
                match handle.recv().await {
                    None => {
                        if handle.is_cancelled() {
                            return Some((Err(VoicekitError::Cancelled), None));
                        }
                        return None;
                    }
                    Some(Incoming::Audio(frame)) => {
                        return Some((Ok(AudioChunk::from_frame(&frame)), Some(handle)));
                    }
                    Some(Incoming::Event(PluginEvent::Done { .. })) => return None,
                    Some(Incoming::Event(PluginEvent::Error { class, message, .. })) => {
                        return Some((Err(classify(class, message)), None))
                    }
                    Some(Incoming::Event(_)) => continue,
                }
            }
        });
        Ok(Box::pin(stream))
    }
}

/// STT provider backed by a plugin `transcribe` request.
pub struct SidecarStt {
    client: Arc<SidecarClient>,
    manifest: ProviderManifest,
}

impl SidecarStt {
    pub fn new(client: Arc<SidecarClient>) -> Self {
        let manifest = adapter_manifest(&client, Modality::Stt, "sidecar-stt");
        SidecarStt { client, manifest }
    }
}

#[async_trait]
impl SttProvider for SidecarStt {
    fn manifest(&self) -> &ProviderManifest {
        &self.manifest
    }

    async fn transcribe(
        &self,
        audio: AudioChunk,
        cancel: CancellationToken,
    ) -> Result<ProviderStream<TranscriptChunk>, VoicekitError> {
        if audio.is_empty() {
            return Err(VoicekitError::Protocol("audio must not be empty".into()));
        }
        let request_id = self.client.new_request_id("stt");
        let frames = vec![AudioFrame::new(
            Timestamp::from_millis(0),
            audio.sample_rate,
            audio.channels,
            audio.pcm,
        )];
        let handle = self
            .client
            .open_request(
                PluginRequest::Transcribe {
                    request_id,
                    sample_rate: frames[0].sample_rate,
                    channels: frames[0].channels,
                },
                frames,
                Some(cancel),
            )
            .await?;
        let stream = stream::unfold(Some(handle), |state| async move {
            let mut handle = state?;
            loop {
                match handle.recv().await {
                    None => {
                        if handle.is_cancelled() {
                            return Some((Err(VoicekitError::Cancelled), None));
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
                    Some(Incoming::Event(PluginEvent::Done { .. })) => return None,
                    Some(Incoming::Event(PluginEvent::Error { class, message, .. })) => {
                        return Some((Err(classify(class, message)), None))
                    }
                    Some(Incoming::Event(_)) | Some(Incoming::Audio(_)) => continue,
                }
            }
        });
        Ok(Box::pin(stream))
    }
}
