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
use parlevox_ipc::{ErrorClassWire, PluginEvent, PluginRequest, MAX_FRAME_BYTES};
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

/// Refuse a request that would not fit one frame, before it reaches the writer.
///
/// The check counts the kind byte the frame layer adds and deliberately never
/// truncates: conversation history is either sent whole or rejected loudly.
fn check_request_fits_frame(request: &PluginRequest) -> Result<(), ParlevoxError> {
    let encoded = serde_json::to_vec(request)
        .map_err(|err| ParlevoxError::Protocol(format!("request encode failed: {err}")))?;
    let frame_len = encoded.len().saturating_add(1);
    if frame_len > MAX_FRAME_BYTES {
        return Err(ParlevoxError::Protocol(format!(
            "request is {frame_len} bytes, above MAX_FRAME_BYTES ({MAX_FRAME_BYTES}); \
             refusing to truncate conversation history"
        )));
    }
    Ok(())
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
        let GenerateRequest { prompt, history } = request;
        if prompt.trim().is_empty() {
            return Err(ParlevoxError::Protocol("prompt must not be empty".into()));
        }
        let request_id = self.client.new_request_id("llm");
        let request = PluginRequest::Generate {
            request_id,
            prompt,
            history,
        };
        // Pre-serialise so an oversized history fails here as a protocol error
        // with a useful message, rather than as a transport error from the
        // writer. Nothing is ever truncated: either the whole history fits one
        // frame or the request is refused.
        check_request_fits_frame(&request)?;
        let handle = self
            .client
            .open_request(request, Vec::new(), Some(cancel))
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

#[cfg(test)]
mod tests {
    use super::*;
    use parlevox_providers::{ChatMessage, Role};

    fn generate_with_history(history: Vec<ChatMessage>) -> PluginRequest {
        PluginRequest::Generate {
            request_id: "r1".into(),
            prompt: "now".into(),
            history,
        }
    }

    #[test]
    fn oversized_history_is_refused_as_a_protocol_error() {
        let request = generate_with_history(vec![ChatMessage {
            role: Role::User,
            text: "x".repeat(MAX_FRAME_BYTES),
        }]);

        let err = check_request_fits_frame(&request).unwrap_err();
        assert!(matches!(err, ParlevoxError::Protocol(_)));
        assert!(err.to_string().contains("MAX_FRAME_BYTES"));
    }

    #[test]
    fn bounded_history_fits_one_frame() {
        let request = generate_with_history(vec![
            ChatMessage {
                role: Role::User,
                text: "hello".into(),
            },
            ChatMessage {
                role: Role::Assistant,
                text: "hi".into(),
            },
        ]);

        assert!(check_request_fits_frame(&request).is_ok());
    }
}
