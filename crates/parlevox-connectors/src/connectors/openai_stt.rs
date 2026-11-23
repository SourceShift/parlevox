//! The `openai-audio-stt` connector.
//!
//! Uploads one utterance as a multipart form and returns a transcript stream.
//! A descriptor that declares the `streaming` capability gets SSE
//! `transcript.text.delta` events; every other provider gets one JSON
//! document. The trait contract — partials, then exactly one final — holds in
//! both shapes.

use std::sync::Arc;

use async_trait::async_trait;
use parlevox_core::ParlevoxError;
use parlevox_providers::{
    AudioChunk, ProviderManifest, ProviderStream, SttProvider, TranscriptChunk,
};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::error::ConnectorError;
use crate::stream::{BodySource, JsonSource, NdjsonSource, SseSource, StreamFormat};
use crate::toolkit::HttpToolkit;
use crate::transport::{MultipartBody, RequestSpec};

/// The `openai-audio-stt` connector.
pub struct OpenAiStt {
    manifest: ProviderManifest,
    endpoint: String,
    model: String,
    streaming: bool,
    toolkit: HttpToolkit,
}

impl OpenAiStt {
    pub(crate) fn new(
        manifest: ProviderManifest,
        endpoint: String,
        model: String,
        toolkit: HttpToolkit,
    ) -> Self {
        let streaming = manifest.supports("streaming");
        OpenAiStt {
            manifest,
            endpoint,
            model,
            streaming,
            toolkit,
        }
    }
}

#[async_trait]
impl SttProvider for OpenAiStt {
    fn manifest(&self) -> &ProviderManifest {
        &self.manifest
    }

    async fn transcribe(
        &self,
        audio: AudioChunk,
        cancel: CancellationToken,
    ) -> Result<ProviderStream<TranscriptChunk>, ParlevoxError> {
        let provider = self.manifest.name.clone();
        if audio.pcm.is_empty() {
            return Err(ParlevoxError::Protocol(format!(
                "provider '{provider}' received empty audio"
            )));
        }
        let mut form = MultipartBody::new()
            .with_text("model", self.model.clone())
            .with_file("file", "audio.pcm", "audio/pcm", pcm_bytes(&audio.pcm));
        let accept = if self.streaming {
            form = form.with_text("stream", "true");
            "text/event-stream"
        } else {
            "application/json"
        };
        let spec = RequestSpec::post(&self.endpoint)
            .with_header("accept", accept)
            .map_err(|error| error.into_provider(&provider))?
            .with_multipart(form);
        let response = self
            .toolkit
            .send(spec, &cancel)
            .await
            .map_err(|error| error.into_provider(&provider))?;
        let format = StreamFormat::from_content_type(response.content_type.as_deref());
        let source = response.into_body_source(cancel);
        Ok(transcript_stream(source, format, &self.toolkit, provider))
    }

    fn take_sidecar_dropped_frames(&self) -> u64 {
        self.toolkit.stats().take_dropped()
    }
}

fn pcm_bytes(pcm: &[i16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(pcm.len().saturating_mul(2));
    for sample in pcm {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

enum TranscriptSource {
    Sse(SseSource),
    Ndjson(NdjsonSource),
    Json(JsonSource),
}

enum TranscriptItem {
    Partial(String),
    Final(String),
}

struct TranscriptState {
    source: Option<TranscriptSource>,
    provider: String,
    accumulated: String,
    final_emitted: bool,
}

impl TranscriptState {
    async fn next_chunk(&mut self) -> Result<Option<TranscriptChunk>, ConnectorError> {
        let Some(mut source) = self.source.take() else {
            return Ok(None);
        };
        let item = match &mut source {
            TranscriptSource::Sse(sse) => next_sse_item(sse).await,
            TranscriptSource::Ndjson(ndjson) => next_ndjson_item(ndjson).await,
            TranscriptSource::Json(json) => next_json_item(json).await,
        };
        match item {
            Ok(Some(TranscriptItem::Partial(text))) => {
                self.accumulated.push_str(&text);
                self.source = Some(source);
                Ok(Some(TranscriptChunk {
                    text,
                    is_final: false,
                }))
            }
            Ok(Some(TranscriptItem::Final(text))) => {
                let text = if text.is_empty() {
                    std::mem::take(&mut self.accumulated)
                } else {
                    text
                };
                self.final_emitted = true;
                Ok(Some(TranscriptChunk {
                    text,
                    is_final: true,
                }))
            }
            Ok(None) => {
                if self.final_emitted {
                    return Ok(None);
                }
                let text = std::mem::take(&mut self.accumulated);
                if text.is_empty() {
                    return Err(ConnectorError::Protocol(
                        "transcript stream ended without text".to_owned(),
                    ));
                }
                self.final_emitted = true;
                Ok(Some(TranscriptChunk {
                    text,
                    is_final: true,
                }))
            }
            Err(error) => Err(error),
        }
    }
}

fn transcript_stream(
    body: BodySource,
    format: StreamFormat,
    toolkit: &HttpToolkit,
    provider: String,
) -> ProviderStream<TranscriptChunk> {
    let max_buffer_bytes = toolkit.max_buffer_bytes();
    let stats = Arc::clone(toolkit.stats());
    let source = match format {
        StreamFormat::Sse => {
            TranscriptSource::Sse(SseSource::new(body, max_buffer_bytes, Arc::clone(&stats)))
        }
        StreamFormat::Ndjson => TranscriptSource::Ndjson(NdjsonSource::new(
            body,
            max_buffer_bytes,
            Arc::clone(&stats),
        )),
        StreamFormat::Json => TranscriptSource::Json(JsonSource::new(body, max_buffer_bytes)),
    };
    let state = TranscriptState {
        source: Some(source),
        provider,
        accumulated: String::new(),
        final_emitted: false,
    };
    Box::pin(futures_util::stream::unfold(
        state,
        |mut state| async move {
            match state.next_chunk().await {
                Ok(Some(chunk)) => Some((Ok(chunk), state)),
                Ok(None) => None,
                Err(error) => {
                    let mapped = error.into_provider(&state.provider);
                    state.source = None;
                    Some((Err(mapped), state))
                }
            }
        },
    ))
}

async fn next_sse_item(source: &mut SseSource) -> Result<Option<TranscriptItem>, ConnectorError> {
    loop {
        let Some(item) = source.next().await else {
            return Ok(None);
        };
        let event = item?;
        if event.data.trim() == "[DONE]" {
            return Ok(None);
        }
        if event.data.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(&event.data).map_err(|_| {
            ConnectorError::Protocol("transcription SSE data is not valid JSON".to_owned())
        })?;
        let kind = event
            .event
            .as_deref()
            .or_else(|| value.get("type").and_then(Value::as_str));
        return decode_item(kind, &value);
    }
}

async fn next_ndjson_item(
    source: &mut NdjsonSource,
) -> Result<Option<TranscriptItem>, ConnectorError> {
    let Some(item) = source.next().await else {
        return Ok(None);
    };
    let value = item?;
    let kind = value.get("type").and_then(Value::as_str);
    decode_item(kind, &value)
}

async fn next_json_item(source: &mut JsonSource) -> Result<Option<TranscriptItem>, ConnectorError> {
    let Some(item) = source.next().await else {
        return Ok(None);
    };
    let value = item?;
    let text = value.get("text").and_then(Value::as_str).ok_or_else(|| {
        ConnectorError::Protocol("transcription response has no 'text'".to_owned())
    })?;
    Ok(Some(TranscriptItem::Final(text.to_owned())))
}

fn decode_item(
    kind: Option<&str>,
    value: &Value,
) -> Result<Option<TranscriptItem>, ConnectorError> {
    match kind {
        Some("transcript.text.delta") => {
            let delta = value.get("delta").and_then(Value::as_str).ok_or_else(|| {
                ConnectorError::Protocol("transcript delta has no 'delta' string".to_owned())
            })?;
            Ok(Some(TranscriptItem::Partial(delta.to_owned())))
        }
        Some("transcript.text.done") => {
            let text = value
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            Ok(Some(TranscriptItem::Final(text)))
        }
        Some(other) => Err(ConnectorError::Protocol(format!(
            "unknown transcription event '{other}'"
        ))),
        None => Err(ConnectorError::Protocol(
            "transcription event has no type".to_owned(),
        )),
    }
}
