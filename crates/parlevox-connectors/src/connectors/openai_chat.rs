//! The `openai-compatible-chat` connector.
//!
//! Builds a `chat/completions` request from the conversation history, then
//! dispatches on the response `content-type`: SSE deltas, NDJSON records, or
//! one JSON document. Every malformed shape becomes
//! [`ParlevoxError::Protocol`], never a panic.

use std::sync::Arc;

use async_trait::async_trait;
use parlevox_core::ParlevoxError;
use parlevox_providers::{GenerateRequest, LlmProvider, ProviderManifest, ProviderStream, Role};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use crate::error::ConnectorError;
use crate::stream::{BodySource, JsonSource, NdjsonSource, SseSource, StreamFormat};
use crate::toolkit::HttpToolkit;
use crate::transport::RequestSpec;

/// The `openai-compatible-chat` connector.
pub struct OpenAiChat {
    manifest: ProviderManifest,
    endpoint: String,
    model: String,
    toolkit: HttpToolkit,
}

impl OpenAiChat {
    pub(crate) fn new(
        manifest: ProviderManifest,
        endpoint: String,
        model: String,
        toolkit: HttpToolkit,
    ) -> Self {
        OpenAiChat {
            manifest,
            endpoint,
            model,
            toolkit,
        }
    }
}

#[async_trait]
impl LlmProvider for OpenAiChat {
    fn manifest(&self) -> &ProviderManifest {
        &self.manifest
    }

    async fn generate(
        &self,
        request: GenerateRequest,
        cancel: CancellationToken,
    ) -> Result<ProviderStream<String>, ParlevoxError> {
        let provider = self.manifest.name.clone();
        let body = self.request_body(&request);
        let spec = RequestSpec::post(&self.endpoint)
            .with_header("accept", "text/event-stream")
            .map_err(|error| error.into_provider(&provider))?
            .with_json(body);
        let response = self
            .toolkit
            .send(spec, &cancel)
            .await
            .map_err(|error| error.into_provider(&provider))?;
        let format = StreamFormat::from_content_type(response.content_type.as_deref());
        let source = response.into_body_source(cancel);
        Ok(chat_stream(source, format, &self.toolkit, provider))
    }

    fn take_sidecar_dropped_frames(&self) -> u64 {
        self.toolkit.stats().take_dropped()
    }
}

impl OpenAiChat {
    fn request_body(&self, request: &GenerateRequest) -> Value {
        let mut messages = Vec::with_capacity(request.history.len().saturating_add(1));
        for message in &request.history {
            messages.push(json!({
                "role": role_name(message.role),
                "content": message.text,
            }));
        }
        messages.push(json!({"role": "user", "content": request.prompt}));
        json!({
            "model": self.model,
            "stream": true,
            "messages": messages,
        })
    }
}

fn role_name(role: Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
    }
}

enum ChatSource {
    Sse(SseSource),
    Ndjson(NdjsonSource),
    Json(JsonSource),
}

struct ChatState {
    source: Option<ChatSource>,
    provider: String,
}

impl ChatState {
    async fn next_delta(&mut self) -> Result<Option<String>, ConnectorError> {
        let Some(mut source) = self.source.take() else {
            return Ok(None);
        };
        let result = match &mut source {
            ChatSource::Sse(sse) => next_sse_delta(sse).await,
            ChatSource::Ndjson(ndjson) => next_ndjson_delta(ndjson).await,
            ChatSource::Json(json) => next_json_delta(json).await,
        };
        match result {
            Ok(Some(delta)) => {
                self.source = Some(source);
                Ok(Some(delta))
            }
            Ok(None) => Ok(None),
            Err(error) => Err(error),
        }
    }
}

fn chat_stream(
    body: BodySource,
    format: StreamFormat,
    toolkit: &HttpToolkit,
    provider: String,
) -> ProviderStream<String> {
    let max_buffer_bytes = toolkit.max_buffer_bytes();
    let stats = Arc::clone(toolkit.stats());
    let source = match format {
        StreamFormat::Sse => {
            ChatSource::Sse(SseSource::new(body, max_buffer_bytes, Arc::clone(&stats)))
        }
        StreamFormat::Ndjson => ChatSource::Ndjson(NdjsonSource::new(
            body,
            max_buffer_bytes,
            Arc::clone(&stats),
        )),
        StreamFormat::Json => ChatSource::Json(JsonSource::new(body, max_buffer_bytes)),
    };
    let state = ChatState {
        source: Some(source),
        provider,
    };
    Box::pin(futures_util::stream::unfold(
        state,
        |mut state| async move {
            match state.next_delta().await {
                Ok(Some(delta)) => Some((Ok(delta), state)),
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

async fn next_sse_delta(source: &mut SseSource) -> Result<Option<String>, ConnectorError> {
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
        if let Some(name) = event.event.as_deref() {
            if name != "message" {
                return Err(ConnectorError::Protocol(format!(
                    "unknown chat SSE event '{name}'"
                )));
            }
        }
        let value: Value = serde_json::from_str(&event.data)
            .map_err(|_| ConnectorError::Protocol("chat SSE data is not valid JSON".to_owned()))?;
        match chat_delta(&value)? {
            Some(delta) if !delta.is_empty() => return Ok(Some(delta)),
            _ => continue,
        }
    }
}

async fn next_ndjson_delta(source: &mut NdjsonSource) -> Result<Option<String>, ConnectorError> {
    loop {
        let Some(item) = source.next().await else {
            return Ok(None);
        };
        let value = item?;
        match chat_delta(&value)? {
            Some(delta) if !delta.is_empty() => return Ok(Some(delta)),
            _ => continue,
        }
    }
}

async fn next_json_delta(source: &mut JsonSource) -> Result<Option<String>, ConnectorError> {
    let Some(item) = source.next().await else {
        return Ok(None);
    };
    let value = item?;
    match chat_delta(&value)? {
        Some(delta) if !delta.is_empty() => Ok(Some(delta)),
        _ => Err(ConnectorError::Protocol(
            "chat response has no content".to_owned(),
        )),
    }
}

/// Pull the assistant text out of one chat response object.
///
/// Streaming deltas live under `choices[0].delta.content`; a non-streaming
/// response uses `choices[0].message.content`. An empty `choices` array is a
/// usage-only record and yields no text; a missing `choices` field is a wire
/// violation.
fn chat_delta(value: &Value) -> Result<Option<String>, ConnectorError> {
    let object = value
        .as_object()
        .ok_or_else(|| ConnectorError::Protocol("chat event is not a JSON object".to_owned()))?;
    let choices = object
        .get("choices")
        .ok_or_else(|| ConnectorError::Protocol("chat event has no 'choices'".to_owned()))?
        .as_array()
        .ok_or_else(|| ConnectorError::Protocol("chat 'choices' is not an array".to_owned()))?;
    let Some(choice) = choices.first() else {
        return Ok(None);
    };
    let delta = choice
        .get("delta")
        .and_then(Value::as_object)
        .or_else(|| choice.get("message").and_then(Value::as_object));
    let Some(delta) = delta else {
        return Ok(None);
    };
    match delta.get("content") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) => Ok(Some(text.clone())),
        Some(_) => Err(ConnectorError::Protocol(
            "chat delta content is not a string".to_owned(),
        )),
    }
}
