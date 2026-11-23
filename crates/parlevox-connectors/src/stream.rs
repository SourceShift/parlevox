//! Body sources: cancellation, deadlines and byte-to-event decoding.
//!
//! A [`BodySource`] is the only path from a transport body to a connector.
//! Every read races the caller's cancellation token and the request's
//! absolute deadline; losing either race drops the underlying response body,
//! which is what aborts the HTTP request. `SseSource`, `NdjsonSource` and
//! `JsonSource` layer a parser over it.

use std::collections::VecDeque;
use std::sync::Arc;

use bytes::Bytes;
use futures_util::StreamExt;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::buffer::ConnectorStats;
use crate::error::ConnectorError;
use crate::ndjson::NdjsonParser;
use crate::sse::{SseEvent, SseParser};
use crate::transport::ByteStream;

/// The body shape implied by a response `content-type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StreamFormat {
    /// Server-Sent Events.
    Sse,
    /// Newline-delimited JSON.
    Ndjson,
    /// One JSON document for the whole body.
    Json,
}

impl StreamFormat {
    /// Dispatch on the response content type; JSON is the fallback.
    pub(crate) fn from_content_type(content_type: Option<&str>) -> Self {
        let Some(content_type) = content_type else {
            return StreamFormat::Json;
        };
        let content_type = content_type.to_ascii_lowercase();
        if content_type.contains("text/event-stream") {
            StreamFormat::Sse
        } else if content_type.contains("ndjson") || content_type.contains("jsonl") {
            StreamFormat::Ndjson
        } else {
            StreamFormat::Json
        }
    }
}

/// A transport body with cancellation and deadline enforcement.
pub(crate) struct BodySource {
    body: Option<ByteStream>,
    cancel: CancellationToken,
    deadline: Option<tokio::time::Instant>,
    stats: Arc<ConnectorStats>,
    done: bool,
}

impl BodySource {
    pub(crate) fn new(
        body: ByteStream,
        cancel: CancellationToken,
        deadline: Option<tokio::time::Instant>,
        stats: Arc<ConnectorStats>,
    ) -> Self {
        BodySource {
            body: Some(body),
            cancel,
            deadline,
            stats,
            done: false,
        }
    }

    /// Read the next body chunk.
    ///
    /// Returns `None` at end of body. A cancellation or an expired deadline
    /// ends the body and yields one error; the underlying body is dropped
    /// before the error is returned so the request is aborted promptly.
    pub(crate) async fn next(&mut self) -> Option<Result<Bytes, ConnectorError>> {
        if self.done {
            return None;
        }
        if self.cancel.is_cancelled() {
            self.abort();
            return Some(Err(ConnectorError::Cancelled));
        }
        let cancel = self.cancel.clone();
        let deadline = self.deadline;
        let Some(body) = self.body.as_mut() else {
            self.done = true;
            return None;
        };
        let outcome = tokio::select! {
            biased;
            () = cancel.cancelled() => None,
            result = read_next(body, deadline) => Some(result),
        };
        match outcome {
            None => {
                self.abort();
                Some(Err(ConnectorError::Cancelled))
            }
            Some(Err(error)) => {
                self.abort();
                Some(Err(error))
            }
            Some(Ok(None)) => {
                self.body = None;
                self.done = true;
                None
            }
            Some(Ok(Some(chunk))) => {
                self.stats.record_bytes(chunk.len());
                Some(Ok(chunk))
            }
        }
    }

    fn abort(&mut self) {
        self.body = None;
        self.done = true;
    }
}

async fn read_next(
    body: &mut ByteStream,
    deadline: Option<tokio::time::Instant>,
) -> Result<Option<Bytes>, ConnectorError> {
    match deadline {
        Some(deadline) => match tokio::time::timeout_at(deadline, body.next()).await {
            Ok(Some(Ok(chunk))) => Ok(Some(chunk)),
            Ok(Some(Err(error))) => Err(error),
            Ok(None) => Ok(None),
            Err(_) => Err(ConnectorError::Transient(
                "request deadline exceeded".to_owned(),
            )),
        },
        None => match body.next().await {
            Some(Ok(chunk)) => Ok(Some(chunk)),
            Some(Err(error)) => Err(error),
            None => Ok(None),
        },
    }
}

/// An SSE body decoded into events.
pub(crate) struct SseSource {
    body: BodySource,
    parser: SseParser,
    pending: VecDeque<SseEvent>,
    stats: Arc<ConnectorStats>,
}

impl SseSource {
    pub(crate) fn new(
        body: BodySource,
        max_buffer_bytes: usize,
        stats: Arc<ConnectorStats>,
    ) -> Self {
        SseSource {
            body,
            parser: SseParser::new(max_buffer_bytes),
            pending: VecDeque::new(),
            stats,
        }
    }

    pub(crate) async fn next(&mut self) -> Option<Result<SseEvent, ConnectorError>> {
        loop {
            if let Some(event) = self.pending.pop_front() {
                return Some(Ok(event));
            }
            match self.body.next().await {
                None => {
                    return match self.parser.finish() {
                        Ok(events) => {
                            if events.is_empty() {
                                None
                            } else {
                                self.stats.record_events(events.len());
                                self.pending.extend(events);
                                continue;
                            }
                        }
                        Err(error) => Some(Err(error)),
                    }
                }
                Some(Err(error)) => return Some(Err(error)),
                Some(Ok(chunk)) => match self.parser.push(&chunk) {
                    Ok(events) => {
                        self.stats.record_events(events.len());
                        self.pending.extend(events);
                    }
                    Err(error) => return Some(Err(error)),
                },
            }
        }
    }
}

/// An NDJSON body decoded into JSON values.
pub(crate) struct NdjsonSource {
    body: BodySource,
    parser: NdjsonParser,
    pending: VecDeque<Value>,
    stats: Arc<ConnectorStats>,
}

impl NdjsonSource {
    pub(crate) fn new(
        body: BodySource,
        max_buffer_bytes: usize,
        stats: Arc<ConnectorStats>,
    ) -> Self {
        NdjsonSource {
            body,
            parser: NdjsonParser::new(max_buffer_bytes),
            pending: VecDeque::new(),
            stats,
        }
    }

    pub(crate) async fn next(&mut self) -> Option<Result<Value, ConnectorError>> {
        loop {
            if let Some(value) = self.pending.pop_front() {
                return Some(Ok(value));
            }
            match self.body.next().await {
                None => {
                    return match self.parser.finish() {
                        Ok(records) => {
                            if records.is_empty() {
                                None
                            } else {
                                self.stats.record_events(records.len());
                                self.pending.extend(records);
                                continue;
                            }
                        }
                        Err(error) => Some(Err(error)),
                    }
                }
                Some(Err(error)) => return Some(Err(error)),
                Some(Ok(chunk)) => match self.parser.push(&chunk) {
                    Ok(records) => {
                        self.stats.record_events(records.len());
                        self.pending.extend(records);
                    }
                    Err(error) => return Some(Err(error)),
                },
            }
        }
    }
}

/// A whole JSON body, read once and bounded.
pub(crate) struct JsonSource {
    body: BodySource,
    max_body_bytes: usize,
    consumed: bool,
}

impl JsonSource {
    pub(crate) fn new(body: BodySource, max_body_bytes: usize) -> Self {
        JsonSource {
            body,
            max_body_bytes,
            consumed: false,
        }
    }

    pub(crate) async fn next(&mut self) -> Option<Result<Value, ConnectorError>> {
        if self.consumed {
            return None;
        }
        self.consumed = true;
        let mut bytes = Vec::new();
        loop {
            match self.body.next().await {
                None => break,
                Some(Err(error)) => return Some(Err(error)),
                Some(Ok(chunk)) => {
                    if bytes.len().saturating_add(chunk.len()) > self.max_body_bytes {
                        return Some(Err(ConnectorError::Capacity(format!(
                            "JSON response exceeds {} bytes",
                            self.max_body_bytes
                        ))));
                    }
                    bytes.extend_from_slice(&chunk);
                }
            }
        }
        Some(
            serde_json::from_slice(&bytes).map_err(|error| {
                ConnectorError::Protocol(format!("invalid JSON response: {error}"))
            }),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_type_dispatch() {
        assert_eq!(
            StreamFormat::from_content_type(Some("text/event-stream; charset=utf-8")),
            StreamFormat::Sse
        );
        assert_eq!(
            StreamFormat::from_content_type(Some("application/x-ndjson")),
            StreamFormat::Ndjson
        );
        assert_eq!(
            StreamFormat::from_content_type(Some("application/json")),
            StreamFormat::Json
        );
        assert_eq!(StreamFormat::from_content_type(None), StreamFormat::Json);
    }
}
