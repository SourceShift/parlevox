//! Injectable HTTP transport.
//!
//! A connector never talks to a socket directly. It builds a
//! [`RequestSpec`], asks a [`Transport`] to execute it, and receives a
//! [`TransportResponse`] whose body is a byte stream. Tests supply
//! [`ReplayTransport`]; production supplies [`ReqwestTransport`].
//!
//! # Secret hygiene
//!
//! Request headers and query values can carry credentials. Their `Debug`
//! representations are redacted and the transport strips request URLs from
//! every error it reports, so a credential cannot reach a log line through a
//! connector failure.

use std::collections::VecDeque;
use std::fmt;
use std::pin::Pin;
use std::sync::{Mutex, MutexGuard, PoisonError};

use async_trait::async_trait;
use bytes::Bytes;
use futures_core::Stream;
use serde_json::Value;

use crate::error::ConnectorError;

/// A boxed stream of response-body chunks.
pub type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, ConnectorError>> + Send>>;

/// The HTTP methods the toolkit uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Method {
    /// `GET`, used by discovery endpoints.
    Get,
    /// `POST`, used by every request that carries a body.
    Post,
}

impl Method {
    /// The wire name of the method.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Method::Get => "GET",
            Method::Post => "POST",
        }
    }
}

/// One validated request header.
///
/// The value is deliberately not part of `Debug`: a header may be the
/// credential, and a struct that prints itself must never print it. Callers
/// that genuinely need the bytes use [`Header::value`].
#[derive(Clone, PartialEq, Eq)]
pub struct Header {
    name: String,
    value: String,
}

impl Header {
    /// Validate and build one header.
    ///
    /// # Errors
    ///
    /// Returns [`ConnectorError::Protocol`] when the name is not an HTTP token
    /// or the value contains a control character, which would otherwise allow
    /// response splitting.
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Result<Self, ConnectorError> {
        let name = name.into();
        let value = value.into();
        if !is_valid_header_name(&name) {
            return Err(ConnectorError::Protocol(
                "invalid HTTP header name".to_owned(),
            ));
        }
        if !is_valid_header_value(&value) {
            return Err(ConnectorError::Protocol(
                "invalid HTTP header value".to_owned(),
            ));
        }
        Ok(Header { name, value })
    }

    /// The header name, lower-case as supplied.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The header value.
    #[must_use]
    pub fn value(&self) -> &str {
        &self.value
    }
}

impl fmt::Debug for Header {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Header")
            .field("name", &self.name)
            .field("value", &"<redacted>")
            .finish()
    }
}

/// Whether `name` is a non-empty RFC 7230 token.
#[must_use]
pub fn is_valid_header_name(name: &str) -> bool {
    !name.is_empty()
        && name.bytes().all(|byte| {
            byte.is_ascii_alphanumeric()
                || matches!(
                    byte,
                    b'!' | b'#'
                        | b'$'
                        | b'%'
                        | b'&'
                        | b'\''
                        | b'*'
                        | b'+'
                        | b'-'
                        | b'.'
                        | b'^'
                        | b'_'
                        | b'`'
                        | b'|'
                        | b'~'
                )
        })
}

/// Whether `value` is safe for a header: visible ASCII and horizontal tab,
/// with no CR, LF or NUL.
#[must_use]
pub fn is_valid_header_value(value: &str) -> bool {
    value
        .bytes()
        .all(|byte| byte == b'\t' || (0x20..=0x7e).contains(&byte))
}

/// One part of a multipart request body.
#[derive(Clone, PartialEq, Eq)]
pub struct MultipartPart {
    /// Form field name.
    pub name: String,
    /// File name for a file part; `None` for a text part.
    pub filename: Option<String>,
    /// Content type for a file part.
    pub content_type: Option<String>,
    /// Raw part bytes.
    pub data: Vec<u8>,
}

impl MultipartPart {
    /// A text form field.
    #[must_use]
    pub fn text(name: impl Into<String>, value: impl Into<String>) -> Self {
        MultipartPart {
            name: name.into(),
            filename: None,
            content_type: None,
            data: value.into().into_bytes(),
        }
    }

    /// A file form field.
    #[must_use]
    pub fn file(
        name: impl Into<String>,
        filename: impl Into<String>,
        content_type: impl Into<String>,
        data: impl Into<Vec<u8>>,
    ) -> Self {
        MultipartPart {
            name: name.into(),
            filename: Some(filename.into()),
            content_type: Some(content_type.into()),
            data: data.into(),
        }
    }
}

impl fmt::Debug for MultipartPart {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MultipartPart")
            .field("name", &self.name)
            .field("filename", &self.filename)
            .field("content_type", &self.content_type)
            .field("bytes", &self.data.len())
            .finish()
    }
}

/// A multipart form body.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MultipartBody {
    parts: Vec<MultipartPart>,
}

impl MultipartBody {
    /// An empty form.
    #[must_use]
    pub fn new() -> Self {
        MultipartBody::default()
    }

    /// Append a text field.
    #[must_use]
    pub fn with_text(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.parts.push(MultipartPart::text(name, value));
        self
    }

    /// Append a file field.
    #[must_use]
    pub fn with_file(
        mut self,
        name: impl Into<String>,
        filename: impl Into<String>,
        content_type: impl Into<String>,
        data: impl Into<Vec<u8>>,
    ) -> Self {
        self.parts
            .push(MultipartPart::file(name, filename, content_type, data));
        self
    }

    /// The parts, in insertion order.
    #[must_use]
    pub fn parts(&self) -> &[MultipartPart] {
        &self.parts
    }

    /// Whether the form has no parts.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }
}

/// The body of a request.
#[derive(Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RequestBody {
    /// No body.
    Empty,
    /// A JSON document.
    Json(Value),
    /// Raw bytes.
    Bytes {
        /// Content type to send with the bytes, when known.
        content_type: Option<String>,
        /// The bytes.
        data: Vec<u8>,
    },
    /// A multipart form.
    Multipart(MultipartBody),
}

impl fmt::Debug for RequestBody {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RequestBody::Empty => f.write_str("RequestBody::Empty"),
            RequestBody::Json(_) => f.write_str("RequestBody::Json(<redacted>)"),
            RequestBody::Bytes { content_type, data } => f
                .debug_struct("RequestBody::Bytes")
                .field("content_type", content_type)
                .field("bytes", &data.len())
                .finish(),
            RequestBody::Multipart(body) => f
                .debug_struct("RequestBody::Multipart")
                .field("parts", &body.parts().len())
                .finish(),
        }
    }
}

/// A fully built request handed to a [`Transport`].
///
/// `Debug` prints the method, the URL with any query string removed, header
/// *names* and query *names* only. Values can be credentials and are never
/// printed.
#[derive(Clone)]
pub struct RequestSpec {
    /// HTTP method.
    pub method: Method,
    /// Absolute request URL, without credential values in the query.
    pub url: String,
    /// Request headers.
    pub headers: Vec<Header>,
    /// Query parameters, structured so the transport can percent-encode them.
    pub query: Vec<(String, String)>,
    /// Request body.
    pub body: RequestBody,
}

impl RequestSpec {
    /// A `GET` request.
    #[must_use]
    pub fn get(url: impl Into<String>) -> Self {
        RequestSpec {
            method: Method::Get,
            url: url.into(),
            headers: Vec::new(),
            query: Vec::new(),
            body: RequestBody::Empty,
        }
    }

    /// A `POST` request.
    #[must_use]
    pub fn post(url: impl Into<String>) -> Self {
        RequestSpec {
            method: Method::Post,
            url: url.into(),
            headers: Vec::new(),
            query: Vec::new(),
            body: RequestBody::Empty,
        }
    }

    /// Append a validated header.
    ///
    /// # Errors
    ///
    /// Returns [`ConnectorError::Protocol`] for an invalid name or value.
    pub fn with_header(
        mut self,
        name: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<Self, ConnectorError> {
        self.headers.push(Header::new(name, value)?);
        Ok(self)
    }

    /// Append one query parameter.
    #[must_use]
    pub fn with_query(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.query.push((name.into(), value.into()));
        self
    }

    /// Replace the body with a JSON document.
    #[must_use]
    pub fn with_json(mut self, value: Value) -> Self {
        self.body = RequestBody::Json(value);
        self
    }

    /// Replace the body with a multipart form.
    #[must_use]
    pub fn with_multipart(mut self, body: MultipartBody) -> Self {
        self.body = RequestBody::Multipart(body);
        self
    }

    /// The URL with any query string or fragment removed.
    ///
    /// This is the only URL form this crate logs or reports.
    #[must_use]
    pub fn safe_url(&self) -> String {
        self.url
            .split(['?', '#'])
            .next()
            .unwrap_or_default()
            .to_owned()
    }

    pub(crate) fn push_header(&mut self, name: &str, value: &str) -> Result<(), ConnectorError> {
        self.headers.push(Header::new(name, value)?);
        Ok(())
    }
}

impl fmt::Debug for RequestSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let header_names: Vec<&str> = self.headers.iter().map(Header::name).collect();
        let query_names: Vec<&str> = self.query.iter().map(|(name, _)| name.as_str()).collect();
        f.debug_struct("RequestSpec")
            .field("method", &self.method)
            .field("url", &self.safe_url())
            .field("headers", &header_names)
            .field("query", &query_names)
            .field("body", &self.body)
            .finish()
    }
}

/// The transport's answer: a status, headers and a streaming body.
pub struct TransportResponse {
    /// HTTP status code.
    pub status: u16,
    /// Response headers.
    pub headers: Vec<Header>,
    /// The body, as it arrives.
    pub body: ByteStream,
}

impl TransportResponse {
    /// Build a response.
    #[must_use]
    pub fn new(status: u16, body: ByteStream) -> Self {
        TransportResponse {
            status,
            headers: Vec::new(),
            body,
        }
    }

    /// Append one response header.
    ///
    /// # Errors
    ///
    /// Returns [`ConnectorError::Protocol`] for an invalid name or value.
    pub fn with_header(
        mut self,
        name: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<Self, ConnectorError> {
        self.headers.push(Header::new(name, value)?);
        Ok(self)
    }

    /// The first value for `name`, matched case-insensitively.
    #[must_use]
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|header| header.name().eq_ignore_ascii_case(name))
            .map(Header::value)
    }

    /// The `content-type` header value, when present.
    #[must_use]
    pub fn content_type(&self) -> Option<&str> {
        self.header("content-type")
    }
}

impl fmt::Debug for TransportResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TransportResponse")
            .field("status", &self.status)
            .field("headers", &self.headers)
            .finish_non_exhaustive()
    }
}

/// Wrap an in-memory byte sequence as a body stream.
pub fn bytes_stream(chunks: impl IntoIterator<Item = Bytes>) -> ByteStream {
    // Materialise the iterator so the stream owns a `Send` collection rather
    // than an arbitrary (possibly non-`Send`) iterator type.
    let chunks: Vec<Bytes> = chunks.into_iter().collect();
    Box::pin(futures_util::stream::iter(chunks.into_iter().map(Ok)))
}

/// The seam every connector is built on.
///
/// Implementations must be safe to call concurrently. A transport failure is a
/// [`ConnectorError::Transport`]; HTTP error *statuses* are not failures here,
/// because the toolkit classifies them.
#[async_trait]
pub trait Transport: Send + Sync + fmt::Debug {
    /// Execute one request and return the response head plus a body stream.
    ///
    /// # Errors
    ///
    /// Returns [`ConnectorError::Transport`] when the request cannot be sent or
    /// the response head cannot be read. Returning the response is not an
    /// error, even for a non-2xx status.
    async fn execute(&self, request: RequestSpec) -> Result<TransportResponse, ConnectorError>;
}

/// A transport that replays a scripted list of responses and records requests.
///
/// This is the whole acceptance-test network: nothing here opens a socket.
#[derive(Debug, Default)]
pub struct ReplayTransport {
    state: Mutex<ReplayState>,
}

#[derive(Debug, Default)]
struct ReplayState {
    responses: VecDeque<TransportResponse>,
    requests: Vec<RequestSpec>,
}

impl ReplayTransport {
    /// Build a replay transport from the responses to serve, in order.
    #[must_use]
    pub fn new(responses: Vec<TransportResponse>) -> Self {
        ReplayTransport {
            state: Mutex::new(ReplayState {
                responses: responses.into(),
                requests: Vec::new(),
            }),
        }
    }

    /// The requests received so far, in order.
    #[must_use]
    pub fn requests(&self) -> Vec<RequestSpec> {
        lock(&self.state).requests.clone()
    }

    /// How many requests have been received.
    #[must_use]
    pub fn request_count(&self) -> usize {
        lock(&self.state).requests.len()
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    // A panic while holding this lock cannot leave a partially updated
    // script; the state is a plain queue. Recover instead of poisoning.
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

#[async_trait]
impl Transport for ReplayTransport {
    async fn execute(&self, request: RequestSpec) -> Result<TransportResponse, ConnectorError> {
        let mut state = lock(&self.state);
        state.requests.push(request);
        state
            .responses
            .pop_front()
            .ok_or_else(|| ConnectorError::Transport("replay script exhausted".to_owned()))
    }
}

/// The production transport, backed by `reqwest` over rustls.
///
/// The client is built without `native-tls` and without a client-wide
/// timeout: the toolkit owns per-request deadlines.
#[derive(Debug, Clone)]
pub struct ReqwestTransport {
    client: reqwest::Client,
}

impl ReqwestTransport {
    /// Build a client.
    ///
    /// # Errors
    ///
    /// Returns [`ConnectorError::Fatal`] when the TLS backend cannot be
    /// initialised.
    pub fn new() -> Result<Self, ConnectorError> {
        let client = reqwest::Client::builder()
            .build()
            .map_err(|err| ConnectorError::Fatal(format!("cannot build HTTP client: {err}")))?;
        Ok(ReqwestTransport { client })
    }
}

#[async_trait]
impl Transport for ReqwestTransport {
    async fn execute(&self, request: RequestSpec) -> Result<TransportResponse, ConnectorError> {
        let mut builder = match request.method {
            Method::Get => self.client.get(&request.url),
            Method::Post => self.client.post(&request.url),
        };
        for (name, value) in &request.query {
            builder = builder.query(&[(name.as_str(), value.as_str())]);
        }
        for header in &request.headers {
            let name = reqwest::header::HeaderName::from_bytes(header.name().as_bytes())
                .map_err(|err| ConnectorError::Protocol(format!("invalid header name: {err}")))?;
            let value = reqwest::header::HeaderValue::from_str(header.value())
                .map_err(|err| ConnectorError::Protocol(format!("invalid header value: {err}")))?;
            builder = builder.header(name, value);
        }
        builder = match request.body {
            RequestBody::Empty => builder,
            RequestBody::Json(value) => builder.json(&value),
            RequestBody::Bytes { content_type, data } => {
                let request = builder.body(data);
                match content_type {
                    Some(content_type) => request.header(
                        reqwest::header::CONTENT_TYPE,
                        reqwest::header::HeaderValue::from_str(&content_type).map_err(|err| {
                            ConnectorError::Protocol(format!("invalid request content type: {err}"))
                        })?,
                    ),
                    None => request,
                }
            }
            RequestBody::Multipart(body) => builder.multipart(to_form(body)?),
        };
        let response = builder.send().await.map_err(|err| {
            // `without_url` keeps any query-string credential out of the log.
            ConnectorError::Transport(format!("request failed: {}", err.without_url()))
        })?;
        let status = response.status().as_u16();
        let headers = response
            .headers()
            .iter()
            .filter_map(|(name, value)| {
                value.to_str().ok().map(|value| Header {
                    name: name.as_str().to_owned(),
                    value: value.to_owned(),
                })
            })
            .collect();
        let body: ByteStream = Box::pin(futures_util::StreamExt::map(
            response.bytes_stream(),
            |chunk| {
                chunk.map_err(|err| {
                    ConnectorError::Transport(format!("body stream failed: {}", err.without_url()))
                })
            },
        ));
        Ok(TransportResponse {
            status,
            headers,
            body,
        })
    }
}

fn to_form(body: MultipartBody) -> Result<reqwest::multipart::Form, ConnectorError> {
    let mut form = reqwest::multipart::Form::new();
    for part in body.parts {
        form = match part.filename {
            Some(filename) => {
                let mut entry = reqwest::multipart::Part::bytes(part.data).file_name(filename);
                if let Some(content_type) = part.content_type {
                    entry = entry.mime_str(&content_type).map_err(|err| {
                        ConnectorError::Protocol(format!("invalid multipart content type: {err}"))
                    })?;
                }
                form.part(part.name, entry)
            }
            None => {
                let text = String::from_utf8(part.data).map_err(|_| {
                    ConnectorError::Protocol("multipart text part is not UTF-8".to_owned())
                })?;
                form.text(part.name, text)
            }
        };
    }
    Ok(form)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_validation_rejects_splitting() {
        assert!(Header::new("authorization", "Bearer abc").is_ok());
        assert!(Header::new("bad header", "x").is_err());
        assert!(Header::new("x-test", "line\r\ninjection").is_err());
        assert!(is_valid_header_name("x-api-key"));
    }

    #[test]
    fn request_debug_redacts_values() {
        let request = RequestSpec::post("https://example.invalid/v1?api_key=secret")
            .with_header("authorization", "Bearer sk-secret")
            .expect("header is valid")
            .with_query("api_key", "plain-secret");
        let debug = format!("{request:?}");
        assert!(!debug.contains("sk-secret"), "{debug}");
        assert!(!debug.contains("plain-secret"), "{debug}");
        assert!(debug.contains("authorization"));
        assert!(debug.contains("api_key"));
        assert!(debug.contains("example.invalid/v1"));
    }

    #[test]
    fn replay_transport_serves_in_order() {
        let first = TransportResponse::new(200, bytes_stream([Bytes::from_static(b"a")]));
        let second = TransportResponse::new(429, bytes_stream(Vec::new()));
        let transport = ReplayTransport::new(vec![first, second]);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime builds");
        runtime.block_on(async {
            let request = RequestSpec::get("https://example.invalid");
            assert_eq!(
                transport
                    .execute(request.clone())
                    .await
                    .expect("first response")
                    .status,
                200
            );
            assert_eq!(
                transport
                    .execute(request)
                    .await
                    .expect("second response")
                    .status,
                429
            );
            assert!(transport
                .execute(RequestSpec::get("https://example.invalid"))
                .await
                .is_err());
        });
        assert_eq!(transport.request_count(), 3);
    }
}
