//! HTTP/JSON connectors for parlevox providers.
//!
//! This crate turns a provider *descriptor* into a callable provider without
//! touching the runtime, the session, the turn machine or the CLI. It owns
//! three things:
//!
//! - a small, injectable HTTP toolkit: auth injection, SSE/NDJSON/JSON body
//!   parsing, per-request deadlines, pre-first-byte retry, cancellation and
//!   bounded buffers with counted drops;
//! - three connectors that speak the OpenAI wire shapes
//!   (`openai-compatible-chat`, `openai-audio-stt`, `openai-audio-tts`);
//! - [`build_provider`], which selects a connector from `adapter.kind` and
//!   returns the matching modality trait object.
//!
//! # Features
//!
//! The default `http` feature enables everything above and pulls in the HTTP
//! client. With `--no-default-features` the crate exposes only
//! [`ConnectorError`]; it builds offline and links no network stack.
//!
//! # Network policy
//!
//! Every connector takes an `Arc<dyn Transport>` through
//! [`ConnectorConfig`]. Tests replay recorded fixtures through
//! [`ReplayTransport`] and never open a socket; [`ReqwestTransport`] is the
//! production transport and is only reached when a caller chooses it.

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod error;

#[cfg(feature = "http")]
mod auth;
#[cfg(feature = "http")]
mod buffer;
#[cfg(feature = "http")]
mod connectors;
#[cfg(feature = "http")]
mod factory;
#[cfg(feature = "http")]
mod ndjson;
#[cfg(feature = "http")]
mod sse;
#[cfg(feature = "http")]
mod stream;
#[cfg(feature = "http")]
mod toolkit;
#[cfg(feature = "http")]
mod transport;

pub use error::ConnectorError;

#[cfg(feature = "http")]
pub use auth::{Auth, Secret, QUERY_TEMPLATE_PLACEHOLDER};
#[cfg(feature = "http")]
pub use buffer::{BoundedQueue, ConnectorStats, ConnectorStatsSnapshot};
#[cfg(feature = "http")]
pub use connectors::{OpenAiChat, OpenAiStt, OpenAiTts};
#[cfg(feature = "http")]
pub use factory::{
    build_provider, BuiltProvider, ConnectorConfig, ConnectorConfigBuilder, Credentials,
};
#[cfg(feature = "http")]
pub use ndjson::NdjsonParser;
#[cfg(feature = "http")]
pub use sse::{SseEvent, SseParser};
#[cfg(feature = "http")]
pub use toolkit::{HttpResponse, HttpToolkit};
#[cfg(feature = "http")]
pub use transport::{
    bytes_stream, ByteStream, Header, Method, MultipartBody, MultipartPart, ReplayTransport,
    RequestBody, RequestSpec, ReqwestTransport, Transport, TransportResponse,
};
