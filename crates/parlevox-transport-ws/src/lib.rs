//! `parlevox-transport-ws` — one WebSocket connection, one voice session.
//!
//! The runtime turn loop is transport-agnostic. This crate is the base
//! transport that makes it reachable: a client opens a WebSocket, sends PCM
//! audio and a small set of JSON control messages, and receives JSON events
//! plus PCM playout. Each connection owns a freshly built
//! [`Session`](parlevox_runtime::Session); no session, event queue, or ingress
//! buffer is shared between connections.
//!
//! # Wire protocol
//!
//! Version 1. Every text frame is one UTF-8 JSON object with a `type`
//! discriminator. Every binary frame is signed 16-bit little-endian PCM, mono
//! by default, at [`WsConfig::sample_rate`] (24 kHz by default).
//!
//! ## Server → client
//!
//! The first message on a connection is always:
//!
//! ```json
//! {"type":"ready","protocol_version":1}
//! ```
//!
//! Then, for each committed utterance, the server sends events in this order:
//!
//! | `type` | Payload |
//! |---|---|
//! | `turn_started` | `turn` |
//! | `transcript_final` | `turn`, `text` |
//! | `reply_delta` | `turn`, `text` (zero or more) |
//! | *binary* | `AudioOut` playout frames, one binary message each |
//! | `turn_done` | `turn`, `audio_frames`, `dropped_frames`, `dropped_ms`, `dropped_events`, `sidecar_dropped_frames`, `interrupted`, `ingress_dropped_frames`, `ingress_dropped_ms` |
//!
//! A failed turn instead carries:
//!
//! ```json
//! {"type":"error","class":"transient","message":"…"}
//! ```
//!
//! `class` is one of `transient`, `fatal`, `protocol`, `cancelled`, or
//! `capacity`. The server never echoes client bytes in an error.
//!
//! ## Client → server
//!
//! | Frame | Meaning |
//! |---|---|
//! | `{"type":"utterance_end"}` | Commit the audio received since the previous commit into one turn |
//! | `{"type":"barge_in"}` | Ask the live turn to stop; a no-op with no turn in flight |
//! | `{"type":"close"}` | Close the session gracefully |
//! | *binary* | s16le mono PCM; one message is one ingress frame |
//!
//! Unknown `type` values, unknown fields, malformed JSON, and odd-length
//! binary payloads are protocol violations: the server sends an `error` event
//! with `class: "protocol"` and closes with code `1008`, leaving session state
//! untouched, and a second `utterance_end` received while a turn is live is
//! answered the same way with an `error` event (`class: "protocol"`) and a
//! `1008` close.
//!
//! # Bounded ingress
//!
//! Client audio is buffered in a message-granular ring bounded by
//! [`WsConfig::ingress_capacity_samples`] (two seconds by default). Overflow
//! follows [`WsConfig::ingress_overflow`]: `DropOldest` evicts whole messages,
//! `DropNewest` rejects the incoming one. Every loss increments the counters
//! echoed in the next `turn_done` as `ingress_dropped_frames` and
//! `ingress_dropped_ms`; memory per connection stays bounded by the configured
//! capacity.
//!
//! # Turn lifecycle and barge-in
//!
//! [`Session::run_turn`](parlevox_runtime::Session::run_turn) borrows the
//! session for the whole turn, so the connection selects over the socket and
//! the pinned turn future. Session events are drained and flushed in order
//! immediately after the turn future resolves. A `barge_in` received while the
//! turn is live travels through the session's public
//! [`BargeInHandle`](parlevox_runtime::BargeInHandle) — the same cancellation
//! path [`Session::request_barge_in`](parlevox_runtime::Session::request_barge_in)
//! uses, and the only one that can be called from another task while the turn
//! holds the session. A `barge_in` with no live turn calls
//! [`Session::request_barge_in`](parlevox_runtime::Session::request_barge_in)
//! and reports `Ok(false)` internally.
//!
//! `close` or server shutdown received mid-turn asks the turn to unwind
//! through that handle, flushes its terminal events, and only then sends the
//! close frame.

#![warn(missing_docs)]

pub mod config;
pub mod ingress;
pub mod protocol;
pub mod server;

pub use config::{
    WsConfig, DEFAULT_CHANNELS, DEFAULT_INGRESS_SECONDS, DEFAULT_MAX_MESSAGE_BYTES,
    DEFAULT_SAMPLE_RATE,
};
pub use ingress::{IngressDrops, IngressRing, IngressStats};
pub use protocol::{class_name, parse_control, ClientControl, ServerEvent, PROTOCOL_VERSION};
pub use server::{SessionFactory, WsTransport};
