//! Wire protocol types.
//!
//! Control messages are JSON text frames with a `type` discriminator. Unknown
//! fields are rejected, so a client cannot silently rely on a field this
//! server does not implement. Client audio and server playout are binary
//! frames of signed 16-bit little-endian PCM; they are intentionally not JSON,
//! so a whole turn of audio never becomes a base64 string.

use parlevox_core::{ErrorClass, ParlevoxError};
use serde::{Deserialize, Serialize};

/// Version reported in the server's opening `ready` message.
pub const PROTOCOL_VERSION: u16 = 1;

/// A control message sent by the client.
///
/// Every variant carries no payload. Any unknown `type`, any unknown field, and
/// any malformed JSON is a protocol violation; [`parse_control`] enforces that
/// by deserialising the tag through a [`deny_unknown_fields`] struct, because
/// serde's internally tagged enums ignore extra fields.
///
/// [`deny_unknown_fields`]: https://serde.rs/container-attrs.html#deny_unknown_fields
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClientControl {
    /// Commit the audio received since the last commit into one turn.
    UtteranceEnd,
    /// Ask the live turn to stop; a no-op when no turn is in flight.
    BargeIn,
    /// Close the session gracefully.
    Close,
}

/// The strict shape every client control frame must have: exactly one string
/// field, `type`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawControl {
    #[serde(rename = "type")]
    kind: String,
}

/// A JSON event sent by the server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerEvent {
    /// First message on every connection, sent before any other event.
    Ready {
        /// Protocol version this server speaks.
        protocol_version: u16,
    },
    /// A committed utterance opened a turn.
    TurnStarted {
        /// One-based turn index, independent per connection.
        turn: u64,
    },
    /// The committed user transcript for `turn`.
    TranscriptFinal {
        /// The turn this transcript belongs to.
        turn: u64,
        /// The transcript text.
        text: String,
    },
    /// One streamed assistant delta.
    ReplyDelta {
        /// The turn this delta belongs to.
        turn: u64,
        /// The delta text.
        text: String,
    },
    /// The turn finished, with its playout and drop accounting.
    TurnDone {
        /// The turn that finished.
        turn: u64,
        /// Audio frames emitted for this turn.
        audio_frames: u64,
        /// Audio frames the session's playout ring dropped.
        dropped_frames: u64,
        /// Duration dropped from the playout ring, in milliseconds.
        dropped_ms: u64,
        /// Events the session's bounded event queue dropped.
        dropped_events: u64,
        /// Frames a sidecar adapter reported dropping for this turn.
        sidecar_dropped_frames: u64,
        /// True when a barge-in ended the turn early.
        interrupted: bool,
        /// Ingress messages lost since the previous commit.
        ingress_dropped_frames: u64,
        /// Duration of those lost ingress messages, in milliseconds.
        ingress_dropped_ms: u64,
    },
    /// A classified error. `class` is one of `transient`, `fatal`,
    /// `protocol`, `cancelled`, or `capacity`.
    Error {
        /// Lowercase class name from [`ErrorClass`].
        class: String,
        /// Human-readable, non-secret description.
        message: String,
    },
}

impl ServerEvent {
    /// The opening `ready` event for this protocol version.
    #[must_use]
    pub fn ready() -> Self {
        ServerEvent::Ready {
            protocol_version: PROTOCOL_VERSION,
        }
    }

    /// An error event carrying the lowercase name of `class`.
    #[must_use]
    pub fn error(class: ErrorClass, message: impl Into<String>) -> Self {
        ServerEvent::Error {
            class: class_name(class).to_string(),
            message: message.into(),
        }
    }

    /// Serialise this event as one JSON text frame.
    ///
    /// # Errors
    ///
    /// Returns a protocol-class [`ParlevoxError`] if the event cannot be
    /// encoded. The event set is built entirely from strings and integers, so
    /// this is a defensive path, not an expected one.
    pub fn to_json(&self) -> Result<String, ParlevoxError> {
        serde_json::to_string(self)
            .map_err(|err| ParlevoxError::Protocol(format!("failed to encode server event: {err}")))
    }
}

/// Lowercase wire name for an [`ErrorClass`].
#[must_use]
pub fn class_name(class: ErrorClass) -> &'static str {
    match class {
        ErrorClass::Transient => "transient",
        ErrorClass::Fatal => "fatal",
        ErrorClass::Protocol => "protocol",
        ErrorClass::Cancelled => "cancelled",
        ErrorClass::Capacity => "capacity",
    }
}

/// Parse one client control frame.
///
/// Returns [`None`] for malformed JSON, an unknown `type`, a non-string
/// `type`, or any field besides `type`. The caller turns that into a protocol
/// error event and a policy close.
#[must_use]
pub fn parse_control(text: &str) -> Option<ClientControl> {
    let raw: RawControl = serde_json::from_str(text).ok()?;
    match raw.kind.as_str() {
        "utterance_end" => Some(ClientControl::UtteranceEnd),
        "barge_in" => Some(ClientControl::BargeIn),
        "close" => Some(ClientControl::Close),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ready_matches_the_documented_wire_form() {
        assert_eq!(
            ServerEvent::ready().to_json().expect("serialises"),
            r#"{"type":"ready","protocol_version":1}"#
        );
    }

    #[test]
    fn control_messages_parse_by_type_only() {
        assert_eq!(
            parse_control(r#"{"type":"utterance_end"}"#),
            Some(ClientControl::UtteranceEnd)
        );
        assert_eq!(
            parse_control(r#"{"type":"barge_in"}"#),
            Some(ClientControl::BargeIn)
        );
        assert_eq!(
            parse_control(r#"{"type":"close"}"#),
            Some(ClientControl::Close)
        );
    }

    #[test]
    fn unknown_or_extra_fields_are_rejected() {
        assert!(parse_control(r#"{"type":"not_a_control"}"#).is_none());
        assert!(parse_control(r#"{"type":"utterance_end","extra":1}"#).is_none());
        assert!(parse_control(r#"{"type":"utterance_end""#).is_none());
        assert!(parse_control("hello").is_none());
    }

    #[test]
    fn turn_done_carries_session_and_ingress_accounting() {
        let json = ServerEvent::TurnDone {
            turn: 3,
            audio_frames: 2,
            dropped_frames: 0,
            dropped_ms: 0,
            dropped_events: 0,
            sidecar_dropped_frames: 0,
            interrupted: true,
            ingress_dropped_frames: 4,
            ingress_dropped_ms: 24,
        }
        .to_json()
        .expect("serialises");
        let value: serde_json::Value = serde_json::from_str(&json).expect("valid json");
        assert_eq!(value["type"], "turn_done");
        assert_eq!(value["turn"], 3);
        assert_eq!(value["interrupted"], true);
        assert_eq!(value["ingress_dropped_frames"], 4);
        assert_eq!(value["ingress_dropped_ms"], 24);
    }

    #[test]
    fn error_class_names_are_lowercase() {
        for class in [
            ErrorClass::Transient,
            ErrorClass::Fatal,
            ErrorClass::Protocol,
            ErrorClass::Cancelled,
            ErrorClass::Capacity,
        ] {
            let name = class_name(class);
            assert_eq!(name, name.to_lowercase());
        }
    }
}
