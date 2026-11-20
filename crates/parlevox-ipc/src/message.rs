//! Control messages shared by the Rust core and a Python sidecar plugin.
//!
//! The frame layer ([`crate::frame`]) carries bytes; this module gives those
//! bytes meaning. Control is JSON, audio is binary: a `Synthesize` request
//! opens a request id, the plugin answers with `AudioStart` followed by binary
//! audio frames carrying the same request id, and `Done` closes the stream.
//! `Cancel` is a first-class message so a barge-in never requires killing the
//! plugin process.

use serde::{Deserialize, Serialize};

use parlevox_core::ErrorClass;
use parlevox_providers::ChatMessage;

/// Wire version. Bump for any breaking change to the message shapes below.
pub const PROTOCOL_VERSION: u32 = 1;

/// ABI identifier exchanged during the `hello`/`ready` handshake.
pub const SIDECAR_ABI: &str = "parlevox-sidecar/1";

/// Provider modality, as it appears on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModalityWire {
    Stt,
    Llm,
    Tts,
    Vad,
}

/// Serialisable mirror of [`ErrorClass`] so a plugin can report why it failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorClassWire {
    Transient,
    Fatal,
    Protocol,
    Cancelled,
    Capacity,
}

impl From<ErrorClass> for ErrorClassWire {
    fn from(class: ErrorClass) -> Self {
        match class {
            ErrorClass::Transient => ErrorClassWire::Transient,
            ErrorClass::Fatal => ErrorClassWire::Fatal,
            ErrorClass::Protocol => ErrorClassWire::Protocol,
            ErrorClass::Cancelled => ErrorClassWire::Cancelled,
            ErrorClass::Capacity => ErrorClassWire::Capacity,
        }
    }
}

impl From<ErrorClassWire> for ErrorClass {
    fn from(class: ErrorClassWire) -> Self {
        match class {
            ErrorClassWire::Transient => ErrorClass::Transient,
            ErrorClassWire::Fatal => ErrorClass::Fatal,
            ErrorClassWire::Protocol => ErrorClass::Protocol,
            ErrorClassWire::Cancelled => ErrorClass::Cancelled,
            ErrorClassWire::Capacity => ErrorClass::Capacity,
        }
    }
}

/// A request from the Rust core to a plugin.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PluginRequest {
    /// Opening handshake. Must be the first message on the stream.
    Hello {
        protocol_version: u32,
        abi: String,
        /// Optional list of modalities the core intends to use.
        #[serde(default)]
        modalities: Vec<ModalityWire>,
    },
    /// Ask the plugin to transcribe audio frames that follow.
    Transcribe {
        request_id: String,
        sample_rate: u32,
        channels: u16,
    },
    /// Ask for a streamed completion.
    Generate {
        request_id: String,
        prompt: String,
        /// Prior turns, oldest first, never including the current `prompt`.
        ///
        /// Older plugins ignore unknown fields and older core builds omit the
        /// field entirely, so `#[serde(default)]` keeps both directions
        /// compatible: a missing history decodes as an empty list.
        #[serde(default)]
        history: Vec<ChatMessage>,
    },
    /// Ask for streamed audio for `text`.
    Synthesize {
        request_id: String,
        text: String,
        sample_rate: u32,
        channels: u16,
    },
    /// Ask for voice-activity events over audio frames that follow.
    Analyze {
        request_id: String,
        sample_rate: u32,
        channels: u16,
    },
    /// Signal that all audio frames for `request_id` have been sent.
    AudioEnd { request_id: String },
    /// Cancel an in-flight request without killing the process.
    Cancel { request_id: String },
    /// Clear every pending request and cancellation tombstone, then answer
    /// with `ready`. Additive: a plugin that does not advertise the `reset`
    /// capability can ignore it, and the client never sends it to such a
    /// plugin.
    Reset,
    /// Graceful process shutdown.
    Shutdown,
}

impl PluginRequest {
    /// The request this message opens or controls, when it has one.
    pub fn request_id(&self) -> Option<&str> {
        match self {
            PluginRequest::Hello { .. } | PluginRequest::Reset | PluginRequest::Shutdown => None,
            PluginRequest::AudioEnd { request_id }
            | PluginRequest::Cancel { request_id }
            | PluginRequest::Transcribe { request_id, .. }
            | PluginRequest::Generate { request_id, .. }
            | PluginRequest::Synthesize { request_id, .. }
            | PluginRequest::Analyze { request_id, .. } => Some(request_id),
        }
    }

    /// Whether audio frames follow and an explicit `AudioEnd` must be sent.
    pub fn needs_audio_end(&self) -> bool {
        matches!(
            self,
            PluginRequest::Transcribe { .. } | PluginRequest::Analyze { .. }
        )
    }
}

/// An event from a plugin to the Rust core.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PluginEvent {
    /// Handshake acknowledgement; carries the plugin's manifest.
    Ready {
        protocol_version: u32,
        abi: String,
        plugin: String,
        /// Free-form manifest: kept as JSON so the wire layer does not pin the
        /// provider crate's manifest shape.
        manifest: serde_json::Value,
        /// Named session-level capabilities, e.g. `"reset"`. Additive: a
        /// plugin that predates the field omits it and decodes as empty, so it
        /// is never pooled.
        #[serde(default)]
        capabilities: Vec<String>,
    },
    /// A partial transcript.
    Partial { request_id: String, text: String },
    /// A committed transcript.
    Final { request_id: String, text: String },
    /// One streamed completion chunk.
    Delta { request_id: String, text: String },
    /// Audio for `request_id` follows as binary frames.
    AudioStart {
        request_id: String,
        sample_rate: u32,
        channels: u16,
    },
    /// A voice-activity transition at `timestamp_ms` on the plugin timeline.
    Vad {
        request_id: String,
        event: VadEventWire,
        timestamp_ms: u64,
    },
    /// Successful end of a request's stream.
    ///
    /// `dropped_frames` is additive: plugins that predate it omit the field and
    /// decode as zero, so an older sidecar still works.
    Done {
        request_id: String,
        #[serde(default)]
        dropped_frames: u64,
    },
    /// Acknowledgement that a `Cancel` was observed.
    CancelAck { request_id: String },
    /// A classified failure. `request_id` is empty for session-level errors.
    Error {
        request_id: String,
        class: ErrorClassWire,
        message: String,
    },
}

/// Voice-activity transition on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VadEventWire {
    SpeechStart,
    SpeechEnd,
}

impl PluginEvent {
    /// The request this event belongs to, when it has one.
    pub fn request_id(&self) -> Option<&str> {
        match self {
            PluginEvent::Ready { .. } => None,
            PluginEvent::Partial { request_id, .. }
            | PluginEvent::Final { request_id, .. }
            | PluginEvent::Delta { request_id, .. }
            | PluginEvent::AudioStart { request_id, .. }
            | PluginEvent::Vad { request_id, .. }
            | PluginEvent::Done { request_id, .. }
            | PluginEvent::CancelAck { request_id }
            | PluginEvent::Error { request_id, .. } => Some(request_id),
        }
    }

    /// Whether this event terminates the request's stream.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            PluginEvent::Done { .. } | PluginEvent::CancelAck { .. } | PluginEvent::Error { .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parlevox_providers::Role;

    #[test]
    fn generate_history_roundtrips_and_legacy_frames_default_to_empty() {
        let request = PluginRequest::Generate {
            request_id: "r1".into(),
            prompt: "now".into(),
            history: vec![
                ChatMessage {
                    role: Role::User,
                    text: "before".into(),
                },
                ChatMessage {
                    role: Role::Assistant,
                    text: "earlier reply".into(),
                },
            ],
        };
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"history\":["));
        assert_eq!(
            serde_json::from_str::<PluginRequest>(&json).unwrap(),
            request
        );

        // A frame from a core build that predates `history` still decodes.
        let legacy: PluginRequest =
            serde_json::from_str(r#"{"type":"generate","request_id":"r1","prompt":"now"}"#)
                .unwrap();
        assert_eq!(
            legacy,
            PluginRequest::Generate {
                request_id: "r1".into(),
                prompt: "now".into(),
                history: Vec::new(),
            }
        );
    }

    #[test]
    fn hello_and_ready_roundtrip() {
        let hello = PluginRequest::Hello {
            protocol_version: PROTOCOL_VERSION,
            abi: SIDECAR_ABI.into(),
            modalities: vec![ModalityWire::Tts],
        };
        let json = serde_json::to_string(&hello).unwrap();
        assert!(json.contains("\"type\":\"hello\""));
        assert_eq!(serde_json::from_str::<PluginRequest>(&json).unwrap(), hello);

        let ready = PluginEvent::Ready {
            protocol_version: PROTOCOL_VERSION,
            abi: SIDECAR_ABI.into(),
            plugin: "fake".into(),
            manifest: serde_json::json!({"name": "fake", "modality": "tts"}),
            capabilities: vec!["reset".into()],
        };
        let json = serde_json::to_string(&ready).unwrap();
        assert!(json.contains("\"type\":\"ready\""));
        assert!(json.contains("\"capabilities\":[\"reset\"]"));
        assert_eq!(serde_json::from_str::<PluginEvent>(&json).unwrap(), ready);
    }

    #[test]
    fn reset_roundtrips_without_a_request_id() {
        let reset = PluginRequest::Reset;
        let json = serde_json::to_string(&reset).unwrap();
        assert_eq!(json, r#"{"type":"reset"}"#);
        assert_eq!(serde_json::from_str::<PluginRequest>(&json).unwrap(), reset);
        assert_eq!(reset.request_id(), None);
        assert!(!reset.needs_audio_end());
    }

    #[test]
    fn legacy_ready_without_capabilities_decodes_as_empty() {
        // A plugin built before the additive field still completes a
        // handshake; an empty capability list means the client must never
        // pool it.
        let legacy: PluginEvent = serde_json::from_str(
            r#"{"type":"ready","protocol_version":1,"abi":"parlevox-sidecar/1",
                "plugin":"old","manifest":{"name":"old","modality":"tts"}}"#,
        )
        .unwrap();
        match legacy {
            PluginEvent::Ready { capabilities, .. } => {
                assert!(capabilities.is_empty());
            }
            other => panic!("expected ready, got {other:?}"),
        }
    }

    #[test]
    fn error_class_wire_mirrors_core() {
        for class in [
            ErrorClass::Transient,
            ErrorClass::Fatal,
            ErrorClass::Protocol,
            ErrorClass::Cancelled,
            ErrorClass::Capacity,
        ] {
            let wire = ErrorClassWire::from(class);
            assert_eq!(ErrorClass::from(wire), class);
        }
    }

    #[test]
    fn terminal_events_are_classified() {
        let done = PluginEvent::Done {
            request_id: "r1".into(),
            dropped_frames: 0,
        };
        assert!(done.is_terminal());
        assert_eq!(done.request_id(), Some("r1"));

        let delta = PluginEvent::Delta {
            request_id: "r1".into(),
            text: "a".into(),
        };
        assert!(!delta.is_terminal());
    }

    #[test]
    fn done_carries_dropped_frames_additively() {
        let done = PluginEvent::Done {
            request_id: "r1".into(),
            dropped_frames: 7,
        };
        let json = serde_json::to_string(&done).unwrap();
        assert!(json.contains("\"dropped_frames\":7"));
        assert_eq!(serde_json::from_str::<PluginEvent>(&json).unwrap(), done);

        // An older plugin that omits the field still decodes, as zero.
        let legacy: PluginEvent =
            serde_json::from_str(r#"{"type":"done","request_id":"r1"}"#).unwrap();
        assert_eq!(
            legacy,
            PluginEvent::Done {
                request_id: "r1".into(),
                dropped_frames: 0,
            }
        );
    }

    #[test]
    fn error_roundtrips_with_class() {
        let ev = PluginEvent::Error {
            request_id: "r2".into(),
            class: ErrorClassWire::Fatal,
            message: "boom".into(),
        };
        let json = serde_json::to_string(&ev).unwrap();
        assert!(json.contains("\"class\":\"fatal\""));
        assert_eq!(serde_json::from_str::<PluginEvent>(&json).unwrap(), ev);
    }
}
