//! Typed session events.
//!
//! These are the only things a transport (CLI, websocket, test harness) needs
//! to render a turn. Audio is carried as an owned [`AudioFrame`], never as
//! base64 or whole-turn JSON.

use voicekit_core::{AudioFrame, ErrorClass};

#[derive(Debug, Clone, PartialEq)]
pub enum SessionEvent {
    /// A transcript was committed and a turn index opened.
    TurnStarted { turn: u64 },
    /// The committed user transcript for `turn`.
    TranscriptFinal { turn: u64, text: String },
    /// One streamed assistant delta.
    ReplyDelta { turn: u64, text: String },
    /// One audio frame ready for playout.
    AudioOut { turn: u64, frame: AudioFrame },
    /// The turn finished. `interrupted` is true after a barge-in.
    TurnDone {
        turn: u64,
        audio_frames: u64,
        dropped_frames: u64,
        dropped_ms: u64,
        interrupted: bool,
    },
    /// The turn failed with a classified error.
    Failed { class: ErrorClass, message: String },
}

impl SessionEvent {
    /// Short label used by the CLI and tests.
    pub fn kind(&self) -> &'static str {
        match self {
            SessionEvent::TurnStarted { .. } => "TurnStarted",
            SessionEvent::TranscriptFinal { .. } => "TranscriptFinal",
            SessionEvent::ReplyDelta { .. } => "ReplyDelta",
            SessionEvent::AudioOut { .. } => "AudioOut",
            SessionEvent::TurnDone { .. } => "TurnDone",
            SessionEvent::Failed { .. } => "Failed",
        }
    }
}
