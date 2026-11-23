//! Versioned observation envelope for the session trace.
//!
//! An [`Observation`] is one JSONL line in a session trace: monotonic
//! milliseconds from the session epoch, a session-unique event id, the turn and
//! generation it belongs to, and a typed [`ObservationKind`]. It deliberately
//! carries no wall-clock time — wall-clock time is metadata for the log itself,
//! while the durations latency analysis depends on must come from the monotonic
//! clock so an NTP correction can never reorder them.
//!
//! # Trace-file contract
//!
//! A trace file begins with exactly one header line carrying
//! [`OBSERVATION_SCHEMA_VERSION`]:
//!
//! ```text
//! {"observation_schema_version":1}
//! ```
//!
//! Every following line is one compact JSON [`Observation`], with no embedded
//! newlines. A reader that does not understand the version must refuse the file
//! rather than guess at field meanings; the constant is the only place the
//! version is defined.

use serde::{Deserialize, Serialize};

use crate::clock::MonoClock;
use crate::error::ErrorClass;
use crate::turn::TurnEventKind;

/// The schema version written into the first line of every trace file.
///
/// The header line is `{"observation_schema_version":1}` (compact JSON, one
/// line). Bump this constant when the meaning of a field changes; additive
/// variants in [`ObservationKind`] do not require a bump.
pub const OBSERVATION_SCHEMA_VERSION: u64 = 1;

/// One typed event in a session trace.
///
/// The fields are declared in the order they serialize, and that order is part
/// of the stable wire form. Unknown fields are rejected, so a typo in a hand
/// written trace line fails loudly instead of being ignored.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    /// Milliseconds elapsed on the session's monotonic clock.
    pub t_ms: u64,
    /// Session-unique, monotonically increasing event id.
    pub event_id: u64,
    /// Index of the user turn this observation belongs to.
    pub turn: u64,
    /// Reply generation; increments when a barge-in supersedes a reply.
    pub generation: u64,
    /// What happened.
    pub kind: ObservationKind,
}

/// The typed payload of an [`Observation`].
///
/// The representation is externally tagged snake_case, chosen so a JSONL line
/// stays flat and unambiguous: a unit kind serializes as the bare string
/// (`"vad_started"`) and a data kind as a one-key object
/// (`{"final_transcript":{"text":"hi"}}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationKind {
    /// A final transcript for the turn was produced.
    FinalTranscript { text: String },
    /// A partial transcript arrived.
    PartialTranscript { text: String },
    /// Voice activity started.
    VadStarted,
    /// Voice activity stopped.
    VadStopped,
    /// One streamed reply delta.
    ReplyDelta { text: String },
    /// An audio frame was queued for playout. `pcm_samples` is the frame length
    /// in samples; the PCM payload itself is not traced.
    AudioFrameQueued {
        sample_rate: u32,
        channels: u16,
        pcm_samples: u64,
    },
    /// The reply finished.
    ReplyFinished,
    /// A barge-in was reported.
    BargeInReported,
    /// A cancellation was observed.
    CancellationObserved,
    /// A classified failure.
    Failure { class: ErrorClass, message: String },
    /// A turn-machine event, for correlation with
    /// [`crate::turn::TurnEventKind`].
    Control { event: TurnEventKind },
}

/// Assigns monotonic event ids and monotonic timestamps in one place.
///
/// One sequencer belongs to one session, so ids are unique within a trace and
/// every timestamp comes from the same epoch.
#[derive(Debug, Default)]
pub struct ObservationSequencer {
    clock: MonoClock,
    next_event_id: u64,
}

impl ObservationSequencer {
    /// Create a sequencer over `clock`, starting at event id `0`.
    #[must_use]
    pub fn new(clock: MonoClock) -> Self {
        ObservationSequencer {
            clock,
            next_event_id: 0,
        }
    }

    /// Stamp `kind` with the next event id and the clock's current time.
    ///
    /// The id counter saturates instead of wrapping, so a trace never repeats an
    /// id even after an implausible number of events.
    #[must_use]
    pub fn stamp(&mut self, turn: u64, generation: u64, kind: ObservationKind) -> Observation {
        let event_id = self.next_event_id;
        self.next_event_id = self.next_event_id.saturating_add(1);
        Observation {
            t_ms: self.clock.elapsed_ms(),
            event_id,
            turn,
            generation,
            kind,
        }
    }
}
