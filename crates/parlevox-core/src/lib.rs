//! `parlevox-core` — transport- and provider-agnostic session runtime.
//!
//! Structural rules this crate exists to enforce:
//! - every duration is measured on a monotonic clock ([`clock`]);
//! - every audio path is a bounded ring with an explicit overflow policy
//!   ([`audio::AudioRing`]);
//! - a turn is an explicit state machine, not a bag of booleans ([`turn`]);
//! - errors carry a recoverability class so callers never guess ([`error`]).

pub mod audio;
pub mod clock;
pub mod error;
pub mod observation;
pub mod turn;

pub use audio::{
    AudioFrame, AudioRing, CreditHandle, CreditSnapshot, CreditStats, CreditWait, OverflowPolicy,
    PushOutcome, RingStats,
};
pub use clock::{DurationMs, MonoClock, Timestamp};
pub use error::{ErrorClass, ParlevoxError};
pub use observation::{
    Observation, ObservationKind, ObservationSequencer, OBSERVATION_SCHEMA_VERSION,
};
pub use turn::{
    AudioEmission, FloorOwner, TurnEvent, TurnEventKind, TurnMachine, TurnState, ALL_EVENT_KINDS,
    ALL_STATES, LEGAL_TRANSITIONS,
};
