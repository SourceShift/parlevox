//! `voicekit-core` — transport- and provider-agnostic session runtime.
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
pub mod turn;

pub use audio::{AudioFrame, AudioRing, OverflowPolicy, PushOutcome, RingStats};
pub use clock::{DurationMs, MonoClock, Timestamp};
pub use error::{ErrorClass, VoicekitError};
pub use turn::{TurnEvent, TurnMachine, TurnState};
