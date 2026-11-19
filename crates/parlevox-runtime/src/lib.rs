//! `parlevox-runtime` — the session actor.
//!
//! One session owns one turn state machine, one monotonic clock, bounded audio
//! rings, and a resolved provider set. The turn loop is transport-agnostic and
//! hot-path-only: it never spawns a process and never imports Python.

pub mod events;
pub mod registry;
pub mod session;

pub use events::{EventPushOutcome, EventQueue, EventQueueStats, SessionEvent};
pub use registry::{ProviderNames, ProviderRegistry, Providers};
pub use session::{demo_utterance, Session, SessionConfig, TurnReport};
