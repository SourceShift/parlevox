//! `parlevox-runtime` — the session actor.
//!
//! One session owns one turn state machine, one monotonic clock, bounded audio
//! rings, and a resolved provider set. The turn loop is transport-agnostic and
//! hot-path-only: it never spawns a process and never imports Python.

pub mod context;
pub mod events;
pub mod pipeline;
pub mod registry;
pub mod session;

/// Opt-in OpenTelemetry turn tracing, enabled by the `otel` feature.
#[cfg(feature = "otel")]
mod telemetry;

pub use context::{ChatContext, ContextStats};
pub use events::{EventPushOutcome, EventQueue, EventQueueStats, SessionEvent};
pub use parlevox_core::{CreditHandle, CreditSnapshot, CreditStats, CreditWait};
pub use pipeline::{PipelineStats, SentenceQueue, SentenceSplitter};
pub use registry::{ProviderNames, ProviderRegistry, Providers};
pub use session::{demo_utterance, Session, SessionConfig, SessionCredits, TurnReport};
