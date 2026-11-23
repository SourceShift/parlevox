//! `parlevox-runtime` — the session actor.
//!
//! One session owns one turn state machine, one monotonic clock, bounded audio
//! rings, and a resolved provider set. The turn loop is transport-agnostic and
//! hot-path-only: it never spawns a process and never imports Python.

pub mod context;
pub mod events;
pub mod pipeline;
pub mod recorder;
pub mod registry;
pub mod session;
pub mod strategy;

/// Opt-in OpenTelemetry turn tracing, enabled by the `otel` feature.
#[cfg(feature = "otel")]
mod telemetry;

pub use context::{ChatContext, ContextStats};
pub use events::{
    observation_kind, EventPushOutcome, EventQueue, EventQueueStats, SessionEvent,
    TimedSessionEvent,
};
pub use parlevox_core::{CreditHandle, CreditSnapshot, CreditStats, CreditWait};
pub use pipeline::{PipelineStats, SentenceQueue, SentenceSplitter};
pub use recorder::{Recorder, RecorderConfig, RecorderStats, REDACTED};
pub use registry::{ProviderNames, ProviderRegistry, Providers};
pub use session::{
    demo_utterance, BargeInHandle, InterruptionRecovery, PartialResume, RecoveryStats, Session,
    SessionConfig, SessionCredits, SpeculationStats, SpeculativeStart, TurnReport,
};
pub use strategy::{StrategyStats, TurnContext, TurnDecision, TurnStrategies, TurnStrategy};
