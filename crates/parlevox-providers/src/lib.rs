//! Streaming provider traits and deterministic in-process fakes.
//!
//! Every modality returns a stream, never a whole-utterance future. A provider
//! that can only produce a final answer wraps it in a one-item stream; the
//! trait does not change shape. Every in-flight call receives a
//! [`CancellationToken`](tokio_util::sync::CancellationToken) so barge-in can stop work deterministically instead of
//! racing timers.
//!
//! The fakes here are the hot-path default: they make `demo --fake` and the
//! runtime tests deterministic and Python-free.

pub mod breaker;
pub mod fakes;
pub mod fallback;
pub mod manifest;
pub mod retry;
pub mod stream;
pub mod traits;

pub use breaker::{BreakerPolicy, BreakerStats, CircuitBreakingLlm};
pub use fakes::{
    FakeLlm, FakeStt, FakeTts, FakeVad, ProviderCall, ProviderCallLog, FAKE_CHANNELS,
    FAKE_FRAME_MS, FAKE_FRAME_SAMPLES, FAKE_LLM_DELTAS, FAKE_PARTIAL_TRANSCRIPT, FAKE_REPLY,
    FAKE_SAMPLE_RATE, FAKE_TRANSCRIPT, FAKE_TTS_FRAMES,
};
pub use fallback::{
    FallbackLlm, FallbackStats, FallbackStatsSnapshot, FallbackStt, FallbackTts, FallbackVad,
};
pub use manifest::{Modality, ProviderManifest};
pub use retry::{RetryPolicy, RetryStats, RetryingLlm};
pub use stream::cancelable_stream;
pub use traits::{
    AudioChunk, ChatMessage, GenerateRequest, LlmProvider, ProviderStream, Role, SttProvider,
    SynthesizeRequest, TranscriptChunk, TtsProvider, VadEvent, VadProvider, VadTransition,
};
