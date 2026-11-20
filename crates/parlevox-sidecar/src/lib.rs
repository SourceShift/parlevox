//! Optional Python plugin host.
//!
//! This crate is the only place in the workspace that spawns a process. It
//! adapts a plugin speaking [`parlevox_ipc`] to the same provider traits the
//! in-process fakes implement, so the runtime never knows the difference.

pub mod client;
pub mod config;
pub mod pool;
pub mod provider;
pub mod supervisor;

pub use client::{Incoming, RequestHandle, SidecarClient};
pub use config::{
    resolve_sdk_path, sdk_is_available, SdkLookupError, SdkSearch, SidecarConfig, SDK_ENV_VAR,
};
pub use pool::{
    IdleEntry, IdleQueue, PoolConfig, PoolLease, PoolStats, PoolStatsSnapshot, PushOutcome,
    RefusalReason, SidecarPool, TakeOutcome,
};
pub use provider::{SidecarLlm, SidecarStt, SidecarTts};
pub use supervisor::{RestartPolicy, SidecarSupervisor};
