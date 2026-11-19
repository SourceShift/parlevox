//! Optional Python plugin host.
//!
//! This crate is the only place in the workspace that spawns a process. It
//! adapts a plugin speaking [`voicekit_ipc`] to the same provider traits the
//! in-process fakes implement, so the runtime never knows the difference.

pub mod client;
pub mod config;
pub mod provider;
pub mod supervisor;

pub use client::{Incoming, RequestHandle, SidecarClient};
pub use config::{sdk_is_available, SidecarConfig};
pub use provider::{SidecarLlm, SidecarStt, SidecarTts};
pub use supervisor::{RestartPolicy, SidecarSupervisor};
