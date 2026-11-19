//! Wire protocol between the Rust core and Python plugins.
//!
//! Design constraints (from the comparison's design rules):
//! - **streaming, never whole-turn**: audio is a first-class frame type, sent
//!   per chunk, not base64 inside a JSON response;
//! - **bounded**: every frame is length-checked against [`MAX_FRAME_BYTES`]
//!   before allocation, so a buggy peer cannot make the core allocate gigabytes;
//! - **cancellable**: every request carries an id; `Cancel` is its own message
//!   and may arrive while the plugin is still emitting.

pub mod frame;
pub mod message;

pub use frame::{read_frame, write_frame, AudioHeader, Frame, FrameError, MAX_FRAME_BYTES};
pub use message::{
    ErrorClassWire, ModalityWire, PluginEvent, PluginRequest, VadEventWire, PROTOCOL_VERSION,
    SIDECAR_ABI,
};
