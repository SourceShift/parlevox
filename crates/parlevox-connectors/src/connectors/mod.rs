//! Connectors that implement the provider modality traits over HTTP.
//!
//! Each connector is selected from `adapter.kind`, never from a provider name.
//! The OpenAI family is the first family; a vendor that speaks the same wire
//! shapes needs data, not code.

mod openai_chat;
mod openai_stt;
mod openai_tts;

pub use openai_chat::OpenAiChat;
pub use openai_stt::OpenAiStt;
pub use openai_tts::OpenAiTts;
