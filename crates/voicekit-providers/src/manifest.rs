//! Provider manifests: one small descriptor per provider.
//!
//! A manifest travels over the sidecar handshake and is what the runtime uses
//! to select a provider. Adding a provider is adding a manifest plus the trait
//! implementation — the runtime and the turn state machine never change.

use serde::{Deserialize, Serialize};
use voicekit_core::{ErrorClass, VoicekitError};

/// What a provider does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    Stt,
    Llm,
    Tts,
    Vad,
}

impl Modality {
    pub fn as_str(self) -> &'static str {
        match self {
            Modality::Stt => "stt",
            Modality::Llm => "llm",
            Modality::Tts => "tts",
            Modality::Vad => "vad",
        }
    }
}

/// Static description of a provider.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderManifest {
    /// Stable identifier used for selection, e.g. `"fake-tts"`.
    pub name: String,
    pub modality: Modality,
    /// Named behaviours the provider supports: `"streaming"`, `"cancel"`, ...
    pub capabilities: Vec<String>,
    /// JSON schema-ish description of the provider's configuration object.
    pub config_schema: serde_json::Value,
}

impl ProviderManifest {
    pub fn new(name: impl Into<String>, modality: Modality) -> Self {
        ProviderManifest {
            name: name.into(),
            modality,
            capabilities: Vec::new(),
            config_schema: serde_json::json!({"type": "object", "properties": {}}),
        }
    }

    pub fn with_capability(mut self, capability: impl Into<String>) -> Self {
        self.capabilities.push(capability.into());
        self
    }

    pub fn supports(&self, capability: &str) -> bool {
        self.capabilities.iter().any(|c| c == capability)
    }

    /// Reject manifests that cannot be selected unambiguously.
    pub fn validate(&self) -> Result<(), VoicekitError> {
        if self.name.trim().is_empty() {
            return Err(VoicekitError::Protocol("manifest name is empty".into()));
        }
        if !self.config_schema.is_object() {
            return Err(VoicekitError::Protocol(format!(
                "manifest '{}' config_schema must be a JSON object",
                self.name
            )));
        }
        Ok(())
    }

    /// Convert a validation failure into the class the session should act on.
    pub fn class_on_invalid(&self) -> ErrorClass {
        ErrorClass::Protocol
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_manifest_has_name_and_object_schema() {
        let m = ProviderManifest::new("fake-tts", Modality::Tts)
            .with_capability("streaming")
            .with_capability("cancel");
        assert!(m.validate().is_ok());
        assert!(m.supports("cancel"));
        assert!(!m.supports("network"));
        assert_eq!(m.modality.as_str(), "tts");
    }

    #[test]
    fn empty_name_is_a_protocol_error() {
        let m = ProviderManifest::new("  ", Modality::Stt);
        let err = m.validate().unwrap_err();
        assert_eq!(err.class(), ErrorClass::Protocol);
    }

    #[test]
    fn non_object_schema_is_rejected() {
        let mut m = ProviderManifest::new("x", Modality::Llm);
        m.config_schema = serde_json::json!([1, 2, 3]);
        assert!(m.validate().is_err());
    }
}
