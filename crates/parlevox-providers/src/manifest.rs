//! Provider manifests: one small descriptor per provider.
//!
//! A manifest travels over the sidecar handshake and is what the runtime uses
//! to select a provider. Adding a provider is adding a manifest plus the trait
//! implementation — the runtime and the turn state machine never change.
//!
//! Version 2 adds an adapter reference, model entries and provider limits. All
//! of the new fields are optional on the wire, so a v1 JSON manifest keeps
//! parsing unchanged.

use std::collections::HashSet;

use parlevox_core::{ErrorClass, ParlevoxError};
use serde::{Deserialize, Serialize};

/// Protocol version assumed when a manifest does not declare one.
pub(crate) fn default_protocol_version() -> u32 {
    1
}

/// What a provider does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    Stt,
    Llm,
    Tts,
    Vad,
}

impl Modality {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Modality::Stt => "stt",
            Modality::Llm => "llm",
            Modality::Tts => "tts",
            Modality::Vad => "vad",
        }
    }
}

/// How a provider is reached: an adapter kind plus an optional endpoint.
///
/// The kind stays a `String` on the wire so a newer descriptor kind does not
/// fail to parse; [`AdapterKind`] is the validated vocabulary a manifest is
/// checked against. Unknown *fields* are rejected so a typo in a descriptor is
/// an error rather than a silently dropped setting.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdapterRef {
    /// Adapter implementation kind, e.g. `"openai-compatible-chat"`.
    #[serde(default)]
    pub kind: String,
    /// Optional base URL or socket endpoint; never a secret.
    #[serde(default)]
    pub endpoint: Option<String>,
}

/// The adapter kinds this build understands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterKind {
    OpenAiCompatibleChat,
    OpenAiAudioStt,
    OpenAiAudioTts,
    DeepgramStreaming,
    ElevenlabsWs,
    CartesiaWs,
    Sidecar,
}

impl AdapterKind {
    /// Every wire name [`AdapterKind`] accepts, in declaration order.
    pub const ALLOWED_KINDS: [&'static str; 7] = [
        "openai-compatible-chat",
        "openai-audio-stt",
        "openai-audio-tts",
        "deepgram-streaming",
        "elevenlabs-ws",
        "cartesia-ws",
        "sidecar",
    ];

    /// Every admissible adapter kind.
    pub const ALL: [AdapterKind; 7] = [
        AdapterKind::OpenAiCompatibleChat,
        AdapterKind::OpenAiAudioStt,
        AdapterKind::OpenAiAudioTts,
        AdapterKind::DeepgramStreaming,
        AdapterKind::ElevenlabsWs,
        AdapterKind::CartesiaWs,
        AdapterKind::Sidecar,
    ];

    /// The wire name of this adapter kind.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            AdapterKind::OpenAiCompatibleChat => "openai-compatible-chat",
            AdapterKind::OpenAiAudioStt => "openai-audio-stt",
            AdapterKind::OpenAiAudioTts => "openai-audio-tts",
            AdapterKind::DeepgramStreaming => "deepgram-streaming",
            AdapterKind::ElevenlabsWs => "elevenlabs-ws",
            AdapterKind::CartesiaWs => "cartesia-ws",
            AdapterKind::Sidecar => "sidecar",
        }
    }
}

impl std::str::FromStr for AdapterKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "openai-compatible-chat" => Ok(AdapterKind::OpenAiCompatibleChat),
            "openai-audio-stt" => Ok(AdapterKind::OpenAiAudioStt),
            "openai-audio-tts" => Ok(AdapterKind::OpenAiAudioTts),
            "deepgram-streaming" => Ok(AdapterKind::DeepgramStreaming),
            "elevenlabs-ws" => Ok(AdapterKind::ElevenlabsWs),
            "cartesia-ws" => Ok(AdapterKind::CartesiaWs),
            "sidecar" => Ok(AdapterKind::Sidecar),
            other => Err(format!("unknown adapter kind '{other}'")),
        }
    }
}

/// Per-model limits used to filter selection.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelLimits {
    /// Maximum prompt tokens the model accepts, when bounded.
    #[serde(default)]
    pub max_input_tokens: Option<u32>,
    /// Sample rates the model accepts, in Hz. Empty means "not declared".
    #[serde(default)]
    pub sample_rates: Vec<u32>,
    /// Channel counts the model accepts. Empty means "not declared".
    #[serde(default)]
    pub channels: Vec<u16>,
}

/// One model served by a provider.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelEntry {
    /// Stable model identifier, e.g. `"gpt-4o-mini"`.
    pub id: String,
    /// Human-facing name; defaults to an empty string when omitted.
    #[serde(default)]
    pub display_name: String,
    /// Named behaviours this model supports.
    #[serde(default)]
    pub capabilities: Vec<String>,
    /// Numeric limits used by selection.
    #[serde(default)]
    pub limits: ModelLimits,
    /// Modality of this model; defaults to the manifest's modality.
    #[serde(default)]
    pub modality: Option<Modality>,
}

/// Provider-wide concurrency limits. Recorded for scheduling; they do not
/// filter selection in F0a.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderLimits {
    /// Maximum number of concurrent calls across the provider.
    #[serde(default)]
    pub max_concurrency: Option<u32>,
    /// Maximum number of in-flight calls across the provider.
    #[serde(default)]
    pub max_inflight: Option<u32>,
}

/// Static description of a provider.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderManifest {
    /// Stable identifier used for selection, e.g. `"fake-tts"`.
    pub name: String,
    pub modality: Modality,
    /// Named behaviours the provider supports: `"streaming"`, `"cancel"`, ...
    pub capabilities: Vec<String>,
    /// JSON schema-ish description of the provider's configuration object.
    pub config_schema: serde_json::Value,
    /// Descriptor format version. Absent in v1 manifests, which default to 1.
    #[serde(default = "default_protocol_version")]
    pub protocol_version: u32,
    /// How to reach the provider. The default means "no adapter declared".
    #[serde(default)]
    pub adapter: AdapterRef,
    /// Models the provider serves.
    #[serde(default)]
    pub models: Vec<ModelEntry>,
    /// Provider-wide limits.
    #[serde(default)]
    pub provider_limits: ProviderLimits,
}

impl ProviderManifest {
    pub fn new(name: impl Into<String>, modality: Modality) -> Self {
        ProviderManifest {
            name: name.into(),
            modality,
            capabilities: Vec::new(),
            config_schema: serde_json::json!({"type": "object", "properties": {}}),
            protocol_version: default_protocol_version(),
            adapter: AdapterRef::default(),
            models: Vec::new(),
            provider_limits: ProviderLimits::default(),
        }
    }

    #[must_use = "builder methods return a modified manifest; chain or assign the result"]
    pub fn with_capability(mut self, capability: impl Into<String>) -> Self {
        self.capabilities.push(capability.into());
        self
    }

    #[must_use]
    pub fn supports(&self, capability: &str) -> bool {
        self.capabilities.iter().any(|c| c == capability)
    }

    /// The union of manifest-level and model-level capabilities.
    #[must_use]
    pub fn capability_union(&self) -> Vec<&str> {
        let mut capabilities: Vec<&str> = self
            .capabilities
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        for model in &self.models {
            capabilities.extend(model.capabilities.iter().map(String::as_str));
        }
        capabilities
    }

    /// Reject manifests that cannot be selected unambiguously.
    ///
    /// # Errors
    ///
    /// Returns a protocol-class error when the name is empty or whitespace,
    /// when `config_schema` is not a JSON object, when an adapter kind is
    /// empty-but-endpointed or unknown, when a model id is empty or repeated,
    /// or when a model modality contradicts the manifest's.
    pub fn validate(&self) -> Result<(), ParlevoxError> {
        if self.name.trim().is_empty() {
            return Err(ParlevoxError::Protocol("manifest name is empty".into()));
        }
        if !self.config_schema.is_object() {
            return Err(ParlevoxError::Protocol(format!(
                "manifest '{}' config_schema must be a JSON object",
                self.name
            )));
        }
        self.validate_adapter()?;
        self.validate_models()?;
        Ok(())
    }

    fn validate_adapter(&self) -> Result<(), ParlevoxError> {
        let kind = self.adapter.kind.trim();
        if kind.is_empty() {
            // An absent adapter is the v1 shape. An endpoint without a kind is
            // not usable and is rejected rather than silently ignored.
            if self.adapter.endpoint.is_some() {
                return Err(ParlevoxError::Protocol(format!(
                    "manifest '{}' declares an adapter endpoint without a kind",
                    self.name
                )));
            }
            return Ok(());
        }
        if kind.parse::<AdapterKind>().is_err() {
            return Err(ParlevoxError::Protocol(format!(
                "manifest '{}' has unknown adapter kind '{}'",
                self.name, self.adapter.kind
            )));
        }
        Ok(())
    }

    fn validate_models(&self) -> Result<(), ParlevoxError> {
        let mut seen = HashSet::with_capacity(self.models.len());
        for model in &self.models {
            if model.id.trim().is_empty() {
                return Err(ParlevoxError::Protocol(format!(
                    "manifest '{}' has a model with an empty id",
                    self.name
                )));
            }
            if !seen.insert(model.id.as_str()) {
                return Err(ParlevoxError::Protocol(format!(
                    "manifest '{}' declares model '{}' more than once",
                    self.name, model.id
                )));
            }
            if let Some(modality) = model.modality {
                if modality != self.modality {
                    return Err(ParlevoxError::Protocol(format!(
                        "manifest '{}' declares model '{}' with modality '{}' but the provider \
                         modality is '{}'",
                        self.name,
                        model.id,
                        modality.as_str(),
                        self.modality.as_str()
                    )));
                }
            }
        }
        Ok(())
    }

    /// Convert a validation failure into the class the session should act on.
    #[must_use]
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
        assert_eq!(m.protocol_version, 1);
        assert_eq!(m.adapter, AdapterRef::default());
        assert!(m.models.is_empty());
        assert_eq!(m.provider_limits, ProviderLimits::default());
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

    #[test]
    fn old_manifest_json_still_parses() {
        // Exactly the v1 shape: none of the v2 fields are present.
        let json = r#"{
            "name": "fake-llm",
            "modality": "llm",
            "capabilities": ["streaming"],
            "config_schema": {"type": "object", "properties": {}}
        }"#;
        let manifest: ProviderManifest = serde_json::from_str(json).expect("v1 JSON parses");
        assert_eq!(manifest.protocol_version, 1);
        assert_eq!(manifest.adapter, AdapterRef::default());
        assert!(manifest.models.is_empty());
        assert_eq!(manifest.provider_limits, ProviderLimits::default());
        assert!(manifest.validate().is_ok());
    }

    #[test]
    fn unknown_adapter_kind_is_refused() {
        let mut m = ProviderManifest::new("x", Modality::Llm);
        m.adapter.kind = "bogus-adapter".into();
        let err = m.validate().unwrap_err();
        assert_eq!(err.class(), ErrorClass::Protocol);
        assert!(err.to_string().contains("bogus-adapter"));
    }

    #[test]
    fn adapter_endpoint_without_kind_is_refused() {
        let mut m = ProviderManifest::new("x", Modality::Llm);
        m.adapter.endpoint = Some("https://example.invalid".into());
        assert!(m.validate().is_err());
    }

    #[test]
    fn duplicate_model_ids_rejected() {
        let mut m = ProviderManifest::new("x", Modality::Llm);
        m.models = vec![
            ModelEntry {
                id: "m1".into(),
                display_name: String::new(),
                capabilities: Vec::new(),
                limits: ModelLimits::default(),
                modality: None,
            },
            ModelEntry {
                id: "m1".into(),
                display_name: String::new(),
                capabilities: Vec::new(),
                limits: ModelLimits::default(),
                modality: None,
            },
        ];
        let err = m.validate().unwrap_err();
        assert!(err.to_string().contains("m1"));
        assert!(err.to_string().contains("more than once"));
    }

    #[test]
    fn contradicting_model_modality_rejected() {
        let mut m = ProviderManifest::new("x", Modality::Llm);
        m.models = vec![ModelEntry {
            id: "m1".into(),
            display_name: String::new(),
            capabilities: Vec::new(),
            limits: ModelLimits::default(),
            modality: Some(Modality::Tts),
        }];
        let err = m.validate().unwrap_err();
        assert!(err.to_string().contains("modality"));
    }

    #[test]
    fn capability_union_covers_manifest_and_models() {
        let mut m = ProviderManifest::new("x", Modality::Llm).with_capability("manifest-cap");
        m.models = vec![ModelEntry {
            id: "m1".into(),
            display_name: String::new(),
            capabilities: vec!["model-cap".into()],
            limits: ModelLimits::default(),
            modality: None,
        }];
        let union = m.capability_union();
        assert!(union.contains(&"manifest-cap"));
        assert!(union.contains(&"model-cap"));
    }

    #[test]
    fn adapter_kind_round_trips_through_str() {
        for kind in AdapterKind::ALL {
            assert_eq!(kind.as_str().parse::<AdapterKind>(), Ok(kind));
        }
        assert!("nope".parse::<AdapterKind>().is_err());
        assert_eq!(AdapterKind::ALLOWED_KINDS.len(), AdapterKind::ALL.len());
    }
}
