//! Provider selection over manifests.
//!
//! Registration is manifest-validated, and selection is by exact name. The
//! session receives a resolved [`Providers`] set, so the turn loop never asks
//! "which provider" mid-turn.

use std::sync::Arc;

use voicekit_core::VoicekitError;
use voicekit_providers::{
    FakeLlm, FakeStt, FakeTts, FakeVad, LlmProvider, ProviderManifest, SttProvider, TtsProvider,
    VadProvider,
};

/// The four providers one session runs with.
#[derive(Clone)]
pub struct Providers {
    pub stt: Arc<dyn SttProvider>,
    pub llm: Arc<dyn LlmProvider>,
    pub tts: Arc<dyn TtsProvider>,
    pub vad: Arc<dyn VadProvider>,
}

/// Provider names to resolve. Defaults match the in-process fakes.
#[derive(Debug, Clone)]
pub struct ProviderNames {
    pub stt: String,
    pub llm: String,
    pub tts: String,
    pub vad: String,
}

impl Default for ProviderNames {
    fn default() -> Self {
        ProviderNames {
            stt: "fake-stt".into(),
            llm: "fake-llm".into(),
            tts: "fake-tts".into(),
            vad: "fake-vad".into(),
        }
    }
}

#[derive(Default)]
pub struct ProviderRegistry {
    stt: Vec<Arc<dyn SttProvider>>,
    llm: Vec<Arc<dyn LlmProvider>>,
    tts: Vec<Arc<dyn TtsProvider>>,
    vad: Vec<Arc<dyn VadProvider>>,
}

fn check(manifest: &ProviderManifest) -> Result<(), VoicekitError> {
    manifest.validate()
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registry pre-loaded with the four deterministic fakes.
    ///
    /// # Errors
    ///
    /// Returns a protocol-class error if a built-in fake manifest fails
    /// validation.
    pub fn fakes() -> Result<Self, VoicekitError> {
        let mut registry = Self::new();
        registry.register_stt(Arc::new(FakeStt::new()))?;
        registry.register_llm(Arc::new(FakeLlm::new()))?;
        registry.register_tts(Arc::new(FakeTts::new()))?;
        registry.register_vad(Arc::new(FakeVad::new()))?;
        Ok(registry)
    }

    /// # Errors
    ///
    /// Returns a protocol-class error when the provider's manifest fails
    /// validation.
    pub fn register_stt(&mut self, provider: Arc<dyn SttProvider>) -> Result<(), VoicekitError> {
        check(provider.manifest())?;
        self.stt.push(provider);
        Ok(())
    }

    /// # Errors
    ///
    /// Returns a protocol-class error when the provider's manifest fails
    /// validation.
    pub fn register_llm(&mut self, provider: Arc<dyn LlmProvider>) -> Result<(), VoicekitError> {
        check(provider.manifest())?;
        self.llm.push(provider);
        Ok(())
    }

    /// # Errors
    ///
    /// Returns a protocol-class error when the provider's manifest fails
    /// validation.
    pub fn register_tts(&mut self, provider: Arc<dyn TtsProvider>) -> Result<(), VoicekitError> {
        check(provider.manifest())?;
        self.tts.push(provider);
        Ok(())
    }

    /// # Errors
    ///
    /// Returns a protocol-class error when the provider's manifest fails
    /// validation.
    pub fn register_vad(&mut self, provider: Arc<dyn VadProvider>) -> Result<(), VoicekitError> {
        check(provider.manifest())?;
        self.vad.push(provider);
        Ok(())
    }

    pub fn manifests(&self) -> Vec<ProviderManifest> {
        let mut out = Vec::new();
        out.extend(self.stt.iter().map(|p| p.manifest().clone()));
        out.extend(self.llm.iter().map(|p| p.manifest().clone()));
        out.extend(self.tts.iter().map(|p| p.manifest().clone()));
        out.extend(self.vad.iter().map(|p| p.manifest().clone()));
        out
    }

    fn missing(modality: &str, name: &str) -> VoicekitError {
        VoicekitError::Protocol(format!("no {modality} provider registered as '{name}'"))
    }

    /// # Errors
    ///
    /// Returns a protocol-class error when no registered STT provider has
    /// exactly this name.
    pub fn select_stt(&self, name: &str) -> Result<Arc<dyn SttProvider>, VoicekitError> {
        self.stt
            .iter()
            .find(|p| p.manifest().name == name)
            .cloned()
            .ok_or_else(|| Self::missing("stt", name))
    }

    /// # Errors
    ///
    /// Returns a protocol-class error when no registered LLM provider has
    /// exactly this name.
    pub fn select_llm(&self, name: &str) -> Result<Arc<dyn LlmProvider>, VoicekitError> {
        self.llm
            .iter()
            .find(|p| p.manifest().name == name)
            .cloned()
            .ok_or_else(|| Self::missing("llm", name))
    }

    /// # Errors
    ///
    /// Returns a protocol-class error when no registered TTS provider has
    /// exactly this name.
    pub fn select_tts(&self, name: &str) -> Result<Arc<dyn TtsProvider>, VoicekitError> {
        self.tts
            .iter()
            .find(|p| p.manifest().name == name)
            .cloned()
            .ok_or_else(|| Self::missing("tts", name))
    }

    /// # Errors
    ///
    /// Returns a protocol-class error when no registered VAD provider has
    /// exactly this name.
    pub fn select_vad(&self, name: &str) -> Result<Arc<dyn VadProvider>, VoicekitError> {
        self.vad
            .iter()
            .find(|p| p.manifest().name == name)
            .cloned()
            .ok_or_else(|| Self::missing("vad", name))
    }

    /// Resolve all four roles at once.
    ///
    /// # Errors
    ///
    /// Returns a protocol-class error naming the first role that no registered
    /// provider matches.
    pub fn resolve(&self, names: &ProviderNames) -> Result<Providers, VoicekitError> {
        Ok(Providers {
            stt: self.select_stt(&names.stt)?,
            llm: self.select_llm(&names.llm)?,
            tts: self.select_tts(&names.tts)?,
            vad: self.select_vad(&names.vad)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fakes_resolve_by_name() {
        let registry = ProviderRegistry::fakes().unwrap();
        let providers = registry.resolve(&ProviderNames::default()).unwrap();
        assert_eq!(providers.stt.manifest().name, "fake-stt");
        assert_eq!(providers.llm.manifest().name, "fake-llm");
        assert_eq!(providers.tts.manifest().name, "fake-tts");
        assert_eq!(providers.vad.manifest().name, "fake-vad");
        assert_eq!(registry.manifests().len(), 4);
    }

    #[test]
    fn unknown_name_is_a_protocol_error() {
        let registry = ProviderRegistry::fakes().unwrap();
        let err = match registry.select_tts("missing") {
            Err(err) => err,
            Ok(_) => panic!("expected unknown provider to be rejected"),
        };
        assert!(matches!(err, VoicekitError::Protocol(_)));
    }
}
