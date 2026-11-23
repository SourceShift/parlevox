//! Descriptor-driven connector construction.
//!
//! [`build_provider`] is the wiring seam: it reads `adapter.kind` from a
//! validated descriptor, checks it against the declared modality, and returns
//! the matching modality trait object. A vendor that speaks an existing wire
//! shape needs a new YAML file, not a new Rust branch.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use parlevox_core::ParlevoxError;
use parlevox_providers::{
    AdapterKind, LlmProvider, Modality, ProviderDescriptor, ProviderManifest, RetryPolicy,
    SttProvider, TtsProvider,
};

use crate::auth::{Auth, Secret};
use crate::connectors::{OpenAiChat, OpenAiStt, OpenAiTts};
use crate::toolkit::HttpToolkit;
use crate::transport::Transport;

/// The default per-request timeout.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
/// The default maximum line or whole-JSON body size.
const DEFAULT_MAX_BUFFER_BYTES: usize = 1 << 20;
/// The default maximum frames buffered ahead of the caller.
const DEFAULT_MAX_BUFFERED_FRAMES: usize = 32;
/// The voice used when a descriptor and its configuration name none.
const DEFAULT_VOICE: &str = "alloy";

/// A resolved provider credential.
///
/// There is no `Display` or `Serialize`, and `Debug` is redacted, so a secret
/// cannot reach a fixture or a log through this type.
#[derive(Clone)]
pub struct Credentials(Secret);

impl Credentials {
    /// Wrap a credential value.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Credentials(Secret::new(value))
    }

    /// Resolve a credential from an environment variable.
    ///
    /// This is the explicit live opt-in; the acceptance tests never call it.
    ///
    /// # Errors
    ///
    /// Returns [`ParlevoxError::Protocol`] naming the *variable* when it is
    /// unset or blank. The value is never included in the message.
    pub fn from_env(variable: &str) -> Result<Self, ParlevoxError> {
        match std::env::var(variable) {
            Ok(value) if !value.trim().is_empty() => Ok(Credentials::new(value)),
            Ok(_) => Err(ParlevoxError::Protocol(format!(
                "environment variable '{variable}' is empty"
            ))),
            Err(_) => Err(ParlevoxError::Protocol(format!(
                "environment variable '{variable}' is not set"
            ))),
        }
    }

    pub(crate) fn secret(&self) -> &Secret {
        &self.0
    }
}

impl fmt::Debug for Credentials {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Credentials(<redacted>)")
    }
}

/// Construction settings shared by every connector.
///
/// Build one with [`ConnectorConfig::builder`]. The defaults are a 30-second
/// request deadline, the provider-crate default [`RetryPolicy`] (no retries),
/// bearer auth with the supplied [`Credentials`], a 1 MiB buffer bound and 32
/// buffered audio frames.
pub struct ConnectorConfig {
    transport: Arc<dyn Transport>,
    timeout: Duration,
    retry: RetryPolicy,
    auth: Option<Auth>,
    max_buffer_bytes: usize,
    max_buffered_frames: usize,
    model: Option<String>,
    voice: Option<String>,
}

impl ConnectorConfig {
    /// Start building a configuration around `transport`.
    pub fn builder(transport: Arc<dyn Transport>) -> ConnectorConfigBuilder {
        ConnectorConfigBuilder {
            transport,
            timeout: DEFAULT_TIMEOUT,
            retry: RetryPolicy::default(),
            auth: None,
            max_buffer_bytes: DEFAULT_MAX_BUFFER_BYTES,
            max_buffered_frames: DEFAULT_MAX_BUFFERED_FRAMES,
            model: None,
            voice: None,
        }
    }
}

/// The builder returned by [`ConnectorConfig::builder`].
#[must_use]
pub struct ConnectorConfigBuilder {
    transport: Arc<dyn Transport>,
    timeout: Duration,
    retry: RetryPolicy,
    auth: Option<Auth>,
    max_buffer_bytes: usize,
    max_buffered_frames: usize,
    model: Option<String>,
    voice: Option<String>,
}

impl ConnectorConfigBuilder {
    /// Set the per-request deadline. Zero disables it.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the retry policy for pre-first-byte failures.
    pub fn retry_policy(mut self, retry: RetryPolicy) -> Self {
        self.retry = retry;
        self
    }

    /// Override the descriptor-derived bearer auth.
    pub fn auth(mut self, auth: Auth) -> Self {
        self.auth = Some(auth);
        self
    }

    /// Bound an unterminated line or a whole JSON body.
    pub fn max_buffer_bytes(mut self, bytes: usize) -> Self {
        self.max_buffer_bytes = bytes;
        self
    }

    /// Bound the frames buffered ahead of the caller.
    pub fn max_buffered_frames(mut self, frames: usize) -> Self {
        self.max_buffered_frames = frames;
        self
    }

    /// Override the descriptor's first model id.
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the TTS voice. Defaults to `"alloy"`.
    pub fn voice(mut self, voice: impl Into<String>) -> Self {
        self.voice = Some(voice.into());
        self
    }

    /// Finish the configuration.
    #[must_use]
    pub fn build(self) -> ConnectorConfig {
        ConnectorConfig {
            transport: self.transport,
            timeout: self.timeout,
            retry: self.retry,
            auth: self.auth,
            max_buffer_bytes: self.max_buffer_bytes,
            max_buffered_frames: self.max_buffered_frames,
            model: self.model,
            voice: self.voice,
        }
    }
}

/// A constructed provider, typed by the modality it serves.
#[non_exhaustive]
pub enum BuiltProvider {
    /// A chat-completion provider.
    Llm(Box<dyn LlmProvider>),
    /// A speech-to-text provider.
    Stt(Box<dyn SttProvider>),
    /// A text-to-speech provider.
    Tts(Box<dyn TtsProvider>),
}

impl BuiltProvider {
    /// The manifest the provider was built from.
    #[must_use]
    pub fn manifest(&self) -> &ProviderManifest {
        match self {
            BuiltProvider::Llm(provider) => provider.manifest(),
            BuiltProvider::Stt(provider) => provider.manifest(),
            BuiltProvider::Tts(provider) => provider.manifest(),
        }
    }
}

impl fmt::Debug for BuiltProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (modality, name) = match self {
            BuiltProvider::Llm(provider) => ("llm", provider.manifest().name.as_str()),
            BuiltProvider::Stt(provider) => ("stt", provider.manifest().name.as_str()),
            BuiltProvider::Tts(provider) => ("tts", provider.manifest().name.as_str()),
        };
        f.debug_struct("BuiltProvider")
            .field("modality", &modality)
            .field("name", &name)
            .finish()
    }
}

/// Build the provider a descriptor names.
///
/// # Errors
///
/// Returns [`ParlevoxError::Protocol`] when the descriptor is invalid, has no
/// endpoint or model, names an adapter kind this crate does not implement as
/// an HTTP connector, or declares a modality the adapter kind does not serve.
pub fn build_provider(
    descriptor: &ProviderDescriptor,
    credentials: Credentials,
    config: ConnectorConfig,
) -> Result<BuiltProvider, ParlevoxError> {
    descriptor.validate()?;
    let kind: AdapterKind = descriptor.adapter.kind.parse().map_err(|error| {
        ParlevoxError::Protocol(format!("provider '{}': {error}", descriptor.name))
    })?;
    let endpoint = descriptor
        .adapter
        .endpoint
        .as_deref()
        .map(str::trim)
        .filter(|endpoint| !endpoint.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| {
            ParlevoxError::Protocol(format!(
                "provider '{}' has no adapter endpoint",
                descriptor.name
            ))
        })?;
    let auth = config
        .auth
        .clone()
        .unwrap_or_else(|| Auth::bearer(credentials.secret().clone()));
    let model = config
        .model
        .clone()
        .or_else(|| descriptor.models.first().map(|model| model.id.clone()))
        .ok_or_else(|| {
            ParlevoxError::Protocol(format!("provider '{}' declares no model", descriptor.name))
        })?;
    let ConnectorConfig {
        transport,
        timeout,
        retry,
        max_buffer_bytes,
        max_buffered_frames,
        voice,
        ..
    } = config;
    let toolkit = HttpToolkit::new(transport, auth, timeout, retry, max_buffer_bytes);
    let manifest = descriptor.to_manifest();
    let built = match kind {
        AdapterKind::OpenAiCompatibleChat => {
            require_modality(descriptor, Modality::Llm, kind)?;
            BuiltProvider::Llm(Box::new(OpenAiChat::new(
                manifest, endpoint, model, toolkit,
            )))
        }
        AdapterKind::OpenAiAudioStt => {
            require_modality(descriptor, Modality::Stt, kind)?;
            BuiltProvider::Stt(Box::new(OpenAiStt::new(manifest, endpoint, model, toolkit)))
        }
        AdapterKind::OpenAiAudioTts => {
            require_modality(descriptor, Modality::Tts, kind)?;
            BuiltProvider::Tts(Box::new(OpenAiTts::new(
                manifest,
                endpoint,
                model,
                voice.unwrap_or_else(|| DEFAULT_VOICE.to_owned()),
                max_buffered_frames,
                toolkit,
            )))
        }
        other => {
            return Err(ParlevoxError::Protocol(format!(
                "adapter kind '{}' is not an HTTP connector",
                other.as_str()
            )))
        }
    };
    Ok(built)
}

fn require_modality(
    descriptor: &ProviderDescriptor,
    expected: Modality,
    kind: AdapterKind,
) -> Result<(), ParlevoxError> {
    if descriptor.modality == expected {
        return Ok(());
    }
    Err(ParlevoxError::Protocol(format!(
        "adapter kind '{}' serves '{}', but provider '{}' declares '{}'",
        kind.as_str(),
        expected.as_str(),
        descriptor.name,
        descriptor.modality.as_str()
    )))
}
