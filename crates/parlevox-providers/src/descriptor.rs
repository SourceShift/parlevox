//! YAML provider descriptors and the directory loader.
//!
//! A descriptor is the on-disk form of a [`ProviderManifest`]: a provider is
//! described entirely by data. YAML handling stays inside this module so the
//! public API talks in manifests and `serde_json::Value`, never `serde_yaml`
//! types.

use std::fs;
use std::path::{Path, PathBuf};

use parlevox_core::ParlevoxError;
use serde::{Deserialize, Serialize};

use crate::manifest::{AdapterRef, Modality, ModelEntry, ProviderLimits, ProviderManifest};

fn default_config_schema() -> serde_json::Value {
    serde_json::json!({"type": "object", "properties": {}})
}

/// A provider described by a YAML file.
///
/// Unknown keys are rejected so a typo in a descriptor is an error instead of
/// a silently ignored setting.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderDescriptor {
    /// Stable identifier used for selection, e.g. `"openai-llm"`.
    pub name: String,
    pub modality: Modality,
    /// How the provider is reached.
    #[serde(default)]
    pub adapter: AdapterRef,
    /// Name of the environment variable holding the API key, never the key.
    #[serde(default)]
    pub api_key_env: Option<String>,
    /// Models the provider serves.
    #[serde(default)]
    pub models: Vec<ModelEntry>,
    /// Named behaviours the provider supports.
    #[serde(default)]
    pub capabilities: Vec<String>,
    /// Provider-wide limits.
    #[serde(default)]
    pub provider_limits: ProviderLimits,
    /// JSON schema-ish description of the provider's configuration object.
    #[serde(default = "default_config_schema")]
    pub config_schema: serde_json::Value,
    /// Descriptor format version.
    #[serde(default = "crate::manifest::default_protocol_version")]
    pub protocol_version: u32,
}

impl ProviderDescriptor {
    /// Read, parse and validate one descriptor.
    ///
    /// # Errors
    ///
    /// Returns a protocol-class error when the file cannot be read, is not
    /// valid YAML for a descriptor, or fails [`ProviderDescriptor::validate`].
    /// Parse errors name the path and line/column; semantic errors name the
    /// path and the offending key or value. The file contents are never
    /// included in the message.
    pub fn from_path(path: &Path) -> Result<Self, ParlevoxError> {
        let text = fs::read_to_string(path).map_err(|err| {
            ParlevoxError::Protocol(format!(
                "cannot read provider descriptor '{}': {err}",
                path.display()
            ))
        })?;
        let descriptor: ProviderDescriptor = serde_yaml::from_str(&text).map_err(|err| {
            let at = err.location().map_or_else(String::new, |loc| {
                format!(" at line {} column {}", loc.line(), loc.column())
            });
            ParlevoxError::Protocol(format!(
                "invalid provider descriptor '{}'{}: {err}",
                path.display(),
                at
            ))
        })?;
        descriptor.validate().map_err(|err| match err {
            ParlevoxError::Protocol(message) => ParlevoxError::Protocol(format!(
                "invalid provider descriptor '{}': {message}",
                path.display()
            )),
            other => other,
        })?;
        Ok(descriptor)
    }

    /// Validate the descriptor as a manifest plus descriptor-only rules.
    ///
    /// # Errors
    ///
    /// Returns a protocol-class error when `api_key_env` is present but blank,
    /// or when the equivalent manifest fails its own validation.
    pub fn validate(&self) -> Result<(), ParlevoxError> {
        if let Some(variable) = &self.api_key_env {
            if variable.trim().is_empty() {
                return Err(ParlevoxError::Protocol(format!(
                    "descriptor '{}' declares a blank api_key_env",
                    self.name
                )));
            }
        }
        self.to_manifest().validate()
    }

    /// Project the descriptor onto the runtime manifest type.
    #[must_use]
    pub fn to_manifest(&self) -> ProviderManifest {
        ProviderManifest {
            name: self.name.clone(),
            modality: self.modality,
            capabilities: self.capabilities.clone(),
            config_schema: self.config_schema.clone(),
            protocol_version: self.protocol_version,
            adapter: self.adapter.clone(),
            models: self.models.clone(),
            provider_limits: self.provider_limits.clone(),
        }
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

    /// Render the configuration schema as pretty JSON.
    #[must_use]
    pub fn config_schema_json(&self) -> String {
        serde_json::to_string_pretty(&self.config_schema)
            .unwrap_or_else(|_| self.config_schema.to_string())
    }
}

/// List `*.yaml` and `*.yml` files in `dir`, sorted by file name.
///
/// # Errors
///
/// A missing directory is not an error and yields an empty list. An unreadable
/// directory or entry yields a protocol-class error naming the directory.
pub fn load_dir(dir: &Path) -> Result<Vec<PathBuf>, ParlevoxError> {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => {
            return Err(ParlevoxError::Protocol(format!(
                "cannot read provider directory '{}': {err}",
                dir.display()
            )))
        }
    };

    let mut paths = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|err| {
            ParlevoxError::Protocol(format!(
                "cannot read an entry in provider directory '{}': {err}",
                dir.display()
            ))
        })?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let is_yaml = path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| {
                extension.eq_ignore_ascii_case("yaml") || extension.eq_ignore_ascii_case("yml")
            });
        if is_yaml {
            paths.push(path);
        }
    }
    paths.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new() -> Self {
            let unique = format!(
                "parlevox-descriptor-test-{}-{}",
                std::process::id(),
                COUNTER.fetch_add(1, Ordering::Relaxed)
            );
            let path = std::env::temp_dir().join(unique);
            fs::create_dir_all(&path).expect("create temp dir");
            TempDir { path }
        }

        fn write(&self, name: &str, contents: &str) -> PathBuf {
            let path = self.path.join(name);
            fs::write(&path, contents).expect("write descriptor");
            path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    const LLM_DESCRIPTOR: &str = r#"
name: openai-llm
modality: llm
protocol_version: 2
adapter:
  kind: openai-compatible-chat
  endpoint: https://api.openai.com/v1/chat/completions
api_key_env: OPENAI_API_KEY
capabilities:
  - streaming
models:
  - id: gpt-4o-mini
    display_name: GPT-4o mini
    capabilities:
      - tools
    limits:
      max_input_tokens: 128000
provider_limits:
  max_concurrency: 8
config_schema:
  type: object
  properties:
    temperature:
      type: number
"#;

    #[test]
    fn descriptor_round_trips_to_manifest() {
        let dir = TempDir::new();
        let path = dir.write("openai-llm.yaml", LLM_DESCRIPTOR);
        let descriptor = ProviderDescriptor::from_path(&path).expect("descriptor parses");
        let manifest = descriptor.to_manifest();
        assert_eq!(manifest.name, descriptor.name);
        assert_eq!(manifest.modality, Modality::Llm);
        assert_eq!(manifest.protocol_version, 2);
        assert_eq!(manifest.adapter.kind, "openai-compatible-chat");
        assert_eq!(manifest.models, descriptor.models);
        assert_eq!(manifest.provider_limits, descriptor.provider_limits);
        assert!(manifest.validate().is_ok());
    }

    #[test]
    fn unknown_adapter_kind_is_refused() {
        let dir = TempDir::new();
        let path = dir.write(
            "bad.yaml",
            "name: bad\nmodality: stt\nadapter:\n  kind: not-a-real-adapter\n",
        );
        let err = ProviderDescriptor::from_path(&path).unwrap_err();
        assert!(err.to_string().contains("not-a-real-adapter"));
    }

    #[test]
    fn malformed_file_reports_path() {
        let dir = TempDir::new();
        let path = dir.write("broken.yaml", "name: [unterminated\nmodality: llm\n");
        let err = ProviderDescriptor::from_path(&path).unwrap_err();
        assert!(
            err.to_string().contains(&path.display().to_string()),
            "message should name the file: {err}"
        );
    }

    #[test]
    fn unknown_key_is_refused() {
        let dir = TempDir::new();
        let path = dir.write(
            "typo.yaml",
            "name: typo\nmodality: llm\ncapabilites:\n  - streaming\n",
        );
        let err = ProviderDescriptor::from_path(&path).unwrap_err();
        assert!(err.to_string().contains("capabilites"));
    }

    #[test]
    fn nested_unknown_key_is_refused() {
        let dir = TempDir::new();
        let adapter_typo = dir.write(
            "adapter-typo.yaml",
            "name: adapter-typo\nmodality: llm\nadapter:\n  kind: sidecar\n  endpont: https://example.invalid\n",
        );
        let err = ProviderDescriptor::from_path(&adapter_typo).unwrap_err();
        assert!(err.to_string().contains("endpont"), "{err}");
        assert!(
            err.to_string()
                .contains(&adapter_typo.display().to_string()),
            "message should name the file: {err}"
        );

        let model_typo = dir.write(
            "model-typo.yaml",
            "name: model-typo\nmodality: llm\nmodels:\n  - id: m1\n    capabilites:\n      - tools\n",
        );
        let err = ProviderDescriptor::from_path(&model_typo).unwrap_err();
        assert!(err.to_string().contains("capabilites"), "{err}");
        assert!(
            err.to_string().contains(&model_typo.display().to_string()),
            "message should name the file: {err}"
        );
    }

    #[test]
    fn semantic_error_names_the_file() {
        let dir = TempDir::new();
        let path = dir.write(
            "semantic.yaml",
            "name: semantic-error-probe\nmodality: llm\nmodels:\n  - id: dup-model\n  - id: dup-model\n",
        );
        let err = ProviderDescriptor::from_path(&path).unwrap_err();
        assert!(err.to_string().contains("more than once"), "{err}");
        assert!(err.to_string().contains("dup-model"), "{err}");
        assert!(
            err.to_string().contains(&path.display().to_string()),
            "message should name the file: {err}"
        );
    }

    #[test]
    fn load_dir_skips_missing_directory_and_sorts() {
        let missing = std::env::temp_dir().join("parlevox-no-such-provider-dir");
        assert_eq!(load_dir(&missing).expect("missing dir is skipped").len(), 0);

        let dir = TempDir::new();
        dir.write("b.yaml", "name: b\nmodality: llm\n");
        dir.write("a.yml", "name: a\nmodality: llm\n");
        dir.write("ignored.txt", "not a descriptor");
        let names: Vec<String> = load_dir(&dir.path)
            .expect("directory loads")
            .iter()
            .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec!["a.yml".to_string(), "b.yaml".to_string()]);
    }
}
