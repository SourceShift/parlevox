//! Layered descriptor registry and deterministic selection.
//!
//! A registry is keyed by `(modality, name)` and remembers the file each
//! descriptor came from. Loading is pure: it reads exactly the layer
//! directories it is handed. [`default_layers`] is the only place that reads
//! environment variables.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use parlevox_core::ParlevoxError;

use crate::descriptor::{load_dir, ProviderDescriptor};
use crate::manifest::Modality;

/// Capability and audio-shape requirements used to select a provider.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Requirements {
    /// Capabilities that must all be declared.
    pub capabilities: Vec<String>,
    /// Required sample rate in Hz, when the caller has one.
    pub sample_rate: Option<u32>,
    /// Required channel count, when the caller has one.
    pub channels: Option<u16>,
}

/// A descriptor together with the file it was loaded from.
#[derive(Debug, Clone)]
pub struct ProviderEntry {
    /// The parsed and validated descriptor.
    pub descriptor: ProviderDescriptor,
    /// The layer file the descriptor came from.
    pub source: PathBuf,
}

/// A deterministic, layered registry of provider descriptors.
#[derive(Debug, Default)]
pub struct ProviderRegistry {
    entries: HashMap<(Modality, String), ProviderEntry>,
}

impl ProviderRegistry {
    /// Load every descriptor in `layers`, in order.
    ///
    /// # Errors
    ///
    /// Returns a protocol-class error when a descriptor cannot be read or
    /// validated, or when the same `(modality, name)` is declared by two files.
    /// A duplicate always fails and names both files; later layers never
    /// silently win.
    pub fn load(layers: &[PathBuf]) -> Result<Self, ParlevoxError> {
        let mut entries: HashMap<(Modality, String), ProviderEntry> = HashMap::new();
        for layer in layers {
            for path in load_dir(layer)? {
                let descriptor = ProviderDescriptor::from_path(&path)?;
                let key = (descriptor.modality, descriptor.name.clone());
                if let Some(existing) = entries.get(&key) {
                    return Err(ParlevoxError::Protocol(format!(
                        "provider '{}' ({}) is declared twice: '{}' and '{}'",
                        descriptor.name,
                        descriptor.modality.as_str(),
                        existing.source.display(),
                        path.display()
                    )));
                }
                entries.insert(
                    key,
                    ProviderEntry {
                        descriptor,
                        source: path,
                    },
                );
            }
        }
        Ok(ProviderRegistry { entries })
    }

    /// The descriptor for `(modality, name)`, if it is registered.
    #[must_use]
    pub fn get(&self, modality: Modality, name: &str) -> Option<&ProviderDescriptor> {
        self.entry(modality, name).map(|entry| &entry.descriptor)
    }

    /// The registry entry for `(modality, name)`, including its source file.
    #[must_use]
    pub fn entry(&self, modality: Modality, name: &str) -> Option<&ProviderEntry> {
        self.entries.get(&(modality, name.to_owned()))
    }

    /// Every entry, sorted by `(modality, name)`.
    #[must_use]
    pub fn all(&self) -> Vec<&ProviderEntry> {
        let mut all: Vec<&ProviderEntry> = self.entries.values().collect();
        all.sort_by(|a, b| {
            a.descriptor
                .modality
                .as_str()
                .cmp(b.descriptor.modality.as_str())
                .then_with(|| a.descriptor.name.cmp(&b.descriptor.name))
        });
        all
    }

    /// Resolve a provider name that is unique across modalities.
    ///
    /// # Errors
    ///
    /// Returns a protocol-class error when no provider has the name, or when
    /// the name exists under more than one modality and is therefore
    /// ambiguous.
    pub fn find(&self, name: &str) -> Result<&ProviderEntry, ParlevoxError> {
        let mut matches: Vec<&ProviderEntry> = self
            .entries
            .values()
            .filter(|entry| entry.descriptor.name == name)
            .collect();
        match matches.len() {
            0 => Err(ParlevoxError::Protocol(format!(
                "unknown provider '{name}'"
            ))),
            1 => Ok(matches.remove(0)),
            _ => {
                let mut modalities: Vec<&str> = matches
                    .iter()
                    .map(|entry| entry.descriptor.modality.as_str())
                    .collect();
                modalities.sort_unstable();
                Err(ParlevoxError::Protocol(format!(
                    "provider '{name}' is ambiguous across modalities: {}",
                    modalities.join(", ")
                )))
            }
        }
    }

    /// Providers of `modality` whose descriptor meets `requirements`, sorted by
    /// name.
    ///
    /// Selection is pure: no network, no clock and no environment reads.
    #[must_use]
    pub fn select(
        &self,
        modality: Modality,
        requirements: &Requirements,
    ) -> Vec<&ProviderDescriptor> {
        let mut selected: Vec<&ProviderDescriptor> = self
            .entries
            .values()
            .filter(|entry| entry.descriptor.modality == modality)
            .filter(|entry| meets_requirements(&entry.descriptor, requirements))
            .map(|entry| &entry.descriptor)
            .collect();
        selected.sort_by(|a, b| a.name.cmp(&b.name));
        selected
    }
}

/// Whether `descriptor` satisfies every requirement.
///
/// Required capabilities are checked against the union of manifest-level and
/// model-level capabilities. A required sample rate or channel count needs at
/// least one model that declares it; an empty limits vector means "not
/// declared" and therefore does not meet such a requirement.
#[must_use]
pub fn meets_requirements(descriptor: &ProviderDescriptor, requirements: &Requirements) -> bool {
    let capabilities = descriptor.capability_union();
    if requirements
        .capabilities
        .iter()
        .any(|required| !capabilities.contains(&required.as_str()))
    {
        return false;
    }

    if requirements.sample_rate.is_none() && requirements.channels.is_none() {
        return true;
    }

    descriptor.models.iter().any(|model| {
        requirements
            .sample_rate
            .is_none_or(|rate| model.limits.sample_rates.contains(&rate))
            && requirements
                .channels
                .is_none_or(|channels| model.limits.channels.contains(&channels))
    })
}

/// The default layer list: the repository descriptors, then
/// `PARLEVOX_PROVIDER_DIRS`, then `$PARLEVOX_HOME/providers`.
///
/// The repository layer is resolved from the compile-time crate location with
/// a working-directory fallback, so no operator-local absolute path is ever
/// written to source.
#[must_use]
pub fn default_layers() -> Vec<PathBuf> {
    let mut layers = Vec::with_capacity(3);

    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let repo_layer = manifest_dir
        .join("..")
        .join("..")
        .join("config")
        .join("providers");
    if repo_layer.is_dir() {
        layers.push(repo_layer);
    } else if let Ok(current) = std::env::current_dir() {
        let fallback = current.join("config").join("providers");
        if fallback.is_dir() {
            layers.push(fallback);
        }
    }

    if let Some(dirs) = std::env::var_os("PARLEVOX_PROVIDER_DIRS") {
        for dir in std::env::split_paths(&dirs) {
            if !dir.as_os_str().is_empty() {
                layers.push(dir);
            }
        }
    }

    if let Some(home) = std::env::var_os("PARLEVOX_HOME") {
        let home = PathBuf::from(home);
        if !home.as_os_str().is_empty() {
            layers.push(home.join("providers"));
        }
    }

    layers
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new() -> Self {
            let unique = format!(
                "parlevox-registry-test-{}-{}",
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

    fn selection_fixture() -> TempDir {
        let dir = TempDir::new();
        dir.write(
            "a.yaml",
            r#"
name: alpha-llm
modality: llm
adapter:
  kind: openai-compatible-chat
capabilities:
  - streaming
models:
  - id: alpha-model
    capabilities:
      - tools
"#,
        );
        dir.write(
            "b.yaml",
            r#"
name: beta-llm
modality: llm
adapter:
  kind: openai-compatible-chat
capabilities:
  - streaming
models:
  - id: beta-model
    capabilities:
      - tools
      - audio
    limits:
      sample_rates: [16000]
      channels: [1]
"#,
        );
        dir
    }

    #[test]
    fn selection_honours_capabilities_and_limits() {
        let dir = selection_fixture();
        let registry =
            ProviderRegistry::load(std::slice::from_ref(&dir.path)).expect("registry loads");

        let tools = Requirements {
            capabilities: vec!["tools".into()],
            ..Requirements::default()
        };
        let both: Vec<&str> = registry
            .select(Modality::Llm, &tools)
            .iter()
            .map(|descriptor| descriptor.name.as_str())
            .collect();
        assert_eq!(both, vec!["alpha-llm", "beta-llm"]);

        let audio = Requirements {
            capabilities: vec!["tools".into(), "audio".into()],
            ..Requirements::default()
        };
        let audio_only: Vec<&str> = registry
            .select(Modality::Llm, &audio)
            .iter()
            .map(|descriptor| descriptor.name.as_str())
            .collect();
        assert_eq!(audio_only, vec!["beta-llm"]);

        let shaped = Requirements {
            capabilities: Vec::new(),
            sample_rate: Some(16000),
            channels: Some(1),
        };
        let shaped_only: Vec<&str> = registry
            .select(Modality::Llm, &shaped)
            .iter()
            .map(|descriptor| descriptor.name.as_str())
            .collect();
        assert_eq!(shaped_only, vec!["beta-llm"]);
    }

    #[test]
    fn duplicate_provider_across_layers_is_an_error() {
        let first = TempDir::new();
        let second = TempDir::new();
        let first_path = first.write(
            "one.yaml",
            "name: dup\nmodality: llm\nadapter:\n  kind: sidecar\n",
        );
        let second_path = second.write(
            "two.yaml",
            "name: dup\nmodality: llm\nadapter:\n  kind: sidecar\n",
        );
        let err = ProviderRegistry::load(&[first.path.clone(), second.path.clone()]).unwrap_err();
        let message = err.to_string();
        assert!(message.contains(&first_path.display().to_string()));
        assert!(message.contains(&second_path.display().to_string()));
        assert!(message.contains("dup"));
    }

    #[test]
    fn missing_layer_dir_is_skipped() {
        let dir = selection_fixture();
        let missing = dir.path.join("does-not-exist");
        let registry =
            ProviderRegistry::load(&[missing, dir.path.clone()]).expect("registry loads");
        assert_eq!(registry.all().len(), 2);
    }

    #[test]
    fn find_unknown_name_is_a_protocol_error() {
        let dir = selection_fixture();
        let registry =
            ProviderRegistry::load(std::slice::from_ref(&dir.path)).expect("registry loads");
        assert!(registry.find("alpha-llm").is_ok());
        assert!(registry.find("nope").is_err());
    }

    #[test]
    fn find_ambiguous_name_across_modalities_is_a_protocol_error() {
        let dir = TempDir::new();
        dir.write(
            "llm.yaml",
            "name: shared\nmodality: llm\nadapter:\n  kind: sidecar\n",
        );
        dir.write(
            "tts.yaml",
            "name: shared\nmodality: tts\nadapter:\n  kind: sidecar\n",
        );
        let registry =
            ProviderRegistry::load(std::slice::from_ref(&dir.path)).expect("registry loads");
        let err = registry.find("shared").unwrap_err();
        assert!(err.to_string().contains("ambiguous"));
        assert!(err.to_string().contains("llm"));
        assert!(err.to_string().contains("tts"));
    }

    #[test]
    fn all_is_sorted_by_modality_then_name() {
        let dir = TempDir::new();
        dir.write(
            "z.yaml",
            "name: zed\nmodality: tts\nadapter:\n  kind: sidecar\n",
        );
        dir.write(
            "a.yaml",
            "name: alpha\nmodality: tts\nadapter:\n  kind: sidecar\n",
        );
        dir.write(
            "m.yaml",
            "name: mid\nmodality: llm\nadapter:\n  kind: sidecar\n",
        );
        let registry =
            ProviderRegistry::load(std::slice::from_ref(&dir.path)).expect("registry loads");
        let names: Vec<&str> = registry
            .all()
            .iter()
            .map(|entry| entry.descriptor.name.as_str())
            .collect();
        assert_eq!(names, vec!["mid", "alpha", "zed"]);
    }

    fn requirements_strategy() -> impl Strategy<Value = Requirements> {
        (
            prop::collection::vec(
                prop::sample::select(vec![
                    "streaming".to_string(),
                    "tools".to_string(),
                    "audio".to_string(),
                    "cancel".to_string(),
                ]),
                0..4,
            ),
            prop::option::of(prop::sample::select(vec![8_000u32, 16_000, 24_000, 44_100])),
            prop::option::of(prop::sample::select(vec![1u16, 2])),
        )
            .prop_map(|(capabilities, sample_rate, channels)| Requirements {
                capabilities,
                sample_rate,
                channels,
            })
    }

    proptest! {
        #[test]
        fn selection_only_returns_providers_meeting_requirements(
            requirements in requirements_strategy()
        ) {
            let dir = selection_fixture();
            let first = ProviderRegistry::load(std::slice::from_ref(&dir.path))
                .expect("first load succeeds");
            let second = ProviderRegistry::load(std::slice::from_ref(&dir.path))
                .expect("second load succeeds");

            let selected = first.select(Modality::Llm, &requirements);
            for descriptor in &selected {
                prop_assert!(meets_requirements(descriptor, &requirements));
            }

            let first_names: Vec<&str> = selected
                .iter()
                .map(|descriptor| descriptor.name.as_str())
                .collect();
            let second_names: Vec<&str> = second
                .select(Modality::Llm, &requirements)
                .iter()
                .map(|descriptor| descriptor.name.as_str())
                .collect();
            prop_assert_eq!(first_names, second_names);
        }
    }
}
