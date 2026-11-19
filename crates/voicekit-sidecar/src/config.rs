//! How to launch the optional Python plugin process.
//!
//! Only this crate is allowed to spawn a process. The configuration is explicit
//! so tests can point at a fake program and keep the timeout small.

use std::path::PathBuf;

/// Default interpreter, overridable with `VOICEKIT_PYTHON`.
pub const DEFAULT_PYTHON: &str = "python3";

/// Module executed inside the SDK package.
pub const SIDECAR_MODULE: &str = "voicekit_sdk.sidecar_main";

#[derive(Debug, Clone)]
pub struct SidecarConfig {
    pub program: String,
    pub args: Vec<String>,
    /// Environment entries applied to the child. `PYTHONPATH` is set by the
    /// constructor so the SDK runs without installation.
    pub env: Vec<(String, String)>,
    pub handshake_timeout_ms: u64,
    pub shutdown_timeout_ms: u64,
}

impl SidecarConfig {
    /// Launch the bundled Python SDK with `PYTHONPATH=sdks/python`.
    pub fn python_sdk() -> Self {
        let sdk_path = default_sdk_path();
        let python =
            std::env::var("VOICEKIT_PYTHON").unwrap_or_else(|_| DEFAULT_PYTHON.to_string());
        let python_path = match std::env::var("PYTHONPATH") {
            Ok(existing) if !existing.is_empty() => {
                format!("{}{}{}", sdk_path.display(), path_separator(), existing)
            }
            _ => sdk_path.display().to_string(),
        };
        SidecarConfig {
            program: python,
            args: vec!["-u".into(), "-m".into(), SIDECAR_MODULE.into()],
            env: vec![("PYTHONPATH".into(), python_path)],
            handshake_timeout_ms: 5_000,
            shutdown_timeout_ms: 2_000,
        }
    }

    /// Launch an arbitrary program. Used by tests to simulate a hung or
    /// crashing plugin without touching the real SDK.
    pub fn program(program: impl Into<String>, args: Vec<String>) -> Self {
        SidecarConfig {
            program: program.into(),
            args,
            env: Vec::new(),
            handshake_timeout_ms: 5_000,
            shutdown_timeout_ms: 2_000,
        }
    }

    pub fn with_handshake_timeout_ms(mut self, ms: u64) -> Self {
        self.handshake_timeout_ms = ms;
        self
    }

    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.push((key.into(), value.into()));
        self
    }
}

/// `sdks/python` relative to this crate, resolved at compile time.
pub fn default_sdk_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../sdks/python")
}

/// Whether the bundled SDK is present. `--sidecar` reports this instead of
/// panicking when the checkout is incomplete.
pub fn sdk_is_available() -> bool {
    default_sdk_path()
        .join("voicekit_sdk")
        .join("__init__.py")
        .is_file()
}

fn path_separator() -> char {
    if cfg!(windows) {
        ';'
    } else {
        ':'
    }
}

/// Sanity check used by the CLI for a friendlier diagnostic.
pub fn describe(config: &SidecarConfig) -> String {
    format!("{} {}", config.program, config.args.join(" "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn python_sdk_config_points_at_the_bundled_sdk() {
        let config = SidecarConfig::python_sdk();
        assert!(config.args.contains(&"-m".to_string()));
        assert!(config.args.contains(&SIDECAR_MODULE.to_string()));
        assert!(config
            .env
            .iter()
            .any(|(k, v)| k == "PYTHONPATH" && v.contains("sdks/python")));
    }

    #[test]
    fn sdk_path_ends_at_sdks_python() {
        let path = default_sdk_path();
        assert!(path.ends_with(Path::new("sdks/python")));
    }
}
