//! How to launch the optional Python plugin process.
//!
//! Only this crate is allowed to spawn a process. The configuration is explicit
//! so tests can point at a fake program and keep the timeout small.

use std::fmt;
use std::path::{Path, PathBuf};

/// Default interpreter, overridable with `PARLEVOX_PYTHON`.
pub const DEFAULT_PYTHON: &str = "python3";

/// Environment override for the directory containing `parlevox_sdk`.
pub const SDK_ENV_VAR: &str = "PARLEVOX_PYTHON_SDK";

/// Module executed inside the SDK package.
pub const SIDECAR_MODULE: &str = "parlevox_sdk.sidecar_main";

/// Inputs to [`resolve_sdk_path`].
///
/// Tests construct this directly instead of mutating the process environment,
/// so resolution stays hermetic and parallel-safe. Production code uses
/// [`SdkSearch::from_process`].
#[derive(Debug, Clone)]
pub struct SdkSearch {
    /// Raw value of [`SDK_ENV_VAR`], if set. An empty value is treated as unset.
    pub env_value: Option<String>,
    /// Path of the running executable, if the platform could report it.
    pub exe_path: Option<PathBuf>,
    /// Crate manifest directory; its `../../sdks/python` is the documented
    /// build-tree fallback.
    pub manifest_dir: PathBuf,
}

impl SdkSearch {
    /// Read the ambient process state: the environment override, the current
    /// executable, and the compile-time manifest directory.
    #[must_use]
    pub fn from_process() -> Self {
        Self {
            env_value: std::env::var(SDK_ENV_VAR).ok(),
            exe_path: std::env::current_exe().ok(),
            manifest_dir: manifest_dir(),
        }
    }
}

/// Why no usable bundled SDK directory was found.
///
/// Marked [`non_exhaustive`](https://doc.rust-lang.org/reference/attributes/type_system.html#the-non_exhaustive-attribute)
/// so future candidates can add variants without breaking downstream matches.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SdkLookupError {
    /// [`SDK_ENV_VAR`] was set to a non-empty path that does not contain the
    /// SDK. The override is authoritative, so this never falls through.
    InvalidOverride {
        /// The environment variable that was honoured.
        variable: &'static str,
        /// The rejected directory.
        path: PathBuf,
    },
    /// Every candidate directory was checked and none contained the SDK.
    NotFound {
        /// Each attempted directory with a human-readable description of where
        /// it came from.
        attempts: Vec<(PathBuf, &'static str)>,
        /// Whether executable-relative candidates were skipped because the
        /// current executable path was unavailable.
        exe_relative_skipped: bool,
    },
}

impl fmt::Display for SdkLookupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOverride { variable, path } => write!(
                f,
                "environment variable {variable} is set to `{}` but that directory does not contain parlevox_sdk/__init__.py",
                path.display()
            ),
            Self::NotFound {
                attempts,
                exe_relative_skipped,
            } => {
                write!(
                    f,
                    "could not locate the bundled parlevox SDK (no parlevox_sdk/__init__.py); tried:"
                )?;
                for (path, source) in attempts {
                    write!(f, " {} ({source});", path.display())?;
                }
                if *exe_relative_skipped {
                    write!(
                        f,
                        " executable-relative candidates were skipped because the current \
                         executable path is unavailable"
                    )?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for SdkLookupError {}

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
    ///
    /// The directory is resolved from the ambient process state by
    /// [`SdkSearch::from_process`]. Resolution is fallible: a set-but-invalid
    /// [`SDK_ENV_VAR`] is authoritative, so this returns an error instead of
    /// silently launching a different SDK than the caller named. Only
    /// [`SdkLookupError::NotFound`] falls back to the documented build-tree
    /// path, where the later spawn reports a genuinely missing checkout. Use
    /// [`SidecarConfig::python_sdk_from`] to resolve hermetically from an
    /// explicit [`SdkSearch`].
    ///
    /// # Errors
    ///
    /// Returns [`SdkLookupError::InvalidOverride`] when [`SDK_ENV_VAR`] is set
    /// to a non-empty path that does not contain `parlevox_sdk/__init__.py`.
    /// A missing SDK is not an error here: it falls back to
    /// [`default_sdk_path`] as documented.
    #[must_use = "the config only takes effect once it is handed to SidecarClient::spawn"]
    pub fn python_sdk() -> Result<Self, SdkLookupError> {
        Self::python_sdk_from(&SdkSearch::from_process())
    }

    /// Launch the bundled Python SDK from an explicit [`SdkSearch`].
    ///
    /// This is the hermetic entry point behind [`SidecarConfig::python_sdk`]:
    /// it reads nothing but `search`, so tests can drive every resolution rule
    /// with a planted temporary tree instead of the process environment. A
    /// set-but-invalid [`SDK_ENV_VAR`] is authoritative and surfaced as an
    /// error; only
    /// [`SdkLookupError::NotFound`] falls back to the documented build-tree
    /// path from [`default_sdk_path`].
    ///
    /// # Errors
    ///
    /// Returns [`SdkLookupError::InvalidOverride`] when `search.env_value` is a
    /// non-empty path that does not contain `parlevox_sdk/__init__.py`.
    #[must_use = "the config only takes effect once it is handed to SidecarClient::spawn"]
    pub fn python_sdk_from(search: &SdkSearch) -> Result<Self, SdkLookupError> {
        match resolve_sdk_path(search) {
            Ok(path) => Ok(Self::python_sdk_at(path)),
            Err(SdkLookupError::NotFound { .. }) => Ok(Self::python_sdk_at(default_sdk_path())),
            Err(err) => Err(err),
        }
    }

    /// Launch the bundled Python SDK from an already resolved directory.
    #[must_use = "builder methods return a modified config; chain or assign the result"]
    pub fn python_sdk_at(sdk_path: PathBuf) -> Self {
        let python =
            std::env::var("PARLEVOX_PYTHON").unwrap_or_else(|_| DEFAULT_PYTHON.to_string());
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

    #[must_use = "builder methods return a modified config; chain or assign the result"]
    pub fn with_handshake_timeout_ms(mut self, ms: u64) -> Self {
        self.handshake_timeout_ms = ms;
        self
    }

    #[must_use = "builder methods return a modified config; chain or assign the result"]
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.push((key.into(), value.into()));
        self
    }
}

/// `sdks/python` relative to this crate, resolved at compile time.
///
/// This is the last-resort build-tree fallback, not the primary lookup; use
/// [`resolve_sdk_path`] for the run-time order.
#[must_use]
pub fn default_sdk_path() -> PathBuf {
    manifest_dir().join("../../sdks/python")
}

/// The crate manifest directory. This is the single documented build-tree
/// anchor; every other candidate is derived from the environment or the
/// executable.
fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Resolve the directory that contains `parlevox_sdk` at run time.
///
/// The search order is:
///
/// 1. [`SDK_ENV_VAR`] when set and non-empty — authoritative: a bad path is an
///    error and never falls through.
/// 2. `sdks/python`, `../sdks/python`, and `../lib/parlevox/sdks/python`
///    relative to the directory containing the running executable.
/// 3. The build-tree fallback, `../../sdks/python` under the crate manifest
///    directory.
///
/// A candidate qualifies when `<candidate>/parlevox_sdk/__init__.py` is a
/// regular file.
///
/// # Errors
///
/// Returns [`SdkLookupError::InvalidOverride`] when the environment override is
/// set but does not contain the SDK, and [`SdkLookupError::NotFound`] — listing
/// every attempted path — when no candidate qualifies.
pub fn resolve_sdk_path(search: &SdkSearch) -> Result<PathBuf, SdkLookupError> {
    if let Some(raw) = search.env_value.as_deref() {
        if !raw.is_empty() {
            let path = PathBuf::from(raw);
            if is_sdk_dir(&path) {
                return Ok(path);
            }
            return Err(SdkLookupError::InvalidOverride {
                variable: SDK_ENV_VAR,
                path,
            });
        }
    }

    let mut attempts: Vec<(PathBuf, &'static str)> = Vec::with_capacity(4);
    if let Some(exe) = search.exe_path.as_deref() {
        if let Some(exe_dir) = exe.parent() {
            let candidates = [
                (exe_dir.join("sdks/python"), "relative to the executable"),
                (
                    exe_dir.join("../sdks/python"),
                    "one level above the executable directory",
                ),
                (
                    exe_dir.join("../lib/parlevox/sdks/python"),
                    "under the installed <prefix>/lib/parlevox layout",
                ),
            ];
            for (path, source) in candidates {
                if is_sdk_dir(&path) {
                    return Ok(path);
                }
                attempts.push((path, source));
            }
        }
    }

    let fallback = search.manifest_dir.join("../../sdks/python");
    if is_sdk_dir(&fallback) {
        return Ok(fallback);
    }
    attempts.push((fallback, "build-tree fallback from the crate manifest"));

    Err(SdkLookupError::NotFound {
        attempts,
        exe_relative_skipped: search.exe_path.is_none(),
    })
}

/// Whether a candidate directory contains the SDK package marker.
fn is_sdk_dir(candidate: &Path) -> bool {
    candidate.join("parlevox_sdk").join("__init__.py").is_file()
}

/// Whether the bundled SDK is present. `--sidecar` reports this instead of
/// panicking when the checkout is incomplete.
pub fn sdk_is_available() -> bool {
    resolve_sdk_path(&SdkSearch::from_process()).is_ok()
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
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_DIR_ID: AtomicU64 = AtomicU64::new(0);

    /// A temporary tree that removes itself on drop. SDK-bearing tests plant a
    /// `parlevox_sdk/__init__.py` marker so they never depend on the real
    /// checkout or on the ambient environment.
    struct TempTree {
        root: PathBuf,
    }

    impl TempTree {
        fn new(label: &str) -> Self {
            let id = NEXT_DIR_ID.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir().join(format!(
                "parlevox-sdk-resolver-{label}-{}-{id}",
                std::process::id()
            ));
            std::fs::create_dir_all(&root).expect("create temp tree");
            Self { root }
        }

        fn root(&self) -> &Path {
            &self.root
        }

        /// Plant `parlevox_sdk/__init__.py` at `relative` under the root.
        fn plant_sdk(&self, relative: &str) {
            let package = self.root.join(relative).join("parlevox_sdk");
            std::fs::create_dir_all(&package).expect("create sdk package");
            std::fs::write(package.join("__init__.py"), b"").expect("write sdk marker");
        }
    }

    impl Drop for TempTree {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    /// `<root>/bin/parlevox`, so the executable directory is `<root>/bin`.
    fn exe_path(tree: &TempTree) -> PathBuf {
        tree.root().join("bin").join("parlevox")
    }

    /// A manifest directory whose `../../sdks/python` is `<root>/sdks/python`.
    ///
    /// The directory itself must exist: path resolution walks through it before
    /// applying `..`, exactly as it would in a real cargo tree.
    fn manifest_dir(tree: &TempTree) -> PathBuf {
        let dir = tree.root().join("crates").join("parlevox-sidecar");
        std::fs::create_dir_all(&dir).expect("create manifest dir");
        dir
    }

    /// The fallback candidate exactly as the resolver constructs it.
    fn manifest_candidate(tree: &TempTree) -> PathBuf {
        manifest_dir(tree).join("../../sdks/python")
    }

    fn search_for(
        env_value: Option<String>,
        exe: Option<&TempTree>,
        manifest: &TempTree,
    ) -> SdkSearch {
        SdkSearch {
            env_value,
            exe_path: exe.map(exe_path),
            manifest_dir: manifest_dir(manifest),
        }
    }

    fn missing_dir() -> String {
        std::env::temp_dir()
            .join("parlevox-sdk-resolver-no-such-directory")
            .display()
            .to_string()
    }

    #[test]
    fn python_sdk_config_points_at_the_bundled_sdk() {
        let tree = TempTree::new("bundled-config");
        tree.plant_sdk("sdks/python");
        let search = search_for(None, None, &tree);

        let config = SidecarConfig::python_sdk_from(&search)
            .expect("the build-tree fallback resolves the planted SDK");

        assert!(config.args.contains(&"-m".to_string()));
        assert!(config.args.contains(&SIDECAR_MODULE.to_string()));
        let python_path = config
            .env
            .iter()
            .find(|(key, _)| key == "PYTHONPATH")
            .map(|(_, value)| value)
            .expect("PYTHONPATH is set");
        assert!(
            python_path.starts_with(&manifest_candidate(&tree).display().to_string()),
            "{python_path}"
        );
        assert!(python_path.contains("sdks/python"), "{python_path}");
    }

    #[test]
    fn python_sdk_from_rejects_invalid_override() {
        let tree = TempTree::new("invalid-override");
        let bad = missing_dir();
        let search = search_for(Some(bad.clone()), None, &tree);

        let err = SidecarConfig::python_sdk_from(&search)
            .expect_err("a set-but-invalid override must not yield a launch config");

        match &err {
            SdkLookupError::InvalidOverride { variable, path } => {
                assert_eq!(*variable, SDK_ENV_VAR);
                assert_eq!(path, &PathBuf::from(&bad));
            }
            other => panic!("expected InvalidOverride, got {other:?}"),
        }
        let rendered = err.to_string();
        assert!(rendered.contains(SDK_ENV_VAR), "{rendered}");
        assert!(rendered.contains(&bad), "{rendered}");
    }

    #[test]
    fn sdk_path_ends_at_sdks_python() {
        let path = default_sdk_path();
        assert!(path.ends_with(Path::new("sdks/python")));
    }

    #[test]
    fn python_sdk_at_honours_the_resolved_directory() {
        let tree = TempTree::new("config-at");
        tree.plant_sdk(".");
        let config = SidecarConfig::python_sdk_at(tree.root().to_path_buf());
        let python_path = config
            .env
            .iter()
            .find(|(key, _)| key == "PYTHONPATH")
            .map(|(_, value)| value)
            .expect("PYTHONPATH is set");
        assert!(python_path.starts_with(&tree.root().display().to_string()));
    }

    #[test]
    fn resolver_env_override_wins() {
        let env_tree = TempTree::new("env-wins");
        env_tree.plant_sdk(".");
        let exe_tree = TempTree::new("env-wins-exe");
        exe_tree.plant_sdk("bin/sdks/python");
        let manifest_tree = TempTree::new("env-wins-manifest");
        manifest_tree.plant_sdk("sdks/python");

        let search = search_for(
            Some(env_tree.root().display().to_string()),
            Some(&exe_tree),
            &manifest_tree,
        );

        assert_eq!(resolve_sdk_path(&search), Ok(env_tree.root().to_path_buf()));
    }

    #[test]
    fn resolver_env_override_invalid_is_an_error() {
        let exe_tree = TempTree::new("env-invalid-exe");
        exe_tree.plant_sdk("bin/sdks/python");
        let manifest_tree = TempTree::new("env-invalid-manifest");
        manifest_tree.plant_sdk("sdks/python");
        let bad = missing_dir();

        let search = search_for(Some(bad.clone()), Some(&exe_tree), &manifest_tree);
        let err = resolve_sdk_path(&search).expect_err("an invalid override must not fall through");

        match &err {
            SdkLookupError::InvalidOverride { variable, path } => {
                assert_eq!(*variable, SDK_ENV_VAR);
                assert_eq!(path, &PathBuf::from(&bad));
            }
            other => panic!("expected InvalidOverride, got {other:?}"),
        }
        let rendered = err.to_string();
        assert!(rendered.contains(SDK_ENV_VAR), "{rendered}");
        assert!(rendered.contains(&bad), "{rendered}");
    }

    #[test]
    fn resolver_empty_env_override_is_ignored() {
        let exe_tree = TempTree::new("env-empty-exe");
        exe_tree.plant_sdk("bin/sdks/python");
        let manifest_tree = TempTree::new("env-empty-manifest");
        manifest_tree.plant_sdk("sdks/python");

        let search = search_for(Some(String::new()), Some(&exe_tree), &manifest_tree);

        assert_eq!(
            resolve_sdk_path(&search),
            Ok(exe_tree.root().join("bin").join("sdks/python"))
        );
    }

    #[test]
    fn resolver_exe_relative_used_when_env_absent() {
        let exe_tree = TempTree::new("exe-relative");
        exe_tree.plant_sdk("bin/sdks/python");
        let manifest_tree = TempTree::new("exe-relative-manifest");
        manifest_tree.plant_sdk("sdks/python");

        let search = search_for(None, Some(&exe_tree), &manifest_tree);

        assert_eq!(
            resolve_sdk_path(&search),
            Ok(exe_tree.root().join("bin").join("sdks/python"))
        );
    }

    #[test]
    fn resolver_manifest_is_last_resort() {
        let exe_tree = TempTree::new("manifest-exe");
        let manifest_tree = TempTree::new("manifest-fallback");
        manifest_tree.plant_sdk("sdks/python");

        let search = search_for(None, Some(&exe_tree), &manifest_tree);

        assert_eq!(
            resolve_sdk_path(&search),
            Ok(manifest_candidate(&manifest_tree))
        );
    }

    #[test]
    fn resolver_failure_lists_every_attempted_path() {
        let exe_tree = TempTree::new("failure-exe");
        let manifest_tree = TempTree::new("failure-manifest");

        let search = search_for(None, Some(&exe_tree), &manifest_tree);
        let err = resolve_sdk_path(&search).expect_err("no candidate exists");

        let exe_dir = exe_tree.root().join("bin");
        let expected = [
            exe_dir.join("sdks/python"),
            exe_dir.join("../sdks/python"),
            exe_dir.join("../lib/parlevox/sdks/python"),
            manifest_candidate(&manifest_tree),
        ];
        let rendered = err.to_string();
        for path in &expected {
            assert!(
                rendered.contains(&path.display().to_string()),
                "error did not list {path:?}: {rendered}"
            );
        }
        match err {
            SdkLookupError::NotFound {
                attempts,
                exe_relative_skipped,
            } => {
                assert_eq!(attempts.len(), expected.len());
                assert!(!exe_relative_skipped);
            }
            other => panic!("expected NotFound, got {other:?}"),
        }
    }

    #[test]
    fn resolver_missing_exe_reports_skipped_candidates() {
        let manifest_tree = TempTree::new("no-exe");

        let search = search_for(None, None, &manifest_tree);
        let err = resolve_sdk_path(&search).expect_err("no candidate exists");

        match &err {
            SdkLookupError::NotFound {
                attempts,
                exe_relative_skipped,
            } => {
                assert_eq!(attempts.len(), 1);
                assert!(*exe_relative_skipped);
            }
            other => panic!("expected NotFound, got {other:?}"),
        }
        assert!(err.to_string().contains("executable path is unavailable"));
    }

    #[derive(Clone, Copy, Debug)]
    enum EnvState {
        Absent,
        Valid,
        Invalid,
    }

    /// Exhaustive matrix over (env state, executable candidate present,
    /// manifest candidate present): each combination resolves to the highest
    /// priority candidate that exists, or fails.
    #[test]
    fn resolver_candidate_matrix() {
        for env in [EnvState::Absent, EnvState::Valid, EnvState::Invalid] {
            for exe_valid in [false, true] {
                for manifest_valid in [false, true] {
                    let env_tree = TempTree::new("matrix-env");
                    if matches!(env, EnvState::Valid) {
                        env_tree.plant_sdk(".");
                    }
                    let exe_tree = TempTree::new("matrix-exe");
                    if exe_valid {
                        exe_tree.plant_sdk("bin/sdks/python");
                    }
                    let manifest_tree = TempTree::new("matrix-manifest");
                    if manifest_valid {
                        manifest_tree.plant_sdk("sdks/python");
                    }

                    let env_value = match env {
                        EnvState::Absent => None,
                        EnvState::Valid => Some(env_tree.root().display().to_string()),
                        EnvState::Invalid => Some(missing_dir()),
                    };
                    let search = search_for(env_value, Some(&exe_tree), &manifest_tree);
                    let resolved = resolve_sdk_path(&search);
                    let case = format!(
                        "env={env:?} exe_valid={exe_valid} manifest_valid={manifest_valid}"
                    );

                    match env {
                        EnvState::Valid => assert_eq!(
                            resolved,
                            Ok(env_tree.root().to_path_buf()),
                            "override must win: {case}"
                        ),
                        EnvState::Invalid => assert!(
                            matches!(resolved, Err(SdkLookupError::InvalidOverride { .. })),
                            "invalid override must be a hard error: {case}"
                        ),
                        EnvState::Absent if exe_valid => assert_eq!(
                            resolved,
                            Ok(exe_tree.root().join("bin").join("sdks/python")),
                            "executable-relative wins when the override is absent: {case}"
                        ),
                        EnvState::Absent if manifest_valid => assert_eq!(
                            resolved,
                            Ok(manifest_candidate(&manifest_tree)),
                            "manifest is the last resort: {case}"
                        ),
                        EnvState::Absent => assert!(
                            matches!(resolved, Err(SdkLookupError::NotFound { .. })),
                            "no candidate means NotFound: {case}"
                        ),
                    }
                }
            }
        }
    }
}
