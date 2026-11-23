//! End-to-end checks for the offline `providers` subcommands.
//!
//! These run the real binary in a subprocess so environment variables cannot
//! race with other tests in this process.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new() -> Self {
        let unique = format!(
            "parlevox-cli-providers-test-{}-{}",
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

fn command() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_parlevox"));
    // The repository descriptors are the baseline; operator state must not
    // change what these tests observe.
    command
        .env_remove("PARLEVOX_PROVIDER_DIRS")
        .env_remove("PARLEVOX_HOME")
        .env_remove("OPENAI_API_KEY");
    command
}

fn run(args: &[&str]) -> Output {
    command().args(args).output().expect("run parlevox")
}

#[test]
fn list_is_sorted_and_shows_source_files() {
    let output = run(&["providers", "list"]);
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 3, "one row per shipped descriptor: {stdout}");

    assert!(lines[0].starts_with("openai-llm\tllm\tdescriptor\topenai-compatible-chat\t"));
    assert!(lines[1].starts_with("openai-stt\tstt\tdescriptor\topenai-audio-stt\t"));
    assert!(lines[2].starts_with("openai-tts\ttts\tdescriptor\topenai-audio-tts\t"));
    assert!(lines[0].contains("openai-llm.yaml"));
    assert!(lines[1].contains("openai-stt.yaml"));
    assert!(lines[2].contains("openai-tts.yaml"));
}

#[test]
fn describe_never_prints_the_key() {
    // A deliberately fake, key-shaped value: the fixture exemption applies
    // because the value itself says "test".
    const PLANTED: &str = "sk-test-planted-value-never-shown-000000";
    let output = command()
        .env("OPENAI_API_KEY", PLANTED)
        .args(["providers", "describe", "openai-llm"])
        .output()
        .expect("run parlevox");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{stdout}{stderr}");
    assert!(
        !combined.contains(PLANTED),
        "the key value must never appear"
    );
    assert!(
        !combined.contains("sk-"),
        "no key-shaped material in output"
    );
    assert!(stdout.contains("<set>"), "the env var is reported as set");
    assert!(stdout.contains("gpt-4o-mini"));
    assert!(stdout.contains("capabilities"));
    assert!(stdout.contains("config_schema"));
}

#[test]
fn describe_marks_missing_key() {
    let output = run(&["providers", "describe", "openai-llm"]);
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("<missing>"));
    assert!(stdout.contains("OPENAI_API_KEY"));
}

#[test]
fn descriptor_loads_from_an_out_of_tree_directory() {
    let dir = TempDir::new();
    dir.write(
        "external.yaml",
        "name: external-llm\n\
         modality: llm\n\
         adapter:\n  kind: sidecar\n\
         capabilities:\n  - streaming\n\
         models:\n  - id: external-model\n",
    );

    let output = command()
        .env("PARLEVOX_PROVIDER_DIRS", &dir.path)
        .args(["providers", "list"])
        .output()
        .expect("run parlevox");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("external-llm"),
        "an out-of-tree descriptor is visible: {stdout}"
    );
}

#[test]
fn unknown_provider_is_an_error() {
    let output = run(&["providers", "describe", "does-not-exist"]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("unknown provider"), "stderr: {stderr}");
}
