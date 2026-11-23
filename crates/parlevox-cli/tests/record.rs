//! `demo --record` integration tests: the trace contract, transcript gating,
//! failure on an unwritable path, and byte-identical stdout.

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use parlevox_providers::{FAKE_LLM_DELTAS, FAKE_TRANSCRIPT};

fn parlevox() -> Command {
    Command::new(env!("CARGO_BIN_EXE_parlevox"))
}

fn unique_path(tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "parlevox-record-{tag}-{}-{nanos}.jsonl",
        std::process::id()
    ))
}

fn run_recorded(path: &PathBuf, transcripts: bool) -> std::process::Output {
    let mut command = parlevox();
    command.args(["demo", "--fake", "--record"]);
    command.arg(path);
    if transcripts {
        command.arg("--record-transcripts");
    }
    command.output().expect("run the parlevox binary")
}

#[test]
fn record_writes_header_then_parseable_observations() {
    let path = unique_path("header");
    let output = run_recorded(&path, false);
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let text = fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines[0], r#"{"observation_schema_version":1}"#);
    assert!(lines.len() > 1, "no observations were recorded");
    for line in &lines[1..] {
        serde_json::from_str::<serde_json::Value>(line).expect("every record parses");
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("recorder records="),
        "recorder counters must be printed to stderr, was: {stderr}"
    );
    fs::remove_file(path).ok();
}

#[test]
fn record_transcripts_gates_text() {
    let plain = unique_path("plain");
    let opted_in = unique_path("transcripts");

    assert!(run_recorded(&plain, false).status.success());
    assert!(run_recorded(&opted_in, true).status.success());

    let plain_text = fs::read_to_string(&plain).unwrap();
    let opted_text = fs::read_to_string(&opted_in).unwrap();

    assert!(
        !plain_text.contains(FAKE_TRANSCRIPT),
        "the final transcript must be absent without --record-transcripts"
    );
    assert!(
        !plain_text.contains(FAKE_LLM_DELTAS[0]),
        "reply deltas must be absent without --record-transcripts"
    );
    assert!(
        opted_text.contains(FAKE_TRANSCRIPT),
        "the final transcript must be present with --record-transcripts"
    );
    assert!(
        opted_text.contains(FAKE_LLM_DELTAS[0]),
        "reply deltas must be present with --record-transcripts"
    );

    fs::remove_file(plain).ok();
    fs::remove_file(opted_in).ok();
}

#[test]
fn an_unwritable_record_path_fails_the_command() {
    // A directory is not a writable trace file, so construction fails before
    // the turn runs.
    let directory = unique_path("directory");
    fs::create_dir_all(&directory).unwrap();

    let output = parlevox()
        .args(["demo", "--fake", "--record"])
        .arg(&directory)
        .output()
        .expect("run the parlevox binary");

    assert!(!output.status.success(), "an unwritable path must fail");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("cannot create trace file"),
        "stderr must name the failure, was: {stderr}"
    );

    fs::remove_dir_all(directory).ok();
}

#[test]
fn record_transcripts_requires_record() {
    let output = parlevox()
        .args(["demo", "--fake", "--record-transcripts"])
        .output()
        .expect("run the parlevox binary");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--record-transcripts requires --record"),
        "stderr must explain the rejected combination, was: {stderr}"
    );
}

#[test]
fn recording_does_not_change_stdout() {
    let baseline = parlevox()
        .args(["demo", "--fake"])
        .output()
        .expect("run the baseline demo");
    let path = unique_path("stdout");
    let recorded = run_recorded(&path, false);

    assert!(baseline.status.success());
    assert!(recorded.status.success());
    assert_eq!(
        baseline.stdout, recorded.stdout,
        "stdout must be byte-identical with and without --record"
    );

    fs::remove_file(path).ok();
}
