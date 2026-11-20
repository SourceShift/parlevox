//! Differential parity for one scripted turn.
//!
//! The fake side runs entirely in process. The sidecar side runs the same
//! utterance through the **real** Python plugin — `SidecarStt`, `SidecarLlm`
//! and `SidecarTts` over the framed protocol, with in-process `FakeVad` — so a
//! divergence in the Python plugin fails the test instead of hiding behind a
//! second set of Rust fakes.
//!
//! The comparison uses a canonical projection: kind, turn, text, PCM samples,
//! sample rate, channels, and every drop counter. Audio timestamps are
//! session-epoch-relative and therefore normalized away.
//!
//! When the SDK is absent or the configured interpreter cannot be probed, the
//! differential test prints an explicit skip and returns. It does not fail and
//! is not `#[ignore]`d, so a plain `cargo test` stays green on a Python-free
//! checkout.

use std::sync::Arc;

use parlevox_core::ParlevoxError;
use parlevox_providers::{FAKE_REPLY, FAKE_TRANSCRIPT, FAKE_TTS_FRAMES};
use parlevox_runtime::{
    ProviderNames, ProviderRegistry, Session, SessionConfig, SessionEvent, TurnReport,
};
use parlevox_sidecar::{sdk_is_available, SidecarClient, SidecarConfig};

fn python_probe() -> bool {
    let python = std::env::var("PARLEVOX_PYTHON").unwrap_or_else(|_| "python3".into());
    std::process::Command::new(python)
        .args(["-c", "import sys"])
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

async fn run_fake() -> (TurnReport, Vec<SessionEvent>) {
    let registry = ProviderRegistry::fakes().unwrap();
    let providers = registry.resolve(&ProviderNames::default()).unwrap();
    let session = Session::new(SessionConfig::default(), providers).unwrap();
    parlevox_cli::run_turn_events(session).await.unwrap()
}

async fn run_sidecar(
    client: Arc<SidecarClient>,
) -> Result<(TurnReport, Vec<SessionEvent>), ParlevoxError> {
    let session = parlevox_cli::sidecar_session(client)?;
    parlevox_cli::run_turn_events(session).await
}

#[tokio::test]
async fn fake_turn_is_deterministic() {
    let (first_report, first_events) = run_fake().await;
    let (second_report, second_events) = run_fake().await;

    assert_eq!(first_report, second_report);
    assert_eq!(
        parlevox_cli::project_events(&first_events),
        parlevox_cli::project_events(&second_events)
    );
    assert_eq!(first_report.transcript, FAKE_TRANSCRIPT);
    assert_eq!(first_report.reply, FAKE_REPLY);
    assert_eq!(first_report.audio_frames as usize, FAKE_TTS_FRAMES);
    assert_eq!(first_report.sidecar_dropped_frames, 0);
}

#[tokio::test]
async fn real_sidecar_matches_the_in_process_turn() {
    if !sdk_is_available() {
        eprintln!("skipping differential parity: sdks/python is not available");
        return;
    }
    if !python_probe() {
        eprintln!("skipping differential parity: configured Python interpreter cannot be probed");
        return;
    }

    let (fake_report, fake_events) = run_fake().await;

    let client = Arc::new(
        SidecarClient::spawn(SidecarConfig::python_sdk().expect("resolve the bundled SDK"))
            .await
            .expect("spawn the Python sidecar"),
    );
    // Capture the turn first, then reap the child unconditionally: a failed
    // comparison must never leak a process.
    let result = run_sidecar(Arc::clone(&client)).await;
    let shutdown = client.shutdown().await;
    let (sidecar_report, sidecar_events) = result.expect("sidecar turn completes");
    shutdown.expect("sidecar child is reaped");

    assert_eq!(
        parlevox_cli::project_events(&sidecar_events),
        parlevox_cli::project_events(&fake_events),
        "the real plugin's event trace diverged from the in-process fakes"
    );
    assert_eq!(
        sidecar_report, fake_report,
        "the real plugin's turn report diverged from the in-process fakes"
    );
    assert_eq!(sidecar_report.transcript, FAKE_TRANSCRIPT);
    assert_eq!(sidecar_report.reply, FAKE_REPLY);
    assert_eq!(sidecar_report.audio_frames as usize, FAKE_TTS_FRAMES);
    assert_eq!(sidecar_report.sidecar_dropped_frames, 0);
}

#[test]
fn renderer_surfaces_dropped_events_and_sidecar_drops() {
    let line = parlevox_cli::render(&SessionEvent::TurnDone {
        turn: 1,
        audio_frames: 2,
        dropped_frames: 0,
        dropped_ms: 0,
        dropped_events: 3,
        sidecar_dropped_frames: 4,
        interrupted: false,
    });
    assert!(line.contains("dropped_events=3"));
    assert!(line.contains("sidecar_dropped_frames=4"));
}
