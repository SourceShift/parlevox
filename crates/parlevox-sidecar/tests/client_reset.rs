//! The additive `reset` control message: acknowledgement, timeout, and the
//! refusal to reset a client that is dead or still owns a request route.
//!
//! These tests drive a stdlib-only scripted plugin, never the network and
//! never a user-installed SDK, so they are hermetic on any checkout with a
//! Python interpreter.

use parlevox_core::ParlevoxError;
use parlevox_ipc::PluginRequest;
use parlevox_sidecar::{SidecarClient, SidecarConfig};

fn python() -> String {
    std::env::var("PARLEVOX_PYTHON").unwrap_or_else(|_| "python3".into())
}

fn python_available() -> bool {
    std::process::Command::new(python())
        .args(["-c", "import sys"])
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn scripted(mode: &str) -> SidecarConfig {
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("scripted_plugin.py");
    SidecarConfig::program(
        python(),
        vec![fixture.display().to_string(), mode.to_string()],
    )
}

#[tokio::test]
async fn reset_is_acknowledged_and_the_process_stays_alive() {
    if !python_available() {
        eprintln!("skipping reset acknowledgement: no Python interpreter");
        return;
    }

    let client = SidecarClient::spawn(scripted("ack")).await.unwrap();
    assert!(client.supports_reset());
    assert!(client.pid().is_some(), "spawn captures the child pid");

    client.reset().await.expect("first reset is acknowledged");
    client
        .reset()
        .await
        .expect("a second reset is acknowledged too");
    assert!(client.is_alive().await, "reset must not kill the process");

    client.shutdown().await.unwrap();
}

#[tokio::test]
async fn reset_times_out_when_the_plugin_never_acknowledges() {
    if !python_available() {
        eprintln!("skipping reset timeout: no Python interpreter");
        return;
    }

    let config = scripted("never_ack").with_reset_timeout_ms(150);
    let client = SidecarClient::spawn(config).await.unwrap();

    let err = client
        .reset()
        .await
        .expect_err("an unacknowledged reset must time out");
    assert!(matches!(err, ParlevoxError::Transport(_)));
    assert!(err.to_string().contains("timed out"), "{err}");

    client.kill().await;
}

#[tokio::test]
async fn reset_refuses_a_client_with_an_open_request_route() {
    if !python_available() {
        eprintln!("skipping dirty-reset refusal: no Python interpreter");
        return;
    }

    let client = SidecarClient::spawn(scripted("hold_requests"))
        .await
        .unwrap();
    let handle = client
        .open_request(
            PluginRequest::Synthesize {
                request_id: client.new_request_id("tts"),
                text: "never answered".into(),
                sample_rate: 16_000,
                channels: 1,
            },
            Vec::new(),
            None,
        )
        .await
        .unwrap();

    let err = client
        .reset()
        .await
        .expect_err("a reset must not silently clear a live request");
    assert!(matches!(err, ParlevoxError::Protocol(_)));
    assert!(err.to_string().contains("request route"), "{err}");

    drop(handle);
    client.kill().await;
}

#[tokio::test]
async fn reset_refuses_a_dead_process() {
    if !python_available() {
        eprintln!("skipping dead-client reset: no Python interpreter");
        return;
    }

    let client = SidecarClient::spawn(scripted("ack")).await.unwrap();
    client.kill().await;
    assert!(!client.is_alive().await);

    let err = client
        .reset()
        .await
        .expect_err("a dead plugin cannot acknowledge a reset");
    assert!(matches!(err, ParlevoxError::Transport(_)));
}

#[tokio::test]
async fn a_client_without_the_capability_is_not_poolable() {
    if !python_available() {
        eprintln!("skipping capability detection: no Python interpreter");
        return;
    }

    let client = SidecarClient::spawn(scripted("no_reset_capability"))
        .await
        .unwrap();
    assert!(!client.supports_reset());
    client.shutdown().await.unwrap();
}
