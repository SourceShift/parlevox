//! Sidecar behaviour: handshake timeout, cancel-without-kill, and restart.
//!
//! These tests spawn a local Python interpreter but never touch the network and
//! never install anything: the SDK runs via `PYTHONPATH=sdks/python`.

use std::sync::Arc;

use tokio_util::sync::CancellationToken;
use voicekit_core::VoicekitError;
use voicekit_ipc::{PluginEvent, PluginRequest};
use voicekit_providers::FAKE_LLM_DELTAS;
use voicekit_sidecar::{Incoming, RestartPolicy, SidecarClient, SidecarConfig, SidecarSupervisor};

fn python() -> String {
    std::env::var("VOICEKIT_PYTHON").unwrap_or_else(|_| "python3".into())
}

#[tokio::test]
async fn hung_plugin_times_out_during_handshake() {
    let config = SidecarConfig::program(
        python(),
        vec!["-c".into(), "import time; time.sleep(30)".into()],
    )
    .with_handshake_timeout_ms(150);
    let err = match SidecarClient::spawn(config).await {
        Err(err) => err,
        Ok(_) => panic!("expected handshake to time out"),
    };
    assert!(matches!(err, VoicekitError::Transport(_)));
}

#[tokio::test]
async fn cancel_ends_the_stream_without_killing_the_process() {
    let client = Arc::new(
        SidecarClient::spawn(SidecarConfig::python_sdk())
            .await
            .unwrap(),
    );

    let cancelled = CancellationToken::new();
    cancelled.cancel();
    let mut handle = client
        .open_request(
            PluginRequest::Synthesize {
                request_id: client.new_request_id("tts"),
                text: "hi".into(),
                sample_rate: 16_000,
                channels: 1,
            },
            Vec::new(),
            Some(cancelled),
        )
        .await
        .unwrap();
    assert!(handle.recv().await.is_none());
    assert!(handle.is_cancelled());
    assert!(client.is_alive().await, "cancel must not kill the plugin");

    // The process still serves new work.
    let mut next = client
        .open_request(
            PluginRequest::Generate {
                request_id: client.new_request_id("llm"),
                prompt: "hello".into(),
            },
            Vec::new(),
            None,
        )
        .await
        .unwrap();
    let mut deltas = Vec::new();
    while let Some(item) = next.recv().await {
        match item {
            Incoming::Event(PluginEvent::Delta { text, .. }) => deltas.push(text),
            Incoming::Event(PluginEvent::Done { .. }) => break,
            Incoming::Event(PluginEvent::Error { message, .. }) => {
                panic!("plugin error: {message}")
            }
            _ => {}
        }
    }
    assert_eq!(deltas, FAKE_LLM_DELTAS.to_vec());
    assert!(client.is_alive().await);

    client.shutdown().await.unwrap();
}

#[tokio::test]
async fn supervisor_restarts_a_dead_plugin() {
    let policy = RestartPolicy {
        max_restarts: 3,
        base_backoff_ms: 20,
        max_backoff_ms: 100,
    };
    let mut supervisor = SidecarSupervisor::new(SidecarConfig::python_sdk(), policy);

    let first = supervisor.ensure().await.unwrap();
    assert!(first.is_alive().await);
    first.kill().await;

    let second = supervisor.ensure().await.unwrap();
    assert!(supervisor.restarts() >= 1);
    assert!(second.is_alive().await);

    let mut handle = second
        .open_request(
            PluginRequest::Generate {
                request_id: second.new_request_id("llm"),
                prompt: "hello".into(),
            },
            Vec::new(),
            None,
        )
        .await
        .unwrap();
    let mut saw_delta = false;
    while let Some(item) = handle.recv().await {
        match item {
            Incoming::Event(PluginEvent::Delta { .. }) => saw_delta = true,
            Incoming::Event(PluginEvent::Done { .. }) => break,
            _ => {}
        }
    }
    assert!(saw_delta);

    supervisor.shutdown().await;
}
