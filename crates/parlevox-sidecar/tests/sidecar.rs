//! Sidecar behaviour: handshake timeout, cancel-without-kill, and restart.
//!
//! These tests spawn a local Python interpreter but never touch the network and
//! never install anything: the SDK runs via `PYTHONPATH=sdks/python`.

use std::sync::Arc;

use futures_util::StreamExt;
use parlevox_core::{AudioFrame, ParlevoxError, Timestamp};
use parlevox_ipc::{PluginEvent, PluginRequest};
use parlevox_providers::{AudioChunk, SttProvider, FAKE_LLM_DELTAS, FAKE_TRANSCRIPT};
use parlevox_sidecar::{
    sdk_is_available, Incoming, RestartPolicy, SidecarClient, SidecarConfig, SidecarStt,
    SidecarSupervisor,
};
use tokio_util::sync::CancellationToken;

/// Mirrors `MAX_PENDING_AUDIO_FRAMES` in `sdks/python/parlevox_sdk/plugin.py`.
const PY_MAX_PENDING_AUDIO_FRAMES: usize = 64;

fn python() -> String {
    std::env::var("PARLEVOX_PYTHON").unwrap_or_else(|_| "python3".into())
}

fn python_ready() -> bool {
    sdk_is_available()
        && std::process::Command::new(python())
            .args(["-c", "import sys"])
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
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
    assert!(matches!(err, ParlevoxError::Transport(_)));
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
async fn transcribe_overflow_reports_dropped_frames_in_done() {
    if !python_ready() {
        eprintln!("skipping sidecar drop telemetry: Python SDK is not available");
        return;
    }

    let client = Arc::new(
        SidecarClient::spawn(SidecarConfig::python_sdk())
            .await
            .unwrap(),
    );

    let overflow = 7usize;
    let total = PY_MAX_PENDING_AUDIO_FRAMES + overflow;
    let frames: Vec<AudioFrame> = (0..total)
        .map(|_| AudioFrame::new(Timestamp::from_millis(0), 16_000, 1, vec![0i16; 4]))
        .collect();

    let mut handle = client
        .open_request(
            PluginRequest::Transcribe {
                request_id: client.new_request_id("stt"),
                sample_rate: 16_000,
                channels: 1,
            },
            frames,
            None,
        )
        .await
        .unwrap();

    let mut reported = None;
    while let Some(item) = handle.recv().await {
        match item {
            Incoming::Event(PluginEvent::Done { dropped_frames, .. }) => {
                reported = Some(dropped_frames);
                break;
            }
            Incoming::Event(PluginEvent::Error { message, .. }) => {
                panic!("plugin error: {message}")
            }
            _ => {}
        }
    }

    let shutdown = client.shutdown().await;
    assert_eq!(
        reported,
        Some(overflow as u64),
        "the plugin must report exactly the frames its bounded buffer evicted"
    );
    shutdown.unwrap();
}

#[tokio::test]
async fn sidecar_stt_captures_the_plugin_dropped_frames() {
    if !python_ready() {
        eprintln!("skipping SidecarStt drop telemetry: Python SDK is not available");
        return;
    }

    let client = Arc::new(
        SidecarClient::spawn(SidecarConfig::python_sdk())
            .await
            .unwrap(),
    );
    let stt = SidecarStt::new(Arc::clone(&client));

    // The adapter streams the utterance in 10 ms frames; send enough to
    // overflow the plugin's 64-frame per-request buffer by `overflow`.
    let overflow = 6usize;
    let total_frames = PY_MAX_PENDING_AUDIO_FRAMES + overflow;
    let sample_rate = 16_000u32;
    let samples = total_frames * (sample_rate as usize / 100);
    let audio = AudioChunk::new(sample_rate, 1, vec![0i16; samples]);

    let mut stream = stt
        .transcribe(audio, CancellationToken::new())
        .await
        .unwrap();
    let mut final_text = None;
    while let Some(item) = stream.next().await {
        let chunk = item.unwrap();
        if chunk.is_final {
            final_text = Some(chunk.text);
        }
    }

    assert_eq!(final_text.as_deref(), Some(FAKE_TRANSCRIPT));
    assert_eq!(
        stt.take_sidecar_dropped_frames(),
        overflow as u64,
        "SidecarStt must surface the plugin's bounded-buffer evictions"
    );
    // Swap semantics: the same count is never reported twice.
    assert_eq!(stt.take_sidecar_dropped_frames(), 0);

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
