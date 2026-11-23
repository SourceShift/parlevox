//! Request stall watchdog: a plugin that accepts a request and then goes
//! silent must fail that request with a transient error instead of hanging the
//! turn.
//!
//! The scripted fixture is stdlib-only and the answering case uses the bundled
//! SDK, so these tests are hermetic on any checkout with a Python interpreter
//! and never touch the network.

use std::sync::Arc;
use std::time::Duration;

use futures_util::StreamExt;
use parlevox_ipc::{ErrorClassWire, PluginEvent, PluginRequest};
use parlevox_providers::{GenerateRequest, LlmProvider, FAKE_LLM_DELTAS};
use parlevox_sidecar::{
    sdk_is_available, Incoming, PoolConfig, RestartPolicy, SidecarClient, SidecarConfig,
    SidecarLlm, SidecarPool, SidecarSupervisor, StallPolicy,
};
use tokio_util::sync::CancellationToken;

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

fn generate_request(client: &SidecarClient, prompt: &str) -> PluginRequest {
    PluginRequest::Generate {
        request_id: client.new_request_id("llm"),
        prompt: prompt.into(),
        history: Vec::new(),
    }
}

/// Whether `pid` still exists, using the POSIX `kill -0` probe.
fn process_exists(pid: u32) -> bool {
    std::process::Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

#[tokio::test]
async fn silent_plugin_times_out_into_a_transient_error() {
    if !python_available() {
        eprintln!("skipping stall watchdog: no Python interpreter");
        return;
    }

    let client = SidecarClient::spawn_watched(
        scripted("hold_requests"),
        StallPolicy {
            idle_timeout_ms: 100,
        },
    )
    .await
    .expect("spawn the watched client");

    let handle = client
        .open_request(
            generate_request(&client, "never answered"),
            Vec::new(),
            None,
        )
        .await
        .expect("open the request");
    let request_id = handle.request_id().to_string();
    let mut handle = handle;

    let (first, second) = tokio::time::timeout(Duration::from_secs(1), async {
        let first = handle.recv().await;
        let second = handle.recv().await;
        (first, second)
    })
    .await
    .expect("the watchdog must fire inside the guard");

    match first {
        Some(Incoming::Event(PluginEvent::Error {
            request_id: got,
            class,
            message,
        })) => {
            assert_eq!(class, ErrorClassWire::Transient);
            assert_eq!(got, request_id, "the error names the stalled request");
            assert!(
                message.contains("100"),
                "the message names the idle bound: {message}"
            );
        }
        other => panic!("expected a synthesized transient error, got {other:?}"),
    }
    assert!(
        second.is_none(),
        "the synthesized error is emitted exactly once"
    );
    assert_eq!(client.stall_events(), 1);
    assert!(client.is_wedged());
    assert!(!client.is_alive().await, "a wedged client is not alive");

    client.kill().await;
}

#[tokio::test]
async fn answering_plugin_never_trips_the_watchdog() {
    if !sdk_is_available() || !python_available() {
        eprintln!("skipping watchdog answering case: Python SDK is not available");
        return;
    }

    let client = Arc::new(
        SidecarClient::spawn_watched(
            SidecarConfig::python_sdk().expect("resolve the bundled SDK"),
            StallPolicy {
                idle_timeout_ms: 1_000,
            },
        )
        .await
        .expect("spawn the watched client"),
    );
    let llm = SidecarLlm::new(Arc::clone(&client));

    let mut stream = llm
        .generate(GenerateRequest::new("hello"), CancellationToken::new())
        .await
        .expect("start a generate stream");
    let mut deltas = Vec::new();
    while let Some(item) = stream.next().await {
        match item {
            Ok(text) => deltas.push(text),
            Err(err) => panic!("answering plugin produced an error: {err}"),
        }
    }

    let expected: Vec<String> = FAKE_LLM_DELTAS
        .iter()
        .map(|text| (*text).to_string())
        .collect();
    assert_eq!(deltas, expected, "the reply streamed to completion");
    assert_eq!(client.stall_events(), 0);
    assert!(!client.is_wedged());
    assert!(client.is_alive().await);

    client.shutdown().await.expect("shut the plugin down");
}

#[tokio::test]
async fn supervisor_replaces_a_wedged_client() {
    if !python_available() {
        eprintln!("skipping supervised stall: no Python interpreter");
        return;
    }

    let policy = RestartPolicy {
        max_restarts: 3,
        base_backoff_ms: 20,
        max_backoff_ms: 100,
    };
    let mut supervisor = SidecarSupervisor::new(scripted("hold_requests"), policy)
        .with_stall_watchdog(StallPolicy {
            idle_timeout_ms: 100,
        });

    let first = supervisor.ensure().await.expect("first ensure spawns");
    let first_pid = first.pid();
    assert!(!first.is_wedged());

    let mut handle = first
        .open_request(generate_request(&first, "never answered"), Vec::new(), None)
        .await
        .expect("open the request");
    let (first_item, second_item) = tokio::time::timeout(Duration::from_secs(1), async {
        let first = handle.recv().await;
        let second = handle.recv().await;
        (first, second)
    })
    .await
    .expect("the watchdog must fire inside the guard");
    assert!(matches!(
        first_item,
        Some(Incoming::Event(PluginEvent::Error { .. }))
    ));
    assert!(second_item.is_none());
    assert!(first.is_wedged());

    let second = supervisor
        .ensure()
        .await
        .expect("a wedged client is replaced");
    assert_ne!(
        second.pid(),
        first_pid,
        "the supervisor spawns a fresh process"
    );
    assert_eq!(supervisor.restarts(), 1);
    assert!(!second.is_wedged());
    assert!(second.is_alive().await);

    supervisor.shutdown().await;
}

#[tokio::test]
async fn cancellation_wins_over_the_idle_bound() {
    if !python_available() {
        eprintln!("skipping cancelled stall: no Python interpreter");
        return;
    }

    let client = SidecarClient::spawn_watched(
        scripted("hold_requests"),
        StallPolicy {
            idle_timeout_ms: 100,
        },
    )
    .await
    .expect("spawn the watched client");

    let token = CancellationToken::new();
    let mut handle = client
        .open_request(
            generate_request(&client, "cancel me"),
            Vec::new(),
            Some(token.clone()),
        )
        .await
        .expect("open the request");
    token.cancel();

    let item = tokio::time::timeout(Duration::from_secs(1), handle.recv())
        .await
        .expect("a cancelled handle resolves immediately");
    assert!(item.is_none(), "cancellation ends the stream");
    assert!(handle.is_cancelled());
    assert_eq!(client.stall_events(), 0, "a cancellation is not a stall");
    assert!(!client.is_wedged());

    client.kill().await;
}

#[tokio::test]
async fn without_the_policy_nothing_changes() {
    if !python_available() {
        eprintln!("skipping unwatched client: no Python interpreter");
        return;
    }

    let client = SidecarClient::spawn(scripted("hold_requests"))
        .await
        .expect("spawn the plain client");
    let mut handle = client
        .open_request(
            generate_request(&client, "never answered"),
            Vec::new(),
            None,
        )
        .await
        .expect("open the request");

    let elapsed = tokio::time::timeout(Duration::from_millis(150), handle.recv()).await;
    assert!(
        elapsed.is_err(),
        "an unwatched request must keep waiting, not fail"
    );
    assert_eq!(client.stall_events(), 0);
    assert!(!client.is_wedged());
    assert!(client.is_alive().await);

    drop(handle);
    client.kill().await;
}

#[tokio::test]
async fn wedged_lease_is_retired_at_release() {
    if !python_available() {
        eprintln!("skipping wedged lease: no Python interpreter");
        return;
    }

    let pool = Arc::new(
        SidecarPool::new(scripted("hold_requests"), PoolConfig::default()).with_stall_watchdog(
            StallPolicy {
                idle_timeout_ms: 100,
            },
        ),
    );
    let lease = pool.acquire().await.expect("acquire a fresh client");
    let pid = lease.client().pid().expect("the spawned process has a pid");

    let mut handle = lease
        .client()
        .open_request(
            generate_request(lease.client(), "never answered"),
            Vec::new(),
            None,
        )
        .await
        .expect("open the request");
    let (first_item, second_item) = tokio::time::timeout(Duration::from_secs(1), async {
        let first = handle.recv().await;
        let second = handle.recv().await;
        (first, second)
    })
    .await
    .expect("the watchdog must fire inside the guard");
    assert!(matches!(
        first_item,
        Some(Incoming::Event(PluginEvent::Error { .. }))
    ));
    assert!(second_item.is_none());
    assert!(lease.client().is_wedged());
    drop(handle);

    let release = lease.release().await;
    assert!(
        release.is_err(),
        "a wedged client cannot acknowledge a reset"
    );
    let stats = pool.stats();
    assert!(
        stats.reset_failures >= 1,
        "the failed reset probe is counted: {stats:?}"
    );
    assert!(
        !process_exists(pid),
        "the wedged process must be reaped at release"
    );

    pool.shutdown().await;
}
