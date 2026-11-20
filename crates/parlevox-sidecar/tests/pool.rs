//! Pool bounds and failure handling, driven by a stdlib-only scripted plugin.
//!
//! The idle cap, the age ceiling, the acquire-time probe reset and the
//! dead-process fallback are all observable through [`PoolStats`] and the
//! child pid, so no request work is needed here. Every test shuts the pool
//! down, so no child process survives a failure.

use std::sync::Arc;
use std::time::Duration;

use parlevox_sidecar::{PoolConfig, SidecarClient, SidecarConfig, SidecarPool};

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
    .with_reset_timeout_ms(1_000)
}

fn pid(lease: &parlevox_sidecar::PoolLease) -> u32 {
    lease.client().pid().expect("spawn captures the child pid")
}

/// Whether a pid still answers `kill -0`. A zombie does, so a killed but
/// unreaped child also counts as alive: the kill path reaps it.
fn process_alive(pid: u32) -> bool {
    std::process::Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stderr(std::process::Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

/// Poll until `pid` is gone, with a generous bound instead of a fixed sleep.
async fn wait_until_process_exits(pid: u32) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while process_alive(pid) {
        assert!(
            tokio::time::Instant::now() < deadline,
            "process {pid} outlived its release by more than 10s"
        );
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}

#[tokio::test]
async fn pool_enforces_idle_cap() {
    if !python_available() {
        eprintln!("skipping idle-cap test: no Python interpreter");
        return;
    }

    let pool = Arc::new(SidecarPool::new(
        scripted("ack"),
        PoolConfig::default().with_max_idle(1),
    ));

    // Two concurrent leases are two processes; releasing the second past the
    // cap must shut the oldest idle process down instead of parking both.
    let first = pool.acquire().await.unwrap();
    let second = pool.acquire().await.unwrap();
    assert_ne!(pid(&first), pid(&second));

    first.release().await.unwrap();
    assert_eq!(pool.stats().evicted, 0);

    second.release().await.unwrap();
    let stats = pool.stats();
    assert!(stats.evicted >= 1, "the idle cap must evict: {stats:?}");
    assert_eq!(stats.spawned, 2);
    assert_eq!(stats.reused, 0);
    assert_eq!(stats.reset_failures, 0);

    pool.shutdown().await;
}

#[tokio::test]
async fn pool_with_zero_idle_counts_configuration_refusal_as_shutdown() {
    if !python_available() {
        eprintln!("skipping zero-idle test: no Python interpreter");
        return;
    }

    let pool = Arc::new(SidecarPool::new(
        scripted("ack"),
        PoolConfig::default().with_max_idle(0),
    ));

    let lease = pool.acquire().await.unwrap();
    lease.release().await.unwrap();

    // The configuration refuses to pool the process, so it is shut down. That
    // is a `shutdown`, not an `evicted`: no idle entry was ever displaced.
    let stats = pool.stats();
    assert_eq!(stats.spawned, 1);
    assert_eq!(stats.reused, 0);
    assert_eq!(stats.evicted, 0);
    assert_eq!(stats.shutdown, 1);

    pool.shutdown().await;
}

#[tokio::test]
async fn pool_expires_idle_processes_past_max_age() {
    if !python_available() {
        eprintln!("skipping max-age test: no Python interpreter");
        return;
    }

    let pool = Arc::new(SidecarPool::new(
        scripted("ack"),
        PoolConfig::default()
            .with_max_idle(2)
            .with_max_age(Duration::from_millis(60)),
    ));

    let lease = pool.acquire().await.unwrap();
    let old_pid = pid(&lease);
    lease.release().await.unwrap();

    tokio::time::sleep(Duration::from_millis(150)).await;

    let fresh = pool.acquire().await.unwrap();
    assert_ne!(
        pid(&fresh),
        old_pid,
        "an expired idle process must never be handed out"
    );
    let stats = pool.stats();
    assert!(stats.expired >= 1, "the age ceiling must expire: {stats:?}");
    assert_eq!(stats.reused, 0);

    fresh.release().await.unwrap();
    pool.shutdown().await;
}

#[tokio::test]
async fn pool_falls_back_to_fresh_spawn_when_reset_fails() {
    if !python_available() {
        eprintln!("skipping reset-failure test: no Python interpreter");
        return;
    }

    // The script acknowledges the release-time reset and rejects the
    // acquire-time probe, which is exactly the dirty-handover path the pool
    // must never take.
    let pool = Arc::new(SidecarPool::new(
        scripted("fail_second_reset"),
        PoolConfig::default().with_max_idle(1),
    ));

    let first = pool.acquire().await.unwrap();
    let first_pid = pid(&first);
    first
        .release()
        .await
        .expect("the first reset is acknowledged");

    let second = pool.acquire().await.unwrap();
    assert_ne!(
        pid(&second),
        first_pid,
        "a process whose probe reset failed must be replaced"
    );
    let stats = pool.stats();
    assert!(
        stats.reset_failures >= 1,
        "the failed probe must be counted: {stats:?}"
    );
    assert_eq!(stats.spawned, 2);
    assert_eq!(stats.reused, 0);

    // The replacement is a fresh process, so its first (release) reset is
    // acknowledged; only the stale process ever failed a probe.
    second
        .release()
        .await
        .expect("the fresh process acknowledges its first reset");
    assert_eq!(pool.stats().reset_failures, 1);

    pool.shutdown().await;
}

#[tokio::test]
async fn pool_replaces_a_dead_idle_process() {
    if !python_available() {
        eprintln!("skipping dead-process test: no Python interpreter");
        return;
    }

    let pool = Arc::new(SidecarPool::new(
        scripted("ack"),
        PoolConfig::default().with_max_idle(1),
    ));

    let lease = pool.acquire().await.unwrap();
    let old_pid = pid(&lease);
    let handle: Arc<SidecarClient> = Arc::clone(lease.client());
    lease.release().await.unwrap();

    handle.kill().await;

    let fresh = pool.acquire().await.unwrap();
    assert_ne!(pid(&fresh), old_pid);
    assert!(
        pool.stats().reset_failures >= 1,
        "a dead pooled process must be counted"
    );

    fresh.release().await.unwrap();
    pool.shutdown().await;
}

#[tokio::test]
async fn a_plugin_without_reset_keeps_one_process_per_session() {
    if !python_available() {
        eprintln!("skipping non-poolable plugin test: no Python interpreter");
        return;
    }

    let pool = Arc::new(SidecarPool::new(
        scripted("no_reset_capability"),
        PoolConfig::default().with_max_idle(2),
    ));

    let first = pool.acquire().await.unwrap();
    assert!(!first.client().supports_reset());
    let first_pid = pid(&first);
    first.release().await.unwrap();

    let second = pool.acquire().await.unwrap();
    assert_ne!(pid(&second), first_pid, "a non-poolable process is reaped");
    let stats = pool.stats();
    assert_eq!(stats.spawned, 2);
    assert_eq!(stats.shutdown, 1);
    assert_eq!(stats.reused, 0);

    second.release().await.unwrap();
    pool.shutdown().await;
    assert_eq!(pool.stats().shutdown, 2);
}

#[tokio::test]
async fn release_cancelled_mid_reset_kills_the_process() {
    if !python_available() {
        eprintln!("skipping cancellation test: no Python interpreter");
        return;
    }

    let pool = Arc::new(SidecarPool::new(
        scripted("never_ack"),
        PoolConfig::default().with_max_idle(2),
    ));

    let lease = pool.acquire().await.unwrap();
    let first_pid = pid(&lease);
    assert!(process_alive(first_pid), "the leased process must be live");

    // The fixture never acknowledges a reset and the configured reset timeout
    // is one second, so this timeout drops the release future while it is
    // suspended mid-reset. Pre-change, `released` was already true and the
    // process was left unowned and unkilled with no counter recording it.
    let cancelled = tokio::time::timeout(Duration::from_millis(100), lease.release()).await;
    assert!(
        cancelled.is_err(),
        "the release must still be suspended when the timeout expires"
    );

    // The detached release must finish on its own: the reset times out, the
    // failure is counted, and the process is killed and reaped.
    wait_until_process_exits(first_pid).await;

    let stats = pool.stats();
    assert_eq!(stats.spawned, 1);
    assert_eq!(stats.reused, 0);
    assert_eq!(stats.reset_failures, 1);
    assert_eq!(stats.evicted, 0);
    assert_eq!(stats.shutdown, 0);

    // The pool is still usable after a cancelled release.
    let next = pool.acquire().await.unwrap();
    let next_pid = pid(&next);
    assert_ne!(
        next_pid, first_pid,
        "the killed process must be replaced, not reused"
    );
    assert_eq!(pool.stats().spawned, 2);

    drop(next);
    wait_until_process_exits(next_pid).await;
    pool.shutdown().await;
}
