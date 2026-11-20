//! Pool bounds and failure handling, driven by a stdlib-only scripted plugin.
//!
//! The idle cap, the age ceiling, the acquire-time probe reset and the
//! dead-process fallback are all observable through [`PoolStats`] and the
//! child pid, so no request work is needed here. Every test shuts the pool
//! down, so no child process survives a failure.

use std::sync::Arc;
use std::time::Duration;

use parlevox_core::ParlevoxError;
use parlevox_sidecar::{PoolConfig, SidecarConfig, SidecarPool};

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
    // A clean release pools the process; no client clone may outlive the lease.
    lease.release().await.unwrap();

    // Kill the now-idle process out of band so the acquire probe must observe
    // the death. SIGKILL leaves an unreaped zombie, which `kill -0` still
    // reports as alive; wait a beat, then let the failed probe and its
    // `client.kill()` reap the child instead of polling for exit.
    let signalled = std::process::Command::new("kill")
        .args(["-9", &old_pid.to_string()])
        .status()
        .expect("send SIGKILL to the idle process");
    assert!(signalled.success(), "SIGKILL must reach the idle process");
    tokio::time::sleep(Duration::from_millis(50)).await;

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
async fn release_refuses_while_a_client_handle_is_alive() {
    if !python_available() {
        eprintln!("skipping release-refusal test: no Python interpreter");
        return;
    }

    let pool = Arc::new(SidecarPool::new(
        scripted("ack"),
        PoolConfig::default().with_max_idle(1),
    ));

    let lease = pool.acquire().await.unwrap();
    let first_pid = pid(&lease);
    // An adapter handle outlives the lease: the process may still be in use, so
    // the release must refuse rather than pool it.
    let handle = Arc::clone(lease.client());

    let err = lease
        .release()
        .await
        .expect_err("an outstanding client handle must refuse the release");
    assert!(
        matches!(&err, ParlevoxError::Protocol(_)),
        "a live handle is a protocol violation, got {err:?}"
    );
    let rendered = err.to_string();
    assert!(
        rendered.starts_with("protocol violation:"),
        "the refusal must be protocol-class: {rendered}"
    );
    assert!(
        rendered.contains("1 handle(s)") && rendered.contains("client()"),
        "the refusal must name the live handles and the fix: {rendered}"
    );

    // The chosen fate: the refused process is killed, never pooled, and counted
    // exactly once while no other counter moves.
    wait_until_process_exits(first_pid).await;
    let stats = pool.stats();
    assert_eq!(stats.release_refused, 1);
    assert_eq!(stats.shutdown, 0);
    assert_eq!(stats.reset_failures, 0);
    assert_eq!(stats.reused, 0);
    assert_eq!(stats.spawned, 1);

    // Dropping the stray handle leaves the pool empty, so the next lease is a
    // fresh process rather than the refused one.
    drop(handle);
    let next = pool.acquire().await.unwrap();
    assert_ne!(
        pid(&next),
        first_pid,
        "the refused process must not be pooled"
    );
    assert_eq!(pool.stats().spawned, 2);

    next.release().await.unwrap();
    pool.shutdown().await;
}

#[tokio::test]
async fn release_with_no_outstanding_clone_still_pools() {
    if !python_available() {
        eprintln!("skipping clean-release test: no Python interpreter");
        return;
    }

    let pool = Arc::new(SidecarPool::new(
        scripted("ack"),
        PoolConfig::default().with_max_idle(1),
    ));

    let lease = pool.acquire().await.unwrap();
    let first_pid = pid(&lease);
    lease
        .release()
        .await
        .expect("a clean release pools the process");
    assert_eq!(pool.stats().release_refused, 0);

    let reacquired = pool.acquire().await.unwrap();
    assert_eq!(pid(&reacquired), first_pid, "a clean release is reused");
    let stats = pool.stats();
    assert_eq!(stats.reused, 1);
    assert_eq!(stats.release_refused, 0);

    reacquired.release().await.unwrap();
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

#[tokio::test]
async fn pool_retires_a_process_at_max_reuses() {
    if !python_available() {
        eprintln!("skipping retirement test: no Python interpreter");
        return;
    }

    // With `max_reuses = 1` a process serves one initial lease plus one reuse
    // and is retired at its second return.
    let pool = Arc::new(SidecarPool::new(
        scripted("ack"),
        PoolConfig::default().with_max_idle(1).with_max_reuses(1),
    ));

    let first = pool.acquire().await.unwrap();
    let first_pid = pid(&first);
    first.release().await.unwrap();

    let reused = pool.acquire().await.unwrap();
    assert_eq!(
        pid(&reused),
        first_pid,
        "the process is reused exactly once"
    );
    assert_eq!(pool.stats().reused, 1);
    reused.release().await.unwrap();

    // The second return reaches the ceiling: the process is retired instead of
    // pooled, and no other fate counter moves for it.
    let stats = pool.stats();
    assert_eq!(stats.retired, 1, "the ceiling retires exactly one process");
    assert_eq!(stats.shutdown, 0, "a retirement is not a shutdown");
    assert_eq!(stats.reset_failures, 0);

    let fresh = pool.acquire().await.unwrap();
    assert_ne!(
        pid(&fresh),
        first_pid,
        "a retired process must never be handed out again"
    );
    assert_eq!(pool.stats().spawned, 2);

    fresh.release().await.unwrap();
    pool.shutdown().await;
}

#[tokio::test]
async fn pool_with_zero_max_reuses_is_unlimited() {
    if !python_available() {
        eprintln!("skipping unlimited-reuse test: no Python interpreter");
        return;
    }

    let pool = Arc::new(SidecarPool::new(
        scripted("ack"),
        PoolConfig::default().with_max_idle(1).with_max_reuses(0),
    ));

    let first = pool.acquire().await.unwrap();
    let stable_pid = pid(&first);
    first.release().await.unwrap();

    // `0` is the opposite of a ceiling, unlike `max_idle = 0`: the same process
    // keeps coming back past a would-be ceiling of one.
    for round in 0..3 {
        let lease = pool.acquire().await.unwrap();
        assert_eq!(
            pid(&lease),
            stable_pid,
            "round {round} must reuse the process"
        );
        lease.release().await.unwrap();
    }

    let stats = pool.stats();
    assert_eq!(stats.retired, 0, "an unlimited ceiling never retires");
    assert_eq!(stats.shutdown, 0);
    assert_eq!(stats.spawned, 1, "the one process serves every round");

    pool.shutdown().await;
}

#[tokio::test]
async fn pool_distinguishes_retirement_from_configuration_refusal() {
    if !python_available() {
        eprintln!("skipping refusal-reason test: no Python interpreter");
        return;
    }

    // Two pools, deliberately: `max_idle = 0` disables pooling, so no process
    // can ever accumulate a reuse in that pool, while an age-based
    // configuration refusal is timing-dependent. Retirement needs pooling
    // enabled plus a reached ceiling, so one pool cannot deterministically
    // produce both fates.
    let retiring = Arc::new(SidecarPool::new(
        scripted("ack"),
        PoolConfig::default().with_max_idle(1).with_max_reuses(1),
    ));
    let first = retiring.acquire().await.unwrap();
    first.release().await.unwrap();
    let reused = retiring.acquire().await.unwrap();
    reused.release().await.unwrap();

    let retired_stats = retiring.stats();
    assert_eq!(retired_stats.retired, 1, "the ceiling is a retirement");
    assert_eq!(retired_stats.shutdown, 0);
    retiring.shutdown().await;

    let disabled = Arc::new(SidecarPool::new(
        scripted("ack"),
        PoolConfig::default().with_max_idle(0),
    ));
    let lease = disabled.acquire().await.unwrap();
    lease.release().await.unwrap();

    let disabled_stats = disabled.stats();
    assert_eq!(
        disabled_stats.shutdown, 1,
        "disabled pooling is a configuration refusal"
    );
    assert_eq!(disabled_stats.retired, 0);
    disabled.shutdown().await;
}
