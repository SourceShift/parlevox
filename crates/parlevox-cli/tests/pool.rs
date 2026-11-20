//! Two complete sessions through one warm pool.
//!
//! The baseline is a directly spawned plugin; the pooled run leases the same
//! process twice. Both sessions are compared through the canonical
//! timestamp-free projection the parity test uses, so reuse must be invisible
//! in the event trace, the report and the audio bytes.
//!
//! Like the differential parity test, this prints an explicit skip when the
//! bundled SDK or interpreter is unavailable. It is not `#[ignore]`d.

use std::sync::Arc;

use parlevox_providers::{FAKE_REPLY, FAKE_TRANSCRIPT};
use parlevox_sidecar::{sdk_is_available, PoolConfig, SidecarClient, SidecarConfig, SidecarPool};

fn python_probe() -> bool {
    let python = std::env::var("PARLEVOX_PYTHON").unwrap_or_else(|_| "python3".into());
    std::process::Command::new(python)
        .args(["-c", "import sys"])
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

#[tokio::test]
async fn pool_reuses_one_process_for_two_sessions() {
    if !sdk_is_available() || !python_probe() {
        eprintln!("skipping pool reuse: the bundled SDK or interpreter is unavailable");
        return;
    }

    // Single-session, fresh-spawn baseline.
    let baseline_client = Arc::new(
        SidecarClient::spawn(SidecarConfig::python_sdk().expect("resolve the bundled SDK"))
            .await
            .expect("spawn the baseline sidecar"),
    );
    let baseline_session = parlevox_cli::sidecar_session(Arc::clone(&baseline_client))
        .expect("build baseline session");
    let (baseline_report, baseline_events) = parlevox_cli::run_turn_events(baseline_session)
        .await
        .expect("the baseline turn completes");
    baseline_client
        .shutdown()
        .await
        .expect("the baseline child is reaped");
    let baseline = parlevox_cli::project_events(&baseline_events);

    let pool = Arc::new(SidecarPool::new(
        SidecarConfig::python_sdk().expect("resolve the bundled SDK"),
        PoolConfig::default().with_max_idle(1),
    ));

    let mut pids = Vec::new();
    let mut projections = Vec::new();
    for _ in 0..2 {
        let lease = pool.acquire().await.expect("acquire a pooled plugin");
        pids.push(lease.client().pid().expect("spawn captures the plugin pid"));

        let session = parlevox_cli::sidecar_session(Arc::clone(lease.client()))
            .expect("build the pooled session");
        let (report, events) = parlevox_cli::run_turn_events(session)
            .await
            .expect("the pooled turn completes");
        assert_eq!(report.transcript, FAKE_TRANSCRIPT);
        assert_eq!(report.reply, FAKE_REPLY);
        projections.push(parlevox_cli::project_events(&events));

        lease.release().await.expect("release an idle plugin");
    }

    let stats = pool.stats();
    pool.shutdown().await;

    assert_eq!(
        pids[0], pids[1],
        "both sessions must run on the same plugin process"
    );
    assert_eq!(stats.spawned, 1, "one process should serve both sessions");
    assert!(stats.reused >= 1, "the second lease must be reused");
    assert_eq!(stats.reset_failures, 0);
    assert_eq!(projections.len(), 2);
    for projection in &projections {
        assert_eq!(
            projection, &baseline,
            "a pooled session diverged from the fresh-spawn baseline"
        );
    }

    assert_eq!(baseline_report.transcript, FAKE_TRANSCRIPT);
}

/// `--pool` only means something when there is a plugin process to pool.
///
/// The fakes spawn no process, so the combination must be refused by name
/// rather than silently ignored or misread as an unknown argument.
#[test]
fn demo_fake_pool_is_rejected() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_parlevox"))
        .args(["demo", "--fake", "--pool"])
        .output()
        .expect("run the parlevox binary");

    assert!(
        !output.status.success(),
        "demo --fake --pool must fail, stdout was: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("demo --fake --pool"),
        "stderr must name the rejected combination, was: {stderr}"
    );
}
