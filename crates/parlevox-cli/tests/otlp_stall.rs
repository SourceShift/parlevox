//! Stall test for the opt-in OTLP/HTTP trace and metric exporters.
//!
//! A localhost listener accepts every export request, reads its head, and then
//! never answers — a genuine black hole. The test asserts the load-bearing
//! property of the pipelined exporters: span end only enqueues, so a turn
//! against an unresponsive collector stays prompt, while the documented exit
//! drain still completes inside its shared budget with both providers
//! installed.
//!
//! This is its own integration-test binary on purpose: the OTLP providers are
//! process-global, so sharing a binary with `otlp_trace.rs` would make the
//! second install a no-op.

#![cfg(feature = "otlp")]

use std::time::Duration;

use parlevox_cli::export::{self, TraceExportConfig};
use parlevox_runtime::{ProviderNames, ProviderRegistry, Session, SessionConfig, SessionEvent};
use tokio::io::AsyncReadExt;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;

/// Hang guard around one turn. It cannot interrupt a turn blocked inside a
/// synchronous export; the promptness assertion on the measured elapsed time
/// is what catches that.
const TURN_GUARD: Duration = Duration::from_secs(5);

/// A turn must never wait for the collector.
const PROMPT_BOUND: Duration = Duration::from_secs(1);

/// Outer guard on the documented exit drain.
const DRAIN_BOUND: Duration = Duration::from_secs(10);

/// Shared-drain bound: traces and metrics must drain inside one five-second
/// budget plus margin. A stacked drain (exporting each signal twice, or
/// flushing then shutting each provider down) measures roughly 8s and fails
/// here; the 10s [`DRAIN_BOUND`] remains the outer hang guard.
const SHARED_DRAIN_BOUND: Duration = Duration::from_secs(7);

/// Ceiling for the collector to observe the exports.
const CAPTURE_TIMEOUT: Duration = Duration::from_secs(15);

/// A collector that accepts every connection, records its request path, and
/// then never answers: no response head, no body, no close.
///
/// Each connection is handled by its own pending subtask so the accept loop
/// keeps running and both the trace and the metric export can be observed.
async fn black_hole(listener: TcpListener, paths: mpsc::UnboundedSender<String>) {
    loop {
        let Ok((mut stream, _peer)) = listener.accept().await else {
            return;
        };
        let paths = paths.clone();
        tokio::spawn(async move {
            if let Some(path) = read_request_path(&mut stream).await {
                let _ = paths.send(path);
            }
            std::future::pending::<()>().await;
        });
    }
}

/// Read until the request head ends, then return the request path.
async fn read_request_path(stream: &mut TcpStream) -> Option<String> {
    let mut buffer = Vec::new();
    let mut chunk = [0_u8; 1024];
    loop {
        let read = stream.read(&mut chunk).await.ok()?;
        if read == 0 {
            return None;
        }
        buffer.extend_from_slice(&chunk[..read]);
        if let Some(index) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&buffer[..index + 4]);
            return head
                .split("\r\n")
                .next()?
                .split_whitespace()
                .nth(1)
                .map(str::to_owned);
        }
        if buffer.len() > 64 * 1024 {
            return None;
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_black_holed_collector_never_delays_a_turn() {
    // Arrange: a localhost "collector" that accepts every connection and then
    // stalls forever, and both exporters pointed at it.
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind a localhost collector");
    let port = listener
        .local_addr()
        .expect("bound listener has an address")
        .port();
    let (paths_tx, mut paths_rx) = mpsc::unbounded_channel();
    let collector = tokio::spawn(black_hole(listener, paths_tx));

    let config = TraceExportConfig {
        endpoint: format!("http://127.0.0.1:{port}"),
        service_name: "parlevox-stall".to_owned(),
    };
    export::install(&config).expect("install the OTLP exporters");

    let registry = ProviderRegistry::fakes().expect("fake registry builds");
    let providers = registry
        .resolve(&ProviderNames::default())
        .expect("fake providers resolve");
    let tracer = export::tracer().expect("an installed exporter yields a tracer");
    let meter = export::meter().expect("an installed exporter yields a meter");
    let session = Session::with_tracer(SessionConfig::default(), providers, tracer)
        .expect("session with the exporter tracer builds")
        .with_meter(meter);

    // Act: one deterministic turn against the black hole.
    let turn_start = tokio::time::Instant::now();
    let turn = tokio::time::timeout(TURN_GUARD, parlevox_cli::run_turn_events(session)).await;
    let turn_elapsed = turn_start.elapsed();

    // Assert promptness BEFORE unwrapping the guard: the timeout polls the
    // inner future before checking its deadline, so a turn that blocks inside
    // span end only returns once the export gives up — the measured elapsed
    // time, not the guard's panic, must be the failure in the negative control.
    assert!(
        turn_elapsed < PROMPT_BOUND,
        "the turn must not wait for the collector (elapsed {turn_elapsed:?})"
    );
    let (_report, events) = turn
        .expect("the turn must finish within the hang guard")
        .expect("the fake turn succeeds");

    // Act: the documented exit drain, on a blocking thread so the bound is a
    // real guard. A failed flush is tolerated (it is reported on stderr).
    let drain_start = tokio::time::Instant::now();
    let drained = tokio::time::timeout(
        DRAIN_BOUND,
        tokio::task::spawn_blocking(export::flush_and_shutdown),
    )
    .await;
    let drain_elapsed = drain_start.elapsed();
    assert!(
        drained.is_ok(),
        "the exit drain must finish within {DRAIN_BOUND:?} (elapsed {drain_elapsed:?})"
    );
    drained
        .expect("the drain guard times out only after the measurement")
        .expect("the drain task must not panic");
    assert!(
        drain_elapsed < SHARED_DRAIN_BOUND,
        "the drain must share one budget across both providers, not stack \
         their timeouts (elapsed {drain_elapsed:?})"
    );

    // Assert the collector really saw the exports: the stall was a genuine
    // black hole, not a no-op exporter. The trace drain posts first; collect
    // whatever else arrives until the capture timeout.
    let mut paths = Vec::new();
    let deadline = tokio::time::Instant::now() + CAPTURE_TIMEOUT;
    while paths.len() < 2 {
        let Some(remaining) = deadline
            .checked_duration_since(tokio::time::Instant::now())
            .filter(|remaining| !remaining.is_zero())
        else {
            break;
        };
        match tokio::time::timeout(remaining, paths_rx.recv()).await {
            Ok(Some(path)) => paths.push(path),
            Ok(None) | Err(_) => break,
        }
    }
    assert_eq!(
        paths.first().map(String::as_str),
        Some("/v1/traces"),
        "the trace drain must post first (paths {paths:?})"
    );
    assert!(
        paths.iter().any(|path| path == "/v1/metrics"),
        "the meter export must still be attempted against the black hole \
         (paths {paths:?})"
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, SessionEvent::TurnDone { .. })),
        "the turn must complete before the drain is measured"
    );

    println!(
        "otlp stall evidence turn_elapsed={turn_elapsed:?} \
         drain_elapsed={drain_elapsed:?} paths={paths:?}"
    );

    collector.abort();
}
