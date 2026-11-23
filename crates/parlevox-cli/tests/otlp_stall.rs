//! Stall test for the opt-in OTLP/HTTP trace exporter.
//!
//! A localhost listener accepts the export request, reads its head, and then
//! never answers — a genuine black hole. The test asserts the load-bearing
//! property of the batch processor: span end only enqueues, so a turn against
//! an unresponsive collector stays prompt, while the documented exit drain
//! still completes inside its bound.
//!
//! This is its own integration-test binary on purpose: the OTLP tracer
//! provider is process-global, so sharing a binary with `otlp_trace.rs` would
//! make the second install a no-op.

#![cfg(feature = "otlp")]

use std::time::Duration;

use parlevox_cli::export::{self, TraceExportConfig};
use parlevox_runtime::{ProviderNames, ProviderRegistry, Session, SessionConfig, SessionEvent};
use tokio::io::AsyncReadExt;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::oneshot;

/// Hang guard around one turn. It cannot interrupt a turn blocked inside a
/// synchronous export; the promptness assertion on the measured elapsed time
/// is what catches that.
const TURN_GUARD: Duration = Duration::from_secs(5);

/// A turn must never wait for the collector.
const PROMPT_BOUND: Duration = Duration::from_secs(1);

/// Bound on the documented exit drain.
const DRAIN_BOUND: Duration = Duration::from_secs(10);

/// Ceiling for the collector to observe the export.
const CAPTURE_TIMEOUT: Duration = Duration::from_secs(15);

/// A collector that accepts one request, records its path, and then never
/// answers: no response head, no body, no close.
async fn black_hole(listener: TcpListener, path_tx: oneshot::Sender<String>) {
    let Ok((mut stream, _peer)) = listener.accept().await else {
        return;
    };
    let Some(path) = read_request_path(&mut stream).await else {
        return;
    };
    let _ = path_tx.send(path);
    std::future::pending::<()>().await;
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
    // Arrange: a localhost "collector" that accepts and then stalls forever,
    // and an exporter pointed at it.
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind a localhost collector");
    let port = listener
        .local_addr()
        .expect("bound listener has an address")
        .port();
    let (path_tx, path_rx) = oneshot::channel();
    let collector = tokio::spawn(black_hole(listener, path_tx));

    let config = TraceExportConfig {
        endpoint: format!("http://127.0.0.1:{port}"),
        service_name: "parlevox-stall".to_owned(),
    };
    export::install(&config).expect("install the OTLP exporter");

    let registry = ProviderRegistry::fakes().expect("fake registry builds");
    let providers = registry
        .resolve(&ProviderNames::default())
        .expect("fake providers resolve");
    let tracer = export::tracer().expect("an installed exporter yields a tracer");
    let session = Session::with_tracer(SessionConfig::default(), providers, tracer)
        .expect("session with the exporter tracer builds");

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

    // Assert the collector really saw a traces POST: the stall was a genuine
    // black hole, not a no-op exporter.
    let path = tokio::time::timeout(CAPTURE_TIMEOUT, path_rx)
        .await
        .expect("the collector must observe the export within the timeout")
        .expect("the collector must capture the request path");
    assert_eq!(path, "/v1/traces");

    assert!(
        events
            .iter()
            .any(|event| matches!(event, SessionEvent::TurnDone { .. })),
        "the turn must complete before the drain is measured"
    );

    println!(
        "otlp stall evidence turn_elapsed={turn_elapsed:?} \
         drain_elapsed={drain_elapsed:?} path={path}"
    );

    collector.abort();
}
