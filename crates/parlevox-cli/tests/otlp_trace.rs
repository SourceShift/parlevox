//! Hermetic wire test for the opt-in OTLP/HTTP trace and metric exporters.
//!
//! A localhost listener plays the collector. The test installs both exporters
//! against that listener, runs one fake-provider turn through a session that
//! carries the installed tracer and meter, flushes, and asserts the exporter
//! really posted a protobuf trace batch to `/v1/traces` and a protobuf metric
//! batch to `/v1/metrics`, both carrying the configured service name.

#![cfg(feature = "otlp")]

use std::time::Duration;

use parlevox_cli::export::{self, TraceExportConfig};
use parlevox_runtime::{ProviderNames, ProviderRegistry, Session, SessionConfig};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, oneshot};

/// Ceiling for the collector to observe both exports.
const CAPTURE_TIMEOUT: Duration = Duration::from_secs(15);

/// Service name the test configures and then looks for on the wire.
const SERVICE_NAME: &str = "parlevox-test";

/// One captured OTLP request, as the collector saw it.
struct CapturedRequest {
    path: String,
    content_type: String,
    body: Vec<u8>,
}

/// Read one HTTP request: request line and headers, then exactly
/// `content-length` body bytes.
async fn read_http_request(stream: &mut TcpStream) -> Option<CapturedRequest> {
    let mut buffer = Vec::new();
    let mut chunk = [0_u8; 2048];
    let header_end = loop {
        let read = stream.read(&mut chunk).await.ok()?;
        if read == 0 {
            return None;
        }
        buffer.extend_from_slice(&chunk[..read]);
        if let Some(index) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
            break index + 4;
        }
        if buffer.len() > 64 * 1024 {
            return None;
        }
    };

    let head = String::from_utf8_lossy(&buffer[..header_end]).into_owned();
    let path = head
        .split("\r\n")
        .next()?
        .split_whitespace()
        .nth(1)?
        .to_owned();
    let content_type = header_value(&head, "content-type").unwrap_or_default();
    let content_length = header_value(&head, "content-length")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);

    let mut body = buffer[header_end..].to_vec();
    while body.len() < content_length {
        let read = stream.read(&mut chunk).await.ok()?;
        if read == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..read]);
    }
    Some(CapturedRequest {
        path,
        content_type,
        body,
    })
}

/// Case-insensitive header lookup over a raw request head.
fn header_value(head: &str, wanted: &str) -> Option<String> {
    head.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        name.eq_ignore_ascii_case(wanted)
            .then(|| value.trim().to_owned())
    })
}

/// Accept and answer requests until the test stops the listener, forwarding
/// every captured request so the test can wait for both signal paths.
///
/// Each response asks for `connection: close`, so the exporter opens a fresh
/// connection per signal and a single sequential accept loop sees both.
async fn serve(
    listener: TcpListener,
    requests: mpsc::UnboundedSender<CapturedRequest>,
    mut stop: oneshot::Receiver<()>,
) {
    let mut requests = Some(requests);
    loop {
        tokio::select! {
            _ = &mut stop => return,
            accepted = listener.accept() => {
                let Ok((mut stream, _peer)) = accepted else {
                    return;
                };
                let Some(request) = read_http_request(&mut stream).await else {
                    continue;
                };
                let response = "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\
                                content-length: 2\r\nconnection: close\r\n\r\n{}";
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.flush().await;
                if let Some(sender) = requests.as_ref() {
                    if sender.send(request).is_err() {
                        requests = None;
                    }
                }
            }
        }
    }
}

/// Wait until both signal paths have arrived, returning them in
/// traces-then-metrics order regardless of arrival order.
async fn collect_signals(
    requests: &mut mpsc::UnboundedReceiver<CapturedRequest>,
) -> (CapturedRequest, CapturedRequest) {
    let mut traces = None;
    let mut metrics = None;
    let deadline = tokio::time::Instant::now() + CAPTURE_TIMEOUT;
    while traces.is_none() || metrics.is_none() {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        let request = tokio::time::timeout(remaining, requests.recv())
            .await
            .expect("both signals must export within the timeout")
            .expect("the collector request channel stays open");
        match request.path.as_str() {
            "/v1/traces" => traces = Some(request),
            "/v1/metrics" => metrics = Some(request),
            _ => {}
        }
    }
    (
        traces.expect("the exporter must post traces"),
        metrics.expect("the exporter must post metrics"),
    )
}

/// Assert one captured request is a protobuf OTLP batch for `signal`, then
/// print the DoD evidence line.
fn assert_signal(request: &CapturedRequest, signal: &str, expected_path: &str) {
    assert_eq!(request.path, expected_path, "{signal} signal path");
    assert!(
        request.content_type.starts_with("application/x-protobuf"),
        "{signal} content-type was {:?}",
        request.content_type
    );
    assert!(
        !request.body.is_empty(),
        "the {signal} export body must be non-empty"
    );
    let body = String::from_utf8_lossy(&request.body);
    assert!(
        body.contains(SERVICE_NAME),
        "the {signal} body must carry the configured service name"
    );
    println!(
        "signal={signal} path={} content-type={} body_len={} service_name={SERVICE_NAME}",
        request.path,
        request.content_type,
        request.body.len(),
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exports_one_turn_to_the_configured_otlp_endpoint() {
    // Arrange: a localhost "collector" and both exporters pointed at it.
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind a localhost collector");
    let port = listener
        .local_addr()
        .expect("bound listener has an address")
        .port();
    let (requests_tx, mut requests_rx) = mpsc::unbounded_channel();
    let (stop_tx, stop_rx) = oneshot::channel();
    let server = tokio::spawn(serve(listener, requests_tx, stop_rx));

    let config = TraceExportConfig {
        endpoint: format!("http://127.0.0.1:{port}"),
        service_name: SERVICE_NAME.to_owned(),
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

    // Act: one deterministic turn, then the documented exit drain.
    let (_report, events) = parlevox_cli::run_turn_events(session)
        .await
        .expect("the fake turn succeeds");
    export::flush_and_shutdown();
    let (traces, metrics) = collect_signals(&mut requests_rx).await;

    // Assert: both HTTP/protobuf signals, each carrying the service name.
    assert_signal(&traces, "traces", "/v1/traces");
    assert_signal(&metrics, "metrics", "/v1/metrics");
    assert!(
        events
            .iter()
            .any(|event| matches!(event, parlevox_runtime::SessionEvent::TurnDone { .. })),
        "the turn must complete before the export is asserted"
    );

    let _ = stop_tx.send(());
    server.await.expect("collector task finishes cleanly");
}
