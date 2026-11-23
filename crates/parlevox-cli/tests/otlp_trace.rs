//! Hermetic wire test for the opt-in OTLP/HTTP trace exporter.
//!
//! A localhost listener plays the collector. The test installs the exporter
//! against that listener, runs one fake-provider turn through
//! [`Session::with_tracer`], flushes, and asserts the exporter really posted a
//! protobuf trace batch to `/v1/traces` carrying the configured service name.

#![cfg(feature = "otlp")]

use std::time::Duration;

use parlevox_cli::export::{self, TraceExportConfig};
use parlevox_runtime::{ProviderNames, ProviderRegistry, Session, SessionConfig};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::oneshot;

/// Ceiling for the collector to observe the export.
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

/// Accept requests until the test stops the listener, capturing the first one.
async fn serve(
    listener: TcpListener,
    capture: oneshot::Sender<CapturedRequest>,
    mut stop: oneshot::Receiver<()>,
) {
    let mut capture = Some(capture);
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
                if let Some(sender) = capture.take() {
                    let _ = sender.send(request);
                }
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exports_one_turn_to_the_configured_otlp_endpoint() {
    // Arrange: a localhost "collector" and an exporter pointed at it.
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind a localhost collector");
    let port = listener
        .local_addr()
        .expect("bound listener has an address")
        .port();
    let (capture_tx, capture_rx) = oneshot::channel();
    let (stop_tx, stop_rx) = oneshot::channel();
    let server = tokio::spawn(serve(listener, capture_tx, stop_rx));

    let config = TraceExportConfig {
        endpoint: format!("http://127.0.0.1:{port}"),
        service_name: SERVICE_NAME.to_owned(),
    };
    export::install(&config).expect("install the OTLP exporter");

    let registry = ProviderRegistry::fakes().expect("fake registry builds");
    let providers = registry
        .resolve(&ProviderNames::default())
        .expect("fake providers resolve");
    let tracer = export::tracer().expect("an installed exporter yields a tracer");
    let session = Session::with_tracer(SessionConfig::default(), providers, tracer)
        .expect("session with the exporter tracer builds");

    // Act: one deterministic turn, then the documented exit flush.
    let (_report, events) = parlevox_cli::run_turn_events(session)
        .await
        .expect("the fake turn succeeds");
    export::flush_and_shutdown();
    let captured = tokio::time::timeout(CAPTURE_TIMEOUT, capture_rx)
        .await
        .expect("the exporter must post within the timeout")
        .expect("the collector must capture one request");

    // Assert: HTTP/protobuf on the traces path, carrying the service name.
    assert_eq!(captured.path, "/v1/traces");
    assert!(
        captured.content_type.starts_with("application/x-protobuf"),
        "content-type was {:?}",
        captured.content_type
    );
    assert!(
        !captured.body.is_empty(),
        "the export body must be non-empty"
    );
    let body = String::from_utf8_lossy(&captured.body);
    assert!(
        body.contains(SERVICE_NAME),
        "the body must carry the configured service name"
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, parlevox_runtime::SessionEvent::TurnDone { .. })),
        "the turn must complete before the export is asserted"
    );

    println!(
        "otlp evidence path={} content-type={} body_len={} service_name={}",
        captured.path,
        captured.content_type,
        captured.body.len(),
        SERVICE_NAME
    );

    let _ = stop_tx.send(());
    server.await.expect("collector task finishes cleanly");
}
