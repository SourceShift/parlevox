//! End-to-end WebSocket transport tests.
//!
//! Every test binds an ephemeral loopback port, drives a real
//! `tokio_tungstenite` client, and uses the in-process fake providers. No test
//! touches the network, the filesystem, or a key.

use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::{Arc, Mutex, PoisonError};
use std::task::{Context, Poll};
use std::time::Duration;

use async_trait::async_trait;
use futures_core::Stream;
use futures_util::{SinkExt, StreamExt};
use parlevox_core::ParlevoxError;
use parlevox_providers::{
    AudioChunk, FakeLlm, FakeStt, FakeVad, Modality, ProviderManifest, ProviderStream,
    SynthesizeRequest, TtsProvider,
};
use parlevox_runtime::{ProviderNames, ProviderRegistry, Session, SessionConfig};
use parlevox_transport_ws::{SessionFactory, WsConfig, WsTransport};
use serde_json::Value;
use tokio::net::TcpStream;
use tokio::sync::oneshot;
use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};
use tokio_util::sync::CancellationToken;

const SAMPLE_RATE: u32 = 24_000;
const CHANNELS: u16 = 1;
const FRAME_SAMPLES: usize = 160;
const TIMEOUT: Duration = Duration::from_secs(5);

type ClientWs = WebSocketStream<MaybeTlsStream<TcpStream>>;

fn fake_session() -> Result<Session, ParlevoxError> {
    let providers = ProviderRegistry::fakes()?.resolve(&ProviderNames::default())?;
    Session::new(
        SessionConfig {
            sample_rate: SAMPLE_RATE,
            ..SessionConfig::default()
        },
        providers,
    )
}

fn fake_factory() -> SessionFactory {
    Arc::new(fake_session)
}

async fn bind_default() -> WsTransport {
    WsTransport::bind(([127, 0, 0, 1], 0).into(), fake_factory())
        .await
        .expect("the transport binds an ephemeral loopback port")
}

async fn connect(addr: SocketAddr) -> ClientWs {
    let (ws, _response) = connect_async(format!("ws://{addr}"))
        .await
        .expect("the client connects");
    ws
}

async fn connect_ready(addr: SocketAddr) -> ClientWs {
    let mut ws = connect(addr).await;
    let ready = next_json(&mut ws).await;
    assert_eq!(ready["type"], "ready");
    assert_eq!(ready["protocol_version"], 1);
    ws
}

async fn next_message(ws: &mut ClientWs) -> Message {
    tokio::time::timeout(TIMEOUT, ws.next())
        .await
        .expect("a message arrives within the timeout")
        .expect("the connection stays open")
        .expect("the frame is valid")
}

async fn next_json(ws: &mut ClientWs) -> Value {
    match next_message(ws).await {
        Message::Text(text) => serde_json::from_str(text.as_str()).expect("server JSON parses"),
        other => panic!("expected a JSON text frame, got {other:?}"),
    }
}

fn pcm_bytes(samples: &[i16]) -> Vec<u8> {
    samples
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect()
}

async fn send_audio(ws: &mut ClientWs, samples: &[i16]) {
    ws.send(Message::binary(pcm_bytes(samples)))
        .await
        .expect("the client sends audio");
}

async fn send_control(ws: &mut ClientWs, json: &'static str) {
    ws.send(Message::text(json))
        .await
        .expect("the client sends control");
}

/// Everything a client saw between `utterance_end` and `turn_done`.
#[derive(Debug)]
struct Turn {
    events: Vec<Value>,
    audio: Vec<Vec<u8>>,
    done: Value,
}

async fn read_turn(ws: &mut ClientWs) -> Turn {
    let mut events = Vec::new();
    let mut audio = Vec::new();
    loop {
        match next_message(ws).await {
            Message::Text(text) => {
                let value: Value = serde_json::from_str(text.as_str()).expect("server JSON parses");
                if value["type"] == "turn_done" {
                    return Turn {
                        events,
                        audio,
                        done: value,
                    };
                }
                events.push(value);
            }
            Message::Binary(bytes) => audio.push(bytes.to_vec()),
            other => panic!("unexpected frame while reading a turn: {other:?}"),
        }
    }
}

fn event_types(turn: &Turn) -> Vec<&str> {
    turn.events
        .iter()
        .filter_map(|event| event["type"].as_str())
        .collect()
}

async fn run_simple_turn(ws: &mut ClientWs) -> Turn {
    send_audio(ws, &vec![1i16; FRAME_SAMPLES]).await;
    send_control(ws, r#"{"type":"utterance_end"}"#).await;
    read_turn(ws).await
}

#[tokio::test]
async fn full_fake_turn_over_websocket() {
    // Arrange
    let transport = bind_default().await;
    let mut ws = connect_ready(transport.local_addr()).await;

    // Act
    send_audio(&mut ws, &vec![1i16; FRAME_SAMPLES]).await;
    send_audio(&mut ws, &vec![2i16; FRAME_SAMPLES]).await;
    send_control(&mut ws, r#"{"type":"utterance_end"}"#).await;
    let turn = read_turn(&mut ws).await;

    // Assert
    let types = event_types(&turn);
    assert_eq!(types.first().copied(), Some("turn_started"));
    assert!(types.contains(&"transcript_final"));
    assert!(types.contains(&"reply_delta"));

    let transcript = turn
        .events
        .iter()
        .find(|event| event["type"] == "transcript_final")
        .expect("a final transcript was emitted");
    assert_eq!(transcript["text"], "hello world");

    assert_eq!(turn.audio.len(), 2, "both fake TTS frames arrive as binary");
    assert_eq!(turn.done["turn"].as_u64(), Some(1));
    assert_eq!(turn.done["audio_frames"].as_u64(), Some(2));
    assert_eq!(turn.done["dropped_frames"].as_u64(), Some(0));
    assert_eq!(turn.done["dropped_ms"].as_u64(), Some(0));
    assert_eq!(turn.done["dropped_events"].as_u64(), Some(0));
    assert_eq!(turn.done["interrupted"].as_bool(), Some(false));
    assert_eq!(turn.done["ingress_dropped_frames"].as_u64(), Some(0));
    assert_eq!(turn.done["ingress_dropped_ms"].as_u64(), Some(0));

    transport.shutdown().await;
}

#[tokio::test]
async fn dropped_ingress_frames_are_counted() {
    // Arrange: the ring holds exactly two of the four messages the client
    // sends, so the oldest two are evicted before the commit.
    let config = WsConfig::with_ingress(SAMPLE_RATE, CHANNELS, FRAME_SAMPLES * 2)
        .expect("a valid ingress configuration");
    let transport =
        WsTransport::bind_with_config(([127, 0, 0, 1], 0).into(), config, fake_factory())
            .await
            .expect("the transport binds");
    let mut ws = connect_ready(transport.local_addr()).await;

    // Act
    for value in [1i16, 2, 3, 4] {
        send_audio(&mut ws, &vec![value; FRAME_SAMPLES]).await;
    }
    send_control(&mut ws, r#"{"type":"utterance_end"}"#).await;
    let turn = read_turn(&mut ws).await;

    // Assert
    let dropped = turn.done["ingress_dropped_frames"]
        .as_u64()
        .expect("a frame count");
    assert!(dropped > 0, "overflow must be visible to the client");
    assert!(
        turn.done["ingress_dropped_ms"].as_u64().unwrap_or(0) > 0,
        "lost audio is also reported as milliseconds"
    );
    assert_eq!(turn.done["turn"].as_u64(), Some(1));
    assert!(
        event_types(&turn).contains(&"turn_started"),
        "the retained audio still produced a turn"
    );
    assert_eq!(turn.audio.len(), 2);

    transport.shutdown().await;
}

#[tokio::test]
async fn malformed_control_closes_with_policy_violation() {
    // Arrange
    let transport = bind_default().await;
    let addr = transport.local_addr();
    let mut ws = connect_ready(addr).await;

    // Act
    send_control(&mut ws, r#"{"type":"not_a_control"}"#).await;
    let error = next_json(&mut ws).await;
    let close = next_message(&mut ws).await;

    // Assert
    assert_eq!(error["type"], "error");
    assert_eq!(error["class"], "protocol");
    match close {
        Message::Close(Some(frame)) => assert_eq!(frame.code, CloseCode::Policy),
        other => panic!("expected a 1008 close frame, got {other:?}"),
    }

    // The listener and its factory are untouched: a second connection serves a
    // complete turn with a fresh session.
    let mut second = connect_ready(addr).await;
    let turn = run_simple_turn(&mut second).await;
    assert_eq!(turn.done["turn"].as_u64(), Some(1));
    assert_eq!(turn.audio.len(), 2);

    transport.shutdown().await;
}

#[tokio::test]
async fn two_connections_have_independent_sessions() {
    // Arrange
    let transport = bind_default().await;
    let addr = transport.local_addr();
    let mut first = connect_ready(addr).await;
    let mut second = connect_ready(addr).await;

    // Act + Assert: both begin at turn 1.
    let first_turn = run_simple_turn(&mut first).await;
    let second_turn = run_simple_turn(&mut second).await;
    assert_eq!(first_turn.done["turn"].as_u64(), Some(1));
    assert_eq!(second_turn.done["turn"].as_u64(), Some(1));

    // Advancing one connection does not advance the other.
    let first_again = run_simple_turn(&mut first).await;
    assert_eq!(first_again.done["turn"].as_u64(), Some(2));
    let second_again = run_simple_turn(&mut second).await;
    assert_eq!(second_again.done["turn"].as_u64(), Some(2));

    transport.shutdown().await;
}

/// A TTS stream that announces "I am producing" on its first poll and then
/// parks until it is cancelled.
///
/// The turn parks inside a genuine provider stream, so the test can barge in
/// while the turn is unreachable from the connection's select loop. The stream
/// registers a waker with the cancellation token, so the barge-in cancels the
/// turn deterministically without any sleep or timing assumption.
struct GatedTtsStream {
    cancel: CancellationToken,
    chunk: Option<AudioChunk>,
    producing: Option<oneshot::Sender<()>>,
    permission: oneshot::Receiver<()>,
    announced: bool,
    done: bool,
}

impl Stream for GatedTtsStream {
    type Item = Result<AudioChunk, ParlevoxError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.done {
            return Poll::Ready(None);
        }
        if !self.announced {
            self.announced = true;
            if let Some(producing) = self.producing.take() {
                let _ = producing.send(());
            }
        }
        {
            let wait = self.cancel.cancelled();
            if std::pin::pin!(wait).poll(cx).is_ready() {
                self.done = true;
                return Poll::Ready(Some(Err(ParlevoxError::Cancelled)));
            }
        }
        match Pin::new(&mut self.permission).poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Ok(())) => {
                self.done = true;
                match self.chunk.take() {
                    Some(chunk) => Poll::Ready(Some(Ok(chunk))),
                    None => Poll::Ready(None),
                }
            }
            Poll::Ready(Err(_)) => {
                self.done = true;
                Poll::Ready(None)
            }
        }
    }
}

/// TTS provider that hands out exactly one [`GatedTtsStream`].
struct GatedTts {
    manifest: ProviderManifest,
    producing: Mutex<Option<oneshot::Sender<()>>>,
    permission: Mutex<Option<oneshot::Receiver<()>>>,
}

impl GatedTts {
    fn new(producing: oneshot::Sender<()>, permission: oneshot::Receiver<()>) -> Self {
        GatedTts {
            manifest: ProviderManifest::new("gated-tts", Modality::Tts),
            producing: Mutex::new(Some(producing)),
            permission: Mutex::new(Some(permission)),
        }
    }
}

#[async_trait]
impl TtsProvider for GatedTts {
    fn manifest(&self) -> &ProviderManifest {
        &self.manifest
    }

    async fn synthesize(
        &self,
        request: SynthesizeRequest,
        cancel: CancellationToken,
    ) -> Result<ProviderStream<AudioChunk>, ParlevoxError> {
        let producing = self
            .producing
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        let permission = self
            .permission
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        let Some(permission) = permission else {
            return Err(ParlevoxError::Protocol(
                "gated tts synthesized more than once".into(),
            ));
        };
        let chunk = AudioChunk::new(request.sample_rate, request.channels, vec![1; 16]);
        Ok(Box::pin(GatedTtsStream {
            cancel,
            chunk: Some(chunk),
            producing,
            permission,
            announced: false,
            done: false,
        }))
    }
}

fn gated_factory(tts: Arc<dyn TtsProvider>) -> SessionFactory {
    Arc::new(move || {
        let mut registry = ProviderRegistry::new();
        registry.register_stt(Arc::new(FakeStt::new()))?;
        registry.register_llm(Arc::new(FakeLlm::new()))?;
        registry.register_tts(Arc::clone(&tts))?;
        registry.register_vad(Arc::new(FakeVad::new()))?;
        let names = ProviderNames {
            stt: "fake-stt".into(),
            llm: "fake-llm".into(),
            tts: "gated-tts".into(),
            vad: "fake-vad".into(),
        };
        let providers = registry.resolve(&names)?;
        Session::new(
            SessionConfig {
                sample_rate: SAMPLE_RATE,
                ..SessionConfig::default()
            },
            providers,
        )
    })
}

#[tokio::test]
async fn barge_in_during_reply_is_reported() {
    // Arrange: a gated TTS parks the turn inside synthesis.
    let (producing_tx, producing_rx) = oneshot::channel();
    let (permission_tx, permission_rx) = oneshot::channel();
    let tts = Arc::new(GatedTts::new(producing_tx, permission_rx));
    let transport = WsTransport::bind_with_config(
        ([127, 0, 0, 1], 0).into(),
        WsConfig::default(),
        gated_factory(tts),
    )
    .await
    .expect("the transport binds");
    let mut ws = connect_ready(transport.local_addr()).await;

    // Act
    send_audio(&mut ws, &vec![1i16; FRAME_SAMPLES]).await;
    send_control(&mut ws, r#"{"type":"utterance_end"}"#).await;
    tokio::time::timeout(TIMEOUT, producing_rx)
        .await
        .expect("the turn reaches synthesis within the timeout")
        .expect("the gated tts announces that it is producing");
    send_control(&mut ws, r#"{"type":"barge_in"}"#).await;
    // Hold the permission sender: the stream cannot finish on its own, so the
    // only way this turn ends is the barge-in cancelling it.
    let _permission_tx = permission_tx;
    let turn = read_turn(&mut ws).await;

    // Assert
    assert_eq!(turn.done["interrupted"].as_bool(), Some(true));
    assert_eq!(turn.done["turn"].as_u64(), Some(1));

    transport.shutdown().await;
}

/// Assert that binding `config` on an ephemeral loopback port fails with a
/// protocol error whose message names `field`.
///
/// Port `0` would bind successfully if it were reached, so an `Err` proves the
/// configuration was rejected before a listener existed.
async fn assert_bind_rejects_naming(config: WsConfig, field: &str) {
    match WsTransport::bind_with_config(([127, 0, 0, 1], 0).into(), config, fake_factory()).await {
        Err(ParlevoxError::Protocol(message)) => {
            assert!(
                message.contains(field),
                "the error must name `{field}`, got `{message}`"
            );
        }
        other => panic!("expected a protocol error naming `{field}`, got {other:?}"),
    }
}

// `WsConfig` is `#[non_exhaustive]`, so integration tests cannot use
// struct-update syntax; each test mutates an owned default instead.

#[tokio::test]
#[allow(clippy::field_reassign_with_default)]
async fn bind_rejects_zero_sample_rate() {
    // Arrange
    let mut config = WsConfig::default();
    config.sample_rate = 0;

    // Act + Assert
    assert_bind_rejects_naming(config, "sample_rate").await;
}

#[tokio::test]
#[allow(clippy::field_reassign_with_default)]
async fn bind_rejects_zero_max_message_bytes() {
    // Arrange
    let mut config = WsConfig::default();
    config.max_message_bytes = 0;

    // Act + Assert
    assert_bind_rejects_naming(config, "max_message_bytes").await;
}

#[tokio::test]
#[allow(clippy::field_reassign_with_default)]
async fn bind_rejects_zero_ingress_capacity() {
    // Arrange
    let mut config = WsConfig::default();
    config.ingress_capacity_samples = 0;

    // Act + Assert
    assert_bind_rejects_naming(config, "ingress_capacity_samples").await;
}
