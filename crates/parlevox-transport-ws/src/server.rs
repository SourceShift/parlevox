//! The WebSocket listener and one-session-per-connection server loop.
//!
//! [`WsTransport::bind`] owns a [`tokio::net::TcpListener`]. Each accepted
//! socket is handed to `serve_connection`, which builds exactly one
//! [`Session`] from the supplied factory. A connection never shares a session,
//! an ingress ring, or an event queue with another connection.
//!
//! A turn is driven by [`Session::run_turn`], which borrows the session for the
//! whole turn. The connection therefore selects over the socket and the pinned
//! turn future: client barge-ins during a turn travel through the session's
//! public [`BargeInHandle`], and every other session event is drained and
//! flushed immediately after the turn future resolves. Wire ordering is
//! preserved; only mid-turn event latency is higher than an event-sink seam
//! would give.

use std::net::SocketAddr;
use std::sync::Arc;

use futures_util::{SinkExt, StreamExt};
use parlevox_core::ParlevoxError;
use parlevox_providers::AudioChunk;
use parlevox_runtime::{BargeInHandle, Session, SessionEvent};
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;
use tokio_tungstenite::tungstenite::protocol::{CloseFrame, WebSocketConfig};
use tokio_tungstenite::tungstenite::{Message, Utf8Bytes};
use tokio_tungstenite::{accept_async_with_config, WebSocketStream};
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

use crate::config::WsConfig;
use crate::ingress::{IngressDrops, IngressRing};
use crate::protocol::{class_name, parse_control, ClientControl, ServerEvent};

/// Builds one [`Session`] per accepted connection.
///
/// The factory must be cheap and independent per call; the transport never
/// reuses a session across sockets.
pub type SessionFactory = Arc<dyn Fn() -> Result<Session, ParlevoxError> + Send + Sync + 'static>;

type Ws = WebSocketStream<TcpStream>;

/// A bound WebSocket listener serving one session per connection.
///
/// Dropping the handle cancels the listener and every live connection without
/// awaiting them. Call [`WsTransport::shutdown`] for a graceful stop that
/// waits for in-flight connections.
#[derive(Debug)]
pub struct WsTransport {
    local_addr: SocketAddr,
    shutdown: CancellationToken,
    tracker: TaskTracker,
}

impl WsTransport {
    /// Bind to `addr` with the default [`WsConfig`].
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`WsTransport::bind_with_config`]. The
    /// default configuration is always valid.
    pub async fn bind(addr: SocketAddr, factory: SessionFactory) -> Result<Self, ParlevoxError> {
        Self::bind_with_config(addr, WsConfig::default(), factory).await
    }

    /// Bind to `addr` with an explicit configuration.
    ///
    /// The configuration is checked with [`WsConfig::validate`] before any
    /// listener exists, so an invalid value fails without touching the network.
    ///
    /// # Errors
    ///
    /// Returns a protocol-class [`ParlevoxError`] naming the offending field
    /// when `config` is invalid, and a transport-class error when the listener
    /// cannot be bound or its local address cannot be read.
    pub async fn bind_with_config(
        addr: SocketAddr,
        config: WsConfig,
        factory: SessionFactory,
    ) -> Result<Self, ParlevoxError> {
        config.validate()?;
        let listener = TcpListener::bind(addr)
            .await
            .map_err(|err| ParlevoxError::Transport(format!("websocket bind failed: {err}")))?;
        let local_addr = listener
            .local_addr()
            .map_err(|err| ParlevoxError::Transport(format!("local addr unavailable: {err}")))?;

        let shutdown = CancellationToken::new();
        let tracker = TaskTracker::new();
        tracker.spawn(accept_loop(
            listener,
            config,
            factory,
            shutdown.clone(),
            tracker.clone(),
        ));

        Ok(WsTransport {
            local_addr,
            shutdown,
            tracker,
        })
    }

    /// The address the listener is actually bound to.
    ///
    /// Useful with port `0`: the kernel assigns a free port and this returns
    /// it.
    #[must_use]
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Stop accepting, close every live connection with `1001` (going away),
    /// and wait for all connection tasks to finish.
    ///
    /// An in-flight turn is asked to unwind through its barge-in handle first,
    /// so its terminal events are still flushed before the close frame.
    pub async fn shutdown(self) {
        self.shutdown.cancel();
        self.tracker.close();
        self.tracker.wait().await;
    }
}

impl Drop for WsTransport {
    fn drop(&mut self) {
        self.shutdown.cancel();
    }
}

async fn accept_loop(
    listener: TcpListener,
    config: WsConfig,
    factory: SessionFactory,
    shutdown: CancellationToken,
    tracker: TaskTracker,
) {
    loop {
        tokio::select! {
            biased;
            () = shutdown.cancelled() => break,
            accepted = listener.accept() => {
                match accepted {
                    Ok((stream, _peer)) => {
                        let config = config.clone();
                        let factory = Arc::clone(&factory);
                        let connection_shutdown = shutdown.clone();
                        tracker.spawn(async move {
                            if let Err(err) =
                                serve_connection(stream, config, factory, connection_shutdown).await
                            {
                                // Only the class: an error message could echo
                                // peer-supplied bytes.
                                tracing::debug!(
                                    target: "parlevox_transport_ws",
                                    class = class_name(err.class()),
                                    "websocket connection ended with an error"
                                );
                            }
                        });
                    }
                    Err(err) => {
                        // Transient accept errors (for example a momentary fd
                        // shortage) must not tear down the listener.
                        tracing::warn!(
                            target: "parlevox_transport_ws",
                            kind = ?err.kind(),
                            "websocket listener accept failed"
                        );
                    }
                }
            }
        }
    }
}

async fn serve_connection(
    stream: TcpStream,
    config: WsConfig,
    factory: SessionFactory,
    shutdown: CancellationToken,
) -> Result<(), ParlevoxError> {
    let ws_config = WebSocketConfig::default()
        .max_message_size(Some(config.max_message_bytes))
        .max_frame_size(Some(config.max_message_bytes));
    let mut ws = accept_async_with_config(stream, Some(ws_config))
        .await
        .map_err(|_| transport_error())?;

    let mut session = match factory() {
        Ok(session) => session,
        Err(err) => {
            let _ = send_event(
                &mut ws,
                &ServerEvent::error(err.class(), "session initialization failed"),
            )
            .await;
            let _ = send_close(&mut ws, CloseCode::Error, "session initialization failed").await;
            return Err(err);
        }
    };
    let barge = session.barge_in_handle();
    let mut ingress = IngressRing::new(
        config.sample_rate,
        config.channels,
        config.ingress_capacity_samples,
        config.ingress_overflow,
    )?;

    send_event(&mut ws, &ServerEvent::ready()).await?;

    loop {
        tokio::select! {
            biased;
            () = shutdown.cancelled() => {
                let _ = send_close(&mut ws, CloseCode::Away, "server shutting down").await;
                return Ok(());
            }
            incoming = ws.next() => {
                let Some(incoming) = incoming else {
                    return Ok(());
                };
                let message = incoming.map_err(|_| transport_error())?;
                match message {
                    Message::Text(text) => match parse_control(text.as_str()) {
                        Some(ClientControl::UtteranceEnd) => {
                            let drops = ingress.take_dropped();
                            let utterance = ingress.drain();
                            tracing::debug!(
                                target: "parlevox_transport_ws",
                                samples = utterance.pcm.len(),
                                "committed one utterance"
                            );
                            let outcome = drive_turn(
                                &mut session,
                                &mut ws,
                                &barge,
                                &mut ingress,
                                &shutdown,
                                utterance,
                                drops,
                            )
                            .await?;
                            if outcome != TurnOutcome::Completed {
                                return Ok(());
                            }
                        }
                        Some(ClientControl::BargeIn) => {
                            // No turn is live, so this reports false and
                            // changes nothing.
                            let _ = session.request_barge_in();
                        }
                        Some(ClientControl::Close) => {
                            let _ = send_close(&mut ws, CloseCode::Normal, "session closed").await;
                            return Ok(());
                        }
                        None => {
                            send_protocol_violation(&mut ws).await?;
                            return Ok(());
                        }
                    },
                    Message::Binary(bytes) => {
                        if ingress.push_bytes(&bytes).is_err() {
                            send_protocol_violation(&mut ws).await?;
                            return Ok(());
                        }
                    }
                    Message::Ping(payload) => {
                        ws.send(Message::Pong(payload)).await.map_err(|_| transport_error())?;
                    }
                    Message::Pong(_) | Message::Frame(_) => {}
                    Message::Close(_) => return Ok(()),
                }
            }
        }
    }
}

/// Why an active turn stopped and what the connection should do next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TurnOutcome {
    /// The turn completed and the connection may take another utterance.
    Completed,
    /// A malformed control or binary frame was answered with error + 1008.
    ProtocolViolation,
    /// The client asked to close; a normal close frame was sent.
    ClientClosed,
    /// The server is shutting down; a going-away close frame was sent.
    ServerShutdown,
}

/// Drive one turn while still reading the socket.
///
/// Binary messages keep filling the ingress ring for the next commit; a
/// `barge_in` uses the session's [`BargeInHandle`] because `run_turn` holds the
/// session's only mutable borrow. `close` and server shutdown mark the turn for
/// a close, ask it to unwind, and flush its terminal events before the close
/// frame is sent.
async fn drive_turn(
    session: &mut Session,
    ws: &mut Ws,
    barge: &BargeInHandle,
    ingress: &mut IngressRing,
    shutdown: &CancellationToken,
    utterance: AudioChunk,
    ingress_drops: IngressDrops,
) -> Result<TurnOutcome, ParlevoxError> {
    let mut close_after_turn = false;
    let mut server_shutdown = false;

    let (result, violation) = {
        let mut turn = std::pin::pin!(session.run_turn(utterance));
        loop {
            tokio::select! {
                biased;
                () = shutdown.cancelled(), if !close_after_turn => {
                    server_shutdown = true;
                    close_after_turn = true;
                    barge.barge_in();
                }
                result = &mut turn => {
                    break (Some(result), false);
                }
                incoming = ws.next(), if !close_after_turn => {
                    let Some(incoming) = incoming else {
                        close_after_turn = true;
                        barge.barge_in();
                        continue;
                    };
                    let message = incoming.map_err(|_| transport_error())?;
                    match message {
                        Message::Text(text) => match parse_control(text.as_str()) {
                            Some(ClientControl::BargeIn) => barge.barge_in(),
                            Some(ClientControl::Close) => {
                                close_after_turn = true;
                                barge.barge_in();
                            }
                            // Committing a second utterance while a turn is
                            // live is a state violation, not something to
                            // queue silently.
                            Some(ClientControl::UtteranceEnd) | None => {
                                send_protocol_violation(ws).await?;
                                break (None, true);
                            }
                        },
                        Message::Binary(bytes) => {
                            if ingress.push_bytes(&bytes).is_err() {
                                send_protocol_violation(ws).await?;
                                break (None, true);
                            }
                        }
                        Message::Ping(payload) => {
                            ws.send(Message::Pong(payload))
                                .await
                                .map_err(|_| transport_error())?;
                        }
                        Message::Pong(_) | Message::Frame(_) => {}
                        Message::Close(_) => {
                            close_after_turn = true;
                            barge.barge_in();
                        }
                    }
                }
            }
        }
    };

    if violation {
        return Ok(TurnOutcome::ProtocolViolation);
    }

    let Some(result) = result else {
        return Ok(TurnOutcome::ProtocolViolation);
    };
    let saw_failed = flush_events(ws, session, ingress_drops).await?;
    if let Err(err) = result {
        if !saw_failed {
            send_event(ws, &ServerEvent::error(err.class(), "turn failed")).await?;
        }
    }

    if server_shutdown {
        let _ = send_close(ws, CloseCode::Away, "server shutting down").await;
        return Ok(TurnOutcome::ServerShutdown);
    }
    if close_after_turn {
        let _ = send_close(ws, CloseCode::Normal, "session closed").await;
        return Ok(TurnOutcome::ClientClosed);
    }
    Ok(TurnOutcome::Completed)
}

/// Translate every buffered session event to the wire, in order.
///
/// Returns whether a `Failed` event was translated, so the caller can supply a
/// fallback error exactly once when the turn future failed without emitting
/// one.
async fn flush_events(
    ws: &mut Ws,
    session: &mut Session,
    ingress_drops: IngressDrops,
) -> Result<bool, ParlevoxError> {
    let mut saw_failed = false;
    for event in session.take_events() {
        match event {
            SessionEvent::TurnStarted { turn } => {
                send_event(ws, &ServerEvent::TurnStarted { turn }).await?;
            }
            SessionEvent::TranscriptFinal { turn, text } => {
                send_event(ws, &ServerEvent::TranscriptFinal { turn, text }).await?;
            }
            SessionEvent::ReplyDelta { turn, text } => {
                send_event(ws, &ServerEvent::ReplyDelta { turn, text }).await?;
            }
            SessionEvent::AudioOut { frame, .. } => {
                let bytes = encode_pcm(&frame.pcm);
                ws.send(Message::binary(bytes))
                    .await
                    .map_err(|_| transport_error())?;
            }
            SessionEvent::TurnDone {
                turn,
                audio_frames,
                dropped_frames,
                dropped_ms,
                dropped_events,
                sidecar_dropped_frames,
                interrupted,
            } => {
                send_event(
                    ws,
                    &ServerEvent::TurnDone {
                        turn,
                        audio_frames,
                        dropped_frames,
                        dropped_ms,
                        dropped_events,
                        sidecar_dropped_frames,
                        interrupted,
                        ingress_dropped_frames: ingress_drops.frames,
                        ingress_dropped_ms: ingress_drops.ms,
                    },
                )
                .await?;
            }
            SessionEvent::Failed { class, message } => {
                saw_failed = true;
                send_event(
                    ws,
                    &ServerEvent::Error {
                        class: class_name(class).to_string(),
                        message,
                    },
                )
                .await?;
            }
        }
    }
    Ok(saw_failed)
}

async fn send_event(ws: &mut Ws, event: &ServerEvent) -> Result<(), ParlevoxError> {
    ws.send(Message::text(event.to_json()?))
        .await
        .map_err(|_| transport_error())
}

async fn send_close(
    ws: &mut Ws,
    code: CloseCode,
    reason: &'static str,
) -> Result<(), ParlevoxError> {
    let frame = CloseFrame {
        code,
        reason: Utf8Bytes::from(reason),
    };
    ws.send(Message::Close(Some(frame)))
        .await
        .map_err(|_| transport_error())
}

/// Answer a wire violation with an error event and a `1008` policy close.
///
/// The message is deliberately generic: the client's bytes are never echoed
/// and no session state is touched.
async fn send_protocol_violation(ws: &mut Ws) -> Result<(), ParlevoxError> {
    send_event(
        ws,
        &ServerEvent::error(
            parlevox_core::ErrorClass::Protocol,
            "malformed control message",
        ),
    )
    .await?;
    send_close(ws, CloseCode::Policy, "protocol violation").await
}

/// Interleaved `i16` samples as little-endian bytes.
fn encode_pcm(samples: &[i16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len().saturating_mul(2));
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

/// A transport failure with no peer-supplied text.
///
/// The underlying tungstenite error can contain fragments of client data (for
/// example an invalid UTF-8 payload). It is intentionally not embedded here,
/// so neither the returned error nor any log line can echo the peer's bytes.
fn transport_error() -> ParlevoxError {
    ParlevoxError::Transport("websocket transport failure".into())
}
