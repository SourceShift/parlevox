//! Spawn, handshake, and multiplex a Python plugin process.
//!
//! This is the only crate in the workspace that may spawn a process. Requests
//! are multiplexed by `request_id`: one reader task routes JSON control events
//! and binary audio frames to the request handle that opened the id, so a
//! `Cancel` can be written from a watcher task without tearing the process
//! down.

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use tokio::io::AsyncReadExt;
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::{mpsc, Mutex};
use tokio_util::sync::CancellationToken;
use voicekit_core::{AudioFrame, Timestamp, VoicekitError};
use voicekit_ipc::{
    read_frame, write_frame, AudioHeader, Frame, FrameError, PluginEvent, PluginRequest,
    PROTOCOL_VERSION, SIDECAR_ABI,
};
use voicekit_providers::ProviderManifest;

use crate::config::SidecarConfig;

/// Anything the plugin sends for an open request.
#[derive(Debug)]
pub enum Incoming {
    Event(PluginEvent),
    Audio(AudioFrame),
}

struct Inner {
    writer: Mutex<ChildStdin>,
    child: Mutex<Child>,
    routes: Arc<StdMutex<HashMap<String, mpsc::Sender<Incoming>>>>,
    alive: Arc<AtomicBool>,
    next_id: AtomicU64,
    manifest: Option<ProviderManifest>,
    shutdown_timeout_ms: u64,
}

/// A live plugin process speaking the framed protocol.
pub struct SidecarClient {
    inner: Arc<Inner>,
}

enum RecvOutcome {
    Cancelled,
    Message(Option<Incoming>),
}

impl SidecarClient {
    /// Spawn the process and complete the `hello`/`ready` handshake.
    ///
    /// # Errors
    ///
    /// Returns a transport-class error when the process cannot be spawned, its
    /// pipes are unavailable, or the handshake times out; and a protocol-class
    /// error when the plugin's `ready` frame is malformed or announces an
    /// incompatible protocol version or ABI.
    pub async fn spawn(config: SidecarConfig) -> Result<Self, VoicekitError> {
        let mut command = Command::new(&config.program);
        command
            .args(&config.args)
            .envs(config.env.iter().map(|(k, v)| (k, v)))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let mut child = command.spawn().map_err(|err| {
            VoicekitError::Transport(format!("failed to spawn '{}': {err}", config.program))
        })?;
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| VoicekitError::Transport("plugin stdin unavailable".into()))?;
        let mut stdout = child
            .stdout
            .take()
            .ok_or_else(|| VoicekitError::Transport("plugin stdout unavailable".into()))?;
        if let Some(stderr) = child.stderr.take() {
            tokio::spawn(async move {
                let mut text = String::new();
                let mut stderr = stderr;
                let _ = stderr.read_to_string(&mut text).await;
                if !text.trim().is_empty() {
                    tracing::debug!(target: "voicekit_sidecar", plugin_stderr = %text, "plugin stderr");
                }
            });
        }

        // Handshake happens before the router owns stdout, so there is exactly
        // one reader at every moment.
        let hello = PluginRequest::Hello {
            protocol_version: PROTOCOL_VERSION,
            abi: SIDECAR_ABI.into(),
            modalities: Vec::new(),
        };
        write_frame(&mut stdin, &json_frame(&hello)?)
            .await
            .map_err(|err| VoicekitError::Transport(format!("hello write failed: {err}")))?;

        let ready = match tokio::time::timeout(
            Duration::from_millis(config.handshake_timeout_ms),
            read_frame(&mut stdout),
        )
        .await
        {
            Err(_) => {
                let _ = child.start_kill();
                let _ = child.wait().await;
                return Err(VoicekitError::Transport(format!(
                    "handshake timed out after {} ms",
                    config.handshake_timeout_ms
                )));
            }
            Ok(Err(err)) => {
                let _ = child.start_kill();
                let _ = child.wait().await;
                return Err(VoicekitError::Transport(format!(
                    "handshake read failed: {err}"
                )));
            }
            Ok(Ok(frame)) => frame,
        };

        let manifest = match ready {
            Frame::Json(value) => match serde_json::from_value::<PluginEvent>(value) {
                Ok(PluginEvent::Ready {
                    protocol_version,
                    abi,
                    plugin,
                    manifest,
                }) => {
                    if protocol_version != PROTOCOL_VERSION {
                        let _ = child.start_kill();
                        let _ = child.wait().await;
                        return Err(VoicekitError::Protocol(format!(
                            "plugin protocol version {protocol_version}, expected {PROTOCOL_VERSION}"
                        )));
                    }
                    if abi != SIDECAR_ABI {
                        let _ = child.start_kill();
                        let _ = child.wait().await;
                        return Err(VoicekitError::Protocol(format!(
                            "plugin abi '{abi}', expected '{SIDECAR_ABI}'"
                        )));
                    }
                    tracing::debug!(target: "voicekit_sidecar", plugin = %plugin, "plugin ready");
                    serde_json::from_value::<ProviderManifest>(manifest).ok()
                }
                Ok(other) => {
                    let _ = child.start_kill();
                    let _ = child.wait().await;
                    return Err(VoicekitError::Protocol(format!(
                        "expected ready during handshake, got {other:?}"
                    )));
                }
                Err(err) => {
                    let _ = child.start_kill();
                    let _ = child.wait().await;
                    return Err(VoicekitError::Protocol(format!(
                        "malformed ready event: {err}"
                    )));
                }
            },
            other => {
                let _ = child.start_kill();
                let _ = child.wait().await;
                return Err(VoicekitError::Protocol(format!(
                    "expected JSON ready frame, got {other:?}"
                )));
            }
        };

        let routes: Arc<StdMutex<HashMap<String, mpsc::Sender<Incoming>>>> =
            Arc::new(StdMutex::new(HashMap::new()));
        let alive = Arc::new(AtomicBool::new(true));
        spawn_router(stdout, Arc::clone(&routes), Arc::clone(&alive));

        Ok(SidecarClient {
            inner: Arc::new(Inner {
                writer: Mutex::new(stdin),
                child: Mutex::new(child),
                routes,
                alive,
                next_id: AtomicU64::new(0),
                manifest,
                shutdown_timeout_ms: config.shutdown_timeout_ms,
            }),
        })
    }

    /// The manifest the plugin announced during `ready`, when parseable.
    pub fn plugin_manifest(&self) -> Option<&ProviderManifest> {
        self.inner.manifest.as_ref()
    }

    /// Unique request id for a modality prefix.
    pub fn new_request_id(&self, prefix: &str) -> String {
        let n = self.inner.next_id.fetch_add(1, Ordering::SeqCst);
        format!("{prefix}-{n}")
    }

    /// Whether the child is still running and its stdout stream is open.
    pub async fn is_alive(&self) -> bool {
        if !self.inner.alive.load(Ordering::SeqCst) {
            return false;
        }
        let mut child = self.inner.child.lock().await;
        match child.try_wait() {
            Ok(Some(_)) => {
                self.inner.alive.store(false, Ordering::SeqCst);
                false
            }
            Ok(None) => true,
            Err(_) => false,
        }
    }

    /// Open a request and return a handle over its event stream.
    ///
    /// `cancel` is watched by a task that writes a `Cancel` frame; the process
    /// is never killed. The handle stops when the token fires or the plugin
    /// reports a terminal event.
    ///
    /// # Errors
    ///
    /// Returns a protocol-class error when the request carries no request id
    /// or cannot be encoded, and a transport-class error when writing the
    /// request or its audio frames fails.
    pub async fn open_request(
        &self,
        request: PluginRequest,
        audio: Vec<AudioFrame>,
        cancel: Option<CancellationToken>,
    ) -> Result<RequestHandle, VoicekitError> {
        let request_id = request
            .request_id()
            .ok_or_else(|| VoicekitError::Protocol("request has no request_id".into()))?
            .to_string();
        let needs_audio_end = request.needs_audio_end();

        let (tx, rx) = mpsc::channel(64);
        self.inner
            .routes
            .lock()
            .expect("routes poisoned")
            .insert(request_id.clone(), tx);

        let write_result = async {
            let mut writer = self.inner.writer.lock().await;
            write_frame(&mut *writer, &json_frame(&request)?)
                .await
                .map_err(|err| VoicekitError::Transport(format!("request write failed: {err}")))?;
            for (seq, frame) in audio.iter().enumerate() {
                let header = AudioHeader {
                    request_id: request_id.clone(),
                    seq: seq as u64,
                    sample_rate: frame.sample_rate,
                    channels: frame.channels,
                };
                let frame = Frame::Audio {
                    header,
                    pcm: frame.pcm.clone(),
                };
                write_frame(&mut *writer, &frame).await.map_err(|err| {
                    VoicekitError::Transport(format!("audio write failed: {err}"))
                })?;
            }
            if needs_audio_end {
                let end = PluginRequest::AudioEnd {
                    request_id: request_id.clone(),
                };
                write_frame(&mut *writer, &json_frame(&end)?)
                    .await
                    .map_err(|err| {
                        VoicekitError::Transport(format!("audio_end write failed: {err}"))
                    })?;
            }
            Ok(())
        }
        .await;

        if let Err(err) = write_result {
            self.inner
                .routes
                .lock()
                .expect("routes poisoned")
                .remove(&request_id);
            return Err(err);
        }

        if let Some(token) = cancel.clone() {
            let writer = Arc::clone(&self.inner);
            let id = request_id.clone();
            tokio::spawn(async move {
                token.cancelled().await;
                let mut writer = writer.writer.lock().await;
                let cancel = PluginRequest::Cancel {
                    request_id: id.clone(),
                };
                if let Ok(frame) = json_frame(&cancel) {
                    let _ = write_frame(&mut *writer, &frame).await;
                }
            });
        }

        Ok(RequestHandle {
            request_id,
            rx,
            cancel,
            cancelled: false,
        })
    }

    /// Politely stop the plugin and reap the child.
    ///
    /// # Errors
    ///
    /// Returns a transport-class error when the child cannot be waited on or
    /// does not exit before the shutdown timeout.
    pub async fn shutdown(&self) -> Result<(), VoicekitError> {
        {
            let mut writer = self.inner.writer.lock().await;
            let frame = json_frame(&PluginRequest::Shutdown)?;
            let _ = write_frame(&mut *writer, &frame).await;
        }
        let mut child = self.inner.child.lock().await;
        let waited = tokio::time::timeout(
            Duration::from_millis(self.inner.shutdown_timeout_ms),
            child.wait(),
        )
        .await;
        self.inner.alive.store(false, Ordering::SeqCst);
        match waited {
            Ok(Ok(_)) => Ok(()),
            Ok(Err(err)) => Err(VoicekitError::Transport(format!(
                "plugin wait failed: {err}"
            ))),
            Err(_) => {
                let _ = child.start_kill();
                let _ = child.wait().await;
                Err(VoicekitError::Transport(format!(
                    "plugin did not exit within {} ms",
                    self.inner.shutdown_timeout_ms
                )))
            }
        }
    }

    /// Force the child down. Used by the supervisor when a restart is needed.
    pub async fn kill(&self) {
        self.inner.alive.store(false, Ordering::SeqCst);
        let mut child = self.inner.child.lock().await;
        let _ = child.start_kill();
        let _ = child.wait().await;
    }
}

/// One request's inbound stream.
pub struct RequestHandle {
    request_id: String,
    rx: mpsc::Receiver<Incoming>,
    cancel: Option<CancellationToken>,
    cancelled: bool,
}

impl RequestHandle {
    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled
            || self
                .cancel
                .as_ref()
                .is_some_and(CancellationToken::is_cancelled)
    }

    /// Next inbound item, or `None` when the stream ended or was cancelled.
    pub async fn recv(&mut self) -> Option<Incoming> {
        if self.cancelled {
            return None;
        }
        let token = self.cancel.clone();
        let outcome = {
            let rx = &mut self.rx;
            tokio::select! {
                biased;
                _ = wait_optional(token.as_ref()) => RecvOutcome::Cancelled,
                message = rx.recv() => RecvOutcome::Message(message),
            }
        };
        match outcome {
            RecvOutcome::Cancelled => {
                self.cancelled = true;
                None
            }
            RecvOutcome::Message(message) => {
                if message.is_none() && token.as_ref().is_some_and(CancellationToken::is_cancelled)
                {
                    self.cancelled = true;
                }
                message
            }
        }
    }
}

async fn wait_optional(token: Option<&CancellationToken>) {
    match token {
        Some(token) => token.cancelled().await,
        None => std::future::pending::<()>().await,
    }
}

fn json_frame<T: serde::Serialize>(message: &T) -> Result<Frame, VoicekitError> {
    let value = serde_json::to_value(message)
        .map_err(|err| VoicekitError::Protocol(format!("message encode failed: {err}")))?;
    Ok(Frame::Json(value))
}

fn spawn_router(
    mut stdout: ChildStdout,
    routes: Arc<StdMutex<HashMap<String, mpsc::Sender<Incoming>>>>,
    alive: Arc<AtomicBool>,
) {
    tokio::spawn(async move {
        loop {
            match read_frame(&mut stdout).await {
                Ok(Frame::Json(value)) => {
                    let Ok(event) = serde_json::from_value::<PluginEvent>(value) else {
                        tracing::warn!(target: "voicekit_sidecar", "dropping unknown control event");
                        continue;
                    };
                    let terminal = event.is_terminal();
                    let request_id = event.request_id().map(str::to_string);
                    if let Some(request_id) = request_id {
                        let sender = routes
                            .lock()
                            .expect("routes poisoned")
                            .get(&request_id)
                            .cloned();
                        if let Some(sender) = sender {
                            let _ = sender.send(Incoming::Event(event)).await;
                            if terminal {
                                routes.lock().expect("routes poisoned").remove(&request_id);
                            }
                        }
                    }
                }
                Ok(Frame::Audio { header, pcm }) => {
                    let sender = routes
                        .lock()
                        .expect("routes poisoned")
                        .get(&header.request_id)
                        .cloned();
                    if let Some(sender) = sender {
                        let frame = AudioFrame::new(
                            Timestamp::from_millis(0),
                            header.sample_rate,
                            header.channels,
                            pcm,
                        );
                        let _ = sender.send(Incoming::Audio(frame)).await;
                    }
                }
                Ok(Frame::Close) => break,
                Err(FrameError::Closed) => break,
                Err(err) => {
                    tracing::warn!(target: "voicekit_sidecar", error = %err, "plugin stream failed");
                    break;
                }
            }
        }
        alive.store(false, Ordering::SeqCst);
        routes.lock().expect("routes poisoned").clear();
    });
}
