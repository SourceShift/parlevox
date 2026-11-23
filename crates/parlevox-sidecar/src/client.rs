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

use parlevox_core::{AudioFrame, ParlevoxError, Timestamp};
use parlevox_ipc::{
    read_frame, write_frame, AudioHeader, ErrorClassWire, Frame, FrameError, PluginEvent,
    PluginRequest, PROTOCOL_VERSION, SIDECAR_ABI,
};
use parlevox_providers::ProviderManifest;
use tokio::io::AsyncReadExt;
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::{mpsc, Mutex};
use tokio_util::sync::CancellationToken;

use crate::config::SidecarConfig;

/// Anything the plugin sends for an open request.
#[derive(Debug)]
pub enum Incoming {
    Event(PluginEvent),
    Audio(AudioFrame),
}

/// Opt-in idle bound for a plugin request.
///
/// With a policy installed, a request that produces no inbound event for
/// [`StallPolicy::idle_timeout_ms`] is declared stalled: it fails with a
/// transient provider error, the client reports itself not alive, and the
/// supervisor and pool replace the process. Without a policy the client
/// behaves exactly as it did before this watchdog existed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StallPolicy {
    /// Maximum time a request may go without any inbound event before the
    /// plugin is declared wedged. This is an idle bound, not a total-duration
    /// bound: every inbound event restarts it.
    pub idle_timeout_ms: u64,
}

impl Default for StallPolicy {
    /// Ten seconds: a streaming plugin silent for that long is wedged.
    fn default() -> Self {
        StallPolicy {
            idle_timeout_ms: 10_000,
        }
    }
}

/// One-shot channel the router uses to answer a waiting [`SidecarClient::reset`].
type ResetWaiter = tokio::sync::oneshot::Sender<Result<(), ParlevoxError>>;

struct Inner {
    writer: Mutex<ChildStdin>,
    child: Mutex<Child>,
    routes: Arc<StdMutex<HashMap<String, mpsc::Sender<Incoming>>>>,
    alive: Arc<AtomicBool>,
    next_id: AtomicU64,
    manifest: Option<ProviderManifest>,
    capabilities: Vec<String>,
    pid: Option<u32>,
    shutdown_timeout_ms: u64,
    reset_timeout_ms: u64,
    /// Installed immediately before a `reset` frame is written and taken
    /// exactly once by the router, so a stray `ready` is harmless.
    reset_ack: Arc<StdMutex<Option<ResetWaiter>>>,
    /// Idle bound installed by [`SidecarClient::spawn_watched`]; `None` keeps
    /// the unwatched behaviour, including the absence of any timer.
    stall_policy: Option<StallPolicy>,
    /// Requests declared stalled since the client was created.
    stall_events: AtomicU64,
    /// Set once a request is declared stalled; the client is then unusable and
    /// both [`SidecarSupervisor`](crate::supervisor::SidecarSupervisor) and the
    /// pool treat it as dead.
    wedged: AtomicBool,
}

/// A live plugin process speaking the framed protocol.
pub struct SidecarClient {
    inner: Arc<Inner>,
}

enum RecvOutcome {
    Cancelled,
    Message(Option<Incoming>),
    /// The idle bound elapsed with no inbound event.
    TimedOut,
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
    pub async fn spawn(config: SidecarConfig) -> Result<Self, ParlevoxError> {
        Self::spawn_with_policy(config, None).await
    }

    /// Spawn exactly like [`SidecarClient::spawn`], then install `policy` on
    /// every request handle this client opens.
    ///
    /// A request that produces no inbound event for the policy's idle bound
    /// fails with a transient provider error and declares the client wedged;
    /// [`SidecarClient::is_alive`] then reports `false`.
    ///
    /// # Errors
    ///
    /// Returns the same transport- and protocol-class errors as
    /// [`SidecarClient::spawn`].
    pub async fn spawn_watched(
        config: SidecarConfig,
        policy: StallPolicy,
    ) -> Result<Self, ParlevoxError> {
        Self::spawn_with_policy(config, Some(policy)).await
    }

    /// Shared spawn path: the handshake is identical whether or not a stall
    /// policy is installed.
    async fn spawn_with_policy(
        config: SidecarConfig,
        stall_policy: Option<StallPolicy>,
    ) -> Result<Self, ParlevoxError> {
        let mut command = Command::new(&config.program);
        command
            .args(&config.args)
            .envs(config.env.iter().map(|(k, v)| (k, v)))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let mut child = command.spawn().map_err(|err| {
            ParlevoxError::Transport(format!("failed to spawn '{}': {err}", config.program))
        })?;
        let pid = child.id();
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| ParlevoxError::Transport("plugin stdin unavailable".into()))?;
        let mut stdout = child
            .stdout
            .take()
            .ok_or_else(|| ParlevoxError::Transport("plugin stdout unavailable".into()))?;
        if let Some(stderr) = child.stderr.take() {
            tokio::spawn(async move {
                let mut text = String::new();
                let mut stderr = stderr;
                let _ = stderr.read_to_string(&mut text).await;
                if !text.trim().is_empty() {
                    tracing::debug!(target: "parlevox_sidecar", plugin_stderr = %text, "plugin stderr");
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
            .map_err(|err| ParlevoxError::Transport(format!("hello write failed: {err}")))?;

        let ready = match tokio::time::timeout(
            Duration::from_millis(config.handshake_timeout_ms),
            read_frame(&mut stdout),
        )
        .await
        {
            Err(_) => {
                let _ = child.start_kill();
                let _ = child.wait().await;
                return Err(ParlevoxError::Transport(format!(
                    "handshake timed out after {} ms",
                    config.handshake_timeout_ms
                )));
            }
            Ok(Err(err)) => {
                let _ = child.start_kill();
                let _ = child.wait().await;
                return Err(ParlevoxError::Transport(format!(
                    "handshake read failed: {err}"
                )));
            }
            Ok(Ok(frame)) => frame,
        };

        let (manifest, capabilities) = match ready {
            Frame::Json(value) => match serde_json::from_value::<PluginEvent>(value) {
                Ok(PluginEvent::Ready {
                    protocol_version,
                    abi,
                    plugin,
                    manifest,
                    capabilities,
                }) => {
                    if protocol_version != PROTOCOL_VERSION {
                        let _ = child.start_kill();
                        let _ = child.wait().await;
                        return Err(ParlevoxError::Protocol(format!(
                            "plugin protocol version {protocol_version}, expected {PROTOCOL_VERSION}"
                        )));
                    }
                    if abi != SIDECAR_ABI {
                        let _ = child.start_kill();
                        let _ = child.wait().await;
                        return Err(ParlevoxError::Protocol(format!(
                            "plugin abi '{abi}', expected '{SIDECAR_ABI}'"
                        )));
                    }
                    tracing::debug!(target: "parlevox_sidecar", plugin = %plugin, "plugin ready");
                    (
                        serde_json::from_value::<ProviderManifest>(manifest).ok(),
                        capabilities,
                    )
                }
                Ok(other) => {
                    let _ = child.start_kill();
                    let _ = child.wait().await;
                    return Err(ParlevoxError::Protocol(format!(
                        "expected ready during handshake, got {other:?}"
                    )));
                }
                Err(err) => {
                    let _ = child.start_kill();
                    let _ = child.wait().await;
                    return Err(ParlevoxError::Protocol(format!(
                        "malformed ready event: {err}"
                    )));
                }
            },
            other => {
                let _ = child.start_kill();
                let _ = child.wait().await;
                return Err(ParlevoxError::Protocol(format!(
                    "expected JSON ready frame, got {other:?}"
                )));
            }
        };

        let routes: Arc<StdMutex<HashMap<String, mpsc::Sender<Incoming>>>> =
            Arc::new(StdMutex::new(HashMap::new()));
        let alive = Arc::new(AtomicBool::new(true));
        let reset_ack: Arc<StdMutex<Option<ResetWaiter>>> = Arc::new(StdMutex::new(None));
        spawn_router(
            stdout,
            Arc::clone(&routes),
            Arc::clone(&alive),
            Arc::clone(&reset_ack),
        );

        Ok(SidecarClient {
            inner: Arc::new(Inner {
                writer: Mutex::new(stdin),
                child: Mutex::new(child),
                routes,
                alive,
                next_id: AtomicU64::new(0),
                manifest,
                capabilities,
                pid,
                shutdown_timeout_ms: config.shutdown_timeout_ms,
                reset_timeout_ms: config.reset_timeout(),
                reset_ack,
                stall_policy,
                stall_events: AtomicU64::new(0),
                wedged: AtomicBool::new(false),
            }),
        })
    }

    /// Requests declared stalled since the client was created.
    pub fn stall_events(&self) -> u64 {
        self.inner.stall_events.load(Ordering::SeqCst)
    }

    /// Whether a request was declared stalled. A wedged client is no longer
    /// usable and reports itself dead to the supervisor and the pool.
    pub fn is_wedged(&self) -> bool {
        self.inner.wedged.load(Ordering::SeqCst)
    }

    /// The manifest the plugin announced during `ready`, when parseable.
    pub fn plugin_manifest(&self) -> Option<&ProviderManifest> {
        self.inner.manifest.as_ref()
    }

    /// OS process id of the plugin, captured at spawn.
    pub fn pid(&self) -> Option<u32> {
        self.inner.pid
    }

    /// Whether the plugin advertised the additive `reset` capability.
    ///
    /// A client is only poolable when this is true; a plugin built before the
    /// capability keeps today's one-process-per-session lifecycle.
    pub fn supports_reset(&self) -> bool {
        self.inner
            .capabilities
            .iter()
            .any(|capability| capability == "reset")
    }

    /// Clear the plugin's per-session state and wait for its `ready`
    /// acknowledgement.
    ///
    /// A reset is only meaningful for an idle client: it fails loudly rather
    /// than clearing state behind a live request. The acknowledgement is the
    /// same payload as the initial handshake, validated for protocol version
    /// and ABI.
    ///
    /// # Errors
    ///
    /// Returns a transport-class error when the caller already reset this
    /// client concurrently, when the child is dead, when the `reset` frame
    /// cannot be written, or when no acknowledgement arrives within
    /// [`SidecarConfig::reset_timeout`]. Returns a protocol-class error when a
    /// request route is still open or the plugin rejects the reset.
    pub async fn reset(&self) -> Result<(), ParlevoxError> {
        if !self.is_alive().await {
            return Err(ParlevoxError::Transport(
                "cannot reset a plugin that is not running".into(),
            ));
        }
        let open_routes = self
            .inner
            .routes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len();
        if open_routes != 0 {
            return Err(ParlevoxError::Protocol(format!(
                "cannot reset with {open_routes} request route(s) still open"
            )));
        }

        // Encode first: nothing may leave a waiter installed if the frame
        // cannot even be built.
        let frame = json_frame(&PluginRequest::Reset)?;
        let (tx, rx) = tokio::sync::oneshot::channel();
        {
            let mut slot = self
                .inner
                .reset_ack
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if slot.is_some() {
                return Err(ParlevoxError::Transport(
                    "a reset is already awaiting acknowledgement".into(),
                ));
            }
            *slot = Some(tx);
        }

        let written = {
            let mut writer = self.inner.writer.lock().await;
            write_frame(&mut *writer, &frame).await
        };
        if let Err(err) = written {
            self.inner
                .reset_ack
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .take();
            return Err(ParlevoxError::Transport(format!(
                "reset write failed: {err}"
            )));
        }

        match tokio::time::timeout(Duration::from_millis(self.inner.reset_timeout_ms), rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => {
                self.inner
                    .reset_ack
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .take();
                Err(ParlevoxError::Transport(
                    "reset acknowledgement channel closed".into(),
                ))
            }
            Err(_) => {
                self.inner
                    .reset_ack
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .take();
                Err(ParlevoxError::Transport(format!(
                    "reset acknowledgement timed out after {} ms",
                    self.inner.reset_timeout_ms
                )))
            }
        }
    }

    /// Unique request id for a modality prefix.
    pub fn new_request_id(&self, prefix: &str) -> String {
        let n = self.inner.next_id.fetch_add(1, Ordering::SeqCst);
        format!("{prefix}-{n}")
    }

    /// Whether the child is still running and its stdout stream is open.
    ///
    /// A client declared wedged by the stall watchdog is reported not alive
    /// before the child is inspected, so the supervisor and pool take their
    /// existing replace paths.
    pub async fn is_alive(&self) -> bool {
        if self.inner.wedged.load(Ordering::SeqCst) {
            return false;
        }
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
    ) -> Result<RequestHandle, ParlevoxError> {
        let request_id = request
            .request_id()
            .ok_or_else(|| ParlevoxError::Protocol("request has no request_id".into()))?
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
                .map_err(|err| ParlevoxError::Transport(format!("request write failed: {err}")))?;
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
                    ParlevoxError::Transport(format!("audio write failed: {err}"))
                })?;
            }
            if needs_audio_end {
                let end = PluginRequest::AudioEnd {
                    request_id: request_id.clone(),
                };
                write_frame(&mut *writer, &json_frame(&end)?)
                    .await
                    .map_err(|err| {
                        ParlevoxError::Transport(format!("audio_end write failed: {err}"))
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

        let watch = self.inner.stall_policy.map(|policy| StallWatch {
            idle: Duration::from_millis(policy.idle_timeout_ms),
            inner: Arc::clone(&self.inner),
        });

        Ok(RequestHandle {
            request_id,
            rx,
            cancel,
            cancelled: false,
            finished: false,
            watch,
        })
    }

    /// Politely stop the plugin and reap the child.
    ///
    /// # Errors
    ///
    /// Returns a transport-class error when the child cannot be waited on or
    /// does not exit before the shutdown timeout.
    pub async fn shutdown(&self) -> Result<(), ParlevoxError> {
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
            Ok(Err(err)) => Err(ParlevoxError::Transport(format!(
                "plugin wait failed: {err}"
            ))),
            Err(_) => {
                let _ = child.start_kill();
                let _ = child.wait().await;
                Err(ParlevoxError::Transport(format!(
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

/// The watchdog state a watched request handle carries.
struct StallWatch {
    idle: Duration,
    inner: Arc<Inner>,
}

/// One request's inbound stream.
pub struct RequestHandle {
    request_id: String,
    rx: mpsc::Receiver<Incoming>,
    cancel: Option<CancellationToken>,
    cancelled: bool,
    /// Set when the stream reached a terminal state (cancellation or a stall),
    /// so later `recv` calls return `None` without re-firing the watchdog.
    finished: bool,
    /// Present only when the client was spawned with a [`StallPolicy`].
    watch: Option<StallWatch>,
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
    ///
    /// When the client carries a [`StallPolicy`], an idle bound is also raced:
    /// the first `recv` that observes it returns a single synthesized
    /// [`PluginEvent::Error`] with a transient class and every later `recv`
    /// returns `None`.
    pub async fn recv(&mut self) -> Option<Incoming> {
        if self.finished || self.cancelled {
            return None;
        }
        let token = self.cancel.clone();
        // Read the bound up front so the immutable borrow of `watch` ends
        // before the select borrows `rx` mutably.
        let idle = self.watch.as_ref().map(|watch| watch.idle);
        let outcome = {
            let rx = &mut self.rx;
            match idle {
                // No policy installed: exactly today's two-arm select, and no
                // timer is constructed.
                None => tokio::select! {
                    biased;
                    _ = wait_optional(token.as_ref()) => RecvOutcome::Cancelled,
                    message = rx.recv() => RecvOutcome::Message(message),
                },
                Some(idle) => tokio::select! {
                    biased;
                    _ = wait_optional(token.as_ref()) => RecvOutcome::Cancelled,
                    message = rx.recv() => RecvOutcome::Message(message),
                    () = tokio::time::sleep(idle) => RecvOutcome::TimedOut,
                },
            }
        };
        match outcome {
            RecvOutcome::Cancelled => {
                self.cancelled = true;
                self.finished = true;
                None
            }
            RecvOutcome::Message(message) => {
                if message.is_none() && token.as_ref().is_some_and(CancellationToken::is_cancelled)
                {
                    self.cancelled = true;
                }
                message
            }
            RecvOutcome::TimedOut => {
                // Exactly once per handle: `finished` is set before the error
                // is returned, so every later recv short-circuits above.
                self.finished = true;
                let Some(watch) = &self.watch else {
                    return None;
                };
                watch.inner.wedged.store(true, Ordering::SeqCst);
                watch.inner.stall_events.fetch_add(1, Ordering::SeqCst);
                // Drop a late event instead of queuing it: the route goes away
                // before the synthesized error reaches the consumer.
                watch
                    .inner
                    .routes
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .remove(&self.request_id);
                let idle_ms = watch.idle.as_millis();
                Some(Incoming::Event(PluginEvent::Error {
                    request_id: self.request_id.clone(),
                    class: ErrorClassWire::Transient,
                    message: format!("plugin produced no event for {idle_ms} ms; declared wedged"),
                }))
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

fn json_frame<T: serde::Serialize>(message: &T) -> Result<Frame, ParlevoxError> {
    let value = serde_json::to_value(message)
        .map_err(|err| ParlevoxError::Protocol(format!("message encode failed: {err}")))?;
    Ok(Frame::Json(value))
}

fn spawn_router(
    mut stdout: ChildStdout,
    routes: Arc<StdMutex<HashMap<String, mpsc::Sender<Incoming>>>>,
    alive: Arc<AtomicBool>,
    reset_ack: Arc<StdMutex<Option<ResetWaiter>>>,
) {
    tokio::spawn(async move {
        loop {
            match read_frame(&mut stdout).await {
                Ok(Frame::Json(value)) => {
                    let Ok(event) = serde_json::from_value::<PluginEvent>(value) else {
                        tracing::warn!(target: "parlevox_sidecar", "dropping unknown control event");
                        continue;
                    };
                    // `reset` is answered by an ordinary `ready`. Resolve a
                    // waiting reset exactly once; a stray `ready` with no
                    // waiter stays harmless. The initial handshake runs before
                    // this router exists, so it can never be stolen.
                    if let PluginEvent::Ready {
                        protocol_version,
                        abi,
                        ..
                    } = &event
                    {
                        let waiter = reset_ack
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .take();
                        if let Some(waiter) = waiter {
                            let result = if *protocol_version != PROTOCOL_VERSION {
                                Err(ParlevoxError::Protocol(format!(
                                    "reset acknowledged protocol version {protocol_version}, \
                                     expected {PROTOCOL_VERSION}"
                                )))
                            } else if abi != SIDECAR_ABI {
                                Err(ParlevoxError::Protocol(format!(
                                    "reset acknowledged abi '{abi}', expected '{SIDECAR_ABI}'"
                                )))
                            } else {
                                Ok(())
                            };
                            let _ = waiter.send(result);
                        }
                        continue;
                    }
                    // A session-level error (empty request id) rejects a
                    // waiting reset. Other errors keep their route.
                    if let PluginEvent::Error {
                        request_id,
                        message,
                        ..
                    } = &event
                    {
                        if request_id.is_empty() {
                            let waiter = reset_ack
                                .lock()
                                .unwrap_or_else(|poisoned| poisoned.into_inner())
                                .take();
                            if let Some(waiter) = waiter {
                                let _ = waiter.send(Err(ParlevoxError::Protocol(format!(
                                    "plugin rejected reset: {message}"
                                ))));
                            }
                            continue;
                        }
                    }
                    let terminal = event.is_terminal();
                    let request_id = event.request_id().map(str::to_string);
                    if let Some(request_id) = request_id {
                        let sender = routes
                            .lock()
                            .expect("routes poisoned")
                            .get(&request_id)
                            .cloned();
                        if let Some(sender) = sender {
                            // Remove a terminal route before handing the event
                            // to its consumer: a caller that releases a
                            // process right after the turn then cannot race a
                            // stale route and mistake a clean process for a
                            // dirty one.
                            if terminal {
                                routes.lock().expect("routes poisoned").remove(&request_id);
                            }
                            let _ = sender.send(Incoming::Event(event)).await;
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
                    tracing::warn!(target: "parlevox_sidecar", error = %err, "plugin stream failed");
                    break;
                }
            }
        }
        alive.store(false, Ordering::SeqCst);
        routes.lock().expect("routes poisoned").clear();
        let waiter = reset_ack
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        if let Some(waiter) = waiter {
            let _ = waiter.send(Err(ParlevoxError::Transport(
                "plugin stream closed before acknowledging reset".into(),
            )));
        }
    });
}
