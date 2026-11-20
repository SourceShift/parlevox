//! `parlevox-cli` — one-turn demo over either provider backend.
//!
//! The argument parser and demo runners live here so integration tests can
//! drive the same code path the binary does. `main.rs` is a thin delegate.
//!
//! `demo --fake` uses only in-process fakes: no process, no Python import, no
//! network. `demo --sidecar` runs the same turn with STT, LLM and TTS served by
//! the optional Python plugin over the framed protocol.

use std::sync::Arc;

use parlevox_core::ParlevoxError;
use parlevox_providers::{FakeVad, LlmProvider, SttProvider, TtsProvider};
use parlevox_runtime::{
    demo_utterance, ProviderNames, ProviderRegistry, Session, SessionConfig, SessionEvent,
    TurnReport,
};
use parlevox_sidecar::{
    resolve_sdk_path, PoolConfig, SdkSearch, SidecarClient, SidecarConfig, SidecarLlm, SidecarPool,
    SidecarStt, SidecarTts,
};

/// Build a session whose STT, LLM and TTS are served by `client` over the
/// framed protocol; VAD stays in process because it is a deterministic fake.
///
/// This is the single construction path shared by the differential parity test
/// and the pool reuse test, so a change to provider wiring cannot make those
/// tests disagree about what a session is.
///
/// # Errors
///
/// Returns a protocol-class error when a server-backed provider's manifest is
/// invalid or the requested names cannot be resolved, and any error from
/// session configuration.
pub fn sidecar_session(client: Arc<SidecarClient>) -> Result<Session, ParlevoxError> {
    let (session, _names) = build_sidecar_session(client)?;
    Ok(session)
}

/// Build the sidecar-backed session and return the provider names beside it.
///
/// The names are what the demo prints, so the pooled path does not have to
/// re-derive them from a second construction path.
///
/// # Errors
///
/// Same as [`sidecar_session`]: an invalid manifest, unresolvable names, or a
/// rejected session configuration.
fn build_sidecar_session(
    client: Arc<SidecarClient>,
) -> Result<(Session, ProviderNames), ParlevoxError> {
    let stt = Arc::new(SidecarStt::new(Arc::clone(&client)));
    let llm = Arc::new(SidecarLlm::new(Arc::clone(&client)));
    let tts = Arc::new(SidecarTts::new(Arc::clone(&client)));
    let mut registry = ProviderRegistry::new();
    registry.register_stt(stt.clone())?;
    registry.register_llm(llm.clone())?;
    registry.register_tts(tts.clone())?;
    registry.register_vad(Arc::new(FakeVad::new()))?;

    let names = ProviderNames {
        stt: stt.manifest().name.clone(),
        llm: llm.manifest().name.clone(),
        tts: tts.manifest().name.clone(),
        vad: "fake-vad".into(),
    };
    let providers = registry.resolve(&names)?;
    let session = Session::new(SessionConfig::default(), providers)?;
    Ok((session, names))
}

/// Canonical, timestamp-free projection of an event trace.
///
/// Audio timestamps are session-epoch-relative, so they are normalized away;
/// kind, turn, text, PCM samples, sample rate, channels and every drop counter
/// are preserved. Two runs are interchangeable exactly when their projections
/// are equal.
#[must_use]
pub fn project_events(events: &[SessionEvent]) -> Vec<String> {
    events
        .iter()
        .map(|event| match event {
            SessionEvent::TurnStarted { turn } => format!("TurnStarted turn={turn}"),
            SessionEvent::TranscriptFinal { turn, text } => {
                format!("TranscriptFinal turn={turn} text={text:?}")
            }
            SessionEvent::ReplyDelta { turn, text } => {
                format!("ReplyDelta turn={turn} text={text:?}")
            }
            SessionEvent::AudioOut { turn, frame } => format!(
                "AudioOut turn={turn} rate={} channels={} pcm={:?}",
                frame.sample_rate, frame.channels, frame.pcm
            ),
            SessionEvent::TurnDone {
                turn,
                audio_frames,
                dropped_frames,
                dropped_ms,
                dropped_events,
                sidecar_dropped_frames,
                interrupted,
            } => format!(
                "TurnDone turn={turn} audio_frames={audio_frames} \
                 dropped_frames={dropped_frames} dropped_ms={dropped_ms} \
                 dropped_events={dropped_events} \
                 sidecar_dropped_frames={sidecar_dropped_frames} \
                 interrupted={interrupted}"
            ),
            SessionEvent::Failed { class, message } => {
                format!("Failed class={class:?} message={message:?}")
            }
        })
        .collect()
}

/// Parse process arguments and run the matching demo.
///
/// # Errors
///
/// Returns a protocol-class error for unrecognised arguments and propagates
/// whatever the selected demo returns (provider, transport, or protocol
/// errors).
pub async fn run_args() -> Result<(), ParlevoxError> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [command, flag, pool] if command == "demo" && flag == "--fake" && pool == "--pool" => {
            usage();
            Err(ParlevoxError::Protocol(
                "demo --fake --pool is rejected: --pool requires --sidecar \
                 (the fakes spawn no plugin process)"
                    .into(),
            ))
        }
        [command, flag, pool] if command == "demo" && flag == "--sidecar" && pool == "--pool" => {
            demo_sidecar_pooled().await
        }
        [command, flag] if command == "demo" && flag == "--fake" => demo_fake().await,
        [command, flag] if command == "demo" && flag == "--sidecar" => demo_sidecar().await,
        [flag] if flag == "--help" || flag == "-h" => {
            usage();
            Ok(())
        }
        _ => {
            usage();
            Err(ParlevoxError::Protocol(format!(
                "unrecognised arguments: {}",
                args.join(" ")
            )))
        }
    }
}

pub fn usage() {
    eprintln!("usage: parlevox demo --fake");
    eprintln!("       parlevox demo --sidecar");
    eprintln!("       parlevox demo --sidecar --pool");
    eprintln!("note: demo --fake --pool is rejected; --pool requires --sidecar");
}

/// Run one turn with the in-process fakes and render it.
///
/// # Errors
///
/// Returns a protocol-class error if a fake manifest or the session
/// configuration is rejected, and propagates any error from the turn.
pub async fn demo_fake() -> Result<(), ParlevoxError> {
    let registry = ProviderRegistry::fakes()?;
    let providers = registry.resolve(&ProviderNames::default())?;
    let session = Session::new(SessionConfig::default(), providers)?;
    println!("providers stt=fake-stt llm=fake-llm tts=fake-tts vad=fake-vad");
    run_and_render(session).await
}

/// Run the same turn with STT, LLM and TTS served by the Python sidecar.
///
/// # Errors
///
/// Returns a transport-class error when the bundled SDK cannot be located
/// (the message names `PARLEVOX_PYTHON_SDK` when that override is honoured but
/// invalid, and otherwise lists every path tried) or when the plugin cannot be
/// spawned or reaped, a protocol-class error for an invalid manifest or session
/// configuration, and any error raised by the turn.
pub async fn demo_sidecar() -> Result<(), ParlevoxError> {
    let sdk_path = resolve_sdk_path(&SdkSearch::from_process())
        .map_err(|err| ParlevoxError::Transport(err.to_string()))?;
    let client = Arc::new(SidecarClient::spawn(SidecarConfig::python_sdk_at(sdk_path)).await?);

    let (session, names) = build_sidecar_session(Arc::clone(&client))?;
    println!(
        "providers stt={} llm={} tts={} vad={}",
        names.stt, names.llm, names.tts, names.vad
    );

    let result = run_and_render(session).await;
    // Always reap the child, even when the turn failed.
    let shutdown = client.shutdown().await;
    result?;
    shutdown
}

/// Run two deterministic turns through one [`SidecarPool`].
///
/// The first turn is rendered exactly as `demo --sidecar` renders it, so stdout
/// stays byte-identical to the single-shot demo. The second turn runs through
/// [`run_turn_events`] and prints nothing, which is what makes the reuse
/// observable only in the pool statistics. Those statistics go to stderr after
/// [`SidecarPool::shutdown`], so `shutdown` counts the real teardown rather
/// than a prediction of it.
///
/// The first lease is checked for the `reset` capability before anything is
/// written to stdout: without it a second lease would spawn a second process
/// while still looking like a pool. That case returns a protocol error naming
/// the missing capability instead of claiming reuse that did not happen.
///
/// # Errors
///
/// Returns a transport-class error when the bundled SDK cannot be located or
/// the plugin cannot be spawned or reaped, a protocol-class error when the
/// plugin does not advertise `reset` or a manifest or session configuration is
/// rejected, and any error raised by either turn. The pool is shut down before
/// returning in every case.
pub async fn demo_sidecar_pooled() -> Result<(), ParlevoxError> {
    let sdk_path = resolve_sdk_path(&SdkSearch::from_process())
        .map_err(|err| ParlevoxError::Transport(err.to_string()))?;
    let pool = Arc::new(SidecarPool::new(
        SidecarConfig::python_sdk_at(sdk_path),
        PoolConfig::default(),
    ));

    let turns = run_pooled_turns(Arc::clone(&pool)).await;
    // Shut the pool down before reading stats so `shutdown` reflects the real
    // teardown, and do it on the failure path too.
    pool.shutdown().await;
    let stats = pool.stats();
    eprintln!(
        "pool spawned={} reused={} evicted={} shutdown={} reset_failures={}",
        stats.spawned, stats.reused, stats.evicted, stats.shutdown, stats.reset_failures
    );
    turns
}

/// The two-leases-over-one-pool body of [`demo_sidecar_pooled`].
///
/// Split out so the caller owns shutdown and statistics for both outcomes.
async fn run_pooled_turns(pool: Arc<SidecarPool>) -> Result<(), ParlevoxError> {
    // Turn 1: the only turn that writes to stdout.
    let lease = pool.acquire().await?;
    if !lease.client().supports_reset() {
        return Err(ParlevoxError::Protocol(
            "the plugin does not advertise the reset capability, so it cannot be \
             reused by --pool"
                .into(),
        ));
    }
    let (session, names) = build_sidecar_session(Arc::clone(lease.client()))?;
    println!(
        "providers stt={} llm={} tts={} vad={}",
        names.stt, names.llm, names.tts, names.vad
    );
    let first = run_and_render(session).await;
    let released = lease.release().await;
    first?;
    released?;

    // Turn 2: the same one-turn demo, deliberately silent.
    let lease = pool.acquire().await?;
    let (session, _names) = build_sidecar_session(Arc::clone(lease.client()))?;
    let second = run_turn_events(session).await;
    let released = lease.release().await;
    second?;
    released?;
    Ok(())
}

/// Drive one deterministic demo turn and return the report plus every event.
///
/// This is the event-producing entry point the parity test compares across the
/// fake and sidecar-adapted provider sets.
///
/// # Errors
///
/// Propagates the classified provider, protocol, transport, or closed-session
/// error from the turn.
pub async fn run_turn_events(
    mut session: Session,
) -> Result<(TurnReport, Vec<SessionEvent>), ParlevoxError> {
    let utterance = demo_utterance(16_000, 1, 40);
    let result = session.run_turn(utterance).await;
    let events = session.take_events();
    Ok((result?, events))
}

async fn run_and_render(mut session: Session) -> Result<(), ParlevoxError> {
    let utterance = demo_utterance(16_000, 1, 40);
    let result = session.run_turn(utterance).await;
    for event in session.take_events() {
        println!("{}", render(&event));
    }
    match result {
        Ok(report) => {
            println!(
                "summary turn={} transcript={:?} reply={:?} audio_frames={} \
                 dropped_frames={} dropped_ms={} dropped_events={} \
                 sidecar_dropped_frames={} interrupted={} \
                 context_messages={} context_evicted_messages={}",
                report.turn,
                report.transcript,
                report.reply,
                report.audio_frames,
                report.dropped_frames,
                report.dropped_ms,
                report.dropped_events,
                report.sidecar_dropped_frames,
                report.interrupted,
                report.context_messages,
                report.context_evicted_messages
            );
            Ok(())
        }
        Err(err) => Err(err),
    }
}

pub fn render(event: &SessionEvent) -> String {
    match event {
        SessionEvent::TurnStarted { turn } => format!("event TurnStarted turn={turn}"),
        SessionEvent::TranscriptFinal { turn, text } => {
            format!("event TranscriptFinal turn={turn} text={text:?}")
        }
        SessionEvent::ReplyDelta { turn, text } => {
            format!("event ReplyDelta turn={turn} text={text:?}")
        }
        SessionEvent::AudioOut { turn, frame } => format!(
            "event AudioOut turn={turn} samples={} rate={} channels={}",
            frame.pcm.len(),
            frame.sample_rate,
            frame.channels
        ),
        SessionEvent::TurnDone {
            turn,
            audio_frames,
            dropped_frames,
            dropped_ms,
            dropped_events,
            sidecar_dropped_frames,
            interrupted,
        } => format!(
            "event TurnDone turn={turn} audio_frames={audio_frames} dropped_frames={dropped_frames} \
             dropped_ms={dropped_ms} dropped_events={dropped_events} \
             sidecar_dropped_frames={sidecar_dropped_frames} interrupted={interrupted}"
        ),
        SessionEvent::Failed { class, message } => {
            format!("event Failed class={class:?} message={message:?}")
        }
    }
}
