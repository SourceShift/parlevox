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
    sdk_is_available, SidecarClient, SidecarConfig, SidecarLlm, SidecarStt, SidecarTts,
};

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
/// Returns a transport-class error when the bundled SDK is unavailable or the
/// plugin cannot be spawned or reaped, a protocol-class error for an invalid
/// manifest or session configuration, and any error raised by the turn.
pub async fn demo_sidecar() -> Result<(), ParlevoxError> {
    if !sdk_is_available() {
        return Err(ParlevoxError::Transport(
            "sdks/python is not available; cannot launch the sidecar plugin".into(),
        ));
    }
    let client = Arc::new(SidecarClient::spawn(SidecarConfig::python_sdk()).await?);

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
                 sidecar_dropped_frames={} interrupted={}",
                report.turn,
                report.transcript,
                report.reply,
                report.audio_frames,
                report.dropped_frames,
                report.dropped_ms,
                report.dropped_events,
                report.sidecar_dropped_frames,
                report.interrupted
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
