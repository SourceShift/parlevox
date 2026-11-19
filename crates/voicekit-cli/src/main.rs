//! `voicekit-cli` — one-turn demo over either provider backend.
//!
//! `demo --fake` uses only in-process fakes: no process, no Python import, no
//! network. `demo --sidecar` runs the same turn with LLM and TTS served by the
//! optional Python plugin over the framed protocol.

use std::sync::Arc;

use voicekit_core::VoicekitError;
use voicekit_providers::{FakeStt, FakeVad, LlmProvider, TtsProvider};
use voicekit_runtime::{
    demo_utterance, ProviderNames, ProviderRegistry, Session, SessionConfig, SessionEvent,
};
use voicekit_sidecar::{sdk_is_available, SidecarClient, SidecarConfig, SidecarLlm, SidecarTts};

#[tokio::main]
async fn main() {
    match run().await {
        Ok(()) => {}
        Err(err) => {
            eprintln!("voicekit: {err}");
            std::process::exit(1);
        }
    }
}

async fn run() -> Result<(), VoicekitError> {
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
            Err(VoicekitError::Protocol(format!(
                "unrecognised arguments: {}",
                args.join(" ")
            )))
        }
    }
}

fn usage() {
    eprintln!("usage: voicekit-cli demo --fake");
    eprintln!("       voicekit-cli demo --sidecar");
}

async fn demo_fake() -> Result<(), VoicekitError> {
    let registry = ProviderRegistry::fakes()?;
    let providers = registry.resolve(&ProviderNames::default())?;
    let session = Session::new(SessionConfig::default(), providers)?;
    println!("providers stt=fake-stt llm=fake-llm tts=fake-tts vad=fake-vad");
    run_and_render(session).await
}

async fn demo_sidecar() -> Result<(), VoicekitError> {
    if !sdk_is_available() {
        return Err(VoicekitError::Transport(
            "sdks/python is not available; cannot launch the sidecar plugin".into(),
        ));
    }
    let client = Arc::new(SidecarClient::spawn(SidecarConfig::python_sdk()).await?);

    let llm = Arc::new(SidecarLlm::new(Arc::clone(&client)));
    let tts = Arc::new(SidecarTts::new(Arc::clone(&client)));
    let mut registry = ProviderRegistry::new();
    registry.register_stt(Arc::new(FakeStt::new()))?;
    registry.register_llm(llm.clone())?;
    registry.register_tts(tts.clone())?;
    registry.register_vad(Arc::new(FakeVad::new()))?;

    let names = ProviderNames {
        stt: "fake-stt".into(),
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

async fn run_and_render(mut session: Session) -> Result<(), VoicekitError> {
    let utterance = demo_utterance(16_000, 1, 40);
    let result = session.run_turn(utterance).await;
    for event in session.take_events() {
        println!("{}", render(&event));
    }
    match result {
        Ok(report) => {
            println!(
                "summary turn={} transcript={:?} reply={:?} audio_frames={} interrupted={}",
                report.turn,
                report.transcript,
                report.reply,
                report.audio_frames,
                report.interrupted
            );
            Ok(())
        }
        Err(err) => Err(err),
    }
}

fn render(event: &SessionEvent) -> String {
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
            interrupted,
        } => format!(
            "event TurnDone turn={turn} audio_frames={audio_frames} dropped_frames={dropped_frames} \
             dropped_ms={dropped_ms} interrupted={interrupted}"
        ),
        SessionEvent::Failed { class, message } => {
            format!("event Failed class={class:?} message={message:?}")
        }
    }
}
