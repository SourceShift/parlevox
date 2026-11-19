//! Fake-vs-sidecar parity for one scripted turn.
//!
//! The sidecar side is an in-process test double: it reproduces the plugin's
//! provider behaviour by round-tripping every delta and audio frame through the
//! same framed protocol the real plugin uses. Nothing here spawns Python,
//! imports the SDK, or touches the network.

use std::io::Cursor;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::StreamExt;
use tokio_util::sync::CancellationToken;
use voicekit_core::VoicekitError;
use voicekit_ipc::{read_frame, write_frame, AudioHeader, Frame, PluginEvent};
use voicekit_providers::{
    cancelable_stream, AudioChunk, FakeStt, FakeTts, FakeVad, GenerateRequest, LlmProvider,
    Modality, ProviderManifest, ProviderStream, SynthesizeRequest, TtsProvider, FAKE_LLM_DELTAS,
    FAKE_REPLY, FAKE_TRANSCRIPT, FAKE_TTS_FRAMES,
};
use voicekit_runtime::{ProviderNames, ProviderRegistry, Session, SessionConfig, SessionEvent};

const DOUBLE_LLM_NAME: &str = "sidecar-double-llm";
const DOUBLE_TTS_NAME: &str = "sidecar-double-tts";

/// LLM double: the plugin's deltas, carried through the JSON framing layer.
struct SidecarDoubleLlm {
    manifest: ProviderManifest,
}

impl SidecarDoubleLlm {
    fn new() -> Self {
        SidecarDoubleLlm {
            manifest: ProviderManifest::new(DOUBLE_LLM_NAME, Modality::Llm)
                .with_capability("streaming")
                .with_capability("cancel"),
        }
    }
}

#[async_trait]
impl LlmProvider for SidecarDoubleLlm {
    fn manifest(&self) -> &ProviderManifest {
        &self.manifest
    }

    async fn generate(
        &self,
        request: GenerateRequest,
        cancel: CancellationToken,
    ) -> Result<ProviderStream<String>, VoicekitError> {
        if request.prompt.trim().is_empty() {
            return Err(VoicekitError::Protocol("prompt must not be empty".into()));
        }
        let mut items = Vec::new();
        for delta in FAKE_LLM_DELTAS {
            let event = PluginEvent::Delta {
                request_id: "llm-0".into(),
                text: delta.to_string(),
            };
            let mut bytes = Vec::new();
            write_frame(
                &mut bytes,
                &Frame::Json(serde_json::to_value(event).unwrap()),
            )
            .await
            .map_err(|err| VoicekitError::Protocol(err.to_string()))?;
            let mut cursor = Cursor::new(bytes);
            let Frame::Json(value) = read_frame(&mut cursor)
                .await
                .map_err(|err| VoicekitError::Protocol(err.to_string()))?
            else {
                return Err(VoicekitError::Protocol("expected JSON frame".into()));
            };
            match serde_json::from_value::<PluginEvent>(value) {
                Ok(PluginEvent::Delta { text, .. }) => items.push(Ok(text)),
                other => {
                    return Err(VoicekitError::Protocol(format!(
                        "expected delta, got {other:?}"
                    )))
                }
            }
        }
        Ok(cancelable_stream(items, cancel))
    }
}

/// TTS double: the plugin's audio chunks, carried through the binary framing
/// layer exactly as the sidecar adapter receives them.
struct SidecarDoubleTts {
    manifest: ProviderManifest,
    inner: FakeTts,
}

impl SidecarDoubleTts {
    fn new() -> Self {
        SidecarDoubleTts {
            manifest: ProviderManifest::new(DOUBLE_TTS_NAME, Modality::Tts)
                .with_capability("streaming")
                .with_capability("binary_pcm")
                .with_capability("cancel"),
            inner: FakeTts::new(),
        }
    }
}

#[async_trait]
impl TtsProvider for SidecarDoubleTts {
    fn manifest(&self) -> &ProviderManifest {
        &self.manifest
    }

    async fn synthesize(
        &self,
        request: SynthesizeRequest,
        cancel: CancellationToken,
    ) -> Result<ProviderStream<AudioChunk>, VoicekitError> {
        if request.text.trim().is_empty() {
            return Err(VoicekitError::Protocol("text must not be empty".into()));
        }
        if request.sample_rate == 0 || request.channels == 0 {
            return Err(VoicekitError::Protocol(
                "sample_rate and channels must be non-zero".into(),
            ));
        }

        let mut source = self
            .inner
            .synthesize(request, CancellationToken::new())
            .await?;
        let mut items = Vec::new();
        while let Some(item) = source.next().await {
            let chunk = item?;
            let frame = Frame::Audio {
                header: AudioHeader {
                    request_id: "tts-0".into(),
                    seq: items.len() as u64,
                    sample_rate: chunk.sample_rate,
                    channels: chunk.channels,
                },
                pcm: chunk.pcm,
            };
            let mut bytes = Vec::new();
            write_frame(&mut bytes, &frame)
                .await
                .map_err(|err| VoicekitError::Protocol(err.to_string()))?;
            let mut cursor = Cursor::new(bytes);
            let Frame::Audio { header, pcm } = read_frame(&mut cursor)
                .await
                .map_err(|err| VoicekitError::Protocol(err.to_string()))?
            else {
                return Err(VoicekitError::Protocol("expected audio frame".into()));
            };
            items.push(Ok(AudioChunk::new(
                header.sample_rate,
                header.channels,
                pcm,
            )));
        }
        Ok(cancelable_stream(items, cancel))
    }
}

async fn run_fake() -> (voicekit_runtime::TurnReport, Vec<SessionEvent>) {
    let registry = ProviderRegistry::fakes().unwrap();
    let providers = registry.resolve(&ProviderNames::default()).unwrap();
    let session = Session::new(SessionConfig::default(), providers).unwrap();
    voicekit_cli::run_turn_events(session).await.unwrap()
}

async fn run_sidecar_double() -> (voicekit_runtime::TurnReport, Vec<SessionEvent>) {
    let llm = Arc::new(SidecarDoubleLlm::new());
    let tts = Arc::new(SidecarDoubleTts::new());
    let mut registry = ProviderRegistry::new();
    registry.register_stt(Arc::new(FakeStt::new())).unwrap();
    registry.register_llm(llm.clone()).unwrap();
    registry.register_tts(tts.clone()).unwrap();
    registry.register_vad(Arc::new(FakeVad::new())).unwrap();

    let names = ProviderNames {
        stt: "fake-stt".into(),
        llm: llm.manifest().name.clone(),
        tts: tts.manifest().name.clone(),
        vad: "fake-vad".into(),
    };
    let providers = registry.resolve(&names).unwrap();
    let session = Session::new(SessionConfig::default(), providers).unwrap();
    voicekit_cli::run_turn_events(session).await.unwrap()
}

type DoneFields = (u64, u64, u64, u64, u64, bool);

fn done_fields(events: &[SessionEvent]) -> DoneFields {
    match events.last() {
        Some(SessionEvent::TurnDone {
            turn,
            audio_frames,
            dropped_frames,
            dropped_ms,
            dropped_events,
            interrupted,
        }) => (
            *turn,
            *audio_frames,
            *dropped_frames,
            *dropped_ms,
            *dropped_events,
            *interrupted,
        ),
        other => panic!("expected a terminal TurnDone, got {other:?}"),
    }
}

#[tokio::test]
async fn fake_and_sidecar_double_produce_identical_event_traces() {
    let (fake_report, fake_events) = run_fake().await;
    let (double_report, double_events) = run_sidecar_double().await;

    let fake_kinds: Vec<&str> = fake_events.iter().map(SessionEvent::kind).collect();
    let double_kinds: Vec<&str> = double_events.iter().map(SessionEvent::kind).collect();
    assert_eq!(fake_kinds, double_kinds, "event kind sequences diverged");
    assert_eq!(fake_kinds.first(), Some(&"TurnStarted"));
    assert_eq!(fake_kinds.last(), Some(&"TurnDone"));

    assert_eq!(fake_report, double_report, "turn reports diverged");
    assert_eq!(fake_report.transcript, FAKE_TRANSCRIPT);
    assert_eq!(fake_report.reply, FAKE_REPLY);
    assert_eq!(fake_report.audio_frames as usize, FAKE_TTS_FRAMES);

    let fake_done = done_fields(&fake_events);
    let double_done = done_fields(&double_events);
    assert_eq!(fake_done, double_done);
    // (turn, audio_frames, dropped_frames, dropped_ms, dropped_events, interrupted)
    assert_eq!(fake_done.1 as usize, FAKE_TTS_FRAMES);
    assert_eq!(fake_done.3, 0, "no playout drops at the default capacity");
    assert_eq!(fake_done.4, 0, "no event drops at the default capacity");
    assert!(!fake_done.5);
}

#[tokio::test]
async fn both_paths_render_the_same_trace() {
    let (_, fake_events) = run_fake().await;
    let (_, double_events) = run_sidecar_double().await;
    let fake_lines: Vec<String> = fake_events.iter().map(voicekit_cli::render).collect();
    let double_lines: Vec<String> = double_events.iter().map(voicekit_cli::render).collect();
    assert_eq!(fake_lines, double_lines);
}

#[test]
fn renderer_surfaces_dropped_events() {
    let line = voicekit_cli::render(&SessionEvent::TurnDone {
        turn: 1,
        audio_frames: 2,
        dropped_frames: 0,
        dropped_ms: 0,
        dropped_events: 3,
        interrupted: false,
    });
    assert!(line.contains("dropped_events=3"));
}
