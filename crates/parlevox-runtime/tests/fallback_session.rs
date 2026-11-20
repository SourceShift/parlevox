//! In-session integration tests for the [`FallbackLlm`] chain.
//!
//! The provider-level suite in `parlevox-providers/tests/fallback.rs` proves the
//! chain's own switching law. These tests prove the same law survives a real
//! [`Session::run_turn`]: a pre-commit failure switches providers inside a turn,
//! a post-commit failure fails the turn without switching, a barge-in cancels the
//! stream without switching, and the turn report sums the whole chain's dropped
//! sidecar frames.
//!
//! The only test double is [`ScriptedLlm`]; the STT, TTS, and VAD roles use the
//! existing deterministic fakes.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use parlevox_core::{ErrorClass, ParlevoxError};
use parlevox_providers::{
    cancelable_stream, FakeStt, FakeTts, FakeVad, FallbackLlm, FallbackStatsSnapshot,
    GenerateRequest, LlmProvider, Modality, ProviderManifest, ProviderStream, FAKE_SAMPLE_RATE,
    FAKE_TTS_FRAMES,
};
use parlevox_runtime::{demo_utterance, Providers, Session, SessionConfig, SessionEvent};
use tokio_util::sync::CancellationToken;

/// Ordered log of the scripted providers this fake was called for.
type CallLog = Arc<Mutex<Vec<&'static str>>>;

/// What a [`ScriptedLlm`] does when `generate` is called.
enum Script {
    /// The `generate` call itself fails before a stream exists.
    FailBeforeCommit(ErrorClass),
    /// The stream yields one committed item and then fails.
    OneItemThenFail(&'static str, ErrorClass),
    /// The stream yields these items and ends cleanly.
    Items(Vec<&'static str>),
}

/// Build the classified error a script asks for.
fn scripted_error(provider: &'static str, class: ErrorClass) -> ParlevoxError {
    match class {
        ErrorClass::Transient => ParlevoxError::provider(provider, "scripted transient failure"),
        ErrorClass::Capacity => ParlevoxError::Capacity("scripted capacity failure".into()),
        ErrorClass::Fatal => ParlevoxError::fatal(provider, "scripted fatal failure"),
        ErrorClass::Protocol => ParlevoxError::Protocol("scripted protocol failure".into()),
        ErrorClass::Cancelled => ParlevoxError::Cancelled,
    }
}

/// An LLM provider with a scripted failure mode, a shared call log, and a
/// take-once dropped-frame counter.
struct ScriptedLlm {
    name: &'static str,
    manifest: ProviderManifest,
    script: Script,
    calls: CallLog,
    dropped: Arc<AtomicU64>,
}

impl ScriptedLlm {
    fn new(name: &'static str, script: Script, dropped_frames: u64) -> Self {
        ScriptedLlm {
            name,
            manifest: ProviderManifest::new(name, Modality::Llm),
            script,
            calls: Arc::new(Mutex::new(Vec::new())),
            dropped: Arc::new(AtomicU64::new(dropped_frames)),
        }
    }

    /// The providers this fake was asked to serve, in call order.
    fn calls(&self) -> Vec<&'static str> {
        self.calls
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    fn record_call(&self) {
        self.calls
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(self.name);
    }
}

#[async_trait]
impl LlmProvider for ScriptedLlm {
    fn manifest(&self) -> &ProviderManifest {
        &self.manifest
    }

    async fn generate(
        &self,
        _request: GenerateRequest,
        cancel: CancellationToken,
    ) -> Result<ProviderStream<String>, ParlevoxError> {
        self.record_call();
        match &self.script {
            Script::FailBeforeCommit(class) => Err(scripted_error(self.name, *class)),
            Script::OneItemThenFail(text, class) => {
                let items = vec![
                    Ok((*text).to_string()),
                    Err(scripted_error(self.name, *class)),
                ];
                Ok(cancelable_stream(items, cancel))
            }
            Script::Items(texts) => {
                let items = texts.iter().map(|text| Ok((*text).to_string())).collect();
                Ok(cancelable_stream(items, cancel))
            }
        }
    }

    fn take_sidecar_dropped_frames(&self) -> u64 {
        self.dropped.swap(0, Ordering::SeqCst)
    }
}

/// Build a session whose LLM role is a two-provider fallback chain, returning
/// the chain so the test can read its counters after the turn.
fn session_with_chain(
    config: SessionConfig,
    primary: &Arc<ScriptedLlm>,
    secondary: &Arc<ScriptedLlm>,
) -> (Session, Arc<FallbackLlm>) {
    let primary_link = Arc::clone(primary);
    let secondary_link = Arc::clone(secondary);
    let links: Vec<Arc<dyn LlmProvider>> = vec![primary_link, secondary_link];
    let chain = Arc::new(FallbackLlm::new(links).expect("a two-provider chain is non-empty"));
    let llm = Arc::clone(&chain);
    let providers = Providers {
        stt: Arc::new(FakeStt::new()),
        llm,
        tts: Arc::new(FakeTts::new()),
        vad: Arc::new(FakeVad::new()),
    };
    let session = Session::new(config, providers).expect("the session config is valid");
    (session, chain)
}

/// The number of audio frames the fake TTS emits for a non-empty reply.
fn expected_audio_frames() -> u64 {
    u64::try_from(FAKE_TTS_FRAMES).expect("the fake frame count fits in u64")
}

#[tokio::test]
async fn pre_commit_transient_switches_provider_inside_a_turn() {
    // Arrange: the primary cannot start; the secondary answers.
    let primary = Arc::new(ScriptedLlm::new(
        "primary",
        Script::FailBeforeCommit(ErrorClass::Transient),
        0,
    ));
    let secondary = Arc::new(ScriptedLlm::new(
        "secondary",
        Script::Items(vec!["fallback ", "reply"]),
        0,
    ));
    let (mut session, chain) = session_with_chain(SessionConfig::default(), &primary, &secondary);

    // Act
    let report = session
        .run_turn(demo_utterance(FAKE_SAMPLE_RATE, 1, 40))
        .await
        .expect("the surviving provider completes the turn");

    // Assert: the reply and its audio come from the surviving provider.
    assert_eq!(report.reply, "fallback reply");
    assert_eq!(report.audio_frames, expected_audio_frames());
    assert!(!report.interrupted);
    assert_eq!(
        chain.stats(),
        FallbackStatsSnapshot {
            switched: 1,
            exhausted: 0,
        }
    );
    assert_eq!(primary.calls(), vec!["primary"]);
    assert_eq!(secondary.calls(), vec!["secondary"]);
}

#[tokio::test]
async fn post_commit_error_fails_the_turn_without_switching() {
    // Arrange: the primary commits one item and then fails transiently.
    let primary = Arc::new(ScriptedLlm::new(
        "primary",
        Script::OneItemThenFail("committed", ErrorClass::Transient),
        0,
    ));
    let secondary = Arc::new(ScriptedLlm::new(
        "secondary",
        Script::Items(vec!["never"]),
        0,
    ));
    let (mut session, chain) = session_with_chain(SessionConfig::default(), &primary, &secondary);

    // Act
    let error = session
        .run_turn(demo_utterance(FAKE_SAMPLE_RATE, 1, 40))
        .await
        .expect_err("a post-commit failure fails the turn");

    // Assert: the error is surfaced unchanged and the chain never switches.
    assert_eq!(error.class(), ErrorClass::Transient);
    assert!(secondary.calls().is_empty());
    assert_eq!(
        chain.stats(),
        FallbackStatsSnapshot {
            switched: 0,
            exhausted: 0,
        }
    );

    let events = session.take_events();
    assert!(matches!(
        events.last(),
        Some(SessionEvent::Failed {
            class: ErrorClass::Transient,
            ..
        })
    ));
    assert!(!events
        .iter()
        .any(|event| matches!(event, SessionEvent::TurnDone { .. })));
    // Only the already-committed item reached the caller.
    let deltas: Vec<&str> = events
        .iter()
        .filter_map(|event| match event {
            SessionEvent::ReplyDelta { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(deltas, vec!["committed"]);
}

#[tokio::test]
async fn barge_in_cancels_mid_stream_without_switching() {
    // Arrange: a four-delta reply and a barge-in after two deltas.
    let primary = Arc::new(ScriptedLlm::new(
        "primary",
        Script::Items(vec!["a", "b", "c", "d"]),
        0,
    ));
    let secondary = Arc::new(ScriptedLlm::new(
        "secondary",
        Script::Items(vec!["never"]),
        0,
    ));
    let config = SessionConfig {
        barge_in_after_deltas: Some(2),
        ..SessionConfig::default()
    };
    let (mut session, chain) = session_with_chain(config, &primary, &secondary);

    // Act
    let report = session
        .run_turn(demo_utterance(FAKE_SAMPLE_RATE, 1, 40))
        .await
        .expect("a barge-in ends the turn without an error");

    // Assert: the interrupted turn never starts the next provider.
    assert!(report.interrupted);
    // The returned result alone is not enough; the call log proves the chain
    // did not fall back after the caller asked to stop.
    assert!(secondary.calls().is_empty());
    assert_eq!(
        chain.stats(),
        FallbackStatsSnapshot {
            switched: 0,
            exhausted: 0,
        }
    );

    let events = session.take_events();
    assert!(matches!(
        events.last(),
        Some(SessionEvent::TurnDone {
            interrupted: true,
            ..
        })
    ));
    assert!(!events
        .iter()
        .any(|event| matches!(event, SessionEvent::Failed { .. })));
}

#[tokio::test]
async fn turn_report_sums_sidecar_dropped_frames_across_the_chain() {
    // Arrange: the failing primary and the surviving secondary each report
    // their own per-request loss.
    let primary = Arc::new(ScriptedLlm::new(
        "primary",
        Script::FailBeforeCommit(ErrorClass::Transient),
        7,
    ));
    let secondary = Arc::new(ScriptedLlm::new(
        "secondary",
        Script::Items(vec!["survived"]),
        2,
    ));
    let (mut session, _chain) = session_with_chain(SessionConfig::default(), &primary, &secondary);

    // Act
    let report = session
        .run_turn(demo_utterance(FAKE_SAMPLE_RATE, 1, 40))
        .await
        .expect("the surviving provider completes the turn");

    // Assert: the chain total, not merely the survivor's 2.
    assert_eq!(report.sidecar_dropped_frames, 9);
}
