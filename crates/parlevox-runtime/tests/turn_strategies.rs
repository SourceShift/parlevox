//! Ordered turn-start strategies: accept, reject, abstention, and counters.
//!
//! Every test drives the public session API over deterministic fakes: a
//! scripted STT that yields one final per call, [`FakeVad`] for the VAD
//! boundary, and [`FakeLlm`]/[`FakeTts`] sharing a [`ProviderCallLog`]. The
//! strategies are local doubles that record the [`TurnContext`] they observe
//! and return a scripted verdict, so consultation order and provider silence
//! are provable without wall-clock timing or spawned tasks.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, PoisonError};

use async_trait::async_trait;
use parlevox_core::{AudioFrame, ParlevoxError, Timestamp, TurnState};
use parlevox_providers::{
    cancelable_stream, AudioChunk, FakeLlm, FakeTts, FakeVad, Modality, ProviderCall,
    ProviderCallLog, ProviderManifest, ProviderStream, SttProvider, TranscriptChunk, FAKE_REPLY,
};
use parlevox_runtime::{
    demo_utterance, Providers, Session, SessionConfig, SessionEvent, StrategyStats, TurnContext,
    TurnDecision, TurnStrategies, TurnStrategy,
};
use proptest::prelude::{prop_assert, prop_assert_eq, prop_oneof, proptest, Just, Strategy};
use tokio_util::sync::CancellationToken;

/// STT double that hands out one scripted final transcript per `transcribe`.
struct ScriptedStt {
    manifest: ProviderManifest,
    finals: Mutex<VecDeque<String>>,
}

impl ScriptedStt {
    fn new(finals: &[&str]) -> Self {
        ScriptedStt {
            manifest: ProviderManifest::new("scripted-stt", Modality::Stt),
            finals: Mutex::new(finals.iter().map(|text| (*text).to_string()).collect()),
        }
    }
}

#[async_trait]
impl SttProvider for ScriptedStt {
    fn manifest(&self) -> &ProviderManifest {
        &self.manifest
    }

    async fn transcribe(
        &self,
        _audio: AudioChunk,
        cancel: CancellationToken,
    ) -> Result<ProviderStream<TranscriptChunk>, ParlevoxError> {
        let text = self
            .finals
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop_front();
        let Some(text) = text else {
            return Err(ParlevoxError::Protocol(
                "scripted stt ran out of finals".into(),
            ));
        };
        Ok(cancelable_stream(
            vec![Ok(TranscriptChunk {
                text,
                is_final: true,
            })],
            cancel,
        ))
    }
}

/// One strategy consultation as observed by a recording double.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Observation {
    name: &'static str,
    transcript: String,
    state: TurnState,
    turn_index: u64,
}

type Observed = Arc<Mutex<Vec<Observation>>>;

fn new_observed() -> Observed {
    Arc::new(Mutex::new(Vec::new()))
}

fn observations(observed: &Observed) -> Vec<Observation> {
    observed
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone()
}

fn observed_names(observed: &Observed) -> Vec<&'static str> {
    observations(observed)
        .into_iter()
        .map(|observation| observation.name)
        .collect()
}

/// A strategy that records the context it sees and returns a fixed verdict.
struct RecordingStrategy {
    name: &'static str,
    decision: TurnDecision,
    observed: Observed,
}

impl RecordingStrategy {
    fn boxed(
        name: &'static str,
        decision: TurnDecision,
        observed: Observed,
    ) -> Box<dyn TurnStrategy> {
        Box::new(RecordingStrategy {
            name,
            decision,
            observed,
        })
    }
}

impl TurnStrategy for RecordingStrategy {
    fn name(&self) -> &'static str {
        self.name
    }

    fn decide(&mut self, context: &TurnContext<'_>) -> TurnDecision {
        self.observed
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(Observation {
                name: self.name,
                transcript: context.transcript.to_string(),
                state: context.state,
                turn_index: context.turn_index,
            });
        self.decision
    }
}

/// Compile-time proof that `T` is `Sync`; fails to compile if it is not.
fn assert_sync<T: Sync>() {}

#[test]
fn session_is_sync() {
    // The strategy list lives inside the session, so `TurnStrategy: Sync` is
    // what keeps `Session` (and its list) `Sync`.
    assert_sync::<Session>();
    assert_sync::<TurnStrategies>();
    assert_sync::<Box<dyn TurnStrategy>>();
}

/// Stable names for a generated plan: verdict `i` is always the strategy named
/// `PLAN_NAMES[i]`, so observed names prove the consultation order without
/// allocating a static name per strategy.
const PLAN_NAMES: [&str; 8] = ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"];

/// Install `plan` as the ordered strategy list, one recording double per
/// verdict.
fn plan_strategies(plan: &[TurnDecision], observed: &Observed) -> TurnStrategies {
    let mut strategies = TurnStrategies::new();
    for (name, decision) in PLAN_NAMES.iter().copied().zip(plan.iter().copied()) {
        strategies.push(RecordingStrategy::boxed(
            name,
            decision,
            Arc::clone(observed),
        ));
    }
    strategies
}

/// Any verdict the decision law ranges over.
fn any_verdict() -> impl Strategy<Value = TurnDecision> {
    prop_oneof![
        Just(TurnDecision::Accept),
        Just(TurnDecision::Reject),
        Just(TurnDecision::Abstain),
    ]
}

/// A strategy that rejects exactly one transcript, exercising across-turn
/// accumulation.
struct RejectIf {
    reject_transcript: String,
}

impl TurnStrategy for RejectIf {
    fn name(&self) -> &'static str {
        "reject-if"
    }

    fn decide(&mut self, context: &TurnContext<'_>) -> TurnDecision {
        if context.transcript == self.reject_transcript.as_str() {
            TurnDecision::Reject
        } else {
            TurnDecision::Accept
        }
    }
}

fn call_log() -> ProviderCallLog {
    Arc::new(Mutex::new(Vec::new()))
}

fn calls(log: &ProviderCallLog) -> Vec<ProviderCall> {
    log.lock().unwrap_or_else(PoisonError::into_inner).clone()
}

/// Build a session over the scripted STT, the logging fakes, and an optional
/// strategy list.
fn session_with(
    finals: &[&str],
    strategies: Option<TurnStrategies>,
    log: ProviderCallLog,
) -> Session {
    let providers = Providers {
        stt: Arc::new(ScriptedStt::new(finals)),
        llm: Arc::new(FakeLlm::new().with_call_log(Arc::clone(&log))),
        tts: Arc::new(FakeTts::new().with_call_log(log)),
        vad: Arc::new(FakeVad::new()),
    };
    let session =
        Session::new(SessionConfig::default(), providers).expect("a valid session config");
    match strategies {
        Some(strategies) => session.with_turn_strategies(strategies),
        None => session,
    }
}

fn utterance() -> AudioChunk {
    demo_utterance(16_000, 1, 40)
}

/// Strip the per-session monotonic timestamp from audio events so two sessions
/// can be compared event-for-event.
fn comparable(events: Vec<SessionEvent>) -> Vec<SessionEvent> {
    events
        .into_iter()
        .map(|event| match event {
            SessionEvent::AudioOut { turn, frame } => SessionEvent::AudioOut {
                turn,
                frame: AudioFrame::new(
                    Timestamp::from_millis(0),
                    frame.sample_rate,
                    frame.channels,
                    frame.pcm,
                ),
            },
            other => other,
        })
        .collect()
}

#[tokio::test]
async fn reject_suppresses_the_turn() {
    // Arrange: a single rejecting strategy over the deterministic fakes.
    let log = call_log();
    let observed = new_observed();
    let mut strategies = TurnStrategies::new();
    strategies.push(RecordingStrategy::boxed(
        "reject-all",
        TurnDecision::Reject,
        Arc::clone(&observed),
    ));
    let mut session = session_with(&["drop this"], Some(strategies), Arc::clone(&log));

    // Act
    let report = session
        .run_turn(utterance())
        .await
        .expect("a rejected turn returns Ok");

    // Assert: the providers are never touched and the report is empty.
    assert!(
        calls(&log).is_empty(),
        "a rejected turn must not call the LLM or TTS"
    );
    assert_eq!(report.turn, 0, "a rejected turn does not advance the index");
    assert_eq!(
        report.transcript, "drop this",
        "the report carries the rejected final transcript"
    );
    assert!(
        report.reply.is_empty(),
        "no reply may escape a rejected turn"
    );
    assert_eq!(report.audio_frames, 0);
    assert_eq!(report.dropped_frames, 0);
    assert_eq!(report.dropped_ms, 0);
    assert_eq!(report.dropped_events, 0);
    assert_eq!(report.sidecar_dropped_frames, 0);
    assert_eq!(report.input_dropped_frames, 0);
    assert_eq!(report.context_messages, 0);
    assert_eq!(report.context_evicted_messages, 0);
    assert!(!report.interrupted);

    // No TurnStarted/TranscriptFinal/ReplyStarted/ReplyDelta/AudioOut/TurnDone.
    let events = session.take_events();
    assert!(
        events.is_empty(),
        "a rejected turn emits no session events: {events:?}"
    );

    assert_eq!(
        session.state(),
        TurnState::Listening,
        "abort rests the machine in Listening"
    );
    assert_eq!(
        session.turn_index(),
        0,
        "the committed turn index is unchanged"
    );
    assert_eq!(
        session.strategy_stats(),
        StrategyStats {
            consulted: 1,
            accepted: 0,
            rejected: 1,
            abstained: 0,
            turns_rejected: 1,
        }
    );
}

#[tokio::test]
async fn rejected_turn_reports_the_real_context_depth() {
    // Arrange: one accepted turn teaches the context two messages (the user
    // prompt and the reply), then a second transcript is rejected.
    let log = call_log();
    let mut strategies = TurnStrategies::new();
    strategies.push(Box::new(RejectIf {
        reject_transcript: "drop this".to_string(),
    }));
    let mut session = session_with(
        &["keep this", "drop this"],
        Some(strategies),
        Arc::clone(&log),
    );

    // Act: turn one is accepted, turn two is rejected without an error.
    let accepted = session
        .run_turn(utterance())
        .await
        .expect("the first turn is accepted");
    let rejected = session
        .run_turn(utterance())
        .await
        .expect("the second turn is rejected without an error");

    // Assert: both reports describe the live context depth a caller reads.
    let depth = session.context_stats().messages;
    assert!(
        depth > 0,
        "the accepted turn populated the context before the rejection"
    );
    assert_eq!(
        accepted.context_messages, depth,
        "the accepted report must match the session's context depth"
    );
    assert_eq!(
        rejected.context_messages, depth,
        "a rejected turn must report the context depth it left intact"
    );
    assert_ne!(
        rejected.context_messages, 0,
        "a rejected turn after a committed turn has real depth, not zero"
    );
}

#[tokio::test]
async fn first_decisive_strategy_wins_and_abstain_falls_through() {
    // Arrange: abstain, then reject, then accept.
    let log = call_log();
    let observed = new_observed();
    let mut strategies = TurnStrategies::new();
    strategies.push(RecordingStrategy::boxed(
        "a",
        TurnDecision::Abstain,
        Arc::clone(&observed),
    ));
    strategies.push(RecordingStrategy::boxed(
        "b",
        TurnDecision::Reject,
        Arc::clone(&observed),
    ));
    strategies.push(RecordingStrategy::boxed(
        "c",
        TurnDecision::Accept,
        Arc::clone(&observed),
    ));
    assert_eq!(strategies.names(), vec!["a", "b", "c"]);
    let mut session = session_with(&["never mind"], Some(strategies), Arc::clone(&log));

    // Act
    let report = session
        .run_turn(utterance())
        .await
        .expect("a rejecting strategy returns Ok");

    // Assert: consultation stops at the first decisive verdict.
    assert_eq!(
        observed_names(&observed),
        vec!["a", "b"],
        "abstain falls through to b, and c is never consulted after b rejects"
    );
    assert!(report.reply.is_empty());
    assert_eq!(report.transcript, "never mind");
    assert!(calls(&log).is_empty());
    assert_eq!(
        session.strategy_stats(),
        StrategyStats {
            consulted: 2,
            accepted: 0,
            rejected: 1,
            abstained: 1,
            turns_rejected: 1,
        }
    );
}

#[tokio::test]
async fn all_abstain_or_empty_list_accepts() {
    // Arrange: every strategy abstains.
    let log = call_log();
    let observed = new_observed();
    let mut strategies = TurnStrategies::new();
    strategies.push(RecordingStrategy::boxed(
        "one",
        TurnDecision::Abstain,
        Arc::clone(&observed),
    ));
    strategies.push(RecordingStrategy::boxed(
        "two",
        TurnDecision::Abstain,
        Arc::clone(&observed),
    ));
    let mut abstaining = session_with(&["hello there"], Some(strategies), Arc::clone(&log));

    // Act
    let report = abstaining
        .run_turn(utterance())
        .await
        .expect("an all-abstain list accepts");

    // Assert: the normal reply runs and every verdict is counted.
    assert_eq!(report.reply, FAKE_REPLY);
    assert_eq!(report.turn, 1);
    assert!(
        !calls(&log).is_empty(),
        "the accepting turn calls the fakes"
    );
    assert_eq!(
        abstaining.strategy_stats(),
        StrategyStats {
            consulted: 2,
            accepted: 0,
            rejected: 0,
            abstained: 2,
            turns_rejected: 0,
        }
    );

    // Arrange: an installed empty list.
    let empty_log = call_log();
    let empty = TurnStrategies::new();
    assert!(empty.is_empty());
    assert_eq!(empty.len(), 0);
    assert!(empty.names().is_empty());
    let mut inert = session_with(&["hello there"], Some(empty), Arc::clone(&empty_log));

    // Act
    let report = inert
        .run_turn(utterance())
        .await
        .expect("an empty list accepts");

    // Assert: inert, and the counters stay zero.
    assert_eq!(report.reply, FAKE_REPLY);
    assert_eq!(report.turn, 1);
    assert_eq!(inert.strategy_stats(), StrategyStats::default());
}

#[tokio::test]
async fn accept_matches_the_no_list_path() {
    // Arrange: one accepting strategy, and the same turn with no list.
    let accept_log = call_log();
    let observed = new_observed();
    let mut strategies = TurnStrategies::new();
    strategies.push(RecordingStrategy::boxed(
        "accept",
        TurnDecision::Accept,
        Arc::clone(&observed),
    ));
    let mut accepting = session_with(&["hello there"], Some(strategies), Arc::clone(&accept_log));

    let plain_log = call_log();
    let mut plain = session_with(&["hello there"], None, Arc::clone(&plain_log));

    // Act
    let accepting_report = accepting
        .run_turn(utterance())
        .await
        .expect("accept runs the reply");
    let accepting_events = comparable(accepting.take_events());
    let accepting_calls = calls(&accept_log);

    let plain_report = plain
        .run_turn(utterance())
        .await
        .expect("no list runs the reply");
    let plain_events = comparable(plain.take_events());
    let plain_calls = calls(&plain_log);

    // Assert: an Accept strategy is observationally identical to no list.
    assert_eq!(
        accepting_events, plain_events,
        "an Accept strategy does not change the event stream"
    );
    assert_eq!(
        accepting_report, plain_report,
        "an Accept strategy does not change the report"
    );
    assert_eq!(
        accepting_calls, plain_calls,
        "an Accept strategy does not change provider calls"
    );
    assert_eq!(observed_names(&observed), vec!["accept"]);
}

#[tokio::test]
async fn the_context_carries_the_final_transcript() {
    // Arrange: a recording Accept strategy over a single scripted final.
    let log = call_log();
    let observed = new_observed();
    let mut strategies = TurnStrategies::new();
    strategies.push(RecordingStrategy::boxed(
        "spy",
        TurnDecision::Accept,
        Arc::clone(&observed),
    ));
    let mut session = session_with(&["what is the weather"], Some(strategies), log);

    // Act
    session.run_turn(utterance()).await.expect("the turn runs");

    // Assert: the context carries the final, the VAD-end state, and the
    // candidate turn index (committed index 0 + 1).
    assert_eq!(
        observations(&observed),
        vec![Observation {
            name: "spy",
            transcript: "what is the weather".to_string(),
            state: TurnState::UserSpeaking,
            turn_index: 1,
        }]
    );
}

#[tokio::test]
async fn stats_accumulate_across_turns() {
    // Arrange: two scripted finals; the strategy rejects the first only.
    let log = call_log();
    let mut strategies = TurnStrategies::new();
    strategies.push(Box::new(RejectIf {
        reject_transcript: "drop this".to_string(),
    }));
    let mut session = session_with(
        &["drop this", "keep this"],
        Some(strategies),
        Arc::clone(&log),
    );

    // Act: turn one is rejected, turn two is accepted.
    let rejected = session
        .run_turn(utterance())
        .await
        .expect("the first turn is rejected without an error");
    assert_eq!(rejected.turn, 0);
    assert_eq!(rejected.transcript, "drop this");
    assert!(rejected.reply.is_empty());
    assert_eq!(session.state(), TurnState::Listening);
    assert_eq!(session.turn_index(), 0);

    let accepted = session
        .run_turn(utterance())
        .await
        .expect("the second turn is accepted");

    // Assert: the accepted turn uses the index the rejected turn did not
    // consume, and the counters sum across both turns.
    assert_eq!(
        accepted.turn, 1,
        "the rejected turn did not consume the index"
    );
    assert_eq!(accepted.reply, FAKE_REPLY);
    assert_eq!(
        session.strategy_stats(),
        StrategyStats {
            consulted: 2,
            accepted: 1,
            rejected: 1,
            abstained: 0,
            turns_rejected: 1,
        }
    );
    assert!(
        calls(&log)
            .iter()
            .any(|call| matches!(call, ProviderCall::LlmDelta(_))),
        "the accepted turn called the LLM"
    );
}

proptest! {
    /// The decision law, for any plan of 0-8 verdicts over one turn.
    #[test]
    fn decision_law_holds_for_any_plan(plan in proptest::collection::vec(any_verdict(), 0..=8)) {
        // Arrange: one session over the deterministic fakes, carrying `plan`.
        let observed = new_observed();
        let log = call_log();
        let strategies = plan_strategies(&plan, &observed);
        let mut session =
            session_with(&["property utterance"], Some(strategies), Arc::clone(&log));
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("a current-thread tokio runtime");

        // Act: exactly one turn, so the counters describe this plan alone.
        let report = runtime
            .block_on(session.run_turn(utterance()))
            .expect("no strategy plan errors");

        // (a) The first non-Abstain verdict decides, and everything after it is
        // skipped; an empty or all-Abstain plan consults the whole plan.
        let decisive = plan
            .iter()
            .position(|decision| *decision != TurnDecision::Abstain);
        let decisive_verdict = decisive.map(|index| plan[index]);
        let expected_consulted = u64::try_from(decisive.map_or(plan.len(), |index| index + 1))
            .expect("a generated plan is far below u64::MAX");
        let expected_accepted = u64::from(decisive_verdict == Some(TurnDecision::Accept));
        let expected_rejected = u64::from(decisive_verdict == Some(TurnDecision::Reject));
        let expected_abstained = expected_consulted - expected_accepted - expected_rejected;

        // (b) The counters sum, and only the decider can reject the turn.
        let stats = session.strategy_stats();
        prop_assert_eq!(stats.consulted, expected_consulted);
        prop_assert_eq!(stats.accepted, expected_accepted);
        prop_assert_eq!(stats.rejected, expected_rejected);
        prop_assert_eq!(stats.abstained, expected_abstained);
        prop_assert_eq!(
            stats.consulted,
            stats.accepted + stats.rejected + stats.abstained
        );
        prop_assert_eq!(stats.turns_rejected, stats.rejected);

        // (c) The observed names are exactly the plan prefix up to and
        // including the decider (or the whole plan when nothing decides).
        let expected_names: Vec<&'static str> = match decisive {
            Some(index) => PLAN_NAMES[..=index].to_vec(),
            None => PLAN_NAMES[..plan.len()].to_vec(),
        };
        prop_assert_eq!(observed_names(&observed), expected_names);

        // (d) A rejecting plan never calls the providers and does not commit;
        // an empty or all-Abstain plan accepts, committing turn 1 exactly as
        // with no list installed.
        if expected_rejected == 1 {
            prop_assert!(report.reply.is_empty());
            prop_assert!(calls(&log).is_empty());
            prop_assert_eq!(report.turn, 0);
            prop_assert_eq!(session.turn_index(), 0);
        } else {
            prop_assert_eq!(report.reply.as_str(), FAKE_REPLY);
            prop_assert!(!calls(&log).is_empty());
            prop_assert_eq!(report.turn, 1);
            prop_assert_eq!(session.turn_index(), 1);
        }
    }
}
