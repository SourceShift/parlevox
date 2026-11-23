//! Explicit turn state machine.
//!
//! Every comparable voice-agent framework encodes turn invariants somewhere:
//! in booleans, in task cancellation, or in prose. Here
//! they are a closed set of states and a table of legal transitions. An illegal
//! event returns [`crate::error::ParlevoxError::Protocol`] instead of being
//! silently absorbed, so a misbehaving plugin fails loudly at the boundary.
//!
//! The legal transitions live in [`LEGAL_TRANSITIONS`] as data; [`TurnMachine::apply`]
//! is a lookup over that table. The checked-in prose contract in
//! `docs/architecture/TURN-CONTRACT.md` is verified against this table by
//! `crates/parlevox-core/tests/turn_contract.rs`, so the document and the code
//! cannot drift apart silently.

use crate::error::ParlevoxError;
use serde::{Deserialize, Serialize};

/// The closed set of turn states.
///
/// The enum is `#[non_exhaustive]` so a future additive state does not break
/// downstream crates; within this crate every `match` stays explicit over all
/// variants so a new state cannot be silently folded into an old one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum TurnState {
    /// Session created, no user activity yet.
    Idle,
    /// Waiting for the user.
    Listening,
    /// User is producing speech right now.
    UserSpeaking,
    /// The endpointing window: a final candidate exists but the turn has not
    /// been committed yet, so speech may still cancel it.
    Committing,
    /// A commit exists; the reply has not started streaming.
    Thinking,
    /// An LLM stream is producing text but has not queued audio yet. Distinct
    /// from [`TurnState::Speaking`], whose definition is "first frame queued".
    AgentGenerating,
    /// The agent's audio is playing.
    Speaking,
    /// Speech was detected while the agent was speaking; the resolution window
    /// has not yet decided between a backchannel and a barge-in.
    EvaluatingBargeIn,
    /// Barge-in happened; the reply is being torn down.
    Interrupted,
    /// Terminal.
    Closed,
}

impl TurnState {
    /// Canonical name of the state, identical to the Rust variant name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            TurnState::Idle => "Idle",
            TurnState::Listening => "Listening",
            TurnState::UserSpeaking => "UserSpeaking",
            TurnState::Committing => "Committing",
            TurnState::Thinking => "Thinking",
            TurnState::AgentGenerating => "AgentGenerating",
            TurnState::Speaking => "Speaking",
            TurnState::EvaluatingBargeIn => "EvaluatingBargeIn",
            TurnState::Interrupted => "Interrupted",
            TurnState::Closed => "Closed",
        }
    }

    /// Who owns the audio floor in this state.
    ///
    /// The mapping is stated in `docs/architecture/TURN-CONTRACT.md` and
    /// checked by the contract test. [`TurnState::Closed`] is a terminal state
    /// the requirements document does not list; it owns no floor.
    #[must_use]
    pub const fn owns_floor(self) -> FloorOwner {
        match self {
            TurnState::Idle | TurnState::Listening | TurnState::UserSpeaking => FloorOwner::User,
            TurnState::Committing => FloorOwner::Nobody,
            TurnState::Thinking | TurnState::AgentGenerating => FloorOwner::AgentUnconfirmed,
            TurnState::Speaking => FloorOwner::AgentConfirmed,
            TurnState::EvaluatingBargeIn => FloorOwner::Contested,
            TurnState::Interrupted | TurnState::Closed => FloorOwner::Nobody,
        }
    }

    /// What audio the agent may emit in this state.
    ///
    /// This is the contract's `May emit audio` column, stated in
    /// `docs/architecture/TURN-CONTRACT.md` and checked by the contract test.
    #[must_use]
    pub const fn audio_emission(self) -> AudioEmission {
        match self {
            TurnState::Idle
            | TurnState::Listening
            | TurnState::UserSpeaking
            | TurnState::Committing
            | TurnState::Thinking
            | TurnState::AgentGenerating
            | TurnState::Closed => AudioEmission::Silent,
            TurnState::Speaking => AudioEmission::Audible,
            TurnState::EvaluatingBargeIn => AudioEmission::Ducked,
            TurnState::Interrupted => AudioEmission::DrainOnly,
        }
    }
}

/// Who currently owns the audio floor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FloorOwner {
    /// The user owns the floor; the agent must stay silent.
    User,
    /// The agent holds the floor but has not confirmed playout yet.
    AgentUnconfirmed,
    /// The agent is producing confirmed audio.
    AgentConfirmed,
    /// Both parties are contending for the floor while a barge-in is being
    /// evaluated; neither has won it yet.
    Contested,
    /// Neither party owns the floor.
    Nobody,
}

impl FloorOwner {
    /// Stable, lowercase label used in contracts and diagnostics.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            FloorOwner::User => "user",
            FloorOwner::AgentUnconfirmed => "agent-unconfirmed",
            FloorOwner::AgentConfirmed => "agent-confirmed",
            FloorOwner::Contested => "contested",
            FloorOwner::Nobody => "nobody",
        }
    }
}

/// What audio the agent may emit in a state.
///
/// This is the contract's `May emit audio` column. The variants distinguish a
/// hard silence from "nothing new is generated but already-queued frames may
/// drain", from "output is attenuated while a barge-in is evaluated".
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AudioEmission {
    /// The agent must stay silent.
    Silent,
    /// Only audio already queued may drain; nothing new is generated.
    DrainOnly,
    /// Output is attenuated while the barge-in resolution window is open.
    Ducked,
    /// The agent may emit audio at full level.
    Audible,
}

impl AudioEmission {
    /// Stable, lowercase label used in contracts and diagnostics.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            AudioEmission::Silent => "no",
            AudioEmission::DrainOnly => "drain only",
            AudioEmission::Ducked => "ducked",
            AudioEmission::Audible => "yes",
        }
    }
}

/// A [`TurnEvent`] with its payload erased, suitable as a transition-table key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnEventKind {
    /// Session is ready to take input.
    Start,
    /// The user started producing speech.
    UserStartedSpeaking,
    /// The user stopped producing speech.
    UserStoppedSpeaking,
    /// Endpointing accepted a final candidate; the turn may still be rejected.
    CommitStarted,
    /// A final transcript was committed (either after VAD silence or manually).
    FinalTranscript,
    /// The endpointing window committed the turn.
    TurnCommitted,
    /// The endpointing window rejected the turn; nothing was committed.
    CommitRejected,
    /// The reply stream produced its first text without queuing audio yet.
    ReplyStreamOpened,
    /// The first agent audio frame was queued for playout.
    ReplyStarted,
    /// The last queued frame finished playing.
    ReplyFinished,
    /// Speech was detected while the agent held the floor.
    SpeechDetected,
    /// The resolution window decided the detection was a backchannel.
    BargeInRejected,
    /// The resolution window decided the detection was a barge-in.
    BargeInConfirmed,
    /// The user interrupted an in-flight reply.
    BargeIn,
    /// Cleanup after an interruption is complete.
    ResumeListening,
    /// The session is being torn down.
    Close,
}

impl TurnEventKind {
    /// Canonical name of the event kind, identical to the Rust variant name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            TurnEventKind::Start => "Start",
            TurnEventKind::UserStartedSpeaking => "UserStartedSpeaking",
            TurnEventKind::UserStoppedSpeaking => "UserStoppedSpeaking",
            TurnEventKind::CommitStarted => "CommitStarted",
            TurnEventKind::FinalTranscript => "FinalTranscript",
            TurnEventKind::TurnCommitted => "TurnCommitted",
            TurnEventKind::CommitRejected => "CommitRejected",
            TurnEventKind::ReplyStreamOpened => "ReplyStreamOpened",
            TurnEventKind::ReplyStarted => "ReplyStarted",
            TurnEventKind::ReplyFinished => "ReplyFinished",
            TurnEventKind::SpeechDetected => "SpeechDetected",
            TurnEventKind::BargeInRejected => "BargeInRejected",
            TurnEventKind::BargeInConfirmed => "BargeInConfirmed",
            TurnEventKind::BargeIn => "BargeIn",
            TurnEventKind::ResumeListening => "ResumeListening",
            TurnEventKind::Close => "Close",
        }
    }

    /// Build a representative event for this kind.
    ///
    /// Payload-carrying variants receive a placeholder payload of the right
    /// type, so a table-driven driver can enumerate every kind.
    #[must_use]
    pub fn sample_event(self) -> TurnEvent {
        match self {
            TurnEventKind::Start => TurnEvent::Start,
            TurnEventKind::UserStartedSpeaking => TurnEvent::UserStartedSpeaking,
            TurnEventKind::UserStoppedSpeaking => TurnEvent::UserStoppedSpeaking,
            TurnEventKind::CommitStarted => TurnEvent::CommitStarted,
            TurnEventKind::FinalTranscript => TurnEvent::FinalTranscript {
                text: String::from("sample"),
            },
            TurnEventKind::TurnCommitted => TurnEvent::TurnCommitted,
            TurnEventKind::CommitRejected => TurnEvent::CommitRejected,
            TurnEventKind::ReplyStreamOpened => TurnEvent::ReplyStreamOpened,
            TurnEventKind::ReplyStarted => TurnEvent::ReplyStarted,
            TurnEventKind::ReplyFinished => TurnEvent::ReplyFinished,
            TurnEventKind::SpeechDetected => TurnEvent::SpeechDetected,
            TurnEventKind::BargeInRejected => TurnEvent::BargeInRejected,
            TurnEventKind::BargeInConfirmed => TurnEvent::BargeInConfirmed,
            TurnEventKind::BargeIn => TurnEvent::BargeIn,
            TurnEventKind::ResumeListening => TurnEvent::ResumeListening,
            TurnEventKind::Close => TurnEvent::Close,
        }
    }

    /// Whether accepting this event opens a new user turn.
    pub(crate) const fn opens_turn(self) -> bool {
        matches!(self, TurnEventKind::FinalTranscript)
    }
}

/// An event the turn machine can consume.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TurnEvent {
    /// Session is ready to take input.
    Start,
    /// The user started producing speech.
    UserStartedSpeaking,
    /// The user stopped producing speech.
    UserStoppedSpeaking,
    /// Endpointing accepted a final candidate; the turn may still be rejected.
    CommitStarted,
    /// A final transcript was committed (either after VAD silence or manually).
    FinalTranscript { text: String },
    /// The endpointing window committed the turn.
    TurnCommitted,
    /// The endpointing window rejected the turn; nothing was committed.
    CommitRejected,
    /// The reply stream produced its first text without queuing audio yet.
    ReplyStreamOpened,
    /// The first agent audio frame was queued for playout.
    ReplyStarted,
    /// The last queued frame finished playing.
    ReplyFinished,
    /// Speech was detected while the agent held the floor.
    SpeechDetected,
    /// The resolution window decided the detection was a backchannel.
    BargeInRejected,
    /// The resolution window decided the detection was a barge-in.
    BargeInConfirmed,
    /// The user interrupted an in-flight reply.
    BargeIn,
    /// Cleanup after an interruption is complete.
    ResumeListening,
    /// The session is being torn down.
    Close,
}

impl TurnEvent {
    /// The payload-free kind of this event.
    #[must_use]
    pub fn kind(&self) -> TurnEventKind {
        match self {
            TurnEvent::Start => TurnEventKind::Start,
            TurnEvent::UserStartedSpeaking => TurnEventKind::UserStartedSpeaking,
            TurnEvent::UserStoppedSpeaking => TurnEventKind::UserStoppedSpeaking,
            TurnEvent::CommitStarted => TurnEventKind::CommitStarted,
            TurnEvent::FinalTranscript { .. } => TurnEventKind::FinalTranscript,
            TurnEvent::TurnCommitted => TurnEventKind::TurnCommitted,
            TurnEvent::CommitRejected => TurnEventKind::CommitRejected,
            TurnEvent::ReplyStreamOpened => TurnEventKind::ReplyStreamOpened,
            TurnEvent::ReplyStarted => TurnEventKind::ReplyStarted,
            TurnEvent::ReplyFinished => TurnEventKind::ReplyFinished,
            TurnEvent::SpeechDetected => TurnEventKind::SpeechDetected,
            TurnEvent::BargeInRejected => TurnEventKind::BargeInRejected,
            TurnEvent::BargeInConfirmed => TurnEventKind::BargeInConfirmed,
            TurnEvent::BargeIn => TurnEventKind::BargeIn,
            TurnEvent::ResumeListening => TurnEventKind::ResumeListening,
            TurnEvent::Close => TurnEventKind::Close,
        }
    }
}

/// Every state, in declaration order.
pub const ALL_STATES: &[TurnState] = &[
    TurnState::Idle,
    TurnState::Listening,
    TurnState::UserSpeaking,
    TurnState::Committing,
    TurnState::Thinking,
    TurnState::AgentGenerating,
    TurnState::Speaking,
    TurnState::EvaluatingBargeIn,
    TurnState::Interrupted,
    TurnState::Closed,
];

/// Every event kind, in declaration order.
pub const ALL_EVENT_KINDS: &[TurnEventKind] = &[
    TurnEventKind::Start,
    TurnEventKind::UserStartedSpeaking,
    TurnEventKind::UserStoppedSpeaking,
    TurnEventKind::CommitStarted,
    TurnEventKind::FinalTranscript,
    TurnEventKind::TurnCommitted,
    TurnEventKind::CommitRejected,
    TurnEventKind::ReplyStreamOpened,
    TurnEventKind::ReplyStarted,
    TurnEventKind::ReplyFinished,
    TurnEventKind::SpeechDetected,
    TurnEventKind::BargeInRejected,
    TurnEventKind::BargeInConfirmed,
    TurnEventKind::BargeIn,
    TurnEventKind::ResumeListening,
    TurnEventKind::Close,
];

/// The single source of truth for legal turn transitions.
///
/// Each row is `(from, event kind, to)`. [`TurnMachine::apply`] accepts an
/// event only when a row matches the machine's current state and the event's
/// kind; every other `(state, event)` pair is rejected.
///
/// [`TurnState::Closed`] is terminal: [`TurnMachine::apply`] rejects every
/// event there with [`ParlevoxError::Closed`] before consulting this table,
/// which is why no row has `Closed` as its `from` state. `Close` is legal from
/// every non-closed state.
pub const LEGAL_TRANSITIONS: &[(TurnState, TurnEventKind, TurnState)] = &[
    // Close is legal from every non-closed state.
    (TurnState::Idle, TurnEventKind::Close, TurnState::Closed),
    (
        TurnState::Listening,
        TurnEventKind::Close,
        TurnState::Closed,
    ),
    (
        TurnState::UserSpeaking,
        TurnEventKind::Close,
        TurnState::Closed,
    ),
    (
        TurnState::Committing,
        TurnEventKind::Close,
        TurnState::Closed,
    ),
    (TurnState::Thinking, TurnEventKind::Close, TurnState::Closed),
    (
        TurnState::AgentGenerating,
        TurnEventKind::Close,
        TurnState::Closed,
    ),
    (TurnState::Speaking, TurnEventKind::Close, TurnState::Closed),
    (
        TurnState::EvaluatingBargeIn,
        TurnEventKind::Close,
        TurnState::Closed,
    ),
    (
        TurnState::Interrupted,
        TurnEventKind::Close,
        TurnState::Closed,
    ),
    // Start a session.
    (TurnState::Idle, TurnEventKind::Start, TurnState::Listening),
    // User speech detection and its no-transcript ending.
    (
        TurnState::Listening,
        TurnEventKind::UserStartedSpeaking,
        TurnState::UserSpeaking,
    ),
    (
        TurnState::UserSpeaking,
        TurnEventKind::UserStoppedSpeaking,
        TurnState::Listening,
    ),
    // A final transcript can arrive from Listening (manual turn) or from
    // UserSpeaking (VAD end). Both open a new turn.
    (
        TurnState::Listening,
        TurnEventKind::FinalTranscript,
        TurnState::Thinking,
    ),
    (
        TurnState::UserSpeaking,
        TurnEventKind::FinalTranscript,
        TurnState::Thinking,
    ),
    // Additive endpointing path: a final candidate enters the commit window,
    // which either commits the turn or rejects it back to Listening.
    (
        TurnState::Listening,
        TurnEventKind::CommitStarted,
        TurnState::Committing,
    ),
    (
        TurnState::UserSpeaking,
        TurnEventKind::CommitStarted,
        TurnState::Committing,
    ),
    (
        TurnState::Committing,
        TurnEventKind::TurnCommitted,
        TurnState::Thinking,
    ),
    (
        TurnState::Committing,
        TurnEventKind::CommitRejected,
        TurnState::Listening,
    ),
    // Additive generation path: a reply stream can open before any audio frame
    // is queued, and can be interrupted before the agent speaks.
    (
        TurnState::Thinking,
        TurnEventKind::ReplyStreamOpened,
        TurnState::AgentGenerating,
    ),
    (
        TurnState::AgentGenerating,
        TurnEventKind::ReplyStarted,
        TurnState::Speaking,
    ),
    (
        TurnState::AgentGenerating,
        TurnEventKind::BargeIn,
        TurnState::Interrupted,
    ),
    (
        TurnState::AgentGenerating,
        TurnEventKind::UserStartedSpeaking,
        TurnState::Interrupted,
    ),
    // Additive barge-in resolution: detection while the agent speaks opens a
    // window that either restores Speaking (backchannel) or interrupts.
    (
        TurnState::Speaking,
        TurnEventKind::SpeechDetected,
        TurnState::EvaluatingBargeIn,
    ),
    (
        TurnState::EvaluatingBargeIn,
        TurnEventKind::BargeInRejected,
        TurnState::Speaking,
    ),
    (
        TurnState::EvaluatingBargeIn,
        TurnEventKind::BargeInConfirmed,
        TurnState::Interrupted,
    ),
    // A reply starts, can finish without ever emitting audio, or is interrupted.
    (
        TurnState::Thinking,
        TurnEventKind::ReplyStarted,
        TurnState::Speaking,
    ),
    (
        TurnState::Thinking,
        TurnEventKind::ReplyFinished,
        TurnState::Listening,
    ),
    (
        TurnState::Thinking,
        TurnEventKind::BargeIn,
        TurnState::Interrupted,
    ),
    (
        TurnState::Thinking,
        TurnEventKind::UserStartedSpeaking,
        TurnState::Interrupted,
    ),
    (
        TurnState::Speaking,
        TurnEventKind::BargeIn,
        TurnState::Interrupted,
    ),
    (
        TurnState::Speaking,
        TurnEventKind::UserStartedSpeaking,
        TurnState::Interrupted,
    ),
    (
        TurnState::Speaking,
        TurnEventKind::ReplyFinished,
        TurnState::Listening,
    ),
    (
        TurnState::Interrupted,
        TurnEventKind::ResumeListening,
        TurnState::Listening,
    ),
    // A reply that finishes while interrupted is already discarded.
    (
        TurnState::Interrupted,
        TurnEventKind::ReplyFinished,
        TurnState::Interrupted,
    ),
    // Duplicate/no-op events that must not disturb the state.
    (
        TurnState::Listening,
        TurnEventKind::UserStoppedSpeaking,
        TurnState::Listening,
    ),
    (
        TurnState::Listening,
        TurnEventKind::ReplyFinished,
        TurnState::Listening,
    ),
];

/// A committed state change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Transition {
    /// State the machine left.
    pub from: TurnState,
    /// State the machine entered.
    pub to: TurnState,
    /// Index of the user turn this transition belongs to. Increments when a new
    /// user turn is committed.
    pub turn: u64,
}

/// The turn state machine.
#[derive(Debug)]
pub struct TurnMachine {
    state: TurnState,
    turn: u64,
}

impl Default for TurnMachine {
    fn default() -> Self {
        Self::new()
    }
}

impl TurnMachine {
    /// Create a machine in [`TurnState::Idle`] with turn index `0`.
    #[must_use]
    pub fn new() -> Self {
        TurnMachine {
            state: TurnState::Idle,
            turn: 0,
        }
    }

    /// The current state.
    #[must_use]
    pub fn state(&self) -> TurnState {
        self.state
    }

    /// Index of the current user turn.
    #[must_use]
    pub fn turn_index(&self) -> u64 {
        self.turn
    }

    /// Apply an event. Returns the transition, or an error for an illegal one.
    ///
    /// The event is accepted only when [`LEGAL_TRANSITIONS`] contains a row for
    /// the current state and the event's kind. Only
    /// [`TurnEventKind::FinalTranscript`] increments the turn index.
    ///
    /// # Errors
    ///
    /// Returns [`ParlevoxError::Closed`] once the machine is terminal, before
    /// consulting [`LEGAL_TRANSITIONS`], and [`ParlevoxError::Protocol`] when
    /// the event is not legal from the current state. A rejected event leaves
    /// the state and turn index unchanged.
    pub fn apply(&mut self, event: TurnEvent) -> Result<Transition, ParlevoxError> {
        if self.state == TurnState::Closed {
            return Err(ParlevoxError::Closed);
        }

        let from = self.state;
        let kind = event.kind();
        let Some(&(_, _, to)) = LEGAL_TRANSITIONS
            .iter()
            .find(|(state, candidate, _)| *state == from && *candidate == kind)
        else {
            return Err(ParlevoxError::Protocol(format!(
                "illegal turn transition: {from:?} + {event:?}"
            )));
        };

        let mut turn = self.turn;
        if kind.opens_turn() {
            turn += 1;
        }

        self.state = to;
        self.turn = turn;
        Ok(Transition { from, to, turn })
    }

    /// End the current turn without a terminal event and return the machine to
    /// a truthful resting state.
    ///
    /// This is an owner-initiated operation, not something a plugin can report,
    /// so it is a method rather than a [`TurnEvent`] variant. It never changes
    /// the turn index: the turn that just ended keeps the id it was committed
    /// with, so the next committed turn increments it normally.
    ///
    /// `Idle` stays `Idle` (no turn has started), `Listening` stays
    /// `Listening`, and `UserSpeaking`, `Committing`, `Thinking`,
    /// `AgentGenerating`, `Speaking`, `EvaluatingBargeIn`, and `Interrupted`
    /// all return to `Listening`.
    ///
    /// # Errors
    ///
    /// Returns [`ParlevoxError::Closed`] once the machine is terminal; a closed
    /// machine is never reopened.
    pub fn abort(&mut self) -> Result<(), ParlevoxError> {
        use TurnState as S;

        match self.state {
            S::Idle | S::Listening => Ok(()),
            S::Closed => Err(ParlevoxError::Closed),
            S::UserSpeaking
            | S::Committing
            | S::Thinking
            | S::AgentGenerating
            | S::Speaking
            | S::EvaluatingBargeIn
            | S::Interrupted => {
                self.state = S::Listening;
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn happy_path_text_turn() {
        let mut m = TurnMachine::new();
        assert_eq!(m.apply(TurnEvent::Start).unwrap().to, TurnState::Listening);
        let t = m
            .apply(TurnEvent::FinalTranscript { text: "hi".into() })
            .unwrap();
        assert_eq!(t.to, TurnState::Thinking);
        assert_eq!(t.turn, 1);

        assert_eq!(
            m.apply(TurnEvent::ReplyStarted).unwrap().to,
            TurnState::Speaking
        );
        assert_eq!(
            m.apply(TurnEvent::ReplyFinished).unwrap().to,
            TurnState::Listening
        );
    }

    #[test]
    fn voice_turn_with_vad() {
        let mut m = TurnMachine::new();
        m.apply(TurnEvent::Start).unwrap();
        m.apply(TurnEvent::UserStartedSpeaking).unwrap();
        assert_eq!(m.state(), TurnState::UserSpeaking);
        m.apply(TurnEvent::UserStoppedSpeaking).unwrap();
        assert_eq!(m.state(), TurnState::Listening);
    }

    #[test]
    fn barge_in_interrupts_speaking() {
        let mut m = TurnMachine::new();
        m.apply(TurnEvent::Start).unwrap();
        m.apply(TurnEvent::FinalTranscript { text: "a".into() })
            .unwrap();
        m.apply(TurnEvent::ReplyStarted).unwrap();
        assert_eq!(
            m.apply(TurnEvent::BargeIn).unwrap().to,
            TurnState::Interrupted
        );
        assert_eq!(
            m.apply(TurnEvent::ResumeListening).unwrap().to,
            TurnState::Listening
        );
    }

    #[test]
    fn two_replies_in_flight_is_a_protocol_error() {
        let mut m = TurnMachine::new();
        m.apply(TurnEvent::Start).unwrap();
        m.apply(TurnEvent::FinalTranscript { text: "a".into() })
            .unwrap();
        m.apply(TurnEvent::ReplyStarted).unwrap();
        let err = m.apply(TurnEvent::ReplyStarted).unwrap_err();
        assert!(matches!(err, ParlevoxError::Protocol(_)));
    }

    #[test]
    fn close_is_legal_from_every_state_and_terminal() {
        let mut m = TurnMachine::new();
        m.apply(TurnEvent::Start).unwrap();
        assert_eq!(m.apply(TurnEvent::Close).unwrap().to, TurnState::Closed);
        assert!(matches!(
            m.apply(TurnEvent::Start).unwrap_err(),
            ParlevoxError::Closed
        ));
    }

    #[test]
    fn final_transcript_while_thinking_is_a_protocol_error() {
        let mut m = TurnMachine::new();
        m.apply(TurnEvent::Start).unwrap();
        m.apply(TurnEvent::FinalTranscript { text: "a".into() })
            .unwrap();
        let err = m
            .apply(TurnEvent::FinalTranscript { text: "a b".into() })
            .unwrap_err();
        assert!(matches!(err, ParlevoxError::Protocol(_)));
        // The rejected event left the machine and the turn index untouched.
        assert_eq!(m.state(), TurnState::Thinking);
        assert_eq!(m.turn_index(), 1);
    }

    #[test]
    fn reply_finished_without_audio_returns_to_listening() {
        let mut m = TurnMachine::new();
        m.apply(TurnEvent::Start).unwrap();
        m.apply(TurnEvent::FinalTranscript { text: "a".into() })
            .unwrap();
        let transition = m.apply(TurnEvent::ReplyFinished).unwrap();
        assert_eq!(transition.from, TurnState::Thinking);
        assert_eq!(transition.to, TurnState::Listening);
    }

    #[test]
    fn commit_started_reaches_committing_then_thinking() {
        let mut m = TurnMachine::new();
        m.apply(TurnEvent::Start).unwrap();
        assert_eq!(
            m.apply(TurnEvent::CommitStarted).unwrap().to,
            TurnState::Committing
        );
        let transition = m.apply(TurnEvent::TurnCommitted).unwrap();
        assert_eq!(transition.from, TurnState::Committing);
        assert_eq!(transition.to, TurnState::Thinking);
        // Only `FinalTranscript` opens a turn today; committing does not.
        assert_eq!(transition.turn, 0);
    }

    #[test]
    fn committing_rejection_returns_to_listening() {
        let mut m = TurnMachine::new();
        m.apply(TurnEvent::Start).unwrap();
        m.apply(TurnEvent::CommitStarted).unwrap();
        assert_eq!(
            m.apply(TurnEvent::CommitRejected).unwrap().to,
            TurnState::Listening
        );
    }

    #[test]
    fn agent_generating_barge_in_interrupts() {
        let mut m = TurnMachine::new();
        m.apply(TurnEvent::Start).unwrap();
        m.apply(TurnEvent::FinalTranscript { text: "a".into() })
            .unwrap();
        assert_eq!(
            m.apply(TurnEvent::ReplyStreamOpened).unwrap().to,
            TurnState::AgentGenerating
        );
        assert_eq!(
            m.apply(TurnEvent::BargeIn).unwrap().to,
            TurnState::Interrupted
        );
    }

    #[test]
    fn speech_detected_evaluates_then_rejects_barge_in() {
        let mut m = TurnMachine::new();
        m.apply(TurnEvent::Start).unwrap();
        m.apply(TurnEvent::FinalTranscript { text: "a".into() })
            .unwrap();
        m.apply(TurnEvent::ReplyStarted).unwrap();
        assert_eq!(
            m.apply(TurnEvent::SpeechDetected).unwrap().to,
            TurnState::EvaluatingBargeIn
        );
        assert_eq!(
            m.apply(TurnEvent::BargeInRejected).unwrap().to,
            TurnState::Speaking
        );
    }

    #[test]
    fn barge_in_confirmed_from_evaluation_interrupts() {
        let mut m = TurnMachine::new();
        m.apply(TurnEvent::Start).unwrap();
        m.apply(TurnEvent::FinalTranscript { text: "a".into() })
            .unwrap();
        m.apply(TurnEvent::ReplyStarted).unwrap();
        m.apply(TurnEvent::SpeechDetected).unwrap();
        assert_eq!(
            m.apply(TurnEvent::BargeInConfirmed).unwrap().to,
            TurnState::Interrupted
        );
    }

    #[test]
    fn new_states_abort_to_listening() {
        for state in [
            TurnState::Committing,
            TurnState::AgentGenerating,
            TurnState::EvaluatingBargeIn,
        ] {
            let mut machine = machine_in(state);
            let turn = machine.turn_index();
            machine.abort().unwrap();
            assert_eq!(
                machine.state(),
                TurnState::Listening,
                "abort from {state:?}"
            );
            assert_eq!(machine.turn_index(), turn, "abort keeps the turn index");
        }
    }

    #[test]
    fn abort_rests_open_states_in_listening_and_refuses_closed() {
        let mut idle = TurnMachine::new();
        idle.abort().unwrap();
        assert_eq!(idle.state(), TurnState::Idle);

        for state in [
            TurnState::Listening,
            TurnState::UserSpeaking,
            TurnState::Committing,
            TurnState::Thinking,
            TurnState::AgentGenerating,
            TurnState::Speaking,
            TurnState::EvaluatingBargeIn,
            TurnState::Interrupted,
        ] {
            let mut machine = machine_in(state);
            let turn = machine.turn_index();
            machine.abort().unwrap();
            assert_eq!(
                machine.state(),
                TurnState::Listening,
                "abort from {state:?}"
            );
            assert_eq!(
                machine.turn_index(),
                turn,
                "abort keeps the ended turn's index"
            );
        }

        let mut closed = machine_in(TurnState::Closed);
        assert!(matches!(closed.abort(), Err(ParlevoxError::Closed)));
        assert_eq!(closed.state(), TurnState::Closed);
    }

    /// Number of variants in [`TurnState`], used to size the exhaustive pair
    /// check. It is derived from [`ALL_STATES`], so a forgotten variant cannot
    /// silently shrink the sweep.
    const STATE_COUNT: usize = ALL_STATES.len();
    /// Number of variants in [`TurnEvent`], derived from [`ALL_EVENT_KINDS`].
    const EVENT_COUNT: usize = ALL_EVENT_KINDS.len();

    /// Every state plus one legal path that reaches it from `Idle`.
    ///
    /// Reaching arbitrary states only through accepted transitions is what
    /// makes the totality and reachability properties meaningful: a state is
    /// never fabricated by poking a private field.
    fn legal_paths() -> Vec<(TurnState, Vec<TurnEvent>)> {
        use TurnState as S;
        vec![
            (S::Idle, vec![]),
            (S::Listening, vec![TurnEvent::Start]),
            (
                S::UserSpeaking,
                vec![TurnEvent::Start, TurnEvent::UserStartedSpeaking],
            ),
            (
                S::Committing,
                vec![TurnEvent::Start, TurnEvent::CommitStarted],
            ),
            (
                S::Thinking,
                vec![
                    TurnEvent::Start,
                    TurnEvent::FinalTranscript { text: "a".into() },
                ],
            ),
            (
                S::AgentGenerating,
                vec![
                    TurnEvent::Start,
                    TurnEvent::FinalTranscript { text: "a".into() },
                    TurnEvent::ReplyStreamOpened,
                ],
            ),
            (
                S::Speaking,
                vec![
                    TurnEvent::Start,
                    TurnEvent::FinalTranscript { text: "a".into() },
                    TurnEvent::ReplyStarted,
                ],
            ),
            (
                S::EvaluatingBargeIn,
                vec![
                    TurnEvent::Start,
                    TurnEvent::FinalTranscript { text: "a".into() },
                    TurnEvent::ReplyStarted,
                    TurnEvent::SpeechDetected,
                ],
            ),
            (
                S::Interrupted,
                vec![
                    TurnEvent::Start,
                    TurnEvent::FinalTranscript { text: "a".into() },
                    TurnEvent::BargeIn,
                ],
            ),
            (S::Closed, vec![TurnEvent::Start, TurnEvent::Close]),
        ]
    }

    /// Every distinct state, derived from [`ALL_STATES`].
    fn all_states() -> Vec<TurnState> {
        ALL_STATES.to_vec()
    }

    /// One instance of every event kind, derived from [`ALL_EVENT_KINDS`].
    fn all_events() -> Vec<TurnEvent> {
        ALL_EVENT_KINDS
            .iter()
            .map(|kind| kind.sample_event())
            .collect()
    }

    /// Replay the legal path to `state` on a fresh machine.
    fn machine_in(state: TurnState) -> TurnMachine {
        let (_, path) = legal_paths()
            .into_iter()
            .find(|(target, _)| *target == state)
            .unwrap_or_else(|| panic!("no legal path recorded for {state:?}"));
        let mut machine = TurnMachine::new();
        for event in path {
            machine
                .apply(event)
                .expect("a recorded legal path must be accepted");
        }
        debug_assert_eq!(machine.state(), state);
        machine
    }

    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn prop_totality_over_every_state_event_pair(
            state_index in 0usize..STATE_COUNT,
            event_index in 0usize..EVENT_COUNT,
        ) {
            let state = all_states()[state_index];
            let event = all_events()[event_index].clone();
            let mut machine = machine_in(state);

            match machine.apply(event) {
                Ok(_) => {}
                Err(ParlevoxError::Protocol(_)) | Err(ParlevoxError::Closed) => {}
                Err(other) => prop_assert!(
                    false,
                    "unexpected error class from {state:?}: {other:?}"
                ),
            }
        }

        #[test]
        fn prop_every_state_reachable_from_idle(rotation in 0usize..STATE_COUNT) {
            let mut paths = legal_paths();
            prop_assert_eq!(paths.len(), STATE_COUNT);
            // Rotating the traversal keeps the property sensitive to ordering
            // without ever skipping a state.
            paths.rotate_left(rotation);

            for (target, path) in paths {
                let mut machine = TurnMachine::new();
                for event in path {
                    machine
                        .apply(event)
                        .expect("a recorded legal path must be accepted");
                }
                prop_assert_eq!(machine.state(), target);
            }
        }

        #[test]
        fn prop_turn_monotonic_over_event_sequences(
            indices in prop::collection::vec(0usize..EVENT_COUNT, 0..64),
        ) {
            let mut machine = TurnMachine::new();
            for index in indices {
                let before = machine.turn_index();
                let opens_turn =
                    matches!(machine.state(), TurnState::Listening | TurnState::UserSpeaking)
                        && matches!(
                            all_events()[index],
                            TurnEvent::FinalTranscript { .. }
                        );
                let event = all_events()[index].clone();

                match machine.apply(event) {
                    Ok(transition) => {
                        prop_assert!(transition.turn >= before);
                        let expected = if opens_turn { before + 1 } else { before };
                        prop_assert_eq!(transition.turn, expected);
                    }
                    Err(_) => prop_assert_eq!(machine.turn_index(), before),
                }
            }
        }
    }

    #[test]
    fn every_state_event_pair_is_total_and_counted() {
        let states = all_states();
        let events = all_events();
        assert_eq!(states.len(), STATE_COUNT);
        assert_eq!(events.len(), EVENT_COUNT);

        let mut checked = 0u64;
        for state in &states {
            for event in &events {
                let mut machine = machine_in(*state);
                let outcome = machine.apply(event.clone());
                match &outcome {
                    Ok(_) | Err(ParlevoxError::Protocol(_)) | Err(ParlevoxError::Closed) => {}
                    other => {
                        panic!("{state:?} + {event:?} produced an unexpected outcome: {other:?}")
                    }
                }
                checked += 1;
            }
        }

        let expected = u64::try_from(states.len() * events.len()).expect("pair count fits in u64");
        assert_eq!(checked, expected);
        println!("exhaustive turn-table coverage: {checked} (state, event) pairs");
    }
}
