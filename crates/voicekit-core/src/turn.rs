//! Explicit turn state machine.
//!
//! Every framework compared in `docs/architecture/COMPARISON.md` encodes turn
//! invariants somewhere: in booleans, in task cancellation, or in prose. Here
//! they are a closed set of states and a table of legal transitions. An illegal
//! event returns [`crate::error::VoicekitError::Protocol`] instead of being
//! silently absorbed, so a misbehaving plugin fails loudly at the boundary.

use crate::error::VoicekitError;
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnState {
    /// Session created, no user activity yet.
    Idle,
    /// Waiting for the user.
    Listening,
    /// User is producing speech right now.
    UserSpeaking,
    /// A final transcript exists; the reply has not started speaking.
    Thinking,
    /// The agent's audio is playing.
    Speaking,
    /// Barge-in happened; the reply is being torn down.
    Interrupted,
    /// Terminal.
    Closed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TurnEvent {
    /// Session is ready to take input.
    Start,
    UserStartedSpeaking,
    UserStoppedSpeaking,
    /// A final transcript was committed (either after VAD silence or manually).
    FinalTranscript { text: String },
    /// The first agent audio frame was queued for playout.
    ReplyStarted,
    /// The last queued frame finished playing.
    ReplyFinished,
    /// The user interrupted an in-flight reply.
    BargeIn,
    /// Cleanup after an interruption is complete.
    ResumeListening,
    Close,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Transition {
    pub from: TurnState,
    pub to: TurnState,
    /// Index of the user turn this transition belongs to. Increments when a new
    /// user turn is committed.
    pub turn: u64,
}

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
    pub fn new() -> Self {
        TurnMachine { state: TurnState::Idle, turn: 0 }
    }

    pub fn state(&self) -> TurnState {
        self.state
    }

    pub fn turn_index(&self) -> u64 {
        self.turn
    }

    /// Apply an event. Returns the transition, or an error for an illegal one.
    /// `Close` is legal from every non-closed state.
    pub fn apply(&mut self, event: TurnEvent) -> Result<Transition, VoicekitError> {
        use TurnState as S;

        if self.state == S::Closed {
            return Err(VoicekitError::Closed);
        }

        let from = self.state;
        let mut turn = self.turn;

        let to = match (&self.state, event) {
            (_, TurnEvent::Close) => S::Closed,

            (S::Idle, TurnEvent::Start) => S::Listening,

            (S::Listening, TurnEvent::UserStartedSpeaking) => S::UserSpeaking,

            // A final transcript can arrive from Listening (manual turn) or
            // from UserSpeaking (VAD end). Both open a new turn.
            (S::Listening | S::UserSpeaking, TurnEvent::FinalTranscript { .. }) => {
                turn += 1;
                S::Thinking
            }

            // Speech ends without a transcript → back to listening.
            (S::UserSpeaking, TurnEvent::UserStoppedSpeaking) => S::Listening,

            (S::Thinking, TurnEvent::ReplyStarted) => S::Speaking,

            // The user may retract/correct while thinking.
            (S::Thinking, TurnEvent::FinalTranscript { .. }) => {
                // Replace the pending turn; do not increment the index.
                S::Thinking
            }

            // Barge-in is legal while the agent is thinking or speaking.
            (S::Thinking | S::Speaking, TurnEvent::BargeIn) => S::Interrupted,
            (S::Thinking | S::Speaking, TurnEvent::UserStartedSpeaking) => S::Interrupted,

            (S::Speaking, TurnEvent::ReplyFinished) => S::Listening,

            (S::Interrupted, TurnEvent::ResumeListening) => S::Listening,
            // A reply that finishes while interrupted is already discarded.
            (S::Interrupted, TurnEvent::ReplyFinished) => S::Interrupted,

            // Duplicate/no-op events that must not disturb the state.
            (S::Listening, TurnEvent::UserStoppedSpeaking) => S::Listening,
            (S::Listening, TurnEvent::ReplyFinished) => S::Listening,

            (state, event) => {
                return Err(VoicekitError::Protocol(format!(
                    "illegal turn transition: {state:?} + {event:?}"
                )))
            }
        };

        self.state = to;
        self.turn = turn;
        Ok(Transition { from, to, turn })
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

        assert_eq!(m.apply(TurnEvent::ReplyStarted).unwrap().to, TurnState::Speaking);
        assert_eq!(m.apply(TurnEvent::ReplyFinished).unwrap().to, TurnState::Listening);
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
        m.apply(TurnEvent::FinalTranscript { text: "a".into() }).unwrap();
        m.apply(TurnEvent::ReplyStarted).unwrap();
        assert_eq!(m.apply(TurnEvent::BargeIn).unwrap().to, TurnState::Interrupted);
        assert_eq!(
            m.apply(TurnEvent::ResumeListening).unwrap().to,
            TurnState::Listening
        );
    }

    #[test]
    fn two_replies_in_flight_is_a_protocol_error() {
        let mut m = TurnMachine::new();
        m.apply(TurnEvent::Start).unwrap();
        m.apply(TurnEvent::FinalTranscript { text: "a".into() }).unwrap();
        m.apply(TurnEvent::ReplyStarted).unwrap();
        let err = m.apply(TurnEvent::ReplyStarted).unwrap_err();
        assert!(matches!(err, VoicekitError::Protocol(_)));
    }

    #[test]
    fn close_is_legal_from_every_state_and_terminal() {
        let mut m = TurnMachine::new();
        m.apply(TurnEvent::Start).unwrap();
        assert_eq!(m.apply(TurnEvent::Close).unwrap().to, TurnState::Closed);
        assert!(matches!(
            m.apply(TurnEvent::Start).unwrap_err(),
            VoicekitError::Closed
        ));
    }

    #[test]
    fn correction_while_thinking_keeps_one_turn() {
        let mut m = TurnMachine::new();
        m.apply(TurnEvent::Start).unwrap();
        m.apply(TurnEvent::FinalTranscript { text: "a".into() }).unwrap();
        let t = m
            .apply(TurnEvent::FinalTranscript { text: "a b".into() })
            .unwrap();
        assert_eq!(t.to, TurnState::Thinking);
        assert_eq!(t.turn, 1);
    }
}
