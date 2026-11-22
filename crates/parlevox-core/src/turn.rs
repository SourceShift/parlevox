//! Explicit turn state machine.
//!
//! Every framework compared in `docs/architecture/COMPARISON.md` encodes turn
//! invariants somewhere: in booleans, in task cancellation, or in prose. Here
//! they are a closed set of states and a table of legal transitions. An illegal
//! event returns [`crate::error::ParlevoxError::Protocol`] instead of being
//! silently absorbed, so a misbehaving plugin fails loudly at the boundary.

use crate::error::ParlevoxError;
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
    FinalTranscript {
        text: String,
    },
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
        TurnMachine {
            state: TurnState::Idle,
            turn: 0,
        }
    }

    pub fn state(&self) -> TurnState {
        self.state
    }

    pub fn turn_index(&self) -> u64 {
        self.turn
    }

    /// Apply an event. Returns the transition, or an error for an illegal one.
    /// `Close` is legal from every non-closed state.
    ///
    /// # Errors
    ///
    /// Returns a protocol-class error when the event is not legal from the
    /// current state, and a closed-class error once the machine is terminal.
    pub fn apply(&mut self, event: TurnEvent) -> Result<Transition, ParlevoxError> {
        use TurnState as S;

        if self.state == S::Closed {
            return Err(ParlevoxError::Closed);
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

            // A reply can finish without ever emitting audio (a text-only turn
            // whose TTS produced no chunks). The machine reports that honestly
            // by going straight back to listening and never entering
            // `Speaking`.
            (S::Thinking, TurnEvent::ReplyFinished) => S::Listening,

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
                return Err(ParlevoxError::Protocol(format!(
                    "illegal turn transition: {state:?} + {event:?}"
                )))
            }
        };

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
    /// `Listening`, and `UserSpeaking`, `Thinking`, `Speaking`, and
    /// `Interrupted` all return to `Listening`.
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
            S::UserSpeaking | S::Thinking | S::Speaking | S::Interrupted => {
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
    fn abort_rests_open_states_in_listening_and_refuses_closed() {
        let mut idle = TurnMachine::new();
        idle.abort().unwrap();
        assert_eq!(idle.state(), TurnState::Idle);

        for state in [
            TurnState::Listening,
            TurnState::UserSpeaking,
            TurnState::Thinking,
            TurnState::Speaking,
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
    /// check. A mismatch with [`all_states`] makes the pair test fail loudly.
    const STATE_COUNT: usize = 7;
    /// Number of variants in [`TurnEvent`], used the same way.
    const EVENT_COUNT: usize = 9;

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
                S::Thinking,
                vec![
                    TurnEvent::Start,
                    TurnEvent::FinalTranscript { text: "a".into() },
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

    /// Every distinct state.
    fn all_states() -> Vec<TurnState> {
        use TurnState as S;
        vec![
            S::Idle,
            S::Listening,
            S::UserSpeaking,
            S::Thinking,
            S::Speaking,
            S::Interrupted,
            S::Closed,
        ]
    }

    /// One instance of every event variant.
    fn all_events() -> Vec<TurnEvent> {
        vec![
            TurnEvent::Start,
            TurnEvent::UserStartedSpeaking,
            TurnEvent::UserStoppedSpeaking,
            TurnEvent::FinalTranscript { text: "x".into() },
            TurnEvent::ReplyStarted,
            TurnEvent::ReplyFinished,
            TurnEvent::BargeIn,
            TurnEvent::ResumeListening,
            TurnEvent::Close,
        ]
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
