//! Pure, synchronous turn arbiter.
//!
//! [`TurnArbiter::step`] is the whole decision layer of a session reduced to one
//! function: it takes a borrowed [`ArbiterModel`] plus one [`Observation`] and
//! returns the next model together with the ordered list of [`Command`]s the
//! runtime should carry out. It does no I/O, takes no lock, reads no clock,
//! allocates only the command `Vec`, and cannot panic on any input.
//!
//! Keeping the decision here rather than in the runtime is what makes a session
//! reproducible: the same model and the same observation always produce the same
//! commands, so a trace can be replayed and a bug reproduced from the trace
//! alone.
//!
//! # Determinism
//!
//! Commands are produced by straight-line code in the fixed order documented on
//! [`TurnArbiter::step`]. No `HashMap` or `HashSet` participates in the decision,
//! so there is no iteration order to leak into the output.
//!
//! # What the arbiter does *not* do
//!
//! It does not speak the wire protocol. [`Command::EmitTurnEvent`] carries a
//! [`TurnEventKind`]; mapping that onto the runtime's `SessionEvent` belongs to
//! the runtime slice, which is deliberately not wired to this module yet.

use serde::Serialize;

use crate::observation::{Observation, ObservationKind};
use crate::turn::{TurnEventKind, TurnState, LEGAL_TRANSITIONS};

/// Attenuation applied while a barge-in is being evaluated, in the same
/// `0..=100` scale the audio path uses for gain.
///
/// `20` is a clear duck: the agent stays audible but the user's speech dominates
/// the mix, which keeps the floor legible during the resolution window.
pub const DUCK_GAIN: u8 = 20;

/// One action the runtime should take after a step.
///
/// The variants are generated only by [`TurnArbiter::step`]. They are an
/// exhaustive enum on purpose: a new command must be handled by every consumer
/// rather than silently ignored by a wildcard arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Command {
    /// Begin generation `generation` for user turn `turn`.
    StartGeneration {
        /// User turn the generation belongs to.
        turn: u64,
        /// Reply generation that is starting.
        generation: u64,
    },
    /// Cancel the superseded reply `generation`.
    CancelGeneration {
        /// Generation being torn down.
        generation: u64,
    },
    /// Start synthesising the reply for user turn `turn`.
    StartSynthesis {
        /// User turn being synthesised.
        turn: u64,
        /// Reply generation being synthesised.
        generation: u64,
    },
    /// Drop audio already queued for user turn `turn`.
    FlushAudioBuffer {
        /// User turn whose queued audio is discarded.
        turn: u64,
    },
    /// Attenuate agent audio to `gain` (`0..=100`, see [`DUCK_GAIN`]).
    DuckAudio {
        /// Target gain on the `0..=100` scale.
        gain: u8,
    },
    /// Restore agent audio to full level.
    RestoreAudio,
    /// Emit a turn event of kind `kind`.
    EmitTurnEvent {
        /// Event kind to emit.
        kind: TurnEventKind,
    },
    /// Bump one of the model's counters by way of a recorded trace entry.
    Record {
        /// Which counter the record accounts for.
        kind: RecordKind,
    },
}

/// What a [`Command::Record`] reports, and therefore which [`ArbiterModel`]
/// counter it is paired with.
///
/// Every accepted, ignored and rejected observation is accounted for exactly
/// once, so `accepted + ignored + rejected` always equals the number of
/// observations fed to the arbiter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordKind {
    /// The observation was not actionable; reports the `ignored` counter.
    ObservationIgnored,
    /// The observation was actionable but illegal in the current state;
    /// reports the `rejected` counter.
    IllegalTransition,
    /// The observation arrived after the session closed; reports the
    /// `rejected` counter.
    ObservationAfterClose,
    /// A new user turn was opened; reports the `accepted` counter.
    TurnOpened,
    /// An in-flight reply was superseded by a barge-in; reports the `accepted`
    /// counter and precedes the generation increment.
    GenerationSuperseded,
}

/// The complete decision state of a session.
///
/// This is a value type: [`TurnArbiter::step`] clones it, advances the clone and
/// returns it, leaving the caller's model untouched. Every field is a public
/// documented part of the model so a trace can be inspected and a test can start
/// from an arbitrary state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArbiterModel {
    /// Current turn state. Written only by [`TurnArbiter::step`].
    pub state: TurnState,
    /// Index of the current user turn; increments when a turn opens. Written
    /// only by [`TurnArbiter::step`].
    pub turn: u64,
    /// Reply generation; increments when a turn opens and again when a reply is
    /// superseded. Written only by [`TurnArbiter::step`].
    pub generation: u64,
    /// Whether agent audio is currently ducked. Written only by
    /// [`TurnArbiter::step`].
    pub ducked: bool,
    /// Observations accepted into a state transition. Written only by
    /// [`TurnArbiter::step`].
    pub accepted: u64,
    /// Observations ignored because they carry no actionable event. Written
    /// only by [`TurnArbiter::step`].
    pub ignored: u64,
    /// Observations rejected as illegal or as arriving after close. Written
    /// only by [`TurnArbiter::step`].
    pub rejected: u64,
}

impl Default for ArbiterModel {
    fn default() -> Self {
        ArbiterModel {
            state: TurnState::Idle,
            turn: 0,
            generation: 0,
            ducked: false,
            accepted: 0,
            ignored: 0,
            rejected: 0,
        }
    }
}

/// Namespace for the pure step function.
///
/// The type has no methods that mutate and no fields; it exists so the entry
/// point reads as [`TurnArbiter::step`] rather than a free function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TurnArbiter;

impl TurnArbiter {
    /// Advance `model` by one observation and return the next model plus the
    /// commands to carry out.
    ///
    /// # Decision rules
    ///
    /// 1. If `model.state` is [`TurnState::Closed`], nothing moves: the model's
    ///    `rejected` counter increments and the step yields
    ///    [`RecordKind::ObservationAfterClose`].
    /// 2. An observation that carries no actionable event (a partial
    ///    transcript, a reply delta, a queued audio frame, a cancellation
    ///    notice, or a failure) increments `ignored` and yields
    ///    [`RecordKind::ObservationIgnored`].
    /// 3. An actionable event with no row in [`LEGAL_TRANSITIONS`] for the
    ///    current state increments `rejected` and yields
    ///    [`RecordKind::IllegalTransition`]; the state, turn, generation and
    ///    duck flag are unchanged.
    /// 4. An accepted event is applied exactly as `TurnMachine::apply` would
    ///    apply it: [`TurnEventKind::FinalTranscript`] opens a turn, and every
    ///    other accepted event only changes the state.
    ///
    /// # Command order
    ///
    /// Commands for one step are produced in this fixed order, omitting the
    /// groups that do not apply:
    ///
    /// 1. generation lifecycle: [`Command::CancelGeneration`] (the superseded
    ///    generation) then [`Command::StartGeneration`] (the new turn and
    ///    generation);
    /// 2. [`Command::StartSynthesis`];
    /// 3. audio: [`Command::FlushAudioBuffer`], [`Command::DuckAudio`],
    ///    [`Command::RestoreAudio`];
    /// 4. [`Command::EmitTurnEvent`];
    /// 5. [`Command::Record`].
    ///
    /// # Effects
    ///
    /// - Opening a turn yields `StartGeneration`, `EmitTurnEvent` and
    ///   `Record(TurnOpened)`.
    /// - First audio (`ReplyStarted`) yields `StartSynthesis` and
    ///   `EmitTurnEvent`.
    /// - Speech detected while speaking yields `DuckAudio` and
    ///   `EmitTurnEvent`, and marks the model ducked.
    /// - Leaving the contested window ([`TurnState::EvaluatingBargeIn`]) for any
    ///   other state restores the ducked audio with `RestoreAudio`.
    /// - Entering [`TurnState::Interrupted`] from any other state yields
    ///   `CancelGeneration`, `FlushAudioBuffer`, `RestoreAudio` when the model
    ///   was ducked, `EmitTurnEvent` and `Record(GenerationSuperseded)`, then
    ///   increments the generation.
    /// - Every other accepted event yields only `EmitTurnEvent`.
    ///
    /// The input is never mutated: the returned model is a distinct value, so a
    /// caller may reuse the old model for a retry or a trace.
    #[must_use]
    pub fn step(model: &ArbiterModel, observation: Observation) -> (ArbiterModel, Vec<Command>) {
        if model.state == TurnState::Closed {
            let mut next = *model;
            next.rejected = next.rejected.saturating_add(1);
            return (
                next,
                vec![Command::Record {
                    kind: RecordKind::ObservationAfterClose,
                }],
            );
        }

        let Some(kind) = actionable_kind(&observation.kind) else {
            let mut next = *model;
            next.ignored = next.ignored.saturating_add(1);
            return (
                next,
                vec![Command::Record {
                    kind: RecordKind::ObservationIgnored,
                }],
            );
        };

        let Some(&(from, _, to)) = LEGAL_TRANSITIONS
            .iter()
            .find(|(state, event, _)| *state == model.state && *event == kind)
        else {
            let mut next = *model;
            next.rejected = next.rejected.saturating_add(1);
            return (
                next,
                vec![Command::Record {
                    kind: RecordKind::IllegalTransition,
                }],
            );
        };

        let mut next = *model;
        next.state = to;
        next.accepted = next.accepted.saturating_add(1);

        let opens_turn = kind.opens_turn();
        let interrupts = to == TurnState::Interrupted && from != TurnState::Interrupted;

        let mut commands = Vec::with_capacity(6);

        // 1. Generation lifecycle: cancel the superseded generation first, then
        //    start the new one. The two branches are mutually exclusive today,
        //    but the order is fixed regardless.
        if interrupts {
            commands.push(Command::CancelGeneration {
                generation: model.generation,
            });
        }
        if opens_turn {
            next.turn = next.turn.saturating_add(1);
            next.generation = next.generation.saturating_add(1);
            commands.push(Command::StartGeneration {
                turn: next.turn,
                generation: next.generation,
            });
        }

        // 2. Synthesis begins with the first queued audio frame.
        if kind == TurnEventKind::ReplyStarted {
            commands.push(Command::StartSynthesis {
                turn: next.turn,
                generation: next.generation,
            });
        }

        // 3. Audio effects: flush what is queued, duck on detection, then
        //    restore. Restoration is tied to leaving the contested window, not
        //    to one event, so every exit from `EvaluatingBargeIn` leaves the
        //    model in a truthful, un-ducked state.
        if interrupts {
            commands.push(Command::FlushAudioBuffer { turn: next.turn });
        }
        if kind == TurnEventKind::SpeechDetected {
            next.ducked = true;
            commands.push(Command::DuckAudio { gain: DUCK_GAIN });
        }
        if next.ducked && to != TurnState::EvaluatingBargeIn {
            next.ducked = false;
            commands.push(Command::RestoreAudio);
        }

        // 4. The turn event mirrors the accepted event.
        commands.push(Command::EmitTurnEvent { kind });

        // 5. Accounting. The superseded generation is bumped *after* the record
        //    so the record reports the generation that was cancelled.
        if interrupts {
            next.generation = next.generation.saturating_add(1);
            commands.push(Command::Record {
                kind: RecordKind::GenerationSuperseded,
            });
        } else if opens_turn {
            commands.push(Command::Record {
                kind: RecordKind::TurnOpened,
            });
        }

        (next, commands)
    }
}

/// Map an observation payload onto the turn event it delivers, if any.
///
/// Only observations that correspond to a committed or observable state change
/// are actionable. Partial transcripts, reply deltas, queued audio frames,
/// cancellation notices and failures are traced but do not move the machine.
fn actionable_kind(kind: &ObservationKind) -> Option<TurnEventKind> {
    match kind {
        ObservationKind::FinalTranscript { .. } => Some(TurnEventKind::FinalTranscript),
        ObservationKind::PartialTranscript { .. } => None,
        ObservationKind::VadStarted => Some(TurnEventKind::UserStartedSpeaking),
        ObservationKind::VadStopped => Some(TurnEventKind::UserStoppedSpeaking),
        ObservationKind::ReplyDelta { .. } => None,
        ObservationKind::AudioFrameQueued { .. } => None,
        ObservationKind::ReplyFinished => Some(TurnEventKind::ReplyFinished),
        ObservationKind::BargeInReported => Some(TurnEventKind::BargeIn),
        ObservationKind::CancellationObserved => None,
        ObservationKind::Failure { .. } => None,
        ObservationKind::Control { event } => Some(*event),
    }
}
