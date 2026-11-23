# Turn contract

Status: hand-written contract, mechanically checked.

This document states the turn state machine that `parlevox-core` implements. It
is **hand-written**, not generated: the Rust transition table in
`crates/parlevox-core/src/turn.rs` (`LEGAL_TRANSITIONS`) is the single source of
truth, and the integration test `crates/parlevox-core/tests/turn_contract.rs`
parses this file and fails when the states, the legal transitions, the error
classes, the turn-index rule, or the floor-ownership table disagree with the
code. Edit both together, or the test will refuse the change.

Canonical state and event names are the Rust variant names shown in `code`.

## States

| State | Meaning |
|---|---|
| `Idle` | Session created, no user activity yet. |
| `Listening` | Waiting for the user. |
| `UserSpeaking` | The user is producing speech right now. |
| `Committing` | The endpointing window: a final candidate exists but the turn is not committed yet, so speech may still cancel it. |
| `Thinking` | A final transcript exists; the reply has not started streaming. |
| `AgentGenerating` | An LLM stream is producing text but has not queued audio yet. |
| `Speaking` | The agent's audio is playing. |
| `EvaluatingBargeIn` | Speech was detected while the agent spoke; the resolution window is open. |
| `Interrupted` | A barge-in happened; the reply is being torn down. |
| `Closed` | Terminal; the session can never leave this state. |

`Committing`, `AgentGenerating`, and `EvaluatingBargeIn` are additive states
from `docs/architecture/TURN-SEMANTICS-REQUIREMENTS.md` §3. Nothing emits the
events that reach them yet; they are part of the contract so the arbiter slice
can wire them without changing the state model.

## Legal transitions

`Close` is legal from every non-closed state. `FinalTranscript` is the only
event that opens a new turn; every other accepted event leaves the turn index
unchanged. `Closed` has no outgoing row because the machine rejects every event
there before consulting the table.

| From | Event | To |
|---|---|---|
| `Idle` | `Close` | `Closed` |
| `Listening` | `Close` | `Closed` |
| `UserSpeaking` | `Close` | `Closed` |
| `Committing` | `Close` | `Closed` |
| `Thinking` | `Close` | `Closed` |
| `AgentGenerating` | `Close` | `Closed` |
| `Speaking` | `Close` | `Closed` |
| `EvaluatingBargeIn` | `Close` | `Closed` |
| `Interrupted` | `Close` | `Closed` |
| `Idle` | `Start` | `Listening` |
| `Listening` | `UserStartedSpeaking` | `UserSpeaking` |
| `UserSpeaking` | `UserStoppedSpeaking` | `Listening` |
| `Listening` | `FinalTranscript` | `Thinking` |
| `UserSpeaking` | `FinalTranscript` | `Thinking` |
| `Listening` | `CommitStarted` | `Committing` |
| `UserSpeaking` | `CommitStarted` | `Committing` |
| `Committing` | `TurnCommitted` | `Thinking` |
| `Committing` | `CommitRejected` | `Listening` |
| `Thinking` | `ReplyStreamOpened` | `AgentGenerating` |
| `AgentGenerating` | `ReplyStarted` | `Speaking` |
| `AgentGenerating` | `BargeIn` | `Interrupted` |
| `AgentGenerating` | `UserStartedSpeaking` | `Interrupted` |
| `Speaking` | `SpeechDetected` | `EvaluatingBargeIn` |
| `EvaluatingBargeIn` | `BargeInRejected` | `Speaking` |
| `EvaluatingBargeIn` | `BargeInConfirmed` | `Interrupted` |
| `Thinking` | `ReplyStarted` | `Speaking` |
| `Thinking` | `ReplyFinished` | `Listening` |
| `Thinking` | `BargeIn` | `Interrupted` |
| `Thinking` | `UserStartedSpeaking` | `Interrupted` |
| `Speaking` | `BargeIn` | `Interrupted` |
| `Speaking` | `UserStartedSpeaking` | `Interrupted` |
| `Speaking` | `ReplyFinished` | `Listening` |
| `Interrupted` | `ResumeListening` | `Listening` |
| `Interrupted` | `ReplyFinished` | `Interrupted` |
| `Listening` | `UserStoppedSpeaking` | `Listening` |
| `Listening` | `ReplyFinished` | `Listening` |

Note: the `May accept user speech` cell for `Committing` states the target
contract from `TURN-SEMANTICS-REQUIREMENTS.md` §3 ("yes (cancels commit)"), not
a row in this table. This additive slice deliberately does not add
`Committing` + `UserStartedSpeaking`, so that pair currently returns
`ParlevoxError::Protocol`; wiring speech-cancel in `Committing` belongs to the
arbiter slice.

## Illegal transitions

Two error classes cover every pair that is not listed above.

- **Closed state.** Once the machine is in `Closed`, every event returns
  `ParlevoxError::Closed` (message `session closed`) before the table is
  consulted. This covers all sixteen events from `Closed`, including `Close`
  itself.
- **Unlisted pair from a non-closed state.** Any event that has no row in the
  legal table returns `ParlevoxError::Protocol` with the message
  `illegal turn transition: <From> + <Event>`, where `<From>` and `<Event>` are
  the debug renderings of the state and the event (including a transcript
  payload when present).

A rejected event is atomic: the state and the turn index are exactly what they
were before the call. An accepted event increments the turn index by one **only**
for `FinalTranscript`; every other accepted event leaves it unchanged.

## Floor ownership

The audio floor is a single-owner contract. `Owns floor` is the
`parlevox_core::FloorOwner` label reported by `TurnState::owns_floor`, and
`May emit audio` is the `parlevox_core::AudioEmission` label reported by
`TurnState::audio_emission`; both are checked by the contract test. The nine
non-terminal rows are quoted from
`docs/architecture/TURN-SEMANTICS-REQUIREMENTS.md` §3 ("Target state graph"),
mapped onto the states that exist today. `Closed` is a terminal-state
interpretation: the requirements table does not list it, so it is assigned
`nobody` / no emit / no accept explicitly here.

| State | Owns floor | May emit audio | May accept user speech |
|---|---|---|---|
| `Idle` | user | no | yes (starts a turn) |
| `Listening` | user | no | yes |
| `UserSpeaking` | user | no | yes |
| `Committing` | nobody | no | yes (cancels commit) |
| `Thinking` | agent-unconfirmed | no | yes (cancels generation) |
| `AgentGenerating` | agent-unconfirmed | no | yes (cancels generation) |
| `Speaking` | agent-confirmed | yes | escalated to barge-in evaluation |
| `EvaluatingBargeIn` | contested | ducked | yes (resolution window) |
| `Interrupted` | nobody | drain only | yes (next turn) |
| `Closed` | nobody | no | no |
