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
| `Thinking` | A final transcript exists; the reply has not started speaking. |
| `Speaking` | The agent's audio is playing. |
| `Interrupted` | A barge-in happened; the reply is being torn down. |
| `Closed` | Terminal; the session can never leave this state. |

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
| `Thinking` | `Close` | `Closed` |
| `Speaking` | `Close` | `Closed` |
| `Interrupted` | `Close` | `Closed` |
| `Idle` | `Start` | `Listening` |
| `Listening` | `UserStartedSpeaking` | `UserSpeaking` |
| `UserSpeaking` | `UserStoppedSpeaking` | `Listening` |
| `Listening` | `FinalTranscript` | `Thinking` |
| `UserSpeaking` | `FinalTranscript` | `Thinking` |
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

## Illegal transitions

Two error classes cover every pair that is not listed above.

- **Closed state.** Once the machine is in `Closed`, every event returns
  `ParlevoxError::Closed` (message `session closed`) before the table is
  consulted. This covers all nine events from `Closed`, including `Close`
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
`parlevox_core::FloorOwner` label reported by `TurnState::owns_floor`. The six
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
| `Thinking` | agent-unconfirmed | no | yes (cancels generation) |
| `Speaking` | agent-confirmed | yes | escalated to barge-in evaluation |
| `Interrupted` | nobody | drain only | yes (next turn) |
| `Closed` | nobody | no | no |
