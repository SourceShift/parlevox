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

## Arbiter commands

`parlevox_core::TurnArbiter::step` turns one `Observation` plus an
`ArbiterModel` into the commands the runtime should carry out. The arbiter is
pure and synchronous: it performs no I/O, takes no lock, reads no clock and
cannot panic, so the same model and observation always produce the same
commands in the same order.

The command set is exhaustive:

| Command | Meaning |
|---|---|
| `StartGeneration { turn, generation }` | Begin reply generation for a user turn. |
| `CancelGeneration { generation }` | Tear down a superseded generation. |
| `StartSynthesis { turn, generation }` | Synthesise the reply for a turn. |
| `FlushAudioBuffer { turn }` | Drop queued audio for a turn. |
| `DuckAudio { gain }` | Attenuate agent audio; `gain` is on the `0..=100` scale and `DUCK_GAIN` is `20`. |
| `RestoreAudio` | Return agent audio to full level. |
| `EmitTurnEvent { kind }` | Emit the turn event that was accepted. |
| `Record { kind }` | Account for the step in one of the model's counters. |

Commands for one step are emitted in a fixed order, omitting groups that do not
apply: generation lifecycle (`CancelGeneration`, then `StartGeneration`),
`StartSynthesis`, audio (`FlushAudioBuffer`, `DuckAudio`, `RestoreAudio`),
`EmitTurnEvent`, and finally `Record`. The order is straight-line code; no hash
map participates.

### Observation-to-event mapping

| Observation | Actionable event |
|---|---|
| `FinalTranscript` | `FinalTranscript` |
| `VadStarted` | `UserStartedSpeaking` |
| `VadStopped` | `UserStoppedSpeaking` |
| `ReplyFinished` | `ReplyFinished` |
| `BargeInReported` | `BargeIn` |
| `Control { event }` | `event` |
| `PartialTranscript`, `ReplyDelta`, `AudioFrameQueued`, `CancellationObserved`, `Failure` | none — ignored |

### Counters

Every observation is accounted for exactly once, so `accepted + ignored +
rejected + dropped` always equals the number of observations stepped through the
arbiter.

| `Record` kind | Counter | When |
|---|---|---|
| `TurnOpened` | `accepted` | A `FinalTranscript` opened a new turn. |
| `GenerationSuperseded` | `accepted` | An in-flight reply was interrupted. |
| `ObservationIgnored` | `ignored` | The observation carried no actionable event. |
| `IllegalTransition` | `rejected` | The event is illegal from the current state. |
| `ObservationAfterClose` | `rejected` | A current observation after `Closed`; the state never leaves `Closed`. |
| `ObservationAhead` | `rejected` | The observation carried a newer `(turn, generation)` pair than the model has reached; see [Fence law](#fence-law). |
| `ObservationDropped` | `dropped` | The observation carried an older `(turn, generation)` pair; see [Fence law](#fence-law). |

An accepted event applies the same transition `TurnMachine::apply` would apply:
only `FinalTranscript` opens a turn, and it also starts the next generation.
Entering `Interrupted` from any other state cancels the superseded generation,
flushes queued audio, restores ducked audio when it was ducked, and increments
the model's generation in the same step; the value-less
`Record(GenerationSuperseded)` marks that the previous generation was
superseded and carries no generation number. Leaving `EvaluatingBargeIn` for any
other state restores ducked audio, so a model that is not in the contested
window is never left ducked.

## Fence law

Every observation is stamped with the `(turn, generation)` pair it belongs to.
The model carries the pair it currently considers live, and
`ArbiterModel::fence_position` classifies an observation against it by comparing
the two pairs lexicographically. The predicate is stated once here; the
implementation is in `crates/parlevox-core/src/arbiter.rs`.

| Position | Predicate | Effect in `step` |
|---|---|---|
| `Current` | `observation.(turn, generation) == model.(turn, generation)` | Admitted to the ordinary closed, non-actionable and legality branches. |
| `Superseded` | `observation.(turn, generation) < model.(turn, generation)` | Dropped: `dropped` increments and the step yields only `Record(ObservationDropped)`. No state, turn, generation or duck flag changes and no state-transition command is emitted. |
| `Ahead` | `observation.(turn, generation) > model.(turn, generation)` | Recorded: `rejected` increments and the step yields only `Record(ObservationAhead)`. The model has not reached that pair, so it cannot act on the observation yet. |

The default model is `(0, 0)`, the pre-generation epoch; an observation stamped
`(0, 0)` is `Current` there, which keeps the `Start`, VAD and
opening-transcript path admitted. No older pair is representable at `(0, 0)`, so
admitting it cannot mask a stale observation.

The fence is evaluated **before** the closed, non-actionable and legality
branches. A superseded observation therefore counts as `dropped` even after the
session has closed; only a current observation after `Closed` yields
`ObservationAfterClose`. The named case the fence exists for is a late
`FinalTranscript` for turn `N-1` arriving after turn `N` has opened: it is
dropped and counted, emits no state-transition command, and leaves the machine
in the state it already held.

Every observation remains accounted for exactly once, so the identity extends
to `accepted + ignored + rejected + dropped == observations`.

## Observation recording

`parlevox-runtime::Recorder` is an opt-in trace writer, installed with
`Session::with_recorder` and enabled on the CLI as
`demo --fake|--sidecar --record <path>`. A session without a recorder clones,
formats and writes nothing on the event path, so the default turn is unchanged.

### Record shape

The file begins with exactly one header line and then one compact JSON
`Observation` per line, matching
`crates/parlevox-core/src/observation.rs`:

```text
{"observation_schema_version":1}
{"t_ms":0,"event_id":0,"turn":1,"generation":1,"kind":{"control":{"event":"start"}}}
```

Fields serialize in the order `t_ms`, `event_id`, `turn`, `generation`, `kind`.
`kind` is externally tagged snake_case: a unit kind is a bare string
(`"reply_finished"`) and a data kind is a one-key object
(`{"final_transcript":{"text":"hi"}}`). The session maps its events with
`parlevox_runtime::observation_kind`: `TurnStarted` → `Control { event: Start }`,
`TranscriptFinal` → `FinalTranscript`, `ReplyDelta` → `ReplyDelta`, `AudioOut` →
`AudioFrameQueued`, `TurnDone` → `ReplyFinished`, `Failed` → `Failure`. A record
is written **before** the event enters the bounded event queue, so an event the
queue later drops is still traced.

### Caps and drops

`RecorderConfig::max_records` (default 100 000) and `max_bytes` (default 16 MiB)
bound every write. A record that would exceed either cap is not written; the
event is counted in `dropped_cap`. A record whose write fails is counted in
`dropped_write`. Neither case returns an error to the turn: recording cannot
fail or end a turn. `RecorderStats` exposes `records_written`, `bytes_written`,
`dropped_cap`, `dropped_write`, `redacted` and `transcripts_suppressed`, and
`records_written + dropped_cap + dropped_write + transcripts_suppressed` equals
the number of events offered to the recorder. Writes are synchronous, so an
operating system blocked on a full disk can still add latency.

### Transcript gating

Without `--record-transcripts` (the default,
`RecorderConfig::record_transcripts == false`), the transcript field is **absent,
never blank**: a final transcript is recorded as the text-free
`Control { event: FinalTranscript }` so replay and the arbiter still see the
turn open, while reply deltas are omitted and counted in
`transcripts_suppressed`. With the flag, text-bearing kinds retain their text.

### Redaction

Every recorded transcript, delta and failure message is scanned for the
credential shapes the repository forbids (`sk-…`, `ghp_…`, `github_pat_…`,
`AKIA…`, `xox[bpa]-…`, and a PEM `-----BEGIN … PRIVATE KEY` block). Each match
is replaced with `[redacted]` and counted in `RecorderStats::redacted`; a PEM
block counts as one match however many bytes it spans.

A PEM match is **block-level**, not header-level. Replacement runs from the
matched `-----BEGIN … PRIVATE KEY` header through the first matching
`-----END … PRIVATE KEY-----` marker inclusive, so the base64 body and the END
marker are removed together with the header. When no END marker exists, the
header's closing dashes and line terminator are replaced together with every
following non-empty line whose content is only base64 bytes (`A-Z`, `a-z`,
`0-9`, `+`, `/`, `=`); the first line with any other content ends the span, so
ordinary prose after the block is not swallowed.

The header itself is anchored on the trailing `PRIVATE KEY` literal, so a bare
`-----BEGIN` followed by ordinary prose is not redacted, and prose after a lone
header on the same line is left intact. Provider configuration and API-key
values are never passed to the recorder, so there is no second path by which key
material could reach a trace.
