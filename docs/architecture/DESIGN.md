# Voicekit — design of the shipped skeleton (v0)

**One sentence:** voicekit is a voice-agent runtime whose hot path — audio
buffers, the turn state machine, provider calls, cancellation — lives in a
single Rust process, with Python reachable only through an optional, framed
sidecar for providers that need a Python SDK.

This document describes **what exists in this repository today**. The
prior-art study that preceded the code is [`COMPARISON.md`](COMPARISON.md)
(it records choices made, not a ranking of other frameworks); the decision
record is
[`../adr/0001-architecture.md`](../adr/0001-architecture.md). Where the two
disagree about scope, the ADR is authoritative.

Status at `62f0850`: 87 Rust tests (including property tests for the audio ring
and the event queue), 27 Python tests, `demo --fake` and `demo --sidecar` both
green with identical event sequences. The advisory-closure run
(`run-1795127680-3383`) and the follow-up idiom pass
(`run-1795129305-30892`) were both approved by the four-family panel (glm,
kimi, codex, minimax); [`panel-verdict.json`](panel-verdict.json) holds the
most recent verdict, copied from its run directory.

---

## 1. Shape of the system

```mermaid
flowchart LR
    subgraph host["one OS process (Rust)"]
        cli["voicekit-cli<br/>demo --fake / --sidecar"]
        rt["voicekit-runtime<br/>Session actor<br/>TurnMachine · MonoClock<br/>AudioRing · EventQueue"]
        pr["voicekit-providers<br/>Stt / Llm / Tts / Vad traits<br/>manifests · in-process fakes"]
        core["voicekit-core<br/>AudioFrame · OverflowPolicy<br/>TurnState table · ErrorClass"]
        cli --> rt
        rt --> pr
        rt --> core
        pr --> core
    end

    subgraph side["optional sidecar (only when requested)"]
        sc["voicekit-sidecar<br/>supervisor · client · adapters"]
        ipc["voicekit-ipc<br/>framed protocol<br/>MAX_FRAME_BYTES = 1 MiB"]
        py["Python plugin<br/>voicekit_sdk.Plugin<br/>sidecar_main"]
    end

    rt -. "provider configured as sidecar" .-> sc
    sc --> ipc --> py
```

The default turn never crosses the right-hand side of that diagram. `demo
--fake` spawns no process, imports no Python, and touches no network; the
sidecar is started only when a session resolves a provider that lives there.

## 2. Crates

| Crate | Owns | Key types |
|---|---|---|
| `voicekit-core` | Audio frame, clock, bounded ring, FSM, error classes | `AudioFrame`, `MonoClock`, `AudioRing`, `OverflowPolicy`, `TurnMachine`, `TurnState`, `ErrorClass` |
| `voicekit-providers` | Streaming traits + manifests + deterministic fakes | `SttProvider`, `LlmProvider`, `TtsProvider`, `VadProvider`, `ProviderManifest`, `Fake*` |
| `voicekit-runtime` | One session: FSM owner, rings, event handoff, provider selection, barge-in | `Session`, `SessionConfig`, `SessionEvent`, `EventQueue`, `ProviderRegistry`, `TurnReport` |
| `voicekit-ipc` | The framed wire protocol — used **only** by the sidecar path | `Frame`, `AudioHeader`, `read_frame`, `write_frame`, `MAX_FRAME_BYTES` |
| `voicekit-sidecar` | Process supervision and the adapter from plugin → provider trait | `SidecarClient`, `SidecarSupervisor`, `SidecarStt/Llm/Tts`, `RestartPolicy` |
| `voicekit-cli` | Argument parsing and the one-turn demo, in a library so tests drive the same path | `run_args`, `demo_fake`, `demo_sidecar` |
| `sdks/python/voicekit_sdk` | Reference Python implementation of the plugin side | `Plugin`, `protocol`, `fakes`, `sidecar_main` |

## 3. One turn

`Session::run_turn` drives VAD → STT → LLM → TTS → playout in sequence, and
the turn's invariants live in one closed transition table (`TurnState`:
`Idle → Listening → UserSpeaking → Thinking → Speaking`, with `Interrupted`
on barge-in and `Closed` terminal). An event that is illegal in the current
state returns `VoicekitError::Protocol` — it is never swallowed, which is the
point: a misbehaving plugin fails loudly at the boundary instead of silently
corrupting a turn.

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Listening: Start
    Listening --> UserSpeaking: UserStartedSpeaking
    UserSpeaking --> Thinking: FinalTranscript
    Thinking --> Speaking: ReplyStarted
    Speaking --> Listening: ReplyFinished
    Speaking --> Interrupted: BargeIn
    Thinking --> Interrupted: BargeIn
    Interrupted --> Listening: ResumeListening
    Listening --> Closed: Close
```

**Barge-in** advances a generation counter and fires the turn's
`CancellationToken`. Provider streams observe the token on their next poll and
end with a single `Cancelled` error, so cancellation is cooperative but
deterministic — no task is killed from outside and no provider future is left
dangling. The runtime reports the interrupted turn as `TurnDone { interrupted:
true }`.

**v0 is half-duplex.** The next stage does not start until the previous one
finishes. Early-start and pipelined decoding (synthesizing before the
transcript is final, overlapping LLM deltas with TTS) are deliberately not
implemented — see the ADR's "Non-goals for v0".

## 4. Invariants (the reason this exists)

1. **Every buffer is bounded and accounted for.** Input audio, playout audio
   and the session→transport event handoff each have an explicit capacity and
   an `OverflowPolicy` (default `DropOldest`). Each surfaces `dropped_frames`,
   `dropped_ms` or `dropped_events`, per turn, as deltas from a captured
   baseline — so a session can tell the difference between "the network was
   fine" and "we quietly threw away 300 ms of speech".
2. **Control events are never dropped silently.** The event queue reserves a
   slot for the terminal event and refuses to drop turn structure
   (`TurnStarted`, `TranscriptFinal`, `TurnDone`, `Failed`); only streaming
   `ReplyDelta`/`AudioOut` traffic is droppable under backpressure.
3. **One monotonic clock per session.** `MonoClock` is the only source of time
   on the audio and turn paths; wall-clock time does not appear there.
4. **The default path has no Python.** If `demo --fake` ever needed an
   interpreter, the architecture would be wrong.
5. **Errors are classified.** `VoicekitError` carries an `ErrorClass` and an
   `is_recoverable()` answer, so a caller can retry, degrade or abort without
   string-matching a message.
6. **The boundary is framed, not textual.** Audio travels as binary frames
   (`u8 header_len | JSON header | LE i16 PCM`), never base64; a declared
   length above `MAX_FRAME_BYTES` is rejected *before* allocation, and odd PCM
   is a protocol error.

## 5. How the ten comparison dimensions landed

Section 1.3 of [`COMPARISON.md`](COMPARISON.md) selects one adopt per
dimension. This is the honest scorecard against the shipped tree:

| # | Dimension | Adopted direction | v0 status |
|---|---|---|---|
| 1 | Transport & audio path | Typed frames over a bounded, lag-counted ring | **Implemented** — `AudioRing`, `RingStats`, `dropped_ms` |
| 2 | Concurrency & isolation | Supervised plugin processes, multiplexed by `request_id` | **Partial** — supervision + restart backoff + id-multiplexed client exist; no warm pool, one plugin process per session |
| 3 | Turn-taking | Rust-owned FSM with composable strategies | **Partial** — explicit FSM and barge-in are real; strategy list and adaptive endpointing are not |
| 4 | Provider abstraction | One small interface per modality + versioned manifest | **Implemented** — traits, `ProviderManifest`, registry lookup |
| 5 | State & context | Rust-owned `ChatContext`, bounded policy | **Deferred** — no context object yet; the skeleton is single-turn |
| 6 | Failure handling | Non-fatal/fatal taxonomy, no replay after commit | **Partial** — `ErrorClass` + `is_recoverable`; no fallback provider chain |
| 7 | Observability | OTel turn span + boundary cost meters | **Deferred** — counters only (`dropped_*`, `EventQueueStats`); no traces |
| 8 | Extensibility cost | New provider = one interface + manifest, no core edits | **Implemented** — register/select by name |
| 9 | Deployment & scale | Warm pools sized by provider concurrency and p95 latency | **Deferred** |
| 10 | Language boundary | Rust owns lifecycle; length-prefixed binary frames with explicit cancel | **Implemented** — `voicekit-ipc` + `voicekit-sidecar` |

Deferred rows are deferrals, not omissions being papered over: they are the
v0.1+ agenda, and the ADR records the boundary.

## 6. Adding a provider

A Rust provider is two things: an implementation of one trait from
`voicekit-providers` (`transcribe`, `generate`, `synthesize` or `analyze`,
each returning a stream rather than a whole-utterance future) and a manifest
declaring name, modality, capabilities and config schema. It is registered
with `ProviderRegistry::register_*` and selected by name at session
construction. The runtime and the FSM do not change.

A Python-only provider implements `voicekit_sdk.Plugin` and is reached through
the sidecar adapter, which presents it to the runtime as the same trait. The
plugin host bounds each in-flight request's pending audio, purges it on
`cancel`, and acknowledges only after the local stream is torn down; the
process survives a cancel and serves the next request.

## 7. Running it

```bash
cargo test --workspace
cargo run -q -p voicekit-cli -- demo --fake      # no Python involved
PYTHONPATH=sdks/python python3 -m pytest sdks/python/tests -q
cargo run -q -p voicekit-cli -- demo --sidecar   # same turn through the plugin
```

Expected: a turn prints two `AudioOut` frames and a `TurnDone` carrying
`dropped_frames`, `dropped_ms`, `dropped_events`, `interrupted`.

## 8. Known gaps

The v0.1 advisory list is closed (run `run-1795127680-3383`): the sidecar's
dropped-frame count now reaches `TurnDone` as `sidecar_dropped_frames`,
`EventQueue` has the same property coverage as `AudioRing`, the plugin's
cancelled-request tombstones are a bounded ledger with counted eviction, and
CLI parity is a differential run of both backends rather than a framing
round-trip.

What remains open:

- Producers cannot observe backpressure — there are no credits; overflow is
  handled by dropping, not by slowing the producer (see ADR §"Backpressure
  and loss accounting", which records this as deliberate for v0.1).
- Half-duplex turn model (see §3 and the ADR's "Non-goals for v0").
- No warm plugin pool: one plugin process per session (dimension 2, partial).
- No turn spans or traces; observability is counters only (dimension 7).
