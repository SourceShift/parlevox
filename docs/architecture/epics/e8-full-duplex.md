# E8 — Full-duplex / early-start STT

Status: not started · Epic `e8` · Index: [EPICS.md](../EPICS.md) ·
Roadmap: [PARITY](../PARITY-LIVEKIT-AGENTS.md)

### Goal

Remove the half-duplex constraint. Today `run_turn` takes one buffered
utterance and runs VAD → STT → LLM → TTS in sequence, so the session is never
listening while it speaks. E8 makes intake concurrent with the reply.

### Scope

- Streaming audio intake that feeds STT while the user speaks, instead of one
  `AudioChunk` per turn.
- Interim transcripts surfaced as events (not just discarded partials).
- Speculative LLM start on stable prefixes (the `speculative_start` flag is
  the seed; the intake path is missing).
- Concurrent VAD during `Speaking` — the precondition for any barge-in policy
  that evaluates speech before cutting.

### This is the architectural gap

Every other epic builds on the current turn shape. E8 changes it: the turn
loop becomes a consumer of a continuous observation stream rather than a
one-shot function. It unblocks **E10-T2** (evaluate/duck barge-in) and makes
`EvaluatingBargeIn` reachable at all.

### Depends on

- E3 (a session factory to run continuously)
- E2 (a streaming transport to receive audio)
- E6 (a worker that can host long-lived sessions)

### Exit criteria

- A user can speak during agent playback and the VAD observes it.
- A half-said sentence that ends with a final transcript produces one turn, not
  two.
- With speculative start disabled, the sequential behaviour stays available
  and byte-identical for tests.

### Links

- Turn contract, T2 gate: [`TURN-SEMANTICS-REQUIREMENTS.md`](../TURN-SEMANTICS-REQUIREMENTS.md)
