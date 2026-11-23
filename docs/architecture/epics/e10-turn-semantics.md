# E10 — Turn semantics: contract, arbiter, conformance, replay

Status: requirements draft (T0–T1 startable now) · Epic `e10` · Index:
[EPICS.md](../EPICS.md) · Roadmap: [PARITY](../PARITY-LIVEKIT-AGENTS.md)

Full specification: [`TURN-SEMANTICS-REQUIREMENTS.md`](../TURN-SEMANTICS-REQUIREMENTS.md)
— requirements TS-1…TS-12, PR-1…PR-6, metrics M1–M6, open questions.

### Goal

Make turn behaviour a *specified, executable and replayable contract* rather
than an emergent property of the async shell. Today the FSM is pure
(`crates/parlevox-core/src/turn.rs:96`) but the policy that drives it — when to
commit, when to cut, when to duck — lives in `session.rs`, is half-duplex, and
cannot be replayed from a log.

Four layers to land:

1. **Normalized ingress** — every input becomes an `Observation { t_ms,
   event_id, turn, generation, kind }` on a monotonic clock.
2. **Pure arbiter** — `TurnArbiter::step(&Model, Observation) -> (Model,
   Vec<Command>)`, additive states `Committing`, `AgentGenerating`,
   `EvaluatingBargeIn`, with the backchannel-vs-barge-in distinction.
3. **Declarative commands + conformance** — `DuckAudio`/`RestoreAudio` and the
   rest are executed by the shell; a conformance suite proves time-to-silence,
   no-orphan-frames and spoken-vs-generated accounting.
4. **Event log + replay** — serializable events, JSONL recording, deterministic
   re-execution over a golden corpus.

### Phases

| Phase | Content | Gate |
|---|---|---|
| T0 | Stated contract (TS-1/T-2), `Observation`, pure arbiter skeleton, `Command` enum, recorder (TS-11) | none — start now |
| T1 | Conformance kit (TS-9), replay harness + golden corpus (TS-12) | T0 |
| T2 | `EvaluatingBargeIn`, `DuckAudio{gain}`/`RestoreAudio`, semantic `TurnDetector` seam (TS-6/TS-7) | E8 (full duplex), E2 (gain support) |
| T3 | Spoken-vs-generated accounting and latency stamps surfaced on the public stream (TS-8) | E7, E9 |

### Depends on

- Nothing hard for T0–T1.
- T2: E8 (concurrent intake), E2 (a transport that can apply gain).
- T3: E7 (event stream), E9 (numbers).

### Exit criteria

- A recorded turn replays to the same state trace and the same command
  sequence.
- A backchannel phrase during `Speaking` does not cut the reply; a genuine
  barge-in does, after ducking — or the decision is absent because T2 has not
  landed, and that is stated, not implied.
- Feed one observation sequence into the arbiter and two different shells; both
  emit the same commands.

### Links

- Spec: [`TURN-SEMANTICS-REQUIREMENTS.md`](../TURN-SEMANTICS-REQUIREMENTS.md)
- Today's code anchors: `crates/parlevox-core/src/turn.rs:96`,
  `crates/parlevox-core/src/clock.rs:1-12`,
  `crates/parlevox-runtime/src/session.rs:1309-1310`
