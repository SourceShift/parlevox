# E6 — Worker and job model

Status: not started · Epic `e6` · Index: [EPICS.md](../EPICS.md) ·
Roadmap: [PARITY](../PARITY-LIVEKIT-AGENTS.md)

### Goal

Run many sessions in one process with an operator story: dispatch, concurrency
limits, warm plugin processes, and a graceful drain.

### Scope

- Single-node `AgentServer` equivalent: `parlevox serve --concurrency N`.
- Job dispatch: one session per job, admission control at the configured
  concurrency, bounded queue with counted rejection.
- Warm plugin pool sized from concurrency and p95 turn latency (closes DESIGN
  dimension 9, currently **Deferred**).
- Drain on SIGTERM: stop admitting, let in-flight turns finish within a bound,
  then close.
- `/healthz` reporting liveness plus pool and queue depth.

### Depends on

- E3 (the console/run surface and the session factory)

### Exit criteria

- N+1 concurrent sessions never exceed N live turns; the rejected job is
  reported, not silently dropped.
- A SIGTERM during an active turn produces a completed `TurnDone` before exit.
- `/healthz` reflects real pool depth; killing a plugin degrades it visibly.

### Links

- DESIGN dimension 9: [`DESIGN.md`](../DESIGN.md) §5
