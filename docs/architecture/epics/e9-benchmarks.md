# E9 — Proof

Status: not started · Epic `e9` · Index: [EPICS.md](../EPICS.md) ·
Roadmap: [PARITY](../PARITY-LIVEKIT-AGENTS.md)

### Goal

Publish numbers, not adjectives. Every latency claim in the README must be
backed by a harness anyone can rerun.

### Scope

- Per-stage latency: VAD-end → STT-final → LLM-first-token → TTS-first-byte →
  first `AudioOut`.
- p50/p95 at 1, 10 and 50 concurrent sessions.
- RSS per session, and per-session steady-state cost of the warm plugin pool.
- A committed harness plus a results page; numbers are published, not asserted
  in tests.

### Depends on

- E1 (a real provider set; fakes measure nothing useful)
- E2 (a transport for the concurrent shapes)

### Exit criteria

- One command reproduces the published table on a documented machine class.
- Regressions beyond a stated tolerance are visible in CI history, not by
  re-reading a blog post.

### Links

- Deployment dimension: [`DESIGN.md`](../DESIGN.md) §5, dimension 9
