# E3 — Agent SDK ergonomics

Status: not started · Epic `e3` · Index: [EPICS.md](../EPICS.md) ·
Roadmap: [PARITY](../PARITY-LIVEKIT-AGENTS.md)

### Goal

Make the migration from `livekit.agents` a rename, not a rewrite. The Rust
session already owns the turn loop; this epic gives it the Python surface and
the two CLI entry points that mirror LiveKit's.

### Scope

- Python `AgentSession` / `Agent` / `RunContext` mirroring the
  `livekit.agents` names (with the same constructor shape:
  `AgentSession(stt=..., llm=..., tts=...)`).
- `parlevox console`: run one session over the E2(a) WebSocket transport and
  pipe audio to a local speaker/mic path.
- `parlevox run`: project runner that wires a session from config.
- Migration table kept honest: every name difference is documented in
  [`PARITY`](../PARITY-LIVEKIT-AGENTS.md) §3.

### Depends on

- E1b (a provider set worth configuring)
- E2 (a transport to run over)

### Exit criteria

- A LiveKit Agents sample app changes imports and constructor names only, and
  runs on `parlevox run`.
- `parlevox console` plays a turn end-to-end with no bespoke glue script.

### Links

- Migration map: [`PARITY-LIVEKIT-AGENTS.md`](../PARITY-LIVEKIT-AGENTS.md) §3
