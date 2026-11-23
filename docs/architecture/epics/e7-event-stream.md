# E7 — Public event stream parity

Status: not started · Epic `e7` · Index: [EPICS.md](../EPICS.md) ·
Roadmap: [PARITY](../PARITY-LIVEKIT-AGENTS.md)

### Goal

Give transports and apps the same typed event story `livekit.agents` exposes,
plus a usage API — instead of a Rust-only `SessionEvent` enum that callers
cannot serialize.

### Scope

- Transport-delivered typed events: `user_input_transcribed`, `speech_created`,
  `agent_state_changed`, `function_tools_executed`, `usage_updated`.
- A usage / `TurnReport` API (latency, drops, token and character counts).
- Child plugin spans and loop-stall detection (closes the remaining half of
  DESIGN dimension 7's Partial).
- Serialization contract for events, versioned like the provider manifests.

### Depends on

- E2 (something to deliver the events to)

### Exit criteria

- An app can render a turn from the event stream alone, without reading
  `TurnReport` internals.
- `agent_state_changed` reflects the turn contract's states, so a LiveKit
  migration sees the same turn story.
- A stalled plugin surfaces as an event, not a hung turn.

### Links

- Turn states and floor ownership:
  [`TURN-SEMANTICS-REQUIREMENTS.md`](../TURN-SEMANTICS-REQUIREMENTS.md) §3
- DESIGN dimension 7: [`DESIGN.md`](../DESIGN.md) §5
