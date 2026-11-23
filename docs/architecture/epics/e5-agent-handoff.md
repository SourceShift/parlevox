# E5 — Multi-agent handoff

Status: not started · Epic `e5` · Index: [EPICS.md](../EPICS.md) ·
Roadmap: [PARITY](../PARITY-LIVEKIT-AGENTS.md)

### Goal

Support more than one agent in a session: an agent registry, a handoff event,
and carry-over of the conversation into the next agent's instructions.

### Scope

- Agent registry in the session: each agent has instructions and (once E4
  lands) tools; the active agent is session state.
- `Handoff` event on the public stream naming the from/to agents.
- Context carry-over into the next agent's instructions — the receiving agent
  sees the committed turns, not a summary invented by the runtime.
- Python `Agent(handoff=...)` ergonomics matching `livekit.agents`.

### Depends on

- E3 (agent objects exist)

### Exit criteria

- A handoff mid-session preserves the conversation in the receiving agent's
  first request, and the event stream shows exactly one `Handoff`.
- A handoff while the agent is speaking rests the turn machine honestly (no
  lost `TurnDone`, no orphaned audio).

### Links

- Surface map: [`PARITY-LIVEKIT-AGENTS.md`](../PARITY-LIVEKIT-AGENTS.md) §1
  (`llm/chat_context.py`, `llm/delegation.py`)
