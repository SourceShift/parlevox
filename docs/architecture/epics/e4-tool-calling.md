# E4 — Tool calling

Status: not started · Epic `e4` · Index: [EPICS.md](../EPICS.md) ·
Roadmap: [PARITY](../PARITY-LIVEKIT-AGENTS.md)

### Goal

Let the agent call functions mid-turn, with bounds the runtime owns rather than
the provider.

### Scope

- Tool schemas passed through the LLM call (the manifest already carries
  protocol shape; this adds the tool payload).
- A bounded tool-call loop in Rust: max rounds, per-call timeout, tool error
  becomes a tool output rather than a failed turn, and every abandoned round is
  counted.
- `@function_tool` decorator in `sdks/python`, matching `livekit.agents`.
- `ToolCallStarted` / `ToolCallUpdated` / `ToolCallEnded`-shaped events on the
  public stream.

### Depends on

- E3 (the agent/SDK surface the decorator plugs into)

### Exit criteria

- A tool that always errors produces a completed turn whose reply explains the
  failure, with the error count visible on the event stream.
- The loop stops at `max_rounds` and says so; it never spins.
- The default no-tool path is byte-identical to today's turn.

### Links

- Event shapes: E7 and [`TURN-SEMANTICS-REQUIREMENTS.md`](../TURN-SEMANTICS-REQUIREMENTS.md)
  (PR-5 covers the `agent_state_changed` shape)
