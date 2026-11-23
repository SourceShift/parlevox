# Parity target: LiveKit Agents

**Locked 2026-11-23.** Parlevox replaces **LiveKit Agents** — the agent
framework (`AgentSession`, `Agent`, plugins, worker/job model, room I/O) — not
the LiveKit SFU, Cloud, SIP or TURN stack. Those are media-infrastructure
products; parlevox interoperates with them by joining a room as a participant
and never reimplements them.

Reference: the upstream `livekit/agents` repository (`livekit-agents` v1.x
Python SDK, 88 plugin packages). All anchors below are relative to
`livekit-agents/livekit/agents/`.

## 1. Surface map

| LiveKit Agents concept | Reference | Parlevox today | Gap |
|---|---|---|---|
| `AgentSession` orchestration | `voice/agent_session.py` | `Session` + `run_turn` (Rust) | Python SDK surface (E3) |
| `Agent` instructions + tools | `voice/agent.py` | prompt in `SessionConfig` | agent object, per-turn overrides (E3) |
| Turn model: endpointing, barge-in, false interruption | `voice/endpointing.py`, `voice/agent_activity.py` | **implemented** (closed FSM, recovery, partial resume, turn strategies) | stated contract, conformance and replay (E10) |
| STT/LLM/TTS contracts + 88 plugins | `plugin.py`, `livekit-plugins/*` | plugin protocol + sidecar process supervision | **zero real adapters** (E1) |
| Room I/O: participant audio in/out | `voice/room_io/` | none | **no transport at all** (E2) |
| Function tools, `function_tool`, `ToolContext`, `StopResponse` | `llm/tool_context.py` | none | tool loop + SDK decorator (E4) |
| `AgentHandoff`, context transfer | `llm/chat_context.py`, `llm/delegation.py` | typed `ChatContext` only | handoff registry + event (E5) |
| `AgentServer`, `JobContext`, warm pool, drain, load | `worker.py`, `job.py` | none | worker/job model (E6) |
| Typed event stream, usage, metrics events | `voice/events.py`, `metrics/` | `SessionEvent` + `TurnReport` + OTel | public event stream over transport, usage API (E7) |
| `cli run console / dev / start` | `cli/cli.py` | `demo --fake/--sidecar` | console mode, project runner (E3/E6) |
| Realtime/duplex models | `llm/realtime*.py`, `llm/duplex.py` | half-duplex by design | early-start STT / duplex (E8) |
| Simulation/scenarios | `simulation/` | none | out of scope for parity v1 |

## 2. Epics

Delivery is tracked in [`EPICS.md`](EPICS.md), which is machine-ingestible by
`mini-ork epics` (one `##` heading per epic, dependency edges as
`- depends on:` lines, dependency graph under its `(context)` section). Each
epic's specification lives in [`epics/`](epics/) and is the source of record;
this roadmap carries the parity narrative and the sequencing, not the detail.

| Epic | One-line intent | Detail |
|---|---|---|
| E1 | Real provider adapters in Python, reached through the sidecar | [e1](epics/e1-real-provider-adapters.md) |
| E1b | Adapter ecosystem: registry, packaging, conformance, BYO creds | [e1b](epics/e1b-adapter-ecosystem.md) |
| E2 | Transports: WS PCM, LiveKit-room bridge, Twilio Media Streams | [e2](epics/e2-transports.md) |
| E3 | Agent SDK ergonomics: `AgentSession`/`Agent`/`RunContext`, console | [e3](epics/e3-agent-sdk.md) |
| E4 | Tool calling: bounded loop, `@function_tool`, tool events | [e4](epics/e4-tool-calling.md) |
| E5 | Multi-agent handoff: registry, `Handoff` event, context carry-over | [e5](epics/e5-agent-handoff.md) |
| E6 | Worker and job model: dispatch, concurrency, warm pool, drain | [e6](epics/e6-worker-job-model.md) |
| E7 | Public event stream parity and usage API | [e7](epics/e7-event-stream.md) |
| E8 | Full-duplex / early-start STT — the one architectural gap | [e8](epics/e8-full-duplex.md) |
| E9 | Proof: benchmark harness, published numbers | [e9](epics/e9-benchmarks.md) |
| E10 | Turn semantics: stated contract, pure arbiter, conformance, replay | [e10](epics/e10-turn-semantics.md) |

Two specifications sit under epics rather than duplicating prose here:
[`PROVIDER-FOUNDATION-REQUIREMENTS.md`](PROVIDER-FOUNDATION-REQUIREMENTS.md)
(E1b's three support tiers, requirements and phases — **F0 is the gate**: a
provider addable with zero Rust code) and
[`TURN-SEMANTICS-REQUIREMENTS.md`](TURN-SEMANTICS-REQUIREMENTS.md) (E10's
TS-1…TS-12, metrics M1–M6 and the target state graph).

The dependency graph lives in
[`EPICS.md#dependency-graph-context`](EPICS.md#dependency-graph-context) so it
is read from the same file the tracker ingests, and cannot drift from the
edges that gate dispatch.

## 3. Migration map

| livekit.agents | parlevox |
|---|---|
| `AgentSession(stt=..., llm=..., tts=...)` | session built from the sidecar provider set |
| `Agent(instructions=..., tools=[...])` | agent definition + `@function_tool` |
| `function_tool` | same name in `parlevox_sdk` |
| `JobContext`, `AgentServer` | `parlevox serve --concurrency N` |
| `room_io` (Joins a LiveKit room) | `--livekit-room` bridge feature |
| `cli run console` | `parlevox console` |

## 4. Non-goals

Own SFU/WebRTC server, SIP stack, TURN, STT/TTS models, noise cancellation,
dashboards, billing, mobile SDKs, multi-node scheduling before single-node
worker works.

## 5. Sequencing and operating model

Dispatch order is read from [`EPICS.md`](EPICS.md), not from this section:
`mini-ork epics ready` returns the epics whose hard edges are satisfied.
Deliverables are developed in **separate git worktrees**, one deliverable per
worktree, so two epics never share a working tree. Land order: E2(a) → E1 → E3, then
E4/E5/E6 fan out; E9 after E2(a)+E1. E10 is split: T0–T1 (contract, envelope,
recorder, pure arbiter, replay) are runtime-local and can start immediately;
T2–T3 wait for E8 and E1. Estimates: E2(a) 1–2 runs, E1 4 runs,
E3 2–3 runs, E4 2–3 runs, E5 1–2, E6 2–3, E7 2, E9 1–2, E8 3–4, E10 6–10 runs
across four phases.

Provider work interleaves rather than waits: **F0** (descriptor + HTTP +
registry, 1–2 runs) has no dependency on transports and should land alongside
E2(a); **F1** (WS connector + Deepgram, 1–2 runs) needs F0 only; **F2** and
**F3** land after E1's first adapters so the conformance kit has real
counterparties. Every provider deliverable is scored against the no-patching
metrics M1–M3 in `PROVIDER-FOUNDATION-REQUIREMENTS.md`.
