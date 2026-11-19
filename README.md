# voicekit

A voice-agent runtime built from a best-of-four comparison of
[LiveKit Agents](https://github.com/livekit/agents),
[Pipecat](https://github.com/pipecat-ai/pipecat),
[TEN Framework](https://github.com/TEN-framework/ten-framework) and
[Vocode](https://github.com/vocodedev/vocode-core).

**The hot path is Rust and in-process.** Audio buffers, the turn state machine,
provider calls and cancellation live in one process, with every queue bounded
and every drop counted. A Python sidecar exists only as an optional escape hatch
for providers that need a Python SDK or a local model — the default turn never
spawns an interpreter.

> **Status:** v0 skeleton. APIs are unstable. See
> [`docs/architecture/DESIGN.md`](docs/architecture/DESIGN.md) for what is
> actually implemented versus deferred.

## Quick start

```bash
# One full fake turn: no Python, no network, no API keys.
cargo run -q -p voicekit-cli -- demo --fake

# The same turn through the Python plugin over the framed sidecar protocol.
PYTHONPATH=sdks/python cargo run -q -p voicekit-cli -- demo --sidecar
```

Expected output: two `AudioOut` frames and a `TurnDone` line carrying
`dropped_frames`, `dropped_ms`, `dropped_events` and `interrupted`.

## Tests

```bash
cargo test --workspace
PYTHONPATH=sdks/python python3 -m pytest sdks/python/tests -q
cargo clippy --workspace --all-targets
```

Tests are hermetic: no network, no API keys, no installed Python package.

## Layout

| Path | What |
|---|---|
| `crates/voicekit-core` | Audio frame, monotonic clock, bounded `AudioRing`, turn FSM, error classes |
| `crates/voicekit-providers` | Streaming STT/LLM/TTS/VAD traits, manifests, deterministic in-process fakes |
| `crates/voicekit-runtime` | One session: FSM owner, rings, bounded event handoff, provider selection, barge-in |
| `crates/voicekit-ipc` | Length-prefixed binary framing used **only** by the sidecar |
| `crates/voicekit-sidecar` | Process supervision and plugin→trait adapters |
| `crates/voicekit-cli` | `demo --fake` / `demo --sidecar` |
| `sdks/python/voicekit_sdk` | Reference Python plugin SDK and sidecar entry point |

## Design

- [`docs/architecture/DESIGN.md`](docs/architecture/DESIGN.md) — the shipped
  architecture, invariants, and the dimension-by-dimension status.
- [`docs/architecture/COMPARISON.md`](docs/architecture/COMPARISON.md) — the
  four-framework audit that motivated it, with the raw lens reports under
  [`docs/architecture/lenses/`](docs/architecture/lenses/).
- [`docs/adr/0001-architecture.md`](docs/adr/0001-architecture.md) — the
  architecture decision record: Rust-first with an optional Python sidecar.

## Roadmap (v0.1+)

Bounded-tombstone and sidecar-drop fixes are tracked in
[`DESIGN.md` §8](docs/architecture/DESIGN.md). Beyond that: a Rust-owned
`ChatContext` for multi-turn sessions, OTel turn spans, composable turn-taking
strategies, warm plugin pools, and early-start/pipelined decoding.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
