# voicekit

[![CI](https://github.com/SourceShift/voicekit/actions/workflows/ci.yml/badge.svg)](https://github.com/SourceShift/voicekit/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A voice-agent runtime for one job: run a live audio turn — listen, transcribe,
think, speak, stop when the user interrupts — with predictable behaviour when
the machine, the network, or a provider can't keep up.

## What it does

A `Session` owns one conversation:

1. audio arrives on a **bounded ring** (16 kHz mono `i16` frames);
2. VAD marks the user as speaking and a transcript is produced;
3. an LLM stream produces the reply;
4. TTS streams audio back out on a second bounded ring;
5. if the user speaks while the reply is playing, barge-in cancels the turn
   cooperatively and returns the session to listening.

Every turn ends with a `TurnDone` that reports what happened — including what
was *dropped*.

## What makes it different

**Loss is measured, not hidden.** Every audio path and the session event queue
has a fixed capacity, an explicit `OverflowPolicy`, and a monotonic dropped
counter. Each turn reports deltas (`dropped_frames`, `dropped_ms`,
`dropped_events`), so you can tell "the network was fine" from "we quietly threw
away 300 ms of speech". Streaming deltas may be dropped under backpressure;
turn-structure events (`TurnStarted`, `TranscriptFinal`, `TurnDone`, `Failed`)
never are.

**The turn state machine is explicit and closed.** `Idle → Listening →
UserSpeaking → Thinking → Speaking`, plus `Interrupted` and `Closed`, live in one
transition table. An event that is illegal in the current state is a typed
`Protocol` error — not a log line. A misbehaving provider fails at the boundary
instead of corrupting turn state.

**Cancellation is cooperative and deterministic.** Barge-in advances a
generation counter and fires the turn's `CancellationToken`. Provider streams
observe it on their next poll and end with a single `Cancelled` error. No task is
killed from outside, no stream is left dangling.

**The default turn has no Python.** Audio buffering, the FSM, provider calls and
cancellation are in-process Rust. `demo --fake` spawns no interpreter, imports
no Python and touches no network.

**Python is reachable, but only through a real protocol.** A Python provider is
an optional sidecar: length-prefixed binary frames (`u8 header_len | JSON header
| LE i16 PCM`), a versioned handshake, a 1 MiB pre-allocation guard that rejects
oversized frames *before* allocating, per-request cancellation, and bounded
restart backoff. Never base64, never whole-turn JSON audio.

**Both backends run the same test.** The CLI demo asserts the same event
sequence whether the turn runs through the in-process fakes or through the Python
plugin over the wire.

## Status

v0 skeleton. APIs are unstable, and the turn model is **half-duplex**: VAD →
STT → LLM → TTS run in sequence, with no early-start or pipelined decoding yet.
[`docs/architecture/DESIGN.md`](docs/architecture/DESIGN.md) §5 scores each
design dimension as implemented, partial, or deferred — and §8 lists the known
gaps, including the ones this README does not hide.

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

## Design docs

- [`docs/architecture/DESIGN.md`](docs/architecture/DESIGN.md) — the shipped
  architecture, its six invariants, and a dimension-by-dimension status table.
- [`docs/adr/0001-architecture.md`](docs/adr/0001-architecture.md) — the
  decision record: Rust-first runtime, optional Python sidecar, drop-not-throttle
  backpressure, half-duplex v0.
- [`docs/architecture/COMPARISON.md`](docs/architecture/COMPARISON.md) — the
  prior-art study that preceded the code. It documents choices made, not a
  ranking of other frameworks; where it disagrees with the ADR, the ADR wins.

## Roadmap (v0.1+)

Close the gaps listed in [`DESIGN.md` §8](docs/architecture/DESIGN.md): thread
sidecar `dropped_frames` into `TurnDone`, bound the sidecar's cancellation
tombstones, property-test the event queue, and add a differential fake-vs-plugin
turn test. Beyond that: a Rust-owned `ChatContext` for multi-turn sessions, OTel
turn spans, composable turn-taking strategies, warm plugin pools, and
early-start/pipelined decoding.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
