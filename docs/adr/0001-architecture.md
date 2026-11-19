# ADR 0001 — Rust-first runtime with an optional Python sidecar

- Status: accepted
- Date: 2026-11-19

## Context

The hot path of a voice agent — audio buffering, end-of-turn detection, turn
state, provider streaming, cancellation — is latency- and correctness-sensitive.
Provider ecosystems, however, are uneven: some useful STT, LLM, TTS, and VAD
implementations exist only as Python SDKs or local Python models.

The runtime must therefore be fast, bounded, and deterministic by default, while
still allowing a Python-only provider to participate without making Python a
requirement for the default turn.

## Decision

1. **The runtime is Rust and in-process.** Turn state, monotonic timing, bounded
   audio rings, provider traits, and the in-process fakes live in Rust. A default
   turn spawns no process, imports no Python, and performs no network I/O.
2. **Provider traits stream.** STT, LLM, TTS, and VAD return partial-result
   streams with a cancellation token on every in-flight call. A final-only
   provider wraps its single result in a one-item stream; the trait does not
   change shape. A manifest (name, modality, capabilities, config schema)
   describes each provider and drives selection.
3. **The Python sidecar is optional and interchangeable.** A separate crate is
   the only code allowed to spawn a process. It performs a versioned
   `hello`/`ready` handshake, carries control as JSON and audio as binary
   little-endian PCM frames with a hard size ceiling, propagates classified
   errors, supports cancellation without killing the process, and restarts a
   crashed plugin with bounded exponential backoff. A sidecar provider adapts to
   the same traits as an in-process provider, so the runtime and the turn state
   machine are unchanged.
4. **Audio is never encoded into control messages.** No base64, no whole-turn
   JSON audio. In-process audio travels in owned `i16` buffers; sidecar audio
   travels in binary frames.
5. **Invariants are enforced, not documented.** Illegal turn transitions remain
   `VoicekitError::Protocol`; every audio path is a bounded ring with an explicit
   overflow policy and dropped-frame/dropped-ms counters; duration measurement
   uses one monotonic clock per session.

## Alternatives considered

### Python core

A fully Python runtime would maximize provider reach and iteration speed, but
it makes tail latency, allocation behaviour, and cancellation semantics depend
on the interpreter and the GIL. It would also make the "no Python for a default
turn" guarantee impossible. Rejected.

### Full Rust, no sidecar

An all-Rust implementation has the simplest deployment and the tightest
invariants, but it excludes providers that exist only as Python packages or
local models until someone ports them. Rejected as the only option; retained as
the default path.

### Rust core with mandatory Python plugins

Making every provider a plugin process would give one uniform provider boundary
and a single language for adapters, but it puts a process boundary — and a
failure/restart domain — on the hot path even for providers that could run
in-process. Rejected; the sidecar is an escape hatch, not the architecture.

## Non-goals for v0

The v0 turn model is **half-duplex**: one turn runs VAD → STT → LLM → TTS in
sequence, and the next stage does not begin until the previous one finishes.
Early-start and pipelined decoding — starting synthesis before the transcript is
final, or overlapping LLM deltas with TTS — are a v0.1 item. They are explicitly
**not implemented** in this version, and the sequential turn loop should not be
read as providing them.

## Consequences

- Two provider paths must be kept behaviourally identical: the in-process fakes
  and the Python fakes share fixed transcripts, PCM, and frame counts, and the
  CLI demo asserts the same event sequence for both.
- The framed protocol is a versioned public surface. `PROTOCOL_VERSION` and the
  ABI string gate the handshake, and the Python SDK mirrors the Rust framing
  byte-for-byte (big-endian length, kind byte, binary PCM).
- A misbehaving plugin fails at the boundary with a classified error instead of
  corrupting turn state. A hung plugin fails the handshake by timeout; a crashed
  plugin is restarted with capped backoff.
- Provider authors implement a trait and a manifest. Adding a provider does not
  require touching the runtime, the turn state machine, or the sidecar unless it
  needs the sidecar.
- Debugging spans two languages when the sidecar is in use; the JSON control
  plane is kept small for that reason, and the in-process path stays available
  for isolating runtime defects.

## Scope

This ADR records the boundary decision only. It does not restate the provider
comparison that motivated it, and it does not describe a design document that
does not yet exist in this checkout.
