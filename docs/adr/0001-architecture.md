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
   `ParlevoxError::Protocol`; every audio path is a bounded ring with an explicit
   overflow policy and dropped-frame/dropped-ms counters; duration measurement
   uses one monotonic clock per session.

## Backpressure and loss accounting

v0.1 is **drop-not-throttle** by design. Credit-based or adaptive backpressure
— a producer blocking until a consumer grants capacity — is explicitly out of
scope for this version. A realtime voice turn cannot wait on a slow consumer,
so every lossy path sheds load and counts it instead.

The contract is uniform across paths:

1. **Every audio and event queue has a capacity, an explicit policy, and a
   counter.** The input and playout audio rings, the session event queue, and
   the sidecar's per-request pending-audio buffer each have a fixed bound, a
   declared overflow policy (`DropOldest` or `DropNewest`), and a monotonic
   dropped counter. No audio or event buffer grows without bound. The plugin's
   `pending_requests`/`pending_dropped` maps are lifecycle-scoped rather than
   capacity-bounded: an entry exists only while its request is open and is
   removed on `audio_end` or `cancel`. The one ledger that outlives requests —
   cancelled request ids — is explicitly bounded (point 4), so a client that
   cancels and walks away cannot grow it forever.
2. **Push returns a typed outcome the producer branches on.** `AudioRing::push`
   and `EventQueue::push` report `Accepted`, `Replaced` (an older entry was
   evicted), or `Dropped` (the incoming entry was rejected). A producer either
   observes a drop and keeps going, or receives a loud `Protocol` error for a
   control-lane refusal that must fail the turn. Silent truncation is not an
   option.
3. **Loss is observable end to end, on the path that can actually lose.**
   Only the plugin's `transcribe`/`analyze` paths buffer audio, so only they can
   evict frames; `generate` and `synthesize` stream eagerly and report
   `dropped_frames: 0`. `SidecarStt` sends an utterance as bounded 10 ms frames,
   making the plugin's 64-frame buffer reachable, and captures the terminal
   `done` count alongside `SidecarLlm` and `SidecarTts`. The runtime aggregates
   all three adapters into `TurnReport` and `SessionEvent::TurnDone` as
   `sidecar_dropped_frames`, distinct from the Rust playout `dropped_frames`.
   An operator can therefore tell local backpressure from loss at the process
   boundary; zero means no loss was measured on that turn, not that the counter
   is unplumbed.
4. **Cancellation tombstones are bounded too.** The Python plugin remembers
   cancelled request ids so a late `audio_end` cannot resurrect suppressed
   work, but that ledger is an explicit-capacity, insertion-ordered set. When
   full, the oldest id is evicted deterministically and the eviction is
   counted. A late frame for a live tombstone stays a silent no-op; one for an
   evicted id is rejected as an unknown request rather than executed.

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
comparison that motivated it.

See also: [`../architecture/DESIGN.md`](../architecture/DESIGN.md) for the
shipped skeleton and the dimension-by-dimension status, and
[`../architecture/COMPARISON.md`](../architecture/COMPARISON.md) for the
four-framework audit that motivated this boundary.
