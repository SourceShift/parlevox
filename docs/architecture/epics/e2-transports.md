# E2 — Transports

Status: not started · Epic `e2` · Index: [EPICS.md](../EPICS.md) ·
Roadmap: [PARITY](../PARITY-LIVEKIT-AGENTS.md)

### Goal

Give the session a way to reach real users. The turn loop is
transport-agnostic today; this epic supplies the shells — and the gain control
that a future duck-don't-cut barge-in policy (E10-T2) needs.

### Scope

- **(a) WebSocket PCM transport**: binary 24 kHz s16le mono in/out, JSON
  control messages, one session per connection, bounded ingress with counted
  drops. This is the base transport; it lands first.
- **(b) LiveKit-room bridge** (optional cargo feature): join a room as a
  participant, subscribe the mic, publish the TTS track. Interop with the
  LiveKit SFU, not a replacement for it.
- **(c) Twilio Media Streams** over the same WS transport: μ-law 8 kHz
  resample in and out.

### Exit criteria

- (a) carries a full fake turn end-to-end and drops are counted, not hidden.
- A session speaks with a real room in (b) without touching the runtime.
- (c) tolerates the Twilio framing without a second transport implementation.
- The transport applies volume/gain to the playout path — the precondition for
  E10's `DuckAudio`/`RestoreAudio` commands.

### Links

- Ducking contract (T2): [`TURN-SEMANTICS-REQUIREMENTS.md`](../TURN-SEMANTICS-REQUIREMENTS.md)
