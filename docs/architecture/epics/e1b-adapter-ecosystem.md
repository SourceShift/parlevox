# E1b — Adapter ecosystem

Status: not started · Epic `e1b` · Index: [EPICS.md](../EPICS.md) ·
Roadmap: [PARITY](../PARITY-LIVEKIT-AGENTS.md)

### Goal

E1 produces adapters; E1b turns them into a *network* rather than a product.
LiveKit's adoption advantage is not the plugin count — it is that the provider
a team already pays for works without leaving the framework. The primitives are
already in the tree (four modality traits, a `ProviderManifest` with
`name`/`modality`/`capabilities`/`config_schema`); the gap is distribution
plumbing, not design.

### Scope

- **Registry**: a name→constructor map plus `parlevox providers list|describe`.
  `describe` renders the manifest's `config_schema`, so required keys and
  defaults are discoverable without docs.
- **Packaging**: one crate (or one sidecar module) per vendor. A third party
  adds *their* provider without touching the runtime or the turn machine.
- **Conformance kit**: one suite every adapter must pass — partials then one
  final, cancellation propagates, errors classified into `ErrorClass`,
  dropped-frame accounting, empty-text synthesis refused.
- **BYO credentials**: env + config resolution with a preflight that probes
  *entitlement*, not key presence.
- **Manifest protocol version**: so an older plugin fails loudly instead of
  misbehaving silently.

### Build phases (F0–F3)

The provider foundation is a sub-programme of this epic, specified in
[`PROVIDER-FOUNDATION-REQUIREMENTS.md`](../PROVIDER-FOUNDATION-REQUIREMENTS.md):

| Phase | Content | Depends |
|---|---|---|
| **F0** | manifest v2, descriptor loader, HTTP connector (OpenAI-compatible), registry + `providers list\|describe`, no-patching test | none — can run alongside E2 |
| **F1** | WebSocket connector toolkit, Deepgram STT, one WS TTS, redaction | F0 |
| **F2** | conformance kit, scaffold + ABI policy, compatibility | F0 |
| **F3** | selection, observability, entitlement preflight | F1, F2 |

**F0 is the gate**: a provider must be addable with zero Rust code before any
breadth work starts.

### Priority tiers (what a team already pays for)

STT Deepgram / AssemblyAI / OpenAI-Whisper · TTS ElevenLabs / Cartesia /
OpenAI / Azure · LLM OpenAI-compatible / Anthropic / Bedrock / Vertex ·
VAD Silero. Roughly eleven adapters cover the majority of LiveKit deployments.

### Exit criteria

The provider document's metrics M1–M5 hold: zero-Rust descriptor addition,
≤1 connector module per protocol family with no runtime edits, sidecar-only
providers touching only `sdks/python`, offline `list|describe|check`, and
green conformance cassettes as the merge gate.

### Links

- [`PROVIDER-FOUNDATION-REQUIREMENTS.md`](../PROVIDER-FOUNDATION-REQUIREMENTS.md)
- Turn-facing adapter invariants: TS-9 in
  [`TURN-SEMANTICS-REQUIREMENTS.md`](../TURN-SEMANTICS-REQUIREMENTS.md)
