# E1 — Real provider adapters

Status: not started · Epic `e1` · Index: [EPICS.md](../EPICS.md) ·
Roadmap: [PARITY](../PARITY-LIVEKIT-AGENTS.md)

### Goal

Replace the deterministic fakes with adapters for the providers teams already
pay for, without changing the runtime. Every adapter is one Python module that
registers with the existing sidecar plugin host and satisfies the existing
modality traits.

### Scope

- One streaming-adapter base in `sdks/python`: env-keyed auth, bounded retries
  mapped onto the existing `ErrorClass` taxonomy, fixture-injectable transport.
- STT: Deepgram, OpenAI, AssemblyAI.
- TTS: ElevenLabs, Cartesia, OpenAI.
- LLM: OpenAI-compatible family, Anthropic.
- Each adapter is one module plus a manifest plus a fixture test; a live call
  is opt-in behind the provider's key env var.

### Not in this epic

- Descriptors, registry, conformance kit, credential preflight → **E1b**.
- Any transport work → **E2**.

### Exit criteria

- `cargo run -p parlevox-cli -- demo --sidecar` runs a full turn on each
  adapter against fixtures, with no network and no key.
- A live turn works for each provider when its key is set; CI never requires
  one.
- Each adapter PR leaves `git diff --stat crates/` empty.

### Links

- Foundation tiers and requirements:
  [`PROVIDER-FOUNDATION-REQUIREMENTS.md`](../PROVIDER-FOUNDATION-REQUIREMENTS.md)
