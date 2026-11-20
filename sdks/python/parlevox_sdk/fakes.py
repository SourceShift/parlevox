"""Provider ABCs, manifests, and deterministic fakes.

The fakes mirror the Rust fakes' fixed transcripts and frame counts so tests
and demos assert stable values instead of timing-dependent output.
"""

from abc import ABC, abstractmethod

from .protocol import samples_to_pcm

FAKE_TRANSCRIPT = "hello world"
FAKE_PARTIAL_TRANSCRIPT = "hello"
FAKE_LLM_DELTAS = ("Hello", ", ", "world", "!")
FAKE_REPLY = "".join(FAKE_LLM_DELTAS)
FAKE_SAMPLE_RATE = 16_000
FAKE_CHANNELS = 1
FAKE_FRAME_MS = 10
FAKE_FRAME_SAMPLES = (FAKE_SAMPLE_RATE * FAKE_FRAME_MS) // 1000  # 160
FAKE_TTS_FRAMES = 2


def fake_pcm(frame_index):
    """Same bounded sawtooth as the Rust fake, as little-endian i16 bytes."""
    samples = []
    for i in range(FAKE_FRAME_SAMPLES):
        base = i + frame_index * FAKE_FRAME_SAMPLES
        samples.append((base * 7) % 201 - 100)
    return samples_to_pcm(samples)


def _manifest(name, modality, capabilities):
    return {
        "name": name,
        "modality": modality,
        "capabilities": list(capabilities),
        "config_schema": {"type": "object", "properties": {}},
    }


class SttProvider(ABC):
    @property
    @abstractmethod
    def manifest(self):
        """Provider manifest dict."""

    @abstractmethod
    def transcribe(self, pcm):
        """Yield ``{"text": str, "final": bool}`` partials, final last."""


class LlmProvider(ABC):
    @property
    @abstractmethod
    def manifest(self):
        """Provider manifest dict."""

    @abstractmethod
    def generate(self, prompt, history=()):
        """Yield reply deltas as strings.

        ``history`` is an optional oldest-first sequence of
        ``{"role": str, "text": str}`` dicts for turns before ``prompt``. The
        plugin passes it only when non-empty and this signature accepts it, so
        a third-party provider may keep the one-argument
        ``generate(self, prompt)`` form.
        """


class TtsProvider(ABC):
    @property
    @abstractmethod
    def manifest(self):
        """Provider manifest dict."""

    @abstractmethod
    def synthesize(self, text):
        """Yield little-endian i16 PCM chunks."""


class VadProvider(ABC):
    @property
    @abstractmethod
    def manifest(self):
        """Provider manifest dict."""

    @abstractmethod
    def analyze(self, pcm):
        """Yield ``{"event": str, "timestamp_ms": int}`` transitions."""


class FakeStt(SttProvider):
    def __init__(self):
        self._manifest = _manifest("python-fake-stt", "stt", ("streaming", "partials", "cancel"))

    @property
    def manifest(self):
        return self._manifest

    def transcribe(self, pcm):
        if not pcm:
            raise ValueError("audio must not be empty")
        yield {"text": FAKE_PARTIAL_TRANSCRIPT, "final": False}
        yield {"text": FAKE_TRANSCRIPT, "final": True}


class FakeLlm(LlmProvider):
    def __init__(self):
        self._manifest = _manifest("python-fake-llm", "llm", ("streaming", "cancel"))

    @property
    def manifest(self):
        return self._manifest

    def generate(self, prompt, history=()):
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        # `history` is accepted for parity with the Rust fake but deliberately
        # does not change the fixed single-turn deltas.
        for delta in FAKE_LLM_DELTAS:
            yield delta


class FakeTts(TtsProvider):
    def __init__(self):
        self._manifest = _manifest("python-fake-tts", "tts", ("streaming", "binary_pcm", "cancel"))

    @property
    def manifest(self):
        return self._manifest

    def synthesize(self, text):
        if not text.strip():
            raise ValueError("text must not be empty")
        for frame_index in range(FAKE_TTS_FRAMES):
            yield fake_pcm(frame_index)


class FakeVad(VadProvider):
    def __init__(self):
        self._manifest = _manifest("python-fake-vad", "vad", ("streaming", "cancel"))

    @property
    def manifest(self):
        return self._manifest

    def analyze(self, pcm):
        if not pcm:
            raise ValueError("audio must not be empty")
        duration_ms = (len(pcm) // 2 // FAKE_CHANNELS) * 1000 // FAKE_SAMPLE_RATE
        yield {"event": "speech_start", "timestamp_ms": 0}
        yield {"event": "speech_end", "timestamp_ms": duration_ms}
