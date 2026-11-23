"""OpenAI-wire text-to-speech adapter.

Endpoint: ``POST {base}/audio/speech`` with ``response_format: pcm``, which
returns raw 24 kHz mono little-endian i16 samples. The adapter linearly
resamples to the caller's requested rate, then chunks the result at 10 ms of
the requested rate. A 16 kHz request over the recorded 480-sample fixture
yields exactly two 160-sample frames, matching the fake provider's frame count
so the demo's ``TurnDone`` stays stable.

Configuration: ``PARLEVOX_OPENAI_API_KEY`` (required, first use),
``PARLEVOX_OPENAI_BASE_URL``, ``PARLEVOX_OPENAI_TTS_MODEL`` (default ``tts-1``),
``PARLEVOX_OPENAI_TTS_VOICE`` (default ``alloy``) and ``PARLEVOX_FIXTURE_DIR``.
"""

from __future__ import annotations

import json
import os

from ..fakes import TtsProvider
from ..protocol import pcm_to_samples, samples_to_pcm
from .base import (
    DEFAULT_BASE_URL_OPENAI,
    default_transport,
    resolve_key,
)

__all__ = ["OpenAITTS", "resample_linear"]

ENV_VAR = "PARLEVOX_OPENAI_API_KEY"
DEFAULT_MODEL = "tts-1"
DEFAULT_VOICE = "alloy"

#: The sample rate OpenAI's ``pcm`` response format uses.
NATIVE_SAMPLE_RATE = 24_000
FRAME_MS = 10


def _manifest():
    return {
        "name": "openai-tts",
        "modality": "tts",
        "capabilities": ["streaming", "binary_pcm", "cancel"],
        "config_schema": {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "voice": {"type": "string"},
                "base_url": {"type": "string"},
            },
        },
    }


def resample_linear(samples, source_rate, target_rate):
    """Linearly resample mono i16 ``samples`` from one rate to another.

    Returns a list of ints clamped to the i16 range. Equal rates are a copy.
    """
    if source_rate == target_rate or not samples:
        return list(samples)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    output_length = int(len(samples) * target_rate / source_rate)
    step = source_rate / target_rate
    out = []
    last = len(samples) - 1
    for index in range(output_length):
        position = index * step
        left = int(position)
        fraction = position - left
        if left >= last:
            value = samples[last]
        else:
            value = samples[left] + (samples[left + 1] - samples[left]) * fraction
        out.append(max(-32768, min(32767, int(round(value)))))
    return out


class OpenAITTS(TtsProvider):
    """Speech synthesis yielding little-endian i16 PCM chunks."""

    ENV_VAR = ENV_VAR
    DEFAULT_MODEL = DEFAULT_MODEL
    DEFAULT_VOICE = DEFAULT_VOICE
    native_sample_rate = NATIVE_SAMPLE_RATE

    def __init__(
        self,
        transport=None,
        *,
        base_url=None,
        model=None,
        voice=None,
        environ=None,
        cancel=None,
    ):
        env = os.environ if environ is None else environ
        self._environ = env
        self.base_url = base_url or env.get(
            "PARLEVOX_OPENAI_BASE_URL", DEFAULT_BASE_URL_OPENAI
        )
        self.model = model or env.get("PARLEVOX_OPENAI_TTS_MODEL", DEFAULT_MODEL)
        self.voice = voice or env.get("PARLEVOX_OPENAI_TTS_VOICE", DEFAULT_VOICE)
        self._cancel = cancel
        self._transport = (
            transport
            if transport is not None
            else default_transport(self.base_url, env, "openai-tts")
        )
        self._manifest = _manifest()

    @property
    def manifest(self):
        return self._manifest

    def synthesize(self, text, sample_rate=NATIVE_SAMPLE_RATE, channels=1):
        if not text.strip():
            raise ValueError("text must not be empty")
        sample_rate = int(sample_rate)
        channels = int(channels)
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")
        key = resolve_key(self.ENV_VAR, self._environ)
        body = json.dumps(
            {
                "model": self.model,
                "voice": self.voice,
                "input": text,
                "response_format": "pcm",
            }
        ).encode("utf-8")
        headers = {
            "authorization": f"Bearer {key}",
            "content-type": "application/json",
            "accept": "audio/pcm",
        }
        response = self._transport.send(
            "POST",
            "/audio/speech",
            headers=headers,
            body=body,
            cancel=self._cancel,
        )
        response.raise_for_status()
        native = list(pcm_to_samples(response.read(cancel=self._cancel)))
        mono = resample_linear(native, self.native_sample_rate, sample_rate)
        frame_samples = max(1, sample_rate * FRAME_MS // 1000)
        for start in range(0, len(mono), frame_samples):
            frame = mono[start : start + frame_samples]
            if channels > 1:
                frame = [sample for sample in frame for _ in range(channels)]
            yield samples_to_pcm(frame)
