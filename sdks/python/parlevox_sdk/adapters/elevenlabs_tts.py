"""ElevenLabs text-to-speech adapter.

Endpoint: ``POST {base}/v1/text-to-speech/{voice_id}/stream`` with
``output_format=pcm_<sample_rate>`` on the query string and
``{"text": ..., "model_id": ...}`` as the JSON body. The credential is sent as
``xi-api-key`` (env ``ELEVENLABS_API_KEY``) and is read at first use, never at
import.

ElevenLabs returns raw 16-bit little-endian mono PCM already at the requested
rate, so the adapter does not resample: it slices the response body into 10 ms
frames of the caller's rate. Every yielded frame is a contiguous slice of the
body, so concatenating them reproduces the response byte-for-byte.

Configuration: ``ELEVENLABS_API_KEY`` (required, first use),
``PARLEVOX_ELEVENLABS_BASE_URL``, ``PARLEVOX_ELEVENLABS_VOICE_ID`` and
``PARLEVOX_ELEVENLABS_MODEL``.
"""

from __future__ import annotations

import json
import os
import urllib.parse

from ..fakes import FAKE_SAMPLE_RATE, TtsProvider
from .base import (
    AdapterError,
    ErrorClass,
    default_transport,
    resolve_key,
)

__all__ = ["ElevenLabsTTS"]

ENV_VAR = "ELEVENLABS_API_KEY"
PROVIDER = "elevenlabs-tts"
DEFAULT_BASE_URL_ELEVENLABS = "https://api.elevenlabs.io"
DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
DEFAULT_MODEL = "eleven_multilingual_v2"

FRAME_MS = 10
#: Bytes in one little-endian i16 sample.
SAMPLE_WIDTH = 2


def _manifest():
    return {
        "name": PROVIDER,
        "modality": "tts",
        "capabilities": ["streaming", "binary_pcm", "cancel"],
        "config_schema": {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "voice_id": {"type": "string"},
                "base_url": {"type": "string"},
            },
        },
    }


def _pcm_format(sample_rate):
    """The ElevenLabs ``output_format`` token for the caller's rate.

    Kept as a module-level seam so a test can prove the request field follows
    the caller instead of a hard-coded constant.
    """
    return f"pcm_{int(sample_rate)}"


def _validate(text, sample_rate, channels):
    """Validate the request and return the caller's integer rate.

    Empty text and non-positive formats are caller defects (``ValueError``);
    a multi-channel request is a protocol mismatch with the mono-only wire
    formats and is classified ``protocol``.
    """
    if not text.strip():
        raise ValueError("text must not be empty")
    sample_rate = int(sample_rate)
    channels = int(channels)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if channels <= 0:
        raise ValueError("channels must be positive")
    if channels != 1:
        raise AdapterError(
            PROVIDER,
            f"channels must be 1 (mono), got {channels}",
            ErrorClass.PROTOCOL,
        )
    return sample_rate


def _frame_chunks(pcm, sample_rate):
    """Slice ``pcm`` into 10 ms frames of the caller's rate, in order."""
    frame_bytes = max(1, sample_rate * FRAME_MS // 1000) * SAMPLE_WIDTH
    for start in range(0, len(pcm), frame_bytes):
        yield pcm[start : start + frame_bytes]


class ElevenLabsTTS(TtsProvider):
    """ElevenLabs speech synthesis yielding little-endian i16 PCM chunks."""

    ENV_VAR = ENV_VAR
    DEFAULT_VOICE_ID = DEFAULT_VOICE_ID
    DEFAULT_MODEL = DEFAULT_MODEL

    def __init__(
        self,
        transport=None,
        *,
        base_url=None,
        voice_id=None,
        model=None,
        environ=None,
        cancel=None,
    ):
        env = os.environ if environ is None else environ
        self._environ = env
        self.base_url = base_url or env.get(
            "PARLEVOX_ELEVENLABS_BASE_URL", DEFAULT_BASE_URL_ELEVENLABS
        )
        self.voice_id = voice_id or env.get(
            "PARLEVOX_ELEVENLABS_VOICE_ID", DEFAULT_VOICE_ID
        )
        self.model = model or env.get("PARLEVOX_ELEVENLABS_MODEL", DEFAULT_MODEL)
        self._cancel = cancel
        self._transport = (
            transport
            if transport is not None
            else default_transport(self.base_url, env, PROVIDER)
        )
        self._manifest = _manifest()

    @property
    def manifest(self):
        return self._manifest

    def synthesize(self, text, sample_rate=FAKE_SAMPLE_RATE, channels=1):
        sample_rate = _validate(text, sample_rate, channels)
        key = resolve_key(self.ENV_VAR, self._environ)
        voice_path = urllib.parse.quote(self.voice_id, safe="")
        query = urllib.parse.urlencode({"output_format": _pcm_format(sample_rate)})
        body = json.dumps({"text": text, "model_id": self.model}).encode("utf-8")
        headers = {
            "xi-api-key": key,
            "content-type": "application/json",
            "accept": "audio/pcm",
        }
        response = self._transport.send(
            "POST",
            f"/v1/text-to-speech/{voice_path}/stream?{query}",
            headers=headers,
            body=body,
            cancel=self._cancel,
        )
        response.raise_for_status()
        payload = response.read(cancel=self._cancel)
        yield from _frame_chunks(payload, sample_rate)
