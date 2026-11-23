"""Cartesia text-to-speech adapter.

Endpoint: ``POST {base}/tts/bytes`` with ``x-api-key`` and
``cartesia-version`` headers (env ``CARTESIA_API_KEY``) and a JSON body naming
``model_id``, ``transcript``, the ``voice`` object and
``output_format: {"container": "raw", "encoding": "pcm_s16le",
"sample_rate": <caller rate>}``. Every field is read at first use, never at
import.

Cartesia returns raw 16-bit little-endian mono PCM already at the requested
rate, so the adapter does not resample: it slices the response body into 10 ms
frames of the caller's rate. Every yielded frame is a contiguous slice of the
body, so concatenating them reproduces the response byte-for-byte.

Configuration: ``CARTESIA_API_KEY`` (required, first use),
``PARLEVOX_CARTESIA_BASE_URL``, ``PARLEVOX_CARTESIA_VOICE``,
``PARLEVOX_CARTESIA_MODEL`` and ``PARLEVOX_CARTESIA_VERSION``.
"""

from __future__ import annotations

import json
import os

from ..fakes import FAKE_SAMPLE_RATE, TtsProvider
from .base import (
    AdapterError,
    ErrorClass,
    default_transport,
    resolve_key,
)

__all__ = ["CartesiaTTS"]

ENV_VAR = "CARTESIA_API_KEY"
PROVIDER = "cartesia-tts"
DEFAULT_BASE_URL_CARTESIA = "https://api.cartesia.ai"
DEFAULT_VOICE = "694f9389-aac1-45b6-b726-9d9369183238"
DEFAULT_MODEL = "sonic-2"
DEFAULT_VERSION = "2024-06-10"

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
                "voice": {"type": "string"},
                "version": {"type": "string"},
                "base_url": {"type": "string"},
            },
        },
    }


def _output_format(sample_rate):
    """The Cartesia ``output_format`` object for the caller's rate.

    Kept as a module-level seam so a test can prove the request field follows
    the caller instead of a hard-coded constant.
    """
    return {
        "container": "raw",
        "encoding": "pcm_s16le",
        "sample_rate": int(sample_rate),
    }


def _validate(text, sample_rate, channels):
    """Validate the request and return the caller's integer rate.

    Empty text and non-positive formats are caller defects (``ValueError``);
    a multi-channel request is a protocol mismatch with the mono-only wire
    format and is classified ``protocol``.
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


class CartesiaTTS(TtsProvider):
    """Cartesia speech synthesis yielding little-endian i16 PCM chunks."""

    ENV_VAR = ENV_VAR
    DEFAULT_VOICE = DEFAULT_VOICE
    DEFAULT_MODEL = DEFAULT_MODEL
    DEFAULT_VERSION = DEFAULT_VERSION

    def __init__(
        self,
        transport=None,
        *,
        base_url=None,
        voice=None,
        model=None,
        version=None,
        environ=None,
        cancel=None,
    ):
        env = os.environ if environ is None else environ
        self._environ = env
        self.base_url = base_url or env.get(
            "PARLEVOX_CARTESIA_BASE_URL", DEFAULT_BASE_URL_CARTESIA
        )
        self.voice = voice or env.get("PARLEVOX_CARTESIA_VOICE", DEFAULT_VOICE)
        self.model = model or env.get("PARLEVOX_CARTESIA_MODEL", DEFAULT_MODEL)
        self.version = version or env.get(
            "PARLEVOX_CARTESIA_VERSION", DEFAULT_VERSION
        )
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
        body = json.dumps(
            {
                "model_id": self.model,
                "transcript": text,
                "voice": {"mode": "id", "id": self.voice},
                "output_format": _output_format(sample_rate),
            }
        ).encode("utf-8")
        headers = {
            "x-api-key": key,
            "cartesia-version": self.version,
            "content-type": "application/json",
            "accept": "audio/pcm",
        }
        response = self._transport.send(
            "POST",
            "/tts/bytes",
            headers=headers,
            body=body,
            cancel=self._cancel,
        )
        response.raise_for_status()
        payload = response.read(cancel=self._cancel)
        yield from _frame_chunks(payload, sample_rate)
