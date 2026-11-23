"""Deepgram pre-recorded speech-to-text adapter.

Endpoint: ``POST {base}/v1/listen?model=<model>&smart_format=true`` with a
16-bit RIFF/WAVE container as the raw request body. The JSON response carries
``results.channels[0].alternatives[0].transcript``; the adapter yields a single
final transcript chunk.

The credential is sent as ``Authorization: Token <key>`` — Deepgram's scheme,
not ``Bearer`` — and is read at first use, never at import.

Configuration: ``DEEPGRAM_API_KEY`` (required, first use),
``PARLEVOX_DEEPGRAM_BASE_URL``, ``PARLEVOX_DEEPGRAM_MODEL`` (default
``nova-2``) and ``PARLEVOX_FIXTURE_DIR``.
"""

from __future__ import annotations

import json
import os
import urllib.parse

from ..fakes import FAKE_CHANNELS, FAKE_SAMPLE_RATE, SttProvider
from .base import (
    AdapterError,
    ErrorClass,
    default_transport,
    resolve_key,
    wav_container,
)

__all__ = ["DeepgramSTT"]

ENV_VAR = "DEEPGRAM_API_KEY"
DEFAULT_MODEL = "nova-2"
DEFAULT_BASE_URL_DEEPGRAM = "https://api.deepgram.com"


def _manifest():
    return {
        "name": "deepgram-stt",
        "modality": "stt",
        "capabilities": ["cancel", "binary_pcm"],
        "config_schema": {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "base_url": {"type": "string"},
            },
        },
    }


def _transcript(payload):
    """Return the transcript from a listen response, or ``''``.

    Every shape mismatch — a non-object payload, a missing/empty/non-list
    ``channels`` or ``alternatives``, a non-object entry, a non-string
    ``transcript`` — is an empty transcript, never an exception: Deepgram omits
    these keys for silent audio, and a silent utterance is not a failure.
    """
    results = payload.get("results") if isinstance(payload, dict) else None
    channels = results.get("channels") if isinstance(results, dict) else None
    if not isinstance(channels, (list, tuple)) or not channels:
        return ""
    channel = channels[0]
    alternatives = channel.get("alternatives") if isinstance(channel, dict) else None
    if not isinstance(alternatives, (list, tuple)) or not alternatives:
        return ""
    alternative = alternatives[0]
    transcript = (
        alternative.get("transcript") if isinstance(alternative, dict) else None
    )
    return transcript if isinstance(transcript, str) else ""


class DeepgramSTT(SttProvider):
    """Deepgram pre-recorded transcription provider yielding one final chunk."""

    ENV_VAR = ENV_VAR
    DEFAULT_MODEL = DEFAULT_MODEL
    sample_rate = FAKE_SAMPLE_RATE
    channels = FAKE_CHANNELS

    def __init__(
        self,
        transport=None,
        *,
        base_url=None,
        model=None,
        environ=None,
        cancel=None,
    ):
        env = os.environ if environ is None else environ
        self._environ = env
        self.base_url = base_url or env.get(
            "PARLEVOX_DEEPGRAM_BASE_URL", DEFAULT_BASE_URL_DEEPGRAM
        )
        self.model = model or env.get("PARLEVOX_DEEPGRAM_MODEL", DEFAULT_MODEL)
        self._cancel = cancel
        self._transport = (
            transport
            if transport is not None
            else default_transport(self.base_url, env, "deepgram-stt")
        )
        self._manifest = _manifest()

    @property
    def manifest(self):
        return self._manifest

    def transcribe(self, pcm, *, sample_rate=None, channels=None):
        if not pcm:
            raise ValueError("audio must not be empty")
        key = resolve_key(self.ENV_VAR, self._environ)
        # Declare the format the caller's PCM actually has. The plugin threads
        # the wire request's ``sample_rate``/``channels``; a direct caller that
        # omits them keeps the adapter defaults.
        rate = self.sample_rate if sample_rate is None else sample_rate
        channel_count = self.channels if channels is None else channels
        body = wav_container(pcm, rate, channel_count)
        query = urllib.parse.urlencode(
            {"model": self.model, "smart_format": "true"}
        )
        headers = {
            "authorization": f"Token {key}",
            "content-type": "audio/wav",
            "accept": "application/json",
        }
        response = self._transport.send(
            "POST",
            "/v1/listen?" + query,
            headers=headers,
            body=body,
            cancel=self._cancel,
        )
        response.raise_for_status()
        try:
            payload = json.loads(response.read(cancel=self._cancel).decode("utf-8"))
        except ValueError:
            raise AdapterError(
                "deepgram-stt",
                "malformed transcription response",
                ErrorClass.PROTOCOL,
            ) from None
        yield {"text": _transcript(payload), "final": True}
