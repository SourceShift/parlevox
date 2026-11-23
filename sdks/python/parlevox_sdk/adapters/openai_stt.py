"""OpenAI-wire speech-to-text adapter.

Endpoint: ``POST {base}/audio/transcriptions`` as ``multipart/form-data`` with
the model and the recorded audio file. The JSON response carries ``text``; the
adapter yields a single final transcript chunk.

Configuration: ``PARLEVOX_OPENAI_API_KEY`` (required, first use),
``PARLEVOX_OPENAI_BASE_URL``, ``PARLEVOX_OPENAI_STT_MODEL`` (default
``whisper-1``) and ``PARLEVOX_FIXTURE_DIR``.
"""

from __future__ import annotations

import json
import os

from ..fakes import FAKE_CHANNELS, FAKE_SAMPLE_RATE, SttProvider
from .base import (
    DEFAULT_BASE_URL_OPENAI,
    AdapterError,
    ErrorClass,
    default_transport,
    resolve_key,
    wav_container,
)

__all__ = ["OpenAISTT"]

ENV_VAR = "PARLEVOX_OPENAI_API_KEY"
DEFAULT_MODEL = "whisper-1"
_BOUNDARY = "----parlevox-boundary-7d1f4e"


def _manifest():
    return {
        "name": "openai-stt",
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


def _multipart(fields, filename, payload, *, content_type="application/octet-stream"):
    """Build a deterministic ``multipart/form-data`` body."""
    parts = []
    for name, value in fields.items():
        parts.append(
            (
                f"--{_BOUNDARY}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode("utf-8")
        )
    parts.append(
        (
            f"--{_BOUNDARY}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8")
    )
    parts.append(bytes(payload))
    parts.append(f"\r\n--{_BOUNDARY}--\r\n".encode("utf-8"))
    return b"".join(parts)


class OpenAISTT(SttProvider):
    """Transcription provider yielding one final chunk."""

    ENV_VAR = ENV_VAR
    DEFAULT_MODEL = DEFAULT_MODEL
    filename = "audio.wav"
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
            "PARLEVOX_OPENAI_BASE_URL", DEFAULT_BASE_URL_OPENAI
        )
        self.model = model or env.get("PARLEVOX_OPENAI_STT_MODEL", DEFAULT_MODEL)
        self._cancel = cancel
        self._transport = (
            transport
            if transport is not None
            else default_transport(self.base_url, env, "openai-stt")
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
        body = _multipart(
            {"model": self.model, "response_format": "json"},
            self.filename,
            wav_container(pcm, rate, channel_count),
            content_type="audio/wav",
        )
        headers = {
            "authorization": f"Bearer {key}",
            "content-type": f"multipart/form-data; boundary={_BOUNDARY}",
            "accept": "application/json",
        }
        response = self._transport.send(
            "POST",
            "/audio/transcriptions",
            headers=headers,
            body=body,
            cancel=self._cancel,
        )
        response.raise_for_status()
        try:
            payload = json.loads(response.read(cancel=self._cancel).decode("utf-8"))
        except ValueError:
            raise AdapterError(
                "openai-stt", "malformed transcription response", ErrorClass.PROTOCOL
            ) from None
        text = payload.get("text", "")
        if not isinstance(text, str):
            raise AdapterError(
                "openai-stt", "transcription response has no text", ErrorClass.PROTOCOL
            )
        yield {"text": text, "final": True}
