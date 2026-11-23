"""AssemblyAI pre-recorded speech-to-text adapter.

Three steps, all standard library only:

1. ``POST {base}/v2/upload`` — the WAV container as the raw body, authorised by
   a bare ``Authorization: <key>`` header. The JSON response carries
   ``upload_url``.
2. ``POST {base}/v2/transcript`` — ``{"audio_url": <upload_url>}`` as JSON. The
   response carries the job ``id``.
3. ``GET {base}/v2/transcript/<id>`` — polled until the job reports
   ``completed`` (one final transcript chunk) or ``error`` (fatal).

Polling is bounded and cancellation-aware: at most :data:`MAX_POLLS` polls,
:data:`DEFAULT_POLL_INTERVAL` seconds apart (about 30 s in total), each wait
issued through the cancel token when one is set. Exhausting the bound is a
``transient`` failure — never a hang. Both knobs are constructor arguments so
tests can drive the loop with ``poll_interval=0`` and an injected ``sleeper``.

Configuration: ``ASSEMBLYAI_API_KEY`` (required, first use) and
``PARLEVOX_ASSEMBLYAI_BASE_URL``.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse

from ..fakes import FAKE_CHANNELS, FAKE_SAMPLE_RATE, SttProvider
from .base import (
    AdapterError,
    ErrorClass,
    default_transport,
    resolve_key,
    wav_container,
)

__all__ = ["AssemblyAISTT"]

ENV_VAR = "ASSEMBLYAI_API_KEY"
DEFAULT_BASE_URL_ASSEMBLYAI = "https://api.assemblyai.com"

#: Seconds between polls. With :data:`MAX_POLLS` this bounds a job to ~30 s.
DEFAULT_POLL_INTERVAL = 0.5
#: Maximum number of polls before the job is reported ``transient``.
MAX_POLLS = 60
#: Terminal statuses; every other status means "not ready yet".
_STATUS_COMPLETED = "completed"
_STATUS_ERROR = "error"


def _manifest():
    return {
        "name": "assemblyai-stt",
        "modality": "stt",
        "capabilities": ["cancel", "binary_pcm"],
        "config_schema": {
            "type": "object",
            "properties": {"base_url": {"type": "string"}},
        },
    }


class AssemblyAISTT(SttProvider):
    """AssemblyAI pre-recorded transcription provider yielding one final chunk."""

    ENV_VAR = ENV_VAR
    PROVIDER = "assemblyai-stt"
    sample_rate = FAKE_SAMPLE_RATE
    channels = FAKE_CHANNELS

    def __init__(
        self,
        transport=None,
        *,
        base_url=None,
        environ=None,
        cancel=None,
        poll_interval=DEFAULT_POLL_INTERVAL,
        max_polls=MAX_POLLS,
        sleeper=None,
    ):
        env = os.environ if environ is None else environ
        self._environ = env
        self.base_url = base_url or env.get(
            "PARLEVOX_ASSEMBLYAI_BASE_URL", DEFAULT_BASE_URL_ASSEMBLYAI
        )
        self.poll_interval = poll_interval
        self.max_polls = max_polls
        self._cancel = cancel
        self._sleeper = sleeper or time.sleep
        self._transport = (
            transport
            if transport is not None
            else default_transport(self.base_url, env, self.PROVIDER)
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
        wav = wav_container(pcm, rate, channel_count)
        auth = {"authorization": key, "accept": "application/json"}

        upload = self._send_json(
            "POST",
            "/v2/upload",
            headers={**auth, "content-type": "application/octet-stream"},
            body=wav,
        )
        upload_url = upload.get("upload_url")
        if not isinstance(upload_url, str) or not upload_url:
            raise AdapterError(
                self.PROVIDER, "upload response has no upload_url", ErrorClass.PROTOCOL
            )

        created = self._send_json(
            "POST",
            "/v2/transcript",
            headers={**auth, "content-type": "application/json"},
            body=json.dumps({"audio_url": upload_url}).encode("utf-8"),
        )
        transcript_id = created.get("id")
        if not isinstance(transcript_id, str) or not transcript_id:
            raise AdapterError(
                self.PROVIDER, "create response has no id", ErrorClass.PROTOCOL
            )
        poll_path = "/v2/transcript/" + urllib.parse.quote(transcript_id, safe="")

        polls = max(1, int(self.max_polls))
        for index in range(polls):
            payload = self._send_json("GET", poll_path, headers=auth, body=None)
            status = payload.get("status")
            if not isinstance(status, str):
                raise AdapterError(
                    self.PROVIDER, "poll response has no status", ErrorClass.PROTOCOL
                )
            if status == _STATUS_COMPLETED:
                text = payload.get("text", "")
                if not isinstance(text, str):
                    raise AdapterError(
                        self.PROVIDER,
                        "completed transcript has no text",
                        ErrorClass.PROTOCOL,
                    )
                yield {"text": text, "final": True}
                return
            if status == _STATUS_ERROR:
                detail = payload.get("error")
                if not isinstance(detail, str) or not detail:
                    detail = "transcription failed"
                # The detail passes through ``AdapterError``'s redaction, so a
                # vendor message that echoes request material cannot leak.
                raise AdapterError(
                    self.PROVIDER,
                    f"transcription failed: {detail}",
                    ErrorClass.FATAL,
                )
            if index + 1 >= polls:
                break
            self._wait_poll()
        raise AdapterError(
            self.PROVIDER,
            f"polling exhausted after {polls} polls",
            ErrorClass.TRANSIENT,
        )

    def _send_json(self, method, path, *, headers, body):
        """Send one request and decode a JSON object, classifying failures."""
        response = self._transport.send(
            method, path, headers=headers, body=body, cancel=self._cancel
        )
        response.raise_for_status()
        raw = response.read(cancel=self._cancel)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except ValueError:
            raise AdapterError(
                self.PROVIDER, "malformed JSON response", ErrorClass.PROTOCOL
            ) from None
        if not isinstance(payload, dict):
            raise AdapterError(
                self.PROVIDER, "malformed JSON response", ErrorClass.PROTOCOL
            )
        return payload

    def _wait_poll(self):
        """Wait one poll interval, aborting promptly when cancelled."""
        if self._cancel is not None:
            if self._cancel.sleep(self.poll_interval):
                raise AdapterError(self.PROVIDER, "cancelled", ErrorClass.CANCELLED)
        else:
            self._sleeper(self.poll_interval)
