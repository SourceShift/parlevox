"""Anthropic Messages LLM adapter.

Endpoint: ``POST {base_url}/v1/messages`` with ``stream: true``. The response
is Server-Sent Events: ``content_block_delta`` frames whose ``delta.type`` is
``text_delta`` carry reply fragments; ``message_delta``/``message_stop`` close
the stream and ``event: error`` reports a mid-stream provider failure.

The Messages API rejects a ``system`` role inside ``messages``: the system
prompt is a top-level string on the request. The sidecar wire has no top-level
system field, so every history entry whose role is ``system`` is hoisted into
that string (joined with a blank line) and never emitted as a message.

Configuration (read from the environment, never at import time):

* ``ANTHROPIC_API_KEY`` — required credential, resolved at first use.
* ``PARLEVOX_ANTHROPIC_BASE_URL`` — defaults to ``https://api.anthropic.com``.
* ``PARLEVOX_ANTHROPIC_MODEL`` — defaults to ``claude-opus-5-5``.
* ``PARLEVOX_FIXTURE_DIR`` — when set, replay fixtures instead of HTTP.
"""

from __future__ import annotations

import json
import os

from ..fakes import LlmProvider
from .base import (
    AdapterError,
    ErrorClass,
    classify_status,
    default_transport,
    resolve_key,
)

__all__ = ["AnthropicLlm"]

ENV_VAR = "ANTHROPIC_API_KEY"
PROVIDER = "anthropic-llm"
DEFAULT_BASE_URL_ANTHROPIC = "https://api.anthropic.com"
DEFAULT_MODEL = "claude-opus-5-5"
#: Pinned even though a newer API version exists: none of the fields the newer
#: version removes are sent, so bumping this stays a one-line change.
ANTHROPIC_VERSION = "2023-06-01"
#: A deliberately bounded voice reply; the constructor may override it.
DEFAULT_MAX_TOKENS = 4096

#: Streaming ``error.type`` values mapped to the HTTP status whose
#: classification they share, so there is one taxonomy instead of two. An
#: unrecognised type is ``protocol``: the wire said something we cannot read.
_STREAM_ERROR_STATUS = {
    "invalid_request_error": 400,
    "authentication_error": 401,
    "billing_error": 402,
    "permission_error": 403,
    "not_found_error": 404,
    "request_too_large": 413,
    "rate_limit_error": 429,
    "api_error": 500,
    "overloaded_error": 529,
}


def _manifest():
    return {
        "name": PROVIDER,
        "modality": "llm",
        "capabilities": ["streaming", "cancel", "history"],
        "config_schema": {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "max_tokens": {"type": "integer"},
                "base_url": {"type": "string"},
            },
        },
    }


def _line_text(raw):
    """Decode one body line and strip its terminator, if any."""
    if isinstance(raw, (bytes, bytearray)):
        line = bytes(raw).decode("utf-8", "replace")
    else:
        line = str(raw)
    if line.endswith("\n"):
        line = line[:-1]
    if line.endswith("\r"):
        line = line[:-1]
    return line


def _classify_stream_error(kind):
    """Map an ``error.type`` string onto the shared :class:`ErrorClass`."""
    status = _STREAM_ERROR_STATUS.get(kind)
    if status is None:
        return ErrorClass.PROTOCOL
    # ``classify_status`` returns ``None`` only for a 2xx status and no entry
    # above is 2xx; the fallback keeps the return type total anyway.
    return classify_status(status) or ErrorClass.PROTOCOL


def _stream_error(payload, provider):
    """Build the classified error for an ``event: error`` frame."""
    error = payload.get("error")
    if not isinstance(error, dict):
        error = {}
    kind = str(error.get("type") or "")
    detail = f"stream error {kind}" if kind else "stream error"
    message = error.get("message")
    if isinstance(message, str) and message:
        detail = f"{detail}: {message}"
    return AdapterError(provider, detail, _classify_stream_error(kind))


def _classify_event(name, data, provider):
    """Turn one accumulated SSE event into an outcome, or ``None`` to skip.

    Outcomes are ``("text", str)`` for a reply fragment and ``("stop", None)``
    for a terminal frame. Comments, ``ping``, ``message_start``,
    ``content_block_start``/``_stop`` and non-text deltas (thinking,
    signature) are skipped deliberately.
    """
    if not data:
        return None
    try:
        payload = json.loads(data)
    except ValueError:
        # The raw frame is never echoed: it may carry request material.
        raise AdapterError(
            provider, "malformed streaming event", ErrorClass.PROTOCOL
        ) from None
    if not isinstance(payload, dict):
        raise AdapterError(provider, "malformed streaming event", ErrorClass.PROTOCOL)
    kind = str(payload.get("type") or name or "")
    if kind == "content_block_delta":
        delta = payload.get("delta")
        if isinstance(delta, dict) and delta.get("type") == "text_delta":
            text = delta.get("text")
            if isinstance(text, str) and text:
                return ("text", text)
        return None
    if kind in ("message_delta", "message_stop"):
        return ("stop", None)
    if kind == "error":
        raise _stream_error(payload, provider)
    return None


def _outcomes(lines, provider, cancel):
    """Yield ``("text", str)``/``("stop", None)`` outcomes from SSE lines.

    Events are dispatched on a blank line, ``data:`` fields are joined with a
    newline as the SSE spec requires, and a pending event is flushed at EOF —
    so a stream that ends mid-event without ``message_stop`` still yields the
    delta it carried. The cancel token is polled before every dispatch, which
    is what makes a cancel land between two events that arrived in one chunk.
    """
    event_name = None
    data_lines = []

    def dispatch():
        nonlocal event_name, data_lines
        name = event_name
        data = "\n".join(data_lines)
        event_name = None
        data_lines = []
        if cancel is not None:
            cancel.raise_if_cancelled(provider)
        return _classify_event(name, data, provider)

    for raw in lines:
        line = _line_text(raw)
        if not line:
            outcome = dispatch()
            if outcome is None:
                continue
            yield outcome
            if outcome[0] == "stop":
                return
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if not separator:
            continue
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_name = value
        elif field == "data":
            data_lines.append(value)
        # ``id``/``retry`` and unknown fields carry nothing this adapter needs.
    if event_name is not None or data_lines:
        outcome = dispatch()
        if outcome is not None:
            yield outcome


def parse_sse_text(lines, provider=PROVIDER, cancel=None):
    """Yield reply deltas (``str``) from an Anthropic SSE line stream.

    ``lines`` yields body lines (without terminators) as ``bytes`` — exactly
    what :meth:`~parlevox_sdk.adapters.base.TransportResponse.iter_lines`
    produces. Already-yielded text is never lost: a malformed frame or an
    error frame raises only after the deltas before it were delivered.
    """
    for kind, payload in _outcomes(lines, provider, cancel):
        if kind == "text":
            yield payload


def _max_tokens(value):
    """Return ``value`` as a positive ``int`` or raise ``ValueError``."""
    try:
        tokens = int(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"max_tokens must be a positive integer, got {value!r}"
        ) from None
    if tokens <= 0:
        raise ValueError(f"max_tokens must be a positive integer, got {value!r}")
    return tokens


class AnthropicLlm(LlmProvider):
    """Anthropic Messages provider yielding reply deltas as strings."""

    ENV_VAR = ENV_VAR
    DEFAULT_MODEL = DEFAULT_MODEL

    def __init__(
        self,
        transport=None,
        *,
        base_url=None,
        model=None,
        max_tokens=None,
        environ=None,
        cancel=None,
    ):
        env = os.environ if environ is None else environ
        self._environ = env
        self.base_url = base_url or env.get(
            "PARLEVOX_ANTHROPIC_BASE_URL", DEFAULT_BASE_URL_ANTHROPIC
        )
        self.model = model or env.get("PARLEVOX_ANTHROPIC_MODEL", DEFAULT_MODEL)
        self.max_tokens = _max_tokens(
            DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens
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

    @staticmethod
    def _system_and_messages(prompt, history):
        """Split history into a top-level system string and turn messages.

        Every ``system`` entry is hoisted — the Messages API rejects that role
        inside ``messages`` — and the current prompt is appended as the final
        user turn.
        """
        system_parts = []
        messages = []
        for item in history or ():
            role = item.get("role", "user")
            text = item.get("text", "")
            if role == "system":
                system_parts.append(text)
            else:
                messages.append({"role": role, "content": text})
        messages.append({"role": "user", "content": prompt})
        return "\n\n".join(system_parts), messages

    def generate(self, prompt, history=()):
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        key = resolve_key(self.ENV_VAR, self._environ)
        system, messages = self._system_and_messages(prompt, history)
        request = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
            "stream": True,
        }
        if system:
            request["system"] = system
        body = json.dumps(request).encode("utf-8")
        headers = {
            "x-api-key": key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
            "accept": "text/event-stream",
        }
        response = self._transport.send(
            "POST",
            "/v1/messages",
            headers=headers,
            body=body,
            cancel=self._cancel,
        )
        response.raise_for_status()
        yield from parse_sse_text(
            response.iter_lines(cancel=self._cancel),
            provider=PROVIDER,
            cancel=self._cancel,
        )
