"""OpenAI-wire chat-completions LLM adapter.

Endpoint: ``POST {base}/chat/completions`` with ``stream: true``. The response
is Server-Sent Events: each ``data:`` line carries a JSON chunk whose
``choices[0].delta.content`` is a reply fragment; ``data: [DONE]`` closes the
stream.

Configuration (read from the environment, never at import time):

* ``PARLEVOX_OPENAI_API_KEY`` — required credential, resolved at first use.
* ``PARLEVOX_OPENAI_BASE_URL`` — defaults to ``https://api.openai.com/v1``.
* ``PARLEVOX_OPENAI_LLM_MODEL`` — defaults to ``gpt-4o-mini``.
* ``PARLEVOX_FIXTURE_DIR`` — when set, replay fixtures instead of HTTP.
"""

from __future__ import annotations

import json
import os

from ..fakes import LlmProvider
from .base import (
    DEFAULT_BASE_URL_OPENAI,
    AdapterError,
    ErrorClass,
    default_transport,
    resolve_key,
)

__all__ = ["OpenAILlm"]

ENV_VAR = "PARLEVOX_OPENAI_API_KEY"
DEFAULT_MODEL = "gpt-4o-mini"


def _manifest():
    return {
        "name": "openai-llm",
        "modality": "llm",
        "capabilities": ["streaming", "cancel", "history"],
        "config_schema": {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "base_url": {"type": "string"},
            },
        },
    }


def parse_sse_content(chunks):
    """Yield content deltas from raw SSE ``bytes`` chunks.

    A chunk may contain a partial line; callers pass whole lines via
    :meth:`TransportResponse.iter_lines`. Malformed JSON is a protocol error,
    never silently skipped, so a broken provider is visible.
    """
    for raw in chunks:
        line = raw.decode("utf-8", "replace").strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            return
        try:
            payload = json.loads(data)
        except ValueError:
            raise AdapterError(
                "openai-llm", "malformed streaming frame", ErrorClass.PROTOCOL
            ) from None
        choices = payload.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if content:
            yield content


class OpenAILlm(LlmProvider):
    """Chat-completions provider yielding reply deltas as strings."""

    ENV_VAR = ENV_VAR
    DEFAULT_MODEL = DEFAULT_MODEL

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
        self.model = model or env.get("PARLEVOX_OPENAI_LLM_MODEL", DEFAULT_MODEL)
        self._cancel = cancel
        self._transport = (
            transport
            if transport is not None
            else default_transport(self.base_url, env, "openai-llm")
        )
        self._manifest = _manifest()

    @property
    def manifest(self):
        return self._manifest

    def _messages(self, prompt, history):
        messages = []
        for item in history or ():
            messages.append(
                {
                    "role": item.get("role", "user"),
                    "content": item.get("text", ""),
                }
            )
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(self, prompt, history=()):
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        key = resolve_key(self.ENV_VAR, self._environ)
        body = json.dumps(
            {
                "model": self.model,
                "stream": True,
                "messages": self._messages(prompt, history),
            }
        ).encode("utf-8")
        headers = {
            "authorization": f"Bearer {key}",
            "content-type": "application/json",
            "accept": "text/event-stream",
        }
        response = self._transport.send(
            "POST",
            "/chat/completions",
            headers=headers,
            body=body,
            cancel=self._cancel,
        )
        response.raise_for_status()
        yield from parse_sse_content(response.iter_lines(cancel=self._cancel))
