"""Wire adapters and their shared runtime.

``base`` owns errors, credential resolution, cancellation and transports.
The OpenAI-wire family adds the chat, transcription and speech adapters. Every
module here is standard library only and import-safe without credentials or a
network.
"""

from .base import (
    DEFAULT_BASE_URL_OPENAI,
    MAX_PUBLIC_MESSAGE,
    REDACTED,
    AdapterConfigError,
    AdapterError,
    CancelToken,
    ErrorClass,
    FixtureTransport,
    HttpTransport,
    MissingCredential,
    Transport,
    TransportResponse,
    classify_status,
    default_transport,
    redact,
    register_secret,
    reset_redaction,
    resolve_key,
)
from .openai_llm import OpenAILlm
from .openai_stt import OpenAISTT
from .openai_tts import OpenAITTS

__all__ = [
    "AdapterConfigError",
    "AdapterError",
    "CancelToken",
    "DEFAULT_BASE_URL_OPENAI",
    "ErrorClass",
    "FixtureTransport",
    "HttpTransport",
    "MAX_PUBLIC_MESSAGE",
    "MissingCredential",
    "OpenAILlm",
    "OpenAISTT",
    "OpenAITTS",
    "REDACTED",
    "Transport",
    "TransportResponse",
    "classify_status",
    "default_transport",
    "redact",
    "register_secret",
    "reset_redaction",
    "resolve_key",
]
