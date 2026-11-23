"""Wire adapters and their shared runtime.

``base`` owns errors, credential resolution, cancellation and transports, plus
the shared ``wav_container`` every speech adapter uploads through. The
OpenAI-wire family adds the chat, transcription and speech adapters; the
pre-recorded vendor adapters add Deepgram and AssemblyAI transcription. Every
module here is standard library only and import-safe without credentials or a
network.
"""

from .assemblyai_stt import AssemblyAISTT
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
    wav_container,
)
from .deepgram_stt import DeepgramSTT
from .openai_llm import OpenAILlm
from .openai_stt import OpenAISTT
from .openai_tts import OpenAITTS

__all__ = [
    "AdapterConfigError",
    "AdapterError",
    "AssemblyAISTT",
    "CancelToken",
    "DEFAULT_BASE_URL_OPENAI",
    "DeepgramSTT",
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
    "wav_container",
]
