"""Parlevox Python sidecar SDK.

Standard library only. Install-free: run with ``PYTHONPATH=sdks/python``.
"""

from .protocol import (
    KIND_AUDIO,
    KIND_CLOSE,
    KIND_JSON,
    MAX_FRAME_BYTES,
    PROTOCOL_VERSION,
    SIDECAR_ABI,
    EmptyFrame,
    FrameTooLarge,
    OddPcm,
    ProtocolError,
    UnknownKind,
    decode_audio,
    encode_audio,
    read_frame,
    write_frame,
)
from .plugin import Plugin
from .fakes import FakeLlm, FakeStt, FakeTts, FakeVad

__all__ = [
    "Plugin",
    "FakeLlm",
    "FakeStt",
    "FakeTts",
    "FakeVad",
    "PROTOCOL_VERSION",
    "SIDECAR_ABI",
    "MAX_FRAME_BYTES",
    "KIND_JSON",
    "KIND_AUDIO",
    "KIND_CLOSE",
    "ProtocolError",
    "FrameTooLarge",
    "EmptyFrame",
    "UnknownKind",
    "OddPcm",
    "read_frame",
    "write_frame",
    "encode_audio",
    "decode_audio",
]
