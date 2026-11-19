"""Framed sidecar protocol, mirroring ``crates/voicekit-ipc/src/frame.rs``.

Layout (all integers big-endian)::

    frame  = u32 body_len | u8 kind | body[body_len - 1]
    kind 1 (json)  : body = utf-8 JSON
    kind 2 (audio) : body = u8 header_len | header_json | pcm_i16_le...
    kind 3 (close) : body = empty

``body_len`` counts the kind byte plus the body. The reader rejects a declared
length above :data:`MAX_FRAME_BYTES` *before* allocating or reading the body.
"""

import json
import struct

PROTOCOL_VERSION = 1
SIDECAR_ABI = "voicekit-sidecar/1"

#: Hard ceiling for one frame (1 MiB), matching the Rust constant.
MAX_FRAME_BYTES = 1024 * 1024

KIND_JSON = 1
KIND_AUDIO = 2
KIND_CLOSE = 3

_KNOWN_KINDS = {KIND_JSON, KIND_AUDIO, KIND_CLOSE}


class ProtocolError(Exception):
    """Base class for frame-level violations."""


class FrameTooLarge(ProtocolError):
    def __init__(self, length, maximum=MAX_FRAME_BYTES):
        super().__init__(f"frame declares {length} bytes, limit is {maximum}")
        self.length = length
        self.maximum = maximum


class EmptyFrame(ProtocolError):
    def __init__(self):
        super().__init__("empty frame")


class UnknownKind(ProtocolError):
    def __init__(self, kind):
        super().__init__(f"unknown frame kind {kind}")
        self.kind = kind


class OddPcm(ProtocolError):
    def __init__(self, length):
        super().__init__(f"audio pcm length {length} is not a multiple of 2")
        self.length = length


class BadHeaderLen(ProtocolError):
    def __init__(self, length):
        super().__init__(f"audio header length {length} exceeds body")
        self.length = length


def _read_exact(stream, n):
    """Read exactly ``n`` bytes or raise EOFError."""
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("peer closed the stream")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(stream):
    """Return ``(kind, payload)`` for one frame.

    The length is validated before any body allocation.
    """
    header = _read_exact(stream, 4)
    (body_len,) = struct.unpack(">I", header)
    if body_len == 0:
        raise EmptyFrame()
    if body_len > MAX_FRAME_BYTES:
        raise FrameTooLarge(body_len)
    body = _read_exact(stream, body_len)
    kind = body[0]
    if kind not in _KNOWN_KINDS:
        raise UnknownKind(kind)
    payload = body[1:]
    if kind == KIND_AUDIO:
        # Mirror the Rust reader: header bounds and even PCM are validated at
        # frame-read time, not left for the caller to discover.
        decode_audio(payload)
    return kind, payload


def encode_frame(kind, payload):
    """Encode one frame; rejects payloads that exceed the ceiling."""
    if kind not in _KNOWN_KINDS:
        raise UnknownKind(kind)
    total = 1 + len(payload)
    if total > MAX_FRAME_BYTES:
        raise FrameTooLarge(total)
    return struct.pack(">I", total) + bytes([kind]) + payload


def write_frame(stream, kind, payload):
    stream.write(encode_frame(kind, payload))
    stream.flush()


def encode_json(message):
    return json.dumps(message, separators=(",", ":")).encode("utf-8")


def decode_json(payload):
    return json.loads(payload.decode("utf-8"))


def encode_audio(header, pcm):
    """Encode a binary audio frame: small JSON header plus little-endian i16.

    ``pcm`` may be a ``bytes``/``bytearray`` of already-packed samples or an
    iterable of integers.
    """
    header_bytes = encode_json(header)
    if len(header_bytes) > 0xFF:
        raise BadHeaderLen(len(header_bytes))
    if not isinstance(pcm, (bytes, bytearray)):
        pcm = struct.pack("<%dh" % len(pcm), *pcm)
    if len(pcm) % 2 != 0:
        raise OddPcm(len(pcm))
    return bytes([len(header_bytes)]) + header_bytes + bytes(pcm)


def decode_audio(payload):
    """Decode an audio payload into ``(header, pcm_bytes)``."""
    if not payload:
        raise BadHeaderLen(0)
    header_len = payload[0]
    if header_len > len(payload) - 1:
        raise BadHeaderLen(header_len)
    header = decode_json(payload[1 : 1 + header_len])
    pcm = payload[1 + header_len :]
    if len(pcm) % 2 != 0:
        raise OddPcm(len(pcm))
    return header, pcm


def pcm_to_samples(pcm):
    """Little-endian i16 bytes to a tuple of ints."""
    if len(pcm) % 2 != 0:
        raise OddPcm(len(pcm))
    return struct.unpack("<%dh" % (len(pcm) // 2), pcm)


def samples_to_pcm(samples):
    return struct.pack("<%dh" % len(samples), *samples)
