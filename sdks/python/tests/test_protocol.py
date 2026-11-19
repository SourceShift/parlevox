"""Protocol conformance: framing, bounds, and audio payload rules."""

import io
import struct

import pytest

from voicekit_sdk.protocol import (
    KIND_AUDIO,
    KIND_CLOSE,
    KIND_JSON,
    MAX_FRAME_BYTES,
    EmptyFrame,
    FrameTooLarge,
    OddPcm,
    UnknownKind,
    decode_audio,
    decode_json,
    encode_audio,
    read_frame,
    write_frame,
)


def test_json_roundtrip():
    stream = io.BytesIO()
    payload = b'{"type":"ready","plugin":"fake"}'
    write_frame(stream, KIND_JSON, payload)
    stream.seek(0)
    kind, decoded = read_frame(stream)
    assert kind == KIND_JSON
    assert decoded == payload
    assert decode_json(decoded)["plugin"] == "fake"


def test_audio_roundtrip_preserves_samples():
    header = {"request_id": "r1", "seq": 7, "sample_rate": 16000, "channels": 1}
    pcm = struct.pack("<4h", 1, -2, 3, -32768)
    stream = io.BytesIO()
    write_frame(stream, KIND_AUDIO, encode_audio(header, pcm))
    stream.seek(0)
    kind, payload = read_frame(stream)
    assert kind == KIND_AUDIO
    decoded_header, decoded_pcm = decode_audio(payload)
    assert decoded_header == header
    assert decoded_pcm == pcm


def test_oversized_frame_rejected_before_allocation():
    stream = io.BytesIO(struct.pack(">I", MAX_FRAME_BYTES + 1))
    with pytest.raises(FrameTooLarge):
        read_frame(stream)
    # The body was never read: the cursor is still right after the length.
    assert stream.tell() == 4


def test_zero_length_rejected():
    with pytest.raises(EmptyFrame):
        read_frame(io.BytesIO(struct.pack(">I", 0)))


def test_unknown_kind_rejected():
    body = bytes([99, 0x00])
    stream = io.BytesIO(struct.pack(">I", len(body)) + body)
    with pytest.raises(UnknownKind):
        read_frame(stream)


def test_odd_pcm_rejected():
    header = b'{"request_id":"r","seq":0,"sample_rate":16000,"channels":1}'
    body = bytes([len(header)]) + header + b"\x00"
    stream = io.BytesIO(struct.pack(">I", 1 + len(body)) + bytes([KIND_AUDIO]) + body)
    with pytest.raises(OddPcm):
        read_frame(stream)


def test_eof_is_reported_not_hidden():
    with pytest.raises(EOFError):
        read_frame(io.BytesIO(b""))


def test_close_frame_roundtrip():
    stream = io.BytesIO()
    write_frame(stream, KIND_CLOSE, b"")
    stream.seek(0)
    kind, payload = read_frame(stream)
    assert kind == KIND_CLOSE
    assert payload == b""


def test_encode_rejects_oversized_payload():
    with pytest.raises(FrameTooLarge):
        write_frame(io.BytesIO(), KIND_JSON, b"x" * MAX_FRAME_BYTES)
