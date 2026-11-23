"""AssemblyAI pre-recorded STT adapter: upload, polling, and redaction.

Every test replays a fixture or an injected transport: none opens a socket or
sleeps, and the AssemblyAI credential literal below is an obvious fake.
"""

import io
import json
import logging
import struct
import traceback
from pathlib import Path

import pytest

from parlevox_sdk.adapters import (
    AdapterError,
    AssemblyAISTT,
    CancelToken,
    ErrorClass,
    FixtureTransport,
    MissingCredential,
    reset_redaction,
)
from parlevox_sdk.adapters.base import Transport, TransportResponse
from parlevox_sdk.fakes import FAKE_CHANNELS, FAKE_SAMPLE_RATE
from parlevox_sdk.plugin import Plugin
from parlevox_sdk.protocol import KIND_AUDIO, KIND_JSON, decode_audio, decode_json, read_frame

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "assemblyai"
PLANTED_KEY = "aai-LEAKCHECK"
PROVIDER = "assemblyai-stt"
TRANSCRIPT_ID = "job-0001"
POLL_PATH = "/v2/transcript/" + TRANSCRIPT_ID
UPLOAD_URL = "https://cdn.example.invalid/uploaded-audio"

_UPLOAD_OK = {
    "status": 200,
    "body": json.dumps({"upload_url": UPLOAD_URL}).encode("utf-8"),
}
_CREATE_OK = {
    "status": 200,
    "body": json.dumps({"id": TRANSCRIPT_ID, "status": "queued"}).encode("utf-8"),
}
_COMPLETED = {
    "status": 200,
    "body": json.dumps({"id": TRANSCRIPT_ID, "status": "completed", "text": "hello world"}).encode(
        "utf-8"
    ),
}


@pytest.fixture(autouse=True)
def _clean_redaction():
    reset_redaction()
    yield
    reset_redaction()


def _env(**overrides):
    env = {"ASSEMBLYAI_API_KEY": PLANTED_KEY, "PARLEVOX_FIXTURE_DIR": str(FIXTURES)}
    env.update(overrides)
    return env


class _RecordingTransport(Transport):
    """Capture each request and replay queued bodies; never opens a socket.

    ``on_request`` runs inside ``_send_once`` so a test can drive the order of
    requests without a real connection or clock.
    """

    def __init__(self, responses=None, provider=PROVIDER, on_request=None, **kwargs):
        super().__init__(provider, **kwargs)
        self.requests = []
        self.on_request = on_request
        self._responses = list(
            responses if responses is not None else [_UPLOAD_OK, _CREATE_OK, _COMPLETED]
        )
        self._index = 0

    def _send_once(self, method, path, *, headers, body, cancel):
        self.requests.append(
            {
                "method": method,
                "path": path,
                "headers": dict(headers),
                "body": body,
            }
        )
        spec = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        if self.on_request is not None:
            self.on_request(method, path, self._index - 1)
        return TransportResponse(
            spec.get("status", 200),
            spec.get("headers") or {"content-type": "application/json"},
            [spec.get("body", b"")],
            provider=self.provider,
        )


class _RejectingTransport(Transport):
    """Replay ``http.client``'s illegal-header ``ValueError`` without a socket."""

    def __init__(self):
        super().__init__(PROVIDER, max_attempts=1, sleeper=lambda _seconds: None)

    def _send_once(self, method, path, *, headers, body, cancel):
        raise ValueError(f"Invalid header value {headers['authorization']!r}")


class _CancelOnSleep(CancelToken):
    """Cancel during the first poll wait, exercising the sleep branch.

    The transport's read is cancellation-free, so the only thing that can end
    the loop early is :meth:`AssemblyAISTT._wait_poll` consulting the token.
    """

    def __init__(self):
        super().__init__()
        self.sleeps = 0

    def sleep(self, seconds):
        self.sleeps += 1
        self.cancel()
        return super().sleep(seconds)


def _wav_fmt(payload):
    assert payload[:4] == b"RIFF"
    assert payload[8:12] == b"WAVE"
    return struct.unpack("<HHIIHH", payload[20:36])


def _messages(blob):
    messages = []
    stream = io.BytesIO(blob)
    while True:
        try:
            kind, payload = read_frame(stream)
        except EOFError:
            return messages
        if kind == KIND_JSON:
            messages.append(decode_json(payload))
        elif kind == KIND_AUDIO:
            messages.append(("audio",) + decode_audio(payload))


def _stt(transport, **kwargs):
    kwargs.setdefault("poll_interval", 0)
    kwargs.setdefault("sleeper", lambda _seconds: None)
    return AssemblyAISTT(transport=transport, environ=_env(), **kwargs)


def test_upload_then_create_request_construction():
    transport = _RecordingTransport()
    stt = _stt(transport)
    pcm = b"\x01\x00" * 160
    assert list(stt.transcribe(pcm)) == [{"text": "hello world", "final": True}]

    assert len(transport.requests) == 3
    upload, create, poll = transport.requests

    assert upload["method"] == "POST"
    assert upload["path"] == "/v2/upload"
    assert upload["headers"]["authorization"] == PLANTED_KEY
    assert upload["headers"]["content-type"] == "application/octet-stream"
    assert upload["headers"]["accept"] == "application/json"
    assert upload["body"][:4] == b"RIFF"
    assert len(upload["body"]) == 44 + len(pcm)

    assert create["method"] == "POST"
    assert create["path"] == "/v2/transcript"
    assert create["headers"]["authorization"] == PLANTED_KEY
    assert create["headers"]["content-type"] == "application/json"
    assert json.loads(create["body"].decode("utf-8")) == {"audio_url": UPLOAD_URL}

    assert poll["method"] == "GET"
    assert poll["path"] == POLL_PATH
    assert poll["headers"]["authorization"] == PLANTED_KEY
    assert poll["headers"]["accept"] == "application/json"
    assert poll["body"] is None

    # The credential is the bare key: no scheme on either write.
    for request in (upload, create, poll):
        assert request["headers"]["authorization"] == PLANTED_KEY
        assert "Bearer" not in request["headers"]["authorization"]
        assert "Token" not in request["headers"]["authorization"]


def test_completed_on_third_poll_returns_transcript():
    transport = FixtureTransport.from_directory(FIXTURES, provider=PROVIDER)
    stt = _stt(transport)
    chunks = list(stt.transcribe(b"\x00\x00" * 160))
    assert chunks == [{"text": "hello world", "final": True}]
    # queued -> processing -> completed: exactly three polls.
    assert transport._cursors["GET " + POLL_PATH] == 3


def test_caller_rate_and_channels_reach_uploaded_wav():
    transport = _RecordingTransport()
    list(_stt(transport).transcribe(b"\x00\x00" * 480, sample_rate=48_000, channels=2))
    (audio_format, channels, sample_rate, _byte_rate, _align, bits) = _wav_fmt(
        transport.requests[0]["body"]
    )
    assert (audio_format, channels, sample_rate, bits) == (1, 2, 48_000, 16)

    default_transport = _RecordingTransport()
    list(_stt(default_transport).transcribe(b"\x00\x00" * 160))
    (_fmt, channels, sample_rate, _byte_rate, _align, _bits) = _wav_fmt(
        default_transport.requests[0]["body"]
    )
    assert (channels, sample_rate) == (FAKE_CHANNELS, FAKE_SAMPLE_RATE)


def test_error_status_is_classified_failure():
    transport = _RecordingTransport(
        responses=[
            _UPLOAD_OK,
            _CREATE_OK,
            {
                "status": 200,
                "body": json.dumps(
                    {"status": "error", "error": "audio could not be decoded"}
                ).encode("utf-8"),
            },
        ]
    )
    with pytest.raises(AdapterError) as info:
        list(_stt(transport).transcribe(b"\x00\x00" * 160))
    error = info.value
    assert error.classification is ErrorClass.FATAL
    assert error.wire_class() == "fatal"
    assert "audio could not be decoded" in str(error)
    assert PLANTED_KEY not in str(error)


def test_polling_exhaustion_is_classified_transient():
    transport = _RecordingTransport(
        responses=[
            _UPLOAD_OK,
            _CREATE_OK,
            {"status": 200, "body": b'{"status": "processing"}'},
        ]
    )
    with pytest.raises(AdapterError) as info:
        list(_stt(transport, max_polls=3).transcribe(b"\x00\x00" * 160))
    error = info.value
    assert error.classification is ErrorClass.TRANSIENT
    assert error.wire_class() == "transient"
    assert "polling exhausted" in str(error)
    assert len([r for r in transport.requests if r["method"] == "GET"]) == 3


def test_cancellation_during_polling():
    token = _CancelOnSleep()
    transport = _RecordingTransport(
        responses=[
            _UPLOAD_OK,
            _CREATE_OK,
            {"status": 200, "body": b'{"status": "queued"}'},
        ]
    )
    with pytest.raises(AdapterError) as info:
        list(_stt(transport, cancel=token).transcribe(b"\x00\x00" * 160))
    error = info.value
    assert error.classification is ErrorClass.CANCELLED
    assert error.wire_class() == "cancelled"
    assert token.sleeps == 1
    assert len([r for r in transport.requests if r["method"] == "GET"]) == 1


def test_invalid_json_or_shape_is_protocol():
    transport = _RecordingTransport(responses=[{"status": 200, "body": b"[1, 2, 3]"}])
    with pytest.raises(AdapterError) as info:
        list(_stt(transport).transcribe(b"\x00\x00" * 160))
    assert info.value.classification is ErrorClass.PROTOCOL
    assert info.value.wire_class() == "protocol"


def _status_transport(status):
    return _RecordingTransport(
        responses=[{"status": status, "body": b"{}"}],
        max_attempts=1,
        sleeper=lambda _seconds: None,
    )


def test_401_maps_to_fatal_and_429_to_rate_limit():
    with pytest.raises(AdapterError) as fatal:
        list(_stt(_status_transport(401)).transcribe(b"\x00\x00" * 160))
    assert fatal.value.classification is ErrorClass.FATAL
    assert fatal.value.wire_class() == "fatal"

    with pytest.raises(AdapterError) as limited:
        list(_stt(_status_transport(429)).transcribe(b"\x00\x00" * 160))
    assert limited.value.classification is ErrorClass.RATE_LIMIT
    assert limited.value.wire_class() == "transient"


def test_missing_key_fails_at_first_use_naming_env_var():
    transport = _RecordingTransport()
    stt = AssemblyAISTT(
        transport=transport, environ={}, poll_interval=0, sleeper=lambda _s: None
    )
    with pytest.raises(MissingCredential) as info:
        list(stt.transcribe(b"\x00\x00" * 160))
    error = info.value
    assert error.classification is ErrorClass.FATAL
    assert error.env_var == "ASSEMBLYAI_API_KEY"
    assert "ASSEMBLYAI_API_KEY" in str(error)
    assert PLANTED_KEY not in str(error)
    assert PLANTED_KEY not in repr(error)
    assert transport.requests == []


def test_planted_key_never_appears_in_failure_text(caplog):
    caplog.set_level(logging.DEBUG)
    cases = {
        "401": _status_transport(401),
        "429": _status_transport(429),
        "transport": _RejectingTransport(),
    }
    rendered = {}
    errors = {}
    for label, transport in cases.items():
        with pytest.raises(AdapterError) as info:
            list(_stt(transport).transcribe(b"\x00\x00" * 160))
        error = info.value
        errors[label] = error
        rendered[label] = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )

    for label, error in errors.items():
        for text in (str(error), repr(error), rendered[label]):
            assert PLANTED_KEY not in text, f"{label} leaked the credential"

    assert "<redacted>" in str(errors["transport"])
    assert errors["transport"].classification is ErrorClass.PROTOCOL
    assert errors["transport"].__cause__ is None
    assert errors["transport"].__context__ is None
    assert PLANTED_KEY not in caplog.text


def test_plugin_threads_caller_rate_into_assemblyai_upload():
    transport = _RecordingTransport()
    plugin = Plugin(stt=_stt(transport))
    out = io.BytesIO()
    pending_requests = {}
    pending_audio = {}
    assert plugin.handle(
        {
            "type": "transcribe",
            "request_id": "aai-rate",
            "sample_rate": 48_000,
            "channels": 2,
        },
        out,
        pending_audio=pending_audio,
        pending_requests=pending_requests,
    )
    pending_audio["aai-rate"] = b"\x00\x00" * 480
    assert plugin.handle(
        {"type": "audio_end", "request_id": "aai-rate"},
        out,
        pending_audio=pending_audio,
        pending_requests=pending_requests,
    )

    (_fmt, channels, sample_rate, _byte_rate, _align, bits) = _wav_fmt(
        transport.requests[0]["body"]
    )
    assert (channels, sample_rate, bits) == (2, 48_000, 16)
    finals = [m for m in _messages(out.getvalue()) if m.get("type") == "final"]
    assert finals == [
        {"type": "final", "request_id": "aai-rate", "text": "hello world"}
    ]
