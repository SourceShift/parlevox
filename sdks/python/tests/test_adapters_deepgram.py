"""Deepgram pre-recorded STT adapter: wire shape, parsing, and redaction.

Every test replays a fixture or an injected transport: none opens a socket,
and the Deepgram credential literal below is an obvious fake.
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
    DeepgramSTT,
    ErrorClass,
    FixtureTransport,
    MissingCredential,
    reset_redaction,
)
from parlevox_sdk.adapters.base import Transport, TransportResponse
from parlevox_sdk.fakes import FAKE_CHANNELS, FAKE_SAMPLE_RATE
from parlevox_sdk.plugin import Plugin
from parlevox_sdk.protocol import KIND_AUDIO, KIND_JSON, decode_audio, decode_json, read_frame

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "deepgram"
PLANTED_KEY = "sk-deepgram-LEAKCHECK"
PROVIDER = "deepgram-stt"

_OK_BODY = b'{"results": {"channels": [{"alternatives": [{"transcript": "ok"}]}]}}'


@pytest.fixture(autouse=True)
def _clean_redaction():
    reset_redaction()
    yield
    reset_redaction()


def _env(**overrides):
    env = {"DEEPGRAM_API_KEY": PLANTED_KEY, "PARLEVOX_FIXTURE_DIR": str(FIXTURES)}
    env.update(overrides)
    return env


class _RecordingTransport(Transport):
    """Capture each request and replay queued bodies; never opens a socket.

    ``on_request`` runs inside ``_send_once`` and receives the request before
    its response is returned; the hook is how a test drives an interleaving.
    """

    def __init__(self, responses=None, provider=PROVIDER, on_request=None, **kwargs):
        super().__init__(provider, **kwargs)
        self.requests = []
        self.on_request = on_request
        self._responses = list(
            responses if responses is not None else [{"status": 200, "body": _OK_BODY}]
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


def test_manifest_declares_stt_and_cancel():
    stt = DeepgramSTT(transport=_RecordingTransport(), environ=_env())
    manifest = stt.manifest
    assert manifest["name"] == PROVIDER
    assert manifest["modality"] == "stt"
    assert "cancel" in manifest["capabilities"]
    assert "binary_pcm" in manifest["capabilities"]


def test_request_construction_url_headers_and_body():
    transport = _RecordingTransport()
    stt = DeepgramSTT(
        transport=transport, environ=_env(PARLEVOX_DEEPGRAM_MODEL="nova-3")
    )
    pcm = b"\x01\x00" * 160
    assert list(stt.transcribe(pcm)) == [{"text": "ok", "final": True}]

    assert len(transport.requests) == 1
    request = transport.requests[0]
    assert request["method"] == "POST"
    assert request["path"] == "/v1/listen?model=nova-3&smart_format=true"
    assert request["headers"]["authorization"] == f"Token {PLANTED_KEY}"
    assert request["headers"]["content-type"] == "audio/wav"
    assert request["headers"]["accept"] == "application/json"
    body = request["body"]
    assert body[:4] == b"RIFF"
    assert len(body) == 44 + len(pcm)
    (_fmt, channels, sample_rate, _byte_rate, _align, bits) = _wav_fmt(body)
    assert channels == FAKE_CHANNELS
    assert sample_rate == FAKE_SAMPLE_RATE
    assert bits == 16


def test_default_model_is_nova_2():
    transport = _RecordingTransport()
    stt = DeepgramSTT(transport=transport, environ=_env())
    list(stt.transcribe(b"\x00\x00" * 160))
    assert transport.requests[0]["path"] == "/v1/listen?model=nova-2&smart_format=true"


def test_canned_response_yields_transcript():
    stt = DeepgramSTT(environ=_env())
    assert isinstance(stt._transport, FixtureTransport)
    chunks = list(stt.transcribe(b"\x00\x00" * 160))
    assert chunks == [{"text": "hello world", "final": True}]


def test_caller_rate_and_channels_reach_wav_header():
    transport = _RecordingTransport()
    stt = DeepgramSTT(transport=transport, environ=_env())
    list(stt.transcribe(b"\x00\x00" * 480, sample_rate=48_000, channels=2))
    (audio_format, channels, sample_rate, _byte_rate, _align, bits) = _wav_fmt(
        transport.requests[0]["body"]
    )
    assert (audio_format, channels, sample_rate, bits) == (1, 2, 48_000, 16)

    # The direct-caller path that omits the options keeps the adapter default.
    default_transport = _RecordingTransport()
    list(
        DeepgramSTT(transport=default_transport, environ=_env()).transcribe(
            b"\x00\x00" * 160
        )
    )
    (_fmt, channels, sample_rate, _byte_rate, _align, _bits) = _wav_fmt(
        default_transport.requests[0]["body"]
    )
    assert (channels, sample_rate) == (FAKE_CHANNELS, FAKE_SAMPLE_RATE)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"results": {}},
        {"results": {"channels": []}},
        {"results": {"channels": [{}]}},
        {"results": {"channels": [{"alternatives": []}]}},
        {"results": {"channels": [{"alternatives": [{}]}]}},
        {"results": {"channels": [{"alternatives": [{"transcript": None}]}]}},
        {"results": {"channels": [{"alternatives": "bogus"}]}},
        {"results": "bogus"},
        {"results": None},
    ],
    ids=[
        "empty-object",
        "no-channels",
        "empty-channels",
        "no-alternatives-key",
        "empty-alternatives",
        "no-transcript-key",
        "null-transcript",
        "non-list-alternatives",
        "non-object-results",
        "null-results",
    ],
)
def test_missing_alternatives_yield_empty_transcript(payload):
    transport = _RecordingTransport(
        responses=[{"status": 200, "body": json.dumps(payload).encode("utf-8")}]
    )
    stt = DeepgramSTT(transport=transport, environ=_env())
    assert list(stt.transcribe(b"\x00\x00" * 160)) == [{"text": "", "final": True}]


def test_malformed_json_is_protocol():
    transport = _RecordingTransport(responses=[{"status": 200, "body": b"not json"}])
    stt = DeepgramSTT(transport=transport, environ=_env())
    with pytest.raises(AdapterError) as info:
        list(stt.transcribe(b"\x00\x00" * 160))
    assert info.value.classification is ErrorClass.PROTOCOL
    assert info.value.wire_class() == "protocol"


def _status_transport(status):
    return _RecordingTransport(
        responses=[{"status": status, "body": b"{}"}],
        max_attempts=1,
        sleeper=lambda _seconds: None,
    )


def test_401_maps_to_fatal_and_429_to_rate_limit():
    stt = DeepgramSTT(transport=_status_transport(401), environ=_env())
    with pytest.raises(AdapterError) as fatal:
        list(stt.transcribe(b"\x00\x00" * 160))
    assert fatal.value.classification is ErrorClass.FATAL
    assert fatal.value.wire_class() == "fatal"

    stt = DeepgramSTT(transport=_status_transport(429), environ=_env())
    with pytest.raises(AdapterError) as limited:
        list(stt.transcribe(b"\x00\x00" * 160))
    assert limited.value.classification is ErrorClass.RATE_LIMIT
    assert limited.value.wire_class() == "transient"


def test_missing_key_fails_at_first_use_naming_env_var():
    transport = _RecordingTransport()
    stt = DeepgramSTT(transport=transport, environ={})
    with pytest.raises(MissingCredential) as info:
        list(stt.transcribe(b"\x00\x00" * 160))
    error = info.value
    assert error.classification is ErrorClass.FATAL
    assert error.env_var == "DEEPGRAM_API_KEY"
    assert "DEEPGRAM_API_KEY" in str(error)
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
        stt = DeepgramSTT(transport=transport, environ=_env())
        with pytest.raises(AdapterError) as info:
            list(stt.transcribe(b"\x00\x00" * 160))
        error = info.value
        errors[label] = error
        rendered[label] = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )

    for label, error in errors.items():
        for text in (str(error), repr(error), rendered[label]):
            assert PLANTED_KEY not in text, f"{label} leaked the credential"

    # The transport-failure branch embeds the header in its detail; the
    # registered secret must have been scrubbed there.
    assert "<redacted>" in str(errors["transport"])
    assert errors["transport"].classification is ErrorClass.PROTOCOL
    assert errors["transport"].__cause__ is None
    assert errors["transport"].__context__ is None
    assert PLANTED_KEY not in caplog.text


def test_plugin_threads_caller_rate_into_deepgram_upload():
    transport = _RecordingTransport(
        responses=[
            {
                "status": 200,
                "body": b'{"results": {"channels": [{"alternatives": [{"transcript": "hi"}]}]}}',
            }
        ]
    )
    plugin = Plugin(stt=DeepgramSTT(transport=transport, environ=_env()))
    out = io.BytesIO()
    pending_requests = {}
    pending_audio = {}
    assert plugin.handle(
        {
            "type": "transcribe",
            "request_id": "dg-rate",
            "sample_rate": 48_000,
            "channels": 2,
        },
        out,
        pending_audio=pending_audio,
        pending_requests=pending_requests,
    )
    pending_audio["dg-rate"] = b"\x00\x00" * 480
    assert plugin.handle(
        {"type": "audio_end", "request_id": "dg-rate"},
        out,
        pending_audio=pending_audio,
        pending_requests=pending_requests,
    )

    (_fmt, channels, sample_rate, _byte_rate, _align, bits) = _wav_fmt(
        transport.requests[0]["body"]
    )
    assert (channels, sample_rate, bits) == (2, 48_000, 16)
    finals = [m for m in _messages(out.getvalue()) if m.get("type") == "final"]
    assert finals == [{"type": "final", "request_id": "dg-rate", "text": "hi"}]


def test_fixture_transport_matches_query_stripped_paths():
    exchange = {
        "method": "POST",
        "path": "/v1/listen",
        "responses": [{"status": 200, "chunks": ["ok"]}],
    }
    transport = FixtureTransport([exchange])
    # A query string must not stop the recorded endpoint from matching ...
    assert (
        transport.send("POST", "/v1/listen?model=nova-2&smart_format=true").read()
        == b"ok"
    )
    # ... and the pre-existing path-suffix semantics must be preserved.
    assert transport.send("POST", "/prefix/v1/listen").read() == b"ok"
    with pytest.raises(AdapterError) as info:
        transport.send("POST", "/v1/other?model=nova-2")
    assert info.value.classification is ErrorClass.PROTOCOL
