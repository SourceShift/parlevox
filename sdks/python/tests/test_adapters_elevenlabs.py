"""ElevenLabs text-to-speech adapter: wire shape, chunking, and redaction.

Every test replays a fixture or an injected transport: none opens a socket,
and the credential literal below is an obvious fake.
"""

import base64
import io
import json
import logging
import traceback
from pathlib import Path

import pytest

from parlevox_sdk.adapters import (
    AdapterError,
    ElevenLabsTTS,
    ErrorClass,
    FixtureTransport,
    MissingCredential,
    reset_redaction,
)
from parlevox_sdk.adapters.base import Transport, TransportResponse
from parlevox_sdk.fakes import (
    FakeLlm,
    FakeStt,
    FakeTts,
    FakeVad,
)
from parlevox_sdk.plugin import Plugin
from parlevox_sdk.protocol import (
    KIND_AUDIO,
    KIND_JSON,
    decode_audio,
    decode_json,
    read_frame,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "elevenlabs"
PLANTED_KEY = "sk-elevenlabs-LEAKCHECK"
PROVIDER = "elevenlabs-tts"
VOICE = "21m00Tcm4TlvDq8ikWAM"
DEFAULT_MODEL = "eleven_multilingual_v2"

_FIXTURE = json.loads((FIXTURES / "tts_stream.json").read_text(encoding="utf-8"))
FIXTURE_BODY = base64.b64decode(_FIXTURE["responses"][0]["body_base64"])
_OK_BODY = b"\x01\x00" * 8


@pytest.fixture(autouse=True)
def _clean_redaction():
    reset_redaction()
    yield
    reset_redaction()


def _env(**overrides):
    env = {"ELEVENLABS_API_KEY": PLANTED_KEY, "PARLEVOX_FIXTURE_DIR": str(FIXTURES)}
    env.update(overrides)
    return env


class _RecordingTransport(Transport):
    """Capture each request and replay queued bodies; never opens a socket."""

    def __init__(self, responses=None, on_request=None, **kwargs):
        super().__init__(PROVIDER, **kwargs)
        self.requests = []
        self.on_request = on_request
        self._responses = list(
            responses
            if responses is not None
            else [{"status": 200, "body": _OK_BODY}]
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
            spec.get("headers") or {"content-type": "audio/pcm"},
            [spec.get("body", b"")],
            provider=self.provider,
        )


class _RejectingTransport(Transport):
    """Replay ``http.client``'s illegal-header ``ValueError`` without a socket."""

    def __init__(self):
        super().__init__(PROVIDER, max_attempts=1, sleeper=lambda _seconds: None)

    def _send_once(self, method, path, *, headers, body, cancel):
        raise ValueError(f"Invalid header value {headers['xi-api-key']!r}")


def _status_transport(status):
    return _RecordingTransport(
        responses=[{"status": status, "body": b""}],
        max_attempts=1,
        sleeper=lambda _seconds: None,
    )


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


def test_manifest_declares_tts_streaming_and_cancel():
    tts = ElevenLabsTTS(transport=_RecordingTransport(), environ=_env())
    manifest = tts.manifest
    assert manifest["name"] == PROVIDER
    assert manifest["modality"] == "tts"
    assert "streaming" in manifest["capabilities"]
    assert "binary_pcm" in manifest["capabilities"]
    assert "cancel" in manifest["capabilities"]


def test_request_construction_path_headers_and_body():
    transport = _RecordingTransport()
    tts = ElevenLabsTTS(transport=transport, environ=_env())
    assert b"".join(tts.synthesize("Hello, world!", sample_rate=16_000, channels=1)) == _OK_BODY

    assert len(transport.requests) == 1
    request = transport.requests[0]
    assert request["method"] == "POST"
    assert (
        request["path"]
        == f"/v1/text-to-speech/{VOICE}/stream?output_format=pcm_16000"
    )
    assert request["headers"]["xi-api-key"] == PLANTED_KEY
    assert request["headers"]["content-type"] == "application/json"
    assert request["headers"]["accept"] == "audio/pcm"
    assert json.loads(request["body"]) == {
        "text": "Hello, world!",
        "model_id": DEFAULT_MODEL,
    }


def test_env_overrides_voice_and_model():
    transport = _RecordingTransport()
    tts = ElevenLabsTTS(
        transport=transport,
        environ=_env(
            PARLEVOX_ELEVENLABS_VOICE_ID="voice-from-env",
            PARLEVOX_ELEVENLABS_MODEL="model-from-env",
        ),
    )
    list(tts.synthesize("hi", sample_rate=16_000, channels=1))
    request = transport.requests[0]
    assert request["path"] == (
        "/v1/text-to-speech/voice-from-env/stream?output_format=pcm_16000"
    )
    assert json.loads(request["body"])["model_id"] == "model-from-env"


def test_fixture_replay_rechunks_response_body():
    tts = ElevenLabsTTS(environ=_env())
    assert isinstance(tts._transport, FixtureTransport)
    frames = list(tts.synthesize("Hello", sample_rate=16_000, channels=1))
    assert b"".join(frames) == FIXTURE_BODY
    # 480 bytes at 16 kHz is 480 samples: one 320-byte frame, one 160-byte.
    assert [len(frame) for frame in frames] == [320, 160]


def test_caller_rate_reaches_output_format():
    transport = _RecordingTransport()
    tts = ElevenLabsTTS(transport=transport, environ=_env())
    list(tts.synthesize("hello", sample_rate=48_000, channels=1))
    request = transport.requests[0]
    assert "output_format=pcm_48000" in request["path"]
    assert "pcm_16000" not in request["path"]


def test_401_is_fatal_and_429_is_rate_limited():
    tts = ElevenLabsTTS(transport=_status_transport(401), environ=_env())
    with pytest.raises(AdapterError) as fatal:
        list(tts.synthesize("hello", sample_rate=16_000, channels=1))
    assert fatal.value.classification is ErrorClass.FATAL
    assert fatal.value.wire_class() == "fatal"

    tts = ElevenLabsTTS(transport=_status_transport(429), environ=_env())
    with pytest.raises(AdapterError) as limited:
        list(tts.synthesize("hello", sample_rate=16_000, channels=1))
    assert limited.value.classification is ErrorClass.RATE_LIMIT
    assert limited.value.wire_class() == "transient"


def test_missing_key_fails_at_first_use_naming_env_var():
    transport = _RecordingTransport()
    tts = ElevenLabsTTS(transport=transport, environ={})
    with pytest.raises(MissingCredential) as info:
        list(tts.synthesize("hello", sample_rate=16_000, channels=1))
    error = info.value
    assert error.classification is ErrorClass.FATAL
    assert error.env_var == "ELEVENLABS_API_KEY"
    assert "ELEVENLABS_API_KEY" in str(error)
    assert PLANTED_KEY not in str(error)
    assert transport.requests == []


def test_multi_channel_is_protocol_error_naming_mono():
    transport = _RecordingTransport()
    tts = ElevenLabsTTS(transport=transport, environ=_env())
    with pytest.raises(AdapterError) as info:
        list(tts.synthesize("hello", sample_rate=16_000, channels=2))
    assert info.value.classification is ErrorClass.PROTOCOL
    assert info.value.wire_class() == "protocol"
    assert "mono" in str(info.value)
    assert "2" in str(info.value)
    assert transport.requests == []


def test_empty_text_and_nonpositive_formats_are_value_errors():
    tts = ElevenLabsTTS(transport=_RecordingTransport(), environ=_env())
    with pytest.raises(ValueError):
        list(tts.synthesize("   ", sample_rate=16_000, channels=1))
    with pytest.raises(ValueError):
        list(tts.synthesize("hello", sample_rate=0, channels=1))
    with pytest.raises(ValueError):
        list(tts.synthesize("hello", sample_rate=16_000, channels=0))


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
        tts = ElevenLabsTTS(transport=transport, environ=_env())
        with pytest.raises(AdapterError) as info:
            list(tts.synthesize("hello", sample_rate=16_000, channels=1))
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


def test_plugin_synthesizes_and_threads_caller_rate():
    transport = _RecordingTransport()
    plugin = Plugin(tts=ElevenLabsTTS(transport=transport, environ=_env()))
    out = io.BytesIO()
    assert plugin.handle(
        {
            "type": "synthesize",
            "request_id": "el1",
            "text": "Hello",
            "sample_rate": 48_000,
            "channels": 1,
        },
        out,
    )

    assert "output_format=pcm_48000" in transport.requests[0]["path"]
    messages = _messages(out.getvalue())
    assert messages[0]["type"] == "audio_start"
    assert messages[0]["sample_rate"] == 48_000
    audio = [message for message in messages if isinstance(message, tuple)]
    assert b"".join(pcm for _tag, _header, pcm in audio) == _OK_BODY
    assert messages[-1]["type"] == "done"
    assert PLANTED_KEY not in repr(messages)


def test_from_env_selects_elevenlabs_and_keeps_fake_vad():
    plugin = Plugin.from_env(
        environ={"PARLEVOX_ADAPTERS": "tts:elevenlabs", **_env()}
    )
    assert isinstance(plugin.tts, ElevenLabsTTS)
    assert isinstance(plugin.vad, FakeVad)
    assert plugin.name == "python-elevenlabs"
    assert (
        plugin.manifest()["capabilities"] == Plugin().manifest()["capabilities"]
    )


def test_from_env_without_variable_is_identical_to_default():
    default = Plugin()
    for environ in (
        {},
        {"PARLEVOX_ADAPTERS": ""},
        {"PARLEVOX_ADAPTERS": "llm:fake,stt:fake,tts:fake,vad:fake"},
    ):
        plugin = Plugin.from_env(environ=environ)
        assert plugin.name == default.name
        assert plugin._ready_payload() == default._ready_payload()
        assert isinstance(plugin.stt, FakeStt)
        assert isinstance(plugin.llm, FakeLlm)
        assert isinstance(plugin.tts, FakeTts)
        assert isinstance(plugin.vad, FakeVad)
