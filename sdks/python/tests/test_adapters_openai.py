"""OpenAI-wire adapters, selection, and plugin integration.

Every test replays fixtures or injected transports: no test opens a socket.
"""

import io
import json
import logging
import struct
import traceback
from pathlib import Path

import pytest

from parlevox_sdk.adapters import (
    AdapterConfigError,
    AdapterError,
    CancelToken,
    ErrorClass,
    FixtureTransport,
    MissingCredential,
    OpenAILlm,
    OpenAISTT,
    OpenAITTS,
    reset_redaction,
)
from parlevox_sdk.adapters.base import HttpTransport, Transport, TransportResponse
from parlevox_sdk.fakes import FAKE_SAMPLE_RATE, FakeLlm, FakeStt, FakeTts, FakeVad
from parlevox_sdk.plugin import Plugin
from parlevox_sdk.protocol import (
    KIND_AUDIO,
    KIND_JSON,
    decode_audio,
    decode_json,
    read_frame,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "openai"
RETRY_FIXTURE = FIXTURES / "retries" / "llm_429_then_ok.json"
PLANTED_KEY = "sk-test-planted-e1a-000000000000"
FAKE_DELTAS = ["Hello", ", ", "world", "!"]


@pytest.fixture(autouse=True)
def _clean_redaction():
    reset_redaction()
    yield
    reset_redaction()


def _env(**overrides):
    env = {
        "PARLEVOX_OPENAI_API_KEY": PLANTED_KEY,
        "PARLEVOX_FIXTURE_DIR": str(FIXTURES),
    }
    env.update(overrides)
    return env


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


def _retry_transport(max_attempts):
    exchanges = [json.loads(RETRY_FIXTURE.read_text(encoding="utf-8"))]
    return FixtureTransport(
        exchanges,
        provider="openai-llm",
        max_attempts=max_attempts,
        sleeper=lambda _seconds: None,
    )


def test_llm_streams_deltas_then_finishes():
    llm = OpenAILlm(environ=_env())
    assert list(llm.generate("hello world")) == FAKE_DELTAS
    assert llm.manifest["name"] == "openai-llm"
    assert llm.manifest["modality"] == "llm"


def test_llm_429_is_rate_limited_internally_and_retried():
    llm = OpenAILlm(transport=_retry_transport(max_attempts=2), environ=_env())
    assert list(llm.generate("hello world")) == FAKE_DELTAS


def test_llm_exhausted_429_reports_rate_limit_and_transient_wire():
    exchange = {
        "method": "POST",
        "path": "/chat/completions",
        "responses": [{"status": 429, "headers": {"retry-after": "0"}, "chunks": []}],
    }
    transport = FixtureTransport(
        [exchange],
        provider="openai-llm",
        max_attempts=3,
        sleeper=lambda _seconds: None,
    )
    llm = OpenAILlm(transport=transport, environ=_env())
    with pytest.raises(AdapterError) as info:
        list(llm.generate("hello world"))
    assert info.value.classification is ErrorClass.RATE_LIMIT
    assert info.value.wire_class() == "transient"
    assert PLANTED_KEY not in str(info.value)


def test_cancellation_is_observed_mid_stream():
    token = CancelToken()

    def cancel_after_first_chunk(_chunk, index):
        if index == 1:
            token.cancel()

    transport = FixtureTransport.from_directory(
        FIXTURES, provider="openai-llm", on_chunk=cancel_after_first_chunk
    )
    llm = OpenAILlm(transport=transport, environ=_env(), cancel=token)
    seen = []
    with pytest.raises(AdapterError) as info:
        for delta in llm.generate("hello world"):
            seen.append(delta)
    assert info.value.classification is ErrorClass.CANCELLED
    assert info.value.wire_class() == "cancelled"
    assert seen == [FAKE_DELTAS[0]]


def test_stt_yields_one_final_transcript():
    stt = OpenAISTT(environ=_env())
    chunks = list(stt.transcribe(b"\x00\x00" * 160))
    assert chunks == [{"text": "hello world", "final": True}]
    assert stt.manifest["modality"] == "stt"


def test_tts_resamples_and_chunks_to_two_frames():
    tts = OpenAITTS(environ=_env())
    frames = list(tts.synthesize("Hello, world!", sample_rate=16_000, channels=1))
    assert len(frames) == 2
    assert all(len(frame) == 160 * 2 for frame in frames)
    assert tts.manifest["modality"] == "tts"


def test_tts_without_options_keeps_native_rate():
    tts = OpenAITTS(environ=_env())
    frames = list(tts.synthesize("Hello, world!"))
    # 480 native 24 kHz samples chunked at 10 ms (240 samples) -> two frames.
    assert [len(frame) // 2 for frame in frames] == [240, 240]


def test_missing_key_fails_at_first_use_naming_the_env_var():
    llm = OpenAILlm(transport=FixtureTransport.from_directory(FIXTURES), environ={})
    with pytest.raises(MissingCredential) as info:
        list(llm.generate("hello"))
    error = info.value
    assert error.classification is ErrorClass.FATAL
    assert error.env_var == "PARLEVOX_OPENAI_API_KEY"
    assert "PARLEVOX_OPENAI_API_KEY" in str(error)
    assert PLANTED_KEY not in str(error)
    assert PLANTED_KEY not in repr(error)


def test_planted_key_never_reaches_debug_logs(caplog):
    caplog.set_level(logging.DEBUG)
    assert list(OpenAILlm(environ=_env()).generate("hello world")) == FAKE_DELTAS
    assert PLANTED_KEY not in caplog.text


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


def test_from_env_selects_openai_family_and_keeps_fake_vad():
    plugin = Plugin.from_env(
        environ={"PARLEVOX_ADAPTERS": "llm:openai,stt:openai,tts:openai", **_env()}
    )
    assert isinstance(plugin.llm, OpenAILlm)
    assert isinstance(plugin.stt, OpenAISTT)
    assert isinstance(plugin.tts, OpenAITTS)
    assert isinstance(plugin.vad, FakeVad)
    assert plugin.name == "python-openai"
    # The wire manifest shape is unchanged; only the plugin name differs.
    assert plugin.manifest()["modality"] == Plugin().manifest()["modality"]
    assert (
        plugin.manifest()["capabilities"] == Plugin().manifest()["capabilities"]
    )


def test_from_env_rejects_unknown_and_malformed_tokens():
    with pytest.raises(AdapterConfigError) as unknown:
        Plugin.from_env(environ={"PARLEVOX_ADAPTERS": "tts:bogus"})
    assert "tts:bogus" in str(unknown.value)
    assert PLANTED_KEY not in str(unknown.value)

    with pytest.raises(AdapterConfigError) as malformed:
        Plugin.from_env(environ={"PARLEVOX_ADAPTERS": "llmopenai"})
    assert "llmopenai" in str(malformed.value)


def test_plugin_reports_adapter_error_on_the_wire():
    exchange = {
        "method": "POST",
        "path": "/chat/completions",
        "responses": [{"status": 429, "headers": {"retry-after": "0"}, "chunks": []}],
    }
    transport = FixtureTransport(
        [exchange],
        provider="openai-llm",
        max_attempts=1,
        sleeper=lambda _seconds: None,
    )
    plugin = Plugin(llm=OpenAILlm(transport=transport, environ=_env()))
    out = io.BytesIO()
    assert plugin.handle({"type": "generate", "request_id": "r1", "prompt": "hi"}, out)

    errors = [m for m in _messages(out.getvalue()) if m.get("type") == "error"]
    assert len(errors) == 1
    assert errors[0]["class"] == "transient"
    assert "429" in errors[0]["message"]
    assert PLANTED_KEY not in errors[0]["message"]


def test_plugin_synthesizes_through_openai_tts():
    plugin = Plugin(tts=OpenAITTS(environ=_env()))
    out = io.BytesIO()
    assert plugin.handle(
        {
            "type": "synthesize",
            "request_id": "t1",
            "text": "Hello, world!",
            "sample_rate": 16_000,
            "channels": 1,
        },
        out,
    )
    messages = _messages(out.getvalue())
    assert messages[0]["type"] == "audio_start"
    assert messages[0]["sample_rate"] == 16_000
    audio = [m for m in messages if isinstance(m, tuple)]
    assert len(audio) == 2
    assert [header["seq"] for _tag, header, _pcm in audio] == [0, 1]
    assert all(len(pcm) == 160 * 2 for _tag, _header, pcm in audio)
    assert messages[-1]["type"] == "done"
    assert PLANTED_KEY not in repr(messages)


def test_plugin_transcribes_through_openai_stt():
    plugin = Plugin(stt=OpenAISTT(environ=_env()))
    out = io.BytesIO()
    pending = {"s1": {"type": "transcribe", "request_id": "s1"}}
    buffered = {"s1": b"\x00\x00" * 160}
    assert plugin.handle(
        {"type": "audio_end", "request_id": "s1"},
        out,
        pending_audio=buffered,
        pending_requests=pending,
    )
    finals = [m for m in _messages(out.getvalue()) if m.get("type") == "final"]
    assert finals == [{"type": "final", "request_id": "s1", "text": "hello world"}]


def test_wire_error_redacts_control_character_key():
    """A newline in the key reaches http.client's ValueError; scrub it whole.

    ``putheader`` refuses the illegal header value before any socket is
    opened, and its ``repr()``-rendered ``ValueError`` carries the credential
    with the newline escaped to a literal ``\\n``. Neither form — nor any
    prefix of either — may appear on the wire.
    """
    key = "sk-test-planted-\n-secret-000000000000"
    escaped = key.encode("unicode_escape").decode("ascii")
    plugin = Plugin(
        llm=OpenAILlm(
            environ={
                "PARLEVOX_OPENAI_API_KEY": key,
                "PARLEVOX_OPENAI_BASE_URL": "http://127.0.0.1:9/v1",
            }
        )
    )
    out = io.BytesIO()
    assert plugin.handle({"type": "generate", "request_id": "p2", "prompt": "hi"}, out)

    events = [m for m in _messages(out.getvalue()) if isinstance(m, dict)]
    assert events and events[-1]["type"] == "error"
    assert events[-1]["class"] == "protocol"
    wire = "\n".join(str(event.get("message", "")) for event in events)
    assert "<redacted>" in wire
    assert key not in wire
    assert escaped not in wire
    for length in range(5, len(key) + 1):
        assert key[:length] not in wire, f"raw key prefix of length {length} leaked"
    for length in range(5, len(escaped) + 1):
        assert escaped[:length] not in wire, f"escaped prefix of length {length} leaked"


def test_direct_generate_control_char_key_is_protocol_and_redacted():
    """A control character in the resolved key must not escape unclassified.

    ``http.client`` renders the illegal header value with ``repr()`` — the
    credential with the control character escaped — and raises ``ValueError``
    before a socket is opened. The adapter must turn that into a ``protocol``
    ``AdapterError`` whose message, ``repr`` and formatted traceback contain
    neither the raw key nor its ``unicode_escape`` form.
    """
    key = "sk-test-planted-e1a-000000000000\n"
    escaped = key.encode("unicode_escape").decode("ascii")
    llm = OpenAILlm(
        base_url="http://127.0.0.1:9",
        environ={"PARLEVOX_OPENAI_API_KEY": key},
    )
    with pytest.raises(AdapterError) as info:
        list(llm.generate("hi"))

    error = info.value
    assert error.classification is ErrorClass.PROTOCOL
    assert error.wire_class() == "protocol"
    assert error.__cause__ is None
    assert error.__context__ is None
    rendered = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    for text in (str(error), repr(error), rendered):
        assert key not in text
        assert escaped not in text
    assert "<redacted>" in str(error)


def _rejecting_transport(value):
    """An HTTP transport whose connector replays ``http.client``'s rejection.

    CPython's ``_is_illegal_header_value`` rejects ``\\n`` and ``\\r`` before a
    socket is opened but permits NUL, so a NUL-bearing key cannot reach the real
    rejection without connecting. The connector raises the same ``ValueError``
    shape for every control character, keeping the test socket-free.
    """

    def connector(method, path, headers, body):
        raise ValueError(f"Invalid header value {headers['authorization']!r}")

    return HttpTransport(
        "http://127.0.0.1:9",
        "openai-llm",
        connector=connector,
        max_attempts=3,
        sleeper=lambda _seconds: None,
    )


@pytest.mark.parametrize("shape", ["suffixed", "embedded"], ids=["suffix", "embedded"])
@pytest.mark.parametrize("control", ["\n", "\r", "\x00"], ids=["lf", "cr", "nul"])
def test_every_illegal_header_character_is_classified_protocol(control, shape):
    core = "sk-test-planted-e1a-000000000000"
    key = core + control if shape == "suffixed" else core[:12] + control + core[12:]
    escaped = key.encode("unicode_escape").decode("ascii")
    llm = OpenAILlm(
        transport=_rejecting_transport(key),
        environ={"PARLEVOX_OPENAI_API_KEY": key},
    )
    with pytest.raises(AdapterError) as info:
        list(llm.generate("hi"))

    error = info.value
    assert error.classification is ErrorClass.PROTOCOL
    assert error.__cause__ is None
    assert error.__context__ is None
    rendered = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    for text in (str(error), repr(error), rendered):
        assert key not in text
        assert escaped not in text


class _RecordingTransport(Transport):
    """Capture the request instead of sending it; never opens a socket."""

    def __init__(self):
        super().__init__("openai-stt")
        self.requests = []

    def _send_once(self, method, path, *, headers, body, cancel):
        self.requests.append(
            {"method": method, "path": path, "headers": headers, "body": body}
        )
        return TransportResponse(
            200,
            {"content-type": "application/json"},
            [b'{"text": "ok"}'],
            provider="openai-stt",
        )


def test_stt_uploads_wav_multipart():
    transport = _RecordingTransport()
    stt = OpenAISTT(transport=transport, environ=_env())
    pcm = b"\x01\x00" * 160
    assert list(stt.transcribe(pcm)) == [{"text": "ok", "final": True}]

    request = transport.requests[0]
    assert request["method"] == "POST"
    assert request["path"] == "/audio/transcriptions"
    content_type = request["headers"]["content-type"]
    assert content_type.startswith("multipart/form-data; boundary=")
    boundary = content_type.split("boundary=", 1)[1]

    body = request["body"]
    assert b'name="model"' in body
    assert b"whisper-1" in body
    marker = b'filename="audio.wav"\r\nContent-Type: audio/wav\r\n\r\n'
    assert marker in body
    payload = body.split(marker, 1)[1]
    payload = payload.rsplit(b"\r\n--" + boundary.encode("ascii") + b"--\r\n", 1)[0]

    assert payload[:4] == b"RIFF"
    assert payload[8:12] == b"WAVE"
    assert b"fmt " in payload
    assert b"data" in payload
    (audio_format, channels, sample_rate, _byte_rate, _align, bits) = struct.unpack(
        "<HHIIHH", payload[20:36]
    )
    assert audio_format == 1  # PCM
    assert channels == 1  # mono
    assert sample_rate == FAKE_SAMPLE_RATE
    assert bits == 16
    assert len(payload) == 44 + len(pcm)


def test_plugin_threads_caller_rate_into_stt_upload():
    """The WAV header must declare the rate/channels the core sent, not defaults.

    The Rust core puts the utterance's own format on ``Transcribe``; a header
    that always claims :data:`FAKE_SAMPLE_RATE` mono would silently mis-declare
    the audio for every other session rate.
    """
    transport = _RecordingTransport()
    plugin = Plugin(stt=OpenAISTT(transport=transport, environ=_env()))
    out = io.BytesIO()
    pending_requests = {}
    pending_audio = {}
    assert plugin.handle(
        {
            "type": "transcribe",
            "request_id": "s-rate",
            "sample_rate": 48_000,
            "channels": 2,
        },
        out,
        pending_audio=pending_audio,
        pending_requests=pending_requests,
    )
    pcm = b"\x00\x00" * 160
    pending_audio["s-rate"] = pcm
    assert plugin.handle(
        {"type": "audio_end", "request_id": "s-rate"},
        out,
        pending_audio=pending_audio,
        pending_requests=pending_requests,
    )

    request = transport.requests[0]
    content_type = request["headers"]["content-type"]
    boundary = content_type.split("boundary=", 1)[1]
    marker = b'filename="audio.wav"\r\nContent-Type: audio/wav\r\n\r\n'
    payload = request["body"].split(marker, 1)[1]
    payload = payload.rsplit(b"\r\n--" + boundary.encode("ascii") + b"--\r\n", 1)[0]

    (audio_format, channels, sample_rate, _byte_rate, _align, bits) = struct.unpack(
        "<HHIIHH", payload[20:36]
    )
    assert audio_format == 1  # PCM
    assert channels == 2
    assert sample_rate == 48_000
    assert bits == 16
    assert len(payload) == 44 + len(pcm)
