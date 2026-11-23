"""Anthropic Messages LLM adapter: wire shape, SSE parsing and redaction.

Every test replays a fixture or an injected transport: none opens a socket,
and the credential literal below is an obvious fake.
"""

import io
import json
import logging
import traceback
from pathlib import Path

import pytest

from parlevox_sdk.adapters import (
    AdapterError,
    AnthropicLlm,
    CancelToken,
    ErrorClass,
    FixtureTransport,
    MissingCredential,
    reset_redaction,
)
from parlevox_sdk.adapters.base import Transport, TransportResponse
from parlevox_sdk.fakes import FakeLlm, FakeStt, FakeTts, FakeVad
from parlevox_sdk.plugin import Plugin
from parlevox_sdk.protocol import (
    KIND_AUDIO,
    KIND_JSON,
    decode_audio,
    decode_json,
    read_frame,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "anthropic"
HAPPY = FIXTURES / "happy"
ERROR = FIXTURES / "error"
RATE_LIMITED = FIXTURES / "rate_limited"
TRUNCATED = FIXTURES / "truncated"

PLANTED_KEY = "sk-ant-test-planted-e1a4-000000000000"
PROVIDER = "anthropic-llm"
DEFAULT_MODEL = "claude-opus-5-5"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096
FAKE_DELTAS = ["Hello", ", ", "world", "!"]

_HAPPY_EXCHANGE = json.loads((HAPPY / "messages.json").read_text(encoding="utf-8"))
HAPPY_CHUNKS = _HAPPY_EXCHANGE["responses"][0]["chunks"]


@pytest.fixture(autouse=True)
def _clean_redaction():
    reset_redaction()
    yield
    reset_redaction()


def _env(fixture=HAPPY, **overrides):
    env = {
        "ANTHROPIC_API_KEY": PLANTED_KEY,
        "PARLEVOX_FIXTURE_DIR": str(fixture),
    }
    env.update(overrides)
    return env


class _RecordingTransport(Transport):
    """Capture each request and replay queued chunks; never opens a socket."""

    def __init__(self, *, status=200, headers=None, chunks=(), **kwargs):
        super().__init__(PROVIDER, **kwargs)
        self.requests = []
        self._status = status
        self._headers = headers or {"content-type": "text/event-stream"}
        self._chunks = [
            chunk.encode("utf-8") if isinstance(chunk, str) else chunk
            for chunk in chunks
        ]

    def _send_once(self, method, path, *, headers, body, cancel):
        self.requests.append(
            {
                "method": method,
                "path": path,
                "headers": dict(headers),
                "body": body,
            }
        )
        return TransportResponse(
            self._status,
            self._headers,
            list(self._chunks),
            provider=self.provider,
        )


class _RejectingTransport(Transport):
    """Replay ``http.client``'s illegal-header ``ValueError`` without a socket."""

    def __init__(self):
        super().__init__(PROVIDER, max_attempts=1, sleeper=lambda _seconds: None)

    def _send_once(self, method, path, *, headers, body, cancel):
        raise ValueError(f"Invalid header value {headers['x-api-key']!r}")


def _status_transport(status, headers=None):
    return _RecordingTransport(
        status=status,
        headers=headers or {},
        max_attempts=1,
        sleeper=lambda _seconds: None,
    )


def _text_event(text):
    return (
        "event: content_block_delta\n"
        'data: {"type":"content_block_delta","index":0,'
        f'"delta":{{"type":"text_delta","text":"{text}"}}}}\n\n'
    )


def _plugin_messages(blob):
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


def test_manifest_declares_llm_streaming_and_cancel():
    llm = AnthropicLlm(transport=_RecordingTransport(), environ=_env())
    manifest = llm.manifest
    assert manifest["name"] == PROVIDER
    assert manifest["modality"] == "llm"
    assert "streaming" in manifest["capabilities"]
    assert "cancel" in manifest["capabilities"]
    assert "history" in manifest["capabilities"]


def test_request_construction_path_headers_and_body():
    transport = _RecordingTransport(chunks=HAPPY_CHUNKS)
    llm = AnthropicLlm(transport=transport, environ=_env())
    assert list(llm.generate("hello world")) == FAKE_DELTAS

    assert len(transport.requests) == 1
    request = transport.requests[0]
    assert request["method"] == "POST"
    assert request["path"] == "/v1/messages"
    assert request["headers"]["x-api-key"] == PLANTED_KEY
    assert request["headers"]["anthropic-version"] == ANTHROPIC_VERSION
    assert request["headers"]["content-type"] == "application/json"
    assert request["headers"]["accept"] == "text/event-stream"
    assert json.loads(request["body"]) == {
        "model": DEFAULT_MODEL,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "messages": [{"role": "user", "content": "hello world"}],
        "stream": True,
    }


def test_env_overrides_model_and_base_url():
    transport = _RecordingTransport()
    llm = AnthropicLlm(
        transport=transport,
        environ=_env(
            PARLEVOX_ANTHROPIC_MODEL="model-from-env",
            PARLEVOX_ANTHROPIC_BASE_URL="http://127.0.0.1:9",
        ),
    )
    assert llm.base_url == "http://127.0.0.1:9"
    list(llm.generate("hi"))
    assert json.loads(transport.requests[0]["body"])["model"] == "model-from-env"

    configured = AnthropicLlm(
        transport=_RecordingTransport(), environ=_env(), max_tokens=64
    )
    assert configured.max_tokens == 64


def test_happy_stream_yields_deltas_in_order_and_stops_at_message_stop():
    llm = AnthropicLlm(environ=_env())
    assert isinstance(llm._transport, FixtureTransport)
    assert list(llm.generate("hello world")) == FAKE_DELTAS


def test_only_text_deltas_are_yielded():
    deltas = list(AnthropicLlm(environ=_env()).generate("hello world"))
    assert deltas == FAKE_DELTAS
    assert "pondering" not in deltas
    assert "" not in deltas
    # Everything after ``message_stop`` is never read.
    assert "after-stop" not in deltas


def test_system_prompt_goes_top_level_and_never_inside_messages():
    transport = _RecordingTransport()
    llm = AnthropicLlm(transport=transport, environ=_env())
    history = [
        {"role": "system", "text": "You are a voice assistant."},
        {"role": "user", "text": "Hi"},
        {"role": "assistant", "text": "Hello!"},
        {"role": "system", "text": "Be brief."},
    ]
    list(llm.generate("What time is it?", history=history))

    body = json.loads(transport.requests[0]["body"])
    assert body["system"] == "You are a voice assistant.\n\nBe brief."
    assert body["messages"] == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "What time is it?"},
    ]
    assert all(message["role"] != "system" for message in body["messages"])
    assert b'"role": "system"' not in transport.requests[0]["body"]


def test_plugin_generate_hoists_system_history_to_top_level():
    transport = _RecordingTransport(chunks=HAPPY_CHUNKS)
    plugin = Plugin(llm=AnthropicLlm(transport=transport, environ=_env()))
    out = io.BytesIO()
    assert plugin.handle(
        {
            "type": "generate",
            "request_id": "g1",
            "prompt": "hello world",
            "history": [
                {"role": "system", "text": "Be brief."},
                {"role": "user", "text": "Hi"},
            ],
        },
        out,
    )

    body = json.loads(transport.requests[0]["body"])
    assert body["system"] == "Be brief."
    assert all(message["role"] != "system" for message in body["messages"])

    messages = _plugin_messages(out.getvalue())
    assert [m["text"] for m in messages if m["type"] == "delta"] == FAKE_DELTAS
    assert messages[-1]["type"] == "done"
    assert PLANTED_KEY not in repr(messages)


def test_truncated_stream_yields_every_delta_without_protocol_error():
    llm = AnthropicLlm(environ=_env(TRUNCATED))
    # No ``message_stop`` and no trailing blank line: the final delta still
    # arrives and EOF ends the stream without a protocol error.
    assert list(llm.generate("hello")) == ["Hello", " world", "cut"]


def test_stream_error_event_is_classified_and_keeps_prior_deltas():
    llm = AnthropicLlm(environ=_env(ERROR))
    seen = []
    with pytest.raises(AdapterError) as info:
        for delta in llm.generate("hello"):
            seen.append(delta)
    assert seen == ["Partial"]
    assert info.value.classification is ErrorClass.TRANSIENT
    assert info.value.wire_class() == "transient"


def test_malformed_event_is_protocol_after_prior_deltas():
    transport = _RecordingTransport(
        chunks=[
            _text_event("kept"),
            "event: content_block_delta\ndata: {not json}\n\n",
        ]
    )
    llm = AnthropicLlm(transport=transport, environ=_env())
    seen = []
    with pytest.raises(AdapterError) as info:
        for delta in llm.generate("hello"):
            seen.append(delta)
    assert seen == ["kept"]
    assert info.value.classification is ErrorClass.PROTOCOL
    assert info.value.wire_class() == "protocol"
    assert "not json" not in str(info.value)


@pytest.mark.parametrize(
    "status,classification,wire",
    [
        (400, ErrorClass.PROTOCOL, "protocol"),
        (401, ErrorClass.FATAL, "fatal"),
        (403, ErrorClass.FATAL, "fatal"),
        (429, ErrorClass.RATE_LIMIT, "transient"),
        (500, ErrorClass.TRANSIENT, "transient"),
        (503, ErrorClass.TRANSIENT, "transient"),
    ],
    ids=["400", "401", "403", "429", "500", "503"],
)
def test_status_mapping(status, classification, wire):
    llm = AnthropicLlm(transport=_status_transport(status), environ=_env())
    with pytest.raises(AdapterError) as info:
        list(llm.generate("hello"))
    assert info.value.classification is classification
    assert info.value.wire_class() == wire
    assert PLANTED_KEY not in str(info.value)


def test_429_reports_numeric_retry_after():
    transport = FixtureTransport.from_directory(
        RATE_LIMITED,
        provider=PROVIDER,
        max_attempts=1,
        sleeper=lambda _seconds: None,
    )
    llm = AnthropicLlm(transport=transport, environ=_env())
    with pytest.raises(AdapterError) as info:
        list(llm.generate("hello"))
    assert info.value.classification is ErrorClass.RATE_LIMIT
    assert info.value.retry_after == 0.0


def test_cancellation_stops_between_events_in_one_chunk():
    """Both events arrive in one chunk; the cancel lands between them."""
    token = CancelToken()
    transport = _RecordingTransport(chunks=[_text_event("first") + _text_event("second")])
    llm = AnthropicLlm(transport=transport, environ=_env(), cancel=token)

    seen = []
    with pytest.raises(AdapterError) as info:
        for delta in llm.generate("hello"):
            seen.append(delta)
            token.cancel()

    assert seen == ["first"]
    assert info.value.classification is ErrorClass.CANCELLED
    assert info.value.wire_class() == "cancelled"
    assert len(transport.requests) == 1


def test_missing_key_fails_at_first_use_naming_env_var():
    transport = _RecordingTransport()
    llm = AnthropicLlm(transport=transport, environ={})
    with pytest.raises(MissingCredential) as info:
        list(llm.generate("hello"))
    error = info.value
    assert error.classification is ErrorClass.FATAL
    assert error.env_var == "ANTHROPIC_API_KEY"
    assert "ANTHROPIC_API_KEY" in str(error)
    assert PLANTED_KEY not in str(error)
    assert transport.requests == []


def test_planted_key_never_appears_in_failure_text(caplog):
    caplog.set_level(logging.DEBUG)
    cases = {
        "401": _status_transport(401),
        "429": FixtureTransport.from_directory(
            RATE_LIMITED,
            provider=PROVIDER,
            max_attempts=1,
            sleeper=lambda _seconds: None,
        ),
        "stream_error": FixtureTransport.from_directory(ERROR, provider=PROVIDER),
        "rejected_header": _RejectingTransport(),
    }
    errors = {}
    rendered = {}
    for label, transport in cases.items():
        llm = AnthropicLlm(transport=transport, environ=_env())
        with pytest.raises(AdapterError) as info:
            list(llm.generate("hello"))
        error = info.value
        errors[label] = error
        rendered[label] = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )

    for label, error in errors.items():
        for text in (str(error), repr(error), rendered[label]):
            assert PLANTED_KEY not in text, f"{label} leaked the credential"

    assert "<redacted>" in str(errors["rejected_header"])
    assert errors["rejected_header"].classification is ErrorClass.PROTOCOL
    assert errors["rejected_header"].__cause__ is None
    assert errors["rejected_header"].__context__ is None
    assert PLANTED_KEY not in caplog.text


def test_rejected_x_api_key_header_is_protocol_and_redacted():
    key = "sk-ant-test-planted-e1a4-000000000000\n"
    escaped = key.encode("unicode_escape").decode("ascii")
    llm = AnthropicLlm(
        transport=_RejectingTransport(),
        environ={"ANTHROPIC_API_KEY": key},
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


def test_from_env_selects_anthropic_and_derives_plugin_name():
    plugin = Plugin.from_env(
        environ={"PARLEVOX_ADAPTERS": "llm:anthropic", **_env()}
    )
    assert isinstance(plugin.llm, AnthropicLlm)
    assert isinstance(plugin.vad, FakeVad)
    assert plugin.name == "python-anthropic"
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
