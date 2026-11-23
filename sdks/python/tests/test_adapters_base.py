"""Adapter base invariants: classification, retries, redaction, no sockets."""

import http.client
import json
import threading
import time
import traceback
from pathlib import Path

import pytest

from parlevox_sdk.adapters.base import (
    AdapterError,
    CancelToken,
    ErrorClass,
    FixtureTransport,
    HttpTransport,
    MissingCredential,
    Transport,
    TransportResponse,
    classify_status,
    redact,
    register_secret,
    reset_redaction,
    resolve_key,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "openai"
PLANTED_KEY = "sk-test-planted-e1a-000000000000"


@pytest.fixture(autouse=True)
def _clean_redaction():
    reset_redaction()
    yield
    reset_redaction()


def test_status_classification_is_total_over_http_codes():
    for status in range(100, 600):
        classification = classify_status(status)
        if 200 <= status < 300:
            assert classification is None
        else:
            assert isinstance(classification, ErrorClass), status

    assert classify_status(429) is ErrorClass.RATE_LIMIT
    assert classify_status(429).wire_class() == "transient"
    assert classify_status(401) is ErrorClass.FATAL
    assert classify_status(400) is ErrorClass.PROTOCOL
    assert classify_status(408) is ErrorClass.TRANSIENT
    assert classify_status(413) is ErrorClass.CAPACITY
    assert classify_status(503) is ErrorClass.TRANSIENT
    assert classify_status(507) is ErrorClass.CAPACITY


def test_rate_limit_reports_transient_on_the_wire():
    error = AdapterError("openai", "throttled", ErrorClass.RATE_LIMIT)
    assert error.classification is ErrorClass.RATE_LIMIT
    assert error.wire_class() == "transient"


class _CannedTransport(Transport):
    """Transport whose ``_send_once`` returns planted statuses, no socket."""

    def __init__(self, statuses, *, retry_after="0", **kwargs):
        self.delays = []
        kwargs.setdefault("sleeper", self.delays.append)
        super().__init__("canned", **kwargs)
        self._statuses = list(statuses)
        self._retry_after = retry_after
        self.calls = 0

    def _send_once(self, method, path, *, headers, body, cancel):
        self.calls += 1
        index = min(self.calls - 1, len(self._statuses) - 1)
        response_headers = (
            {} if self._retry_after is None else {"retry-after": self._retry_after}
        )
        return TransportResponse(
            self._statuses[index], response_headers, [], provider="canned"
        )


@pytest.mark.parametrize("configured", [0, 1, 2, 3, 4, 5])
def test_retry_attempts_are_exactly_bounded(configured):
    transport = _CannedTransport([429], max_attempts=configured)
    response = transport.send("POST", "/x")
    assert response.status == 429
    assert transport.calls == max(1, configured)
    assert len(transport.delays) == max(0, max(1, configured) - 1)


def test_cancellation_during_backoff_is_prompt():
    transport = _CannedTransport(
        [503],
        retry_after=None,
        max_attempts=5,
        backoff_base=1.0,
        backoff_cap=2.0,
    )
    token = CancelToken()
    threading.Timer(0.02, token.cancel).start()
    started = time.monotonic()
    with pytest.raises(AdapterError) as info:
        transport.send("POST", "/x", cancel=token)
    elapsed = time.monotonic() - started
    assert info.value.classification is ErrorClass.CANCELLED
    assert elapsed < 0.5
    assert transport.calls == 1


def test_redaction_is_a_property_of_every_message_shape():
    register_secret(PLANTED_KEY)
    shapes = [
        f"provider rejected key {PLANTED_KEY}",
        f"Authorization: Bearer {PLANTED_KEY}",
        f"https://api.example/v1?api_key={PLANTED_KEY}",
        repr({"api_key": PLANTED_KEY, "nested": [PLANTED_KEY]}),
    ]
    for shape in shapes:
        error = AdapterError("openai", shape, ErrorClass.FATAL)
        assert PLANTED_KEY not in str(error)
        assert PLANTED_KEY not in repr(error)
        try:
            raise error
        except AdapterError as caught:
            formatted = "".join(
                traceback.format_exception(type(caught), caught, caught.__traceback__)
            )
        assert PLANTED_KEY not in formatted
        assert "<redacted>" in str(error)

    # Defence in depth: a credential shape never resolved through resolve_key
    # is still scrubbed.
    assert PLANTED_KEY not in redact(f"leaked {PLANTED_KEY}")


def test_missing_credential_names_only_the_env_var():
    with pytest.raises(MissingCredential) as info:
        resolve_key("PARLEVOX_OPENAI_API_KEY", environ={})
    error = info.value
    assert error.classification is ErrorClass.FATAL
    assert error.env_var == "PARLEVOX_OPENAI_API_KEY"
    assert "PARLEVOX_OPENAI_API_KEY" in str(error)
    assert PLANTED_KEY not in str(error)


def test_fixture_mode_never_constructs_an_http_connection(monkeypatch):
    def _boom(*args, **kwargs):  # pragma: no cover - only called on failure
        raise AssertionError("fixture mode must not open a socket")

    monkeypatch.setattr(http.client, "HTTPConnection", _boom)
    monkeypatch.setattr(http.client, "HTTPSConnection", _boom)

    transport = FixtureTransport.from_directory(FIXTURES)
    response = transport.send("POST", "/audio/speech")
    assert response.status == 200
    assert response.read()


def test_fixture_transport_matches_method_and_path_suffix():
    exchange = {
        "method": "POST",
        "path": "/chat/completions",
        "responses": [{"status": 200, "chunks": ["ok"]}],
    }
    transport = FixtureTransport([exchange])
    assert transport.send("POST", "/v1/chat/completions").read() == b"ok"
    with pytest.raises(AdapterError) as info:
        transport.send("POST", "/audio/speech")
    assert info.value.classification is ErrorClass.PROTOCOL


def test_fixture_responses_advance_then_repeat():
    exchange = {
        "method": "POST",
        "path": "/x",
        "responses": [
            {"status": 429, "chunks": []},
            {"status": 200, "chunks": ["done"]},
        ],
    }
    transport = FixtureTransport([exchange], max_attempts=1, sleeper=lambda _: None)
    assert transport.send("POST", "/x").status == 429
    assert transport.send("POST", "/x").status == 200
    assert transport.send("POST", "/x").status == 200


def test_http_transport_retries_transient_before_first_byte():
    calls = []

    def connector(method, path, headers, body):
        calls.append(path)
        status = 503 if len(calls) < 3 else 200
        return TransportResponse(status, {}, [b"body"], provider="http")

    transport = HttpTransport(
        "http://127.0.0.1:9",
        "http",
        connector=connector,
        max_attempts=3,
        sleeper=lambda _: None,
    )
    response = transport.send("POST", "/chat/completions")
    assert response.status == 200
    assert len(calls) == 3


def test_http_transport_does_not_echo_response_bodies_on_error():
    transport = HttpTransport(
        "http://127.0.0.1:9",
        "http",
        connector=lambda *args: TransportResponse(401, {}, [b"secret body"]),
        max_attempts=1,
    )
    response = transport.send("POST", "/x")
    with pytest.raises(AdapterError) as info:
        response.raise_for_status()
    assert info.value.classification is ErrorClass.FATAL
    assert "secret body" not in str(info.value)


class _TransportErrorOnEveryAttempt(Transport):
    """Transport whose ``_send_once`` raises an ``OSError`` carrying a key."""

    def __init__(self):
        super().__init__("probe", max_attempts=3, backoff_base=0.0, backoff_cap=0.0)
        self.calls = 0

    def _send_once(self, method, path, *, headers, body, cancel):
        self.calls += 1
        raise OSError(f"dial failed with key {PLANTED_KEY}")


class _CancelDuringWait:
    """Cancel token that reports cancellation as soon as the retry waits."""

    def raise_if_cancelled(self, provider="adapter"):
        return None

    def sleep(self, seconds):
        return True


def test_cancellation_during_retry_wait_does_not_inherit_transport_error():
    """A retry-wait cancellation must not chain the raw transport error.

    The wait runs after the ``except`` handler rather than inside it, so the
    ``AdapterError(CANCELLED)`` has no ``__cause__``/``__context__`` and a
    formatted traceback cannot render the credential-bearing ``OSError``.
    """
    transport = _TransportErrorOnEveryAttempt()

    with pytest.raises(AdapterError) as info:
        transport.send("POST", "/x", cancel=_CancelDuringWait())

    error = info.value
    assert error.classification is ErrorClass.CANCELLED
    assert error.__cause__ is None
    assert error.__context__ is None
    assert error.__suppress_context__ is False
    assert transport.calls == 1

    rendered = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    escaped = PLANTED_KEY.encode("unicode_escape").decode("ascii")
    for text in (str(error), repr(error), rendered):
        assert PLANTED_KEY not in text
        assert escaped not in text


def test_unexpected_connector_error_surfaces_redacted():
    """A programmer error outside the retry taxonomy is fatal, not swallowed.

    A connector raising anything other than ``OSError``/``HTTPException`` is a
    defect in the caller's transport, not a provider condition: it must be
    classified ``fatal``, must not be retried, and must not carry the resolved
    credential into the public message or an uncaught traceback.
    """
    register_secret(PLANTED_KEY)
    calls = []

    def connector(method, path, headers, body):
        calls.append(path)
        raise TypeError(f"programmer error: {PLANTED_KEY}")

    transport = HttpTransport(
        "http://127.0.0.1:9",
        "http",
        connector=connector,
        max_attempts=3,
        sleeper=lambda _seconds: None,
    )
    with pytest.raises(AdapterError) as info:
        transport.send("POST", "/chat/completions")

    error = info.value
    assert error.classification is ErrorClass.FATAL
    assert error.wire_class() == "fatal"
    assert len(calls) == 1
    assert "programmer error" in str(error)
    assert error.__cause__ is None
    assert error.__context__ is None
    rendered = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    assert PLANTED_KEY not in str(error)
    assert PLANTED_KEY not in repr(error)
    assert PLANTED_KEY not in rendered


def _stream_that_fails_after_first_chunk(error):
    """A fixture exchange whose second body chunk read raises ``error``."""

    def hook(_chunk, index):
        if index == 1:
            raise error

    return FixtureTransport(
        [
            {
                "method": "POST",
                "path": "/x",
                "responses": [{"status": 200, "chunks": ["first", "second"]}],
            }
        ],
        on_chunk=hook,
    )


def test_midstream_read_errors_are_classified():
    transport = _stream_that_fails_after_first_chunk(OSError("connection reset"))
    seen = []
    with pytest.raises(AdapterError) as info:
        for chunk in transport.send("POST", "/x").iter_bytes():
            seen.append(chunk)
    assert seen == [b"first"]
    assert info.value.classification is ErrorClass.TRANSIENT
    assert info.value.wire_class() == "transient"

    transport = _stream_that_fails_after_first_chunk(ValueError("malformed chunk"))
    with pytest.raises(AdapterError) as info:
        list(transport.send("POST", "/x").iter_bytes())
    assert info.value.classification is ErrorClass.PROTOCOL

    original = AdapterError("fixture", "provider said no", ErrorClass.CAPACITY)
    transport = _stream_that_fails_after_first_chunk(original)
    with pytest.raises(AdapterError) as info:
        list(transport.send("POST", "/x").iter_bytes())
    assert info.value is original
