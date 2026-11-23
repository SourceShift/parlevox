"""Provider-neutral adapter runtime: errors, secrets, transports, retries.

Standard library only. This module is the shared seam for every wire adapter:

* :class:`ErrorClass` mirrors the Rust ``ErrorClass`` taxonomy (plus an
  internal ``rate_limit`` class, see below) and maps onto the wire strings the
  sidecar plugin may report.
* :class:`AdapterError` and its subclasses carry a classification, the provider
  name, and a redacted, truncated public message. No string produced by an
  adapter can contain a resolved credential.
* :class:`CancelToken` is a cooperative cancellation primitive checked before
  connecting, between reads, and during retry backoff.
* :func:`classify_status` is a total HTTP-status mapping.
* :class:`Transport` owns the bounded retry loop; :class:`HttpTransport`
  implements it over :mod:`http.client` and :class:`FixtureTransport` replays
  recorded exchanges from disk without ever opening a socket.

Wire-class compromise
---------------------
The Rust wire enum (``crates/parlevox-ipc::ErrorClassWire``) has exactly
``transient``, ``fatal``, ``protocol``, ``cancelled`` and ``capacity`` — there
is no ``rate_limit`` variant. The Python side still models HTTP 429 as an
internal :attr:`ErrorClass.RATE_LIMIT` so retry policy and tests can name it,
but :meth:`ErrorClass.wire_class` reports it as ``transient``: a throttled
provider is retryable, and ``crates/`` is out of this task's scope.
"""

from __future__ import annotations

import base64
import enum
import http.client
import json
import logging
import os
import re
import threading
import time
import urllib.parse
from abc import ABC, abstractmethod
from pathlib import Path

__all__ = [
    "AdapterConfigError",
    "AdapterError",
    "CancelToken",
    "DEFAULT_BASE_URL_OPENAI",
    "DEFAULT_BACKOFF_BASE",
    "DEFAULT_BACKOFF_CAP",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_TIMEOUT",
    "ErrorClass",
    "FixtureTransport",
    "HttpTransport",
    "MissingCredential",
    "REDACTED",
    "Transport",
    "TransportResponse",
    "classify_status",
    "default_transport",
    "redact",
    "register_secret",
    "reset_redaction",
    "resolve_key",
]

_LOG = logging.getLogger("parlevox.adapters")

#: Replacement written wherever a credential would otherwise appear.
REDACTED = "<redacted>"

#: Longest public error detail; longer provider text is truncated.
MAX_PUBLIC_MESSAGE = 300

#: Base URL for the OpenAI wire API; override with ``PARLEVOX_OPENAI_BASE_URL``.
DEFAULT_BASE_URL_OPENAI = "https://api.openai.com/v1"

DEFAULT_TIMEOUT = 10.0
#: Total attempts, including the first. ``<= 1`` disables retries.
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BACKOFF_BASE = 0.1
DEFAULT_BACKOFF_CAP = 2.0

_CREDENTIAL_SHAPES = re.compile(r"sk-[A-Za-z0-9_-]{20,}")
#: ``Bearer <token>``. The token class also consumes backslash escapes, so a
#: value that was rendered with ``repr()`` (``b'Bearer abort\\n...'``) is masked
#: whole instead of stopping at the first escaped control character.
_BEARER_VALUE = re.compile(r"(?i)(bearer\s+)[^\s'\"]+")
#: Header-shaped ``Authorization: <token>`` / ``Authorization=<token>`` text.
_AUTHORIZATION_HEADER = re.compile(
    r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?[^\s'\"]+"
)
#: Credential-bearing query/pair parameters such as ``api_key=``/``token=``.
_QUERY_SECRET = re.compile(
    r"(?i)([?&;](?:api_key|access_token|token)=)[^&\s'\"]+"
)
_known_secrets = set()
_secrets_lock = threading.Lock()


class ErrorClass(enum.Enum):
    """Failure taxonomy, mirroring the Rust classes plus an internal 429.

    ``RATE_LIMIT`` is deliberately internal: the wire has no such variant, so
    :meth:`wire_class` reports it as ``transient`` (see the module docstring).
    """

    TRANSIENT = "transient"
    FATAL = "fatal"
    PROTOCOL = "protocol"
    CANCELLED = "cancelled"
    CAPACITY = "capacity"
    RATE_LIMIT = "rate_limit"

    def wire_class(self):
        """The string this class is reported as on the sidecar wire."""
        if self is ErrorClass.RATE_LIMIT:
            return ErrorClass.TRANSIENT.value
        return self.value


# -- secrets ------------------------------------------------------------


def register_secret(value):
    """Remember a resolved secret so every later message is scrubbed of it.

    The literal value is registered together with its ``unicode_escape`` and
    JSON-escaped forms. ``http.client`` renders an illegal header value with
    ``repr()``, turning a control character in the credential into a
    two-character ``\\n`` sequence that a literal-only match would miss.
    """
    if not value:
        return
    if not isinstance(value, str):
        value = str(value)
    forms = {value}
    try:
        forms.add(value.encode("unicode_escape").decode("ascii"))
    except (UnicodeError, AttributeError):  # pragma: no cover - str always encodes
        pass
    try:
        encoded = json.dumps(value)
    except (TypeError, ValueError):  # pragma: no cover - str always serialises
        pass
    else:
        # The inner text of the JSON string literal, without the quotes.
        forms.add(encoded[1:-1])
    with _secrets_lock:
        _known_secrets.update(forms)


def reset_redaction():
    """Forget every registered secret. Intended for tests only."""
    with _secrets_lock:
        _known_secrets.clear()


def redact(text):
    """Return ``text`` with known secrets and credential shapes replaced.

    Literal registered secrets are removed first (longest first, so a secret
    that contains another is fully scrubbed), then credential-shaped tokens and
    bearer credentials that were never registered are removed defensively.
    """
    if text is None:
        return text
    out = str(text)
    with _secrets_lock:
        secrets = sorted(_known_secrets, key=len, reverse=True)
    for secret in secrets:
        out = out.replace(secret, REDACTED)
    out = _CREDENTIAL_SHAPES.sub(REDACTED, out)
    out = _BEARER_VALUE.sub(r"\1" + REDACTED, out)
    out = _AUTHORIZATION_HEADER.sub(r"\1" + REDACTED, out)
    out = _QUERY_SECRET.sub(r"\1" + REDACTED, out)
    return out


def resolve_key(env_var, environ=None):
    """Read ``env_var`` at first use and return the credential.

    Raises :class:`MissingCredential` (classified ``fatal``) naming only the
    environment variable when the value is absent or empty. This is never
    called at import time: importing an adapter must not require a key.
    """
    env = os.environ if environ is None else environ
    value = env.get(env_var, "")
    if not value:
        raise MissingCredential(
            "adapter",
            f"missing credential in {env_var}",
            env_var=env_var,
        )
    register_secret(value)
    return value


# -- errors -------------------------------------------------------------


class AdapterError(Exception):
    """A classified provider failure with a credential-safe public message.

    The raw detail is kept private and only ever exposed through
    :meth:`public_message`, which redacts secrets and truncates vendor text.
    ``Exception.args`` holds the already-redacted message, so a formatted
    traceback cannot leak a key.
    """

    def __init__(
        self,
        provider,
        message,
        classification=ErrorClass.FATAL,
        *,
        env_var=None,
        retry_after=None,
    ):
        self.provider = provider or "adapter"
        self.classification = classification
        self.env_var = env_var
        self.retry_after = retry_after
        self._detail = str(message)
        super().__init__(self.public_message())

    def public_message(self):
        """Redacted, truncated message safe for logs, errors and the wire."""
        detail = redact(self._detail)
        message = f"{self.provider}: {detail}" if self.provider else detail
        if len(message) > MAX_PUBLIC_MESSAGE:
            message = message[:MAX_PUBLIC_MESSAGE] + "…"
        return message

    def wire_class(self):
        """The wire string the plugin should report for this failure."""
        return self.classification.wire_class()

    def __str__(self):
        return self.public_message()

    def __repr__(self):
        return (
            f"{type(self).__name__}(provider={self.provider!r}, "
            f"class={self.classification.value!r}, "
            f"message={self.public_message()!r})"
        )


class MissingCredential(AdapterError):
    """A required environment-provided credential was absent at first use."""

    def __init__(self, provider, message, *, env_var, classification=ErrorClass.FATAL):
        super().__init__(provider, message, classification, env_var=env_var)


class AdapterConfigError(AdapterError):
    """Adapter selection or fixture configuration is invalid."""

    def __init__(self, provider, message, classification=ErrorClass.FATAL):
        super().__init__(provider, message, classification)


# -- cancellation -------------------------------------------------------


class CancelToken:
    """Cooperative cancellation, safe to poll from multiple threads.

    :meth:`sleep` waits at most ``seconds`` and returns ``True`` if the token
    was cancelled while waiting, which lets retry backoff abort promptly.
    """

    def __init__(self):
        self._event = threading.Event()

    def cancel(self):
        self._event.set()

    @property
    def cancelled(self):
        return self._event.is_set()

    def sleep(self, seconds):
        """Wait ``seconds``; return ``True`` if cancelled during the wait."""
        if seconds <= 0:
            return self._event.is_set()
        return self._event.wait(seconds)

    def raise_if_cancelled(self, provider="adapter"):
        if self._event.is_set():
            raise AdapterError(provider, "cancelled", ErrorClass.CANCELLED)


# -- HTTP status classification ----------------------------------------


def classify_status(status):
    """Map an HTTP status to :class:`ErrorClass`.

    Total over every integer: ``100..=599`` all return either ``None`` (a 2xx
    success) or exactly one :class:`ErrorClass` member, and the function never
    raises. Out-of-range codes fall back to ``fatal``.

    Table: 429 -> rate_limit (wire: transient); 401/403/404 -> fatal;
    400/422 -> protocol; 408/409/425 -> transient; 413/507 -> capacity;
    5xx -> transient; every other non-2xx -> protocol.
    """
    try:
        status = int(status)
    except (TypeError, ValueError):
        return ErrorClass.FATAL
    if 200 <= status < 300:
        return None
    if status == 429:
        return ErrorClass.RATE_LIMIT
    if status in (401, 403, 404):
        return ErrorClass.FATAL
    if status in (400, 422):
        return ErrorClass.PROTOCOL
    if status in (408, 409, 425):
        return ErrorClass.TRANSIENT
    if status in (413, 507):
        return ErrorClass.CAPACITY
    if 500 <= status < 600:
        return ErrorClass.TRANSIENT
    if 100 <= status < 200:
        return ErrorClass.PROTOCOL
    if 300 <= status < 400:
        return ErrorClass.PROTOCOL
    if 400 <= status < 500:
        return ErrorClass.PROTOCOL
    return ErrorClass.FATAL


# -- transport ----------------------------------------------------------


class TransportResponse:
    """A decoded response: status, lower-cased headers, lazy body chunks."""

    def __init__(self, status, headers=None, chunks=(), *, provider="", close=None):
        self.status = int(status)
        self.headers = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
        self.provider = provider or "adapter"
        self._chunks = chunks
        self._close = close

    @property
    def ok(self):
        return 200 <= self.status < 300

    @property
    def error_class(self):
        return classify_status(self.status)

    @property
    def retry_after(self):
        """Numeric ``Retry-After`` in seconds, or ``None`` when absent/invalid."""
        raw = self.headers.get("retry-after")
        if raw is None:
            return None
        try:
            seconds = float(str(raw).strip())
        except (TypeError, ValueError):
            return None
        return seconds if seconds >= 0 else None

    def iter_bytes(self, cancel=None):
        """Yield body chunks, classifying failures that occur after the first.

        A body read can fail after the status line has already arrived, where
        the retry loop in :meth:`Transport.send` can no longer help. Such a
        failure is mapped to the same taxonomy as a status: a transport-level
        ``OSError``/``HTTPException`` is ``transient`` and a malformed chunk is
        ``protocol``. An :class:`AdapterError` raised by the iterator is
        already classified and is re-raised unchanged.
        """
        iterator = iter(self._chunks)
        while True:
            try:
                chunk = next(iterator)
            except StopIteration:
                return
            except AdapterError:
                raise
            except (OSError, http.client.HTTPException) as err:
                raise AdapterError(
                    self.provider,
                    f"transport failure mid-stream: {type(err).__name__}",
                    ErrorClass.TRANSIENT,
                ) from None
            except ValueError:
                raise AdapterError(
                    self.provider,
                    "malformed response chunk",
                    ErrorClass.PROTOCOL,
                ) from None
            if cancel is not None:
                cancel.raise_if_cancelled(self.provider)
            yield chunk

    def iter_lines(self, cancel=None):
        """Yield body lines (without line terminators) as ``bytes``."""
        buffer = b""
        for chunk in self.iter_bytes(cancel):
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                yield line.rstrip(b"\r")
        if buffer:
            yield buffer.rstrip(b"\r")

    def read(self, cancel=None):
        return b"".join(self.iter_bytes(cancel))

    def raise_for_status(self):
        """Raise a classified :class:`AdapterError` for a non-2xx status.

        The response body is never echoed: only the numeric status is named.
        """
        classification = classify_status(self.status)
        if classification is None:
            return
        raise AdapterError(
            self.provider,
            f"HTTP {self.status}",
            classification,
            retry_after=self.retry_after,
        )

    def close(self):
        if self._close is not None:
            try:
                self._close()
            except Exception:  # noqa: BLE001 - closing must not mask a failure
                pass


class Transport(ABC):
    """Base transport owning the bounded retry loop.

    Subclasses implement :meth:`_send_once`. The loop retries a transient or
    rate-limited response (and pre-first-byte transport errors) up to
    ``max_attempts`` total attempts, honouring a numeric ``Retry-After`` capped
    at ``backoff_cap``. Cancellation is checked before every attempt and the
    sleep between attempts aborts promptly.
    """

    def __init__(
        self,
        provider="adapter",
        *,
        max_attempts=DEFAULT_MAX_ATTEMPTS,
        backoff_base=DEFAULT_BACKOFF_BASE,
        backoff_cap=DEFAULT_BACKOFF_CAP,
        sleeper=None,
    ):
        self.provider = provider or "adapter"
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self._sleeper = sleeper or time.sleep

    def send(self, method, path, *, headers=None, body=None, cancel=None):
        """Send one request, retrying within the configured bound."""
        attempts = max(1, int(self.max_attempts))
        for attempt in range(attempts):
            if cancel is not None:
                cancel.raise_if_cancelled(self.provider)
            failure = None
            retry = False
            try:
                response = self._send_once(
                    method, path, headers=headers or {}, body=body, cancel=cancel
                )
            except AdapterError:
                raise
            except (OSError, http.client.HTTPException) as err:
                if attempt + 1 >= attempts:
                    # Captured rather than raised here: a raise inside an
                    # ``except`` handler leaves ``__context__`` pointing at the
                    # raw error, whose text may carry request material. The
                    # classified error is raised after the handler instead.
                    failure = AdapterError(
                        self.provider,
                        f"transport failure: {type(err).__name__}",
                        ErrorClass.TRANSIENT,
                    )
                else:
                    _LOG.debug(
                        "%s %s attempt=%d class=transient (retrying)",
                        method,
                        path,
                        attempt + 1,
                    )
                    # The wait is deferred until after the handler: a
                    # cancellation raised by ``_wait`` inside the ``except``
                    # arm would carry the raw transport error (whose text may
                    # include request material) as ``__context__``.
                    retry = True
            except ValueError as err:
                # ``http.client`` rejects an illegal header value — a control
                # character in the resolved credential — with a ``ValueError``
                # before any socket is opened. It is a caller-side defect:
                # ``protocol`` and never retried. The message is redacted when
                # the AdapterError renders its public text.
                failure = AdapterError(
                    self.provider,
                    f"invalid request: {type(err).__name__}: {err}",
                    ErrorClass.PROTOCOL,
                )
            except Exception as err:  # noqa: BLE001 - every escape must be classified
                # Anything else (for example a ``TypeError`` from a misbehaving
                # connector) is a programmer error: surface it as ``fatal``
                # rather than letting it escape the adapter unclassified.
                failure = AdapterError(
                    self.provider,
                    f"programmer error: {type(err).__name__}: {err}",
                    ErrorClass.FATAL,
                )
            if failure is not None:
                # Raised outside the handler with ``from None`` so both
                # ``__cause__`` and ``__context__`` stay ``None`` and no raw
                # transport text can reach a formatted traceback.
                raise failure from None
            if retry:
                # Waiting outside the handler keeps a cancellation free of
                # ``__cause__``/``__context__``; the retry arguments and the
                # debug line above are unchanged.
                self._wait(attempt, None, cancel)
                continue

            classification = classify_status(response.status)
            if (
                classification in (ErrorClass.TRANSIENT, ErrorClass.RATE_LIMIT)
                and attempt + 1 < attempts
            ):
                retry_after = response.retry_after
                response.close()
                _LOG.debug(
                    "%s %s attempt=%d class=%s (retrying)",
                    method,
                    path,
                    attempt + 1,
                    classification.value,
                )
                self._wait(attempt, retry_after, cancel)
                continue
            _LOG.debug(
                "%s %s status=%d class=%s",
                method,
                path,
                response.status,
                classification.value if classification is not None else "ok",
            )
            return response
        # Unreachable: the loop either returns or raises.
        raise AdapterError(
            self.provider, "transport exhausted retries", ErrorClass.TRANSIENT
        )

    def _wait(self, attempt, retry_after, cancel):
        if retry_after is not None:
            delay = float(retry_after)
        else:
            delay = self.backoff_base * (2**attempt)
        delay = min(max(delay, 0.0), float(self.backoff_cap))
        if cancel is not None:
            if cancel.sleep(delay):
                raise AdapterError(self.provider, "cancelled", ErrorClass.CANCELLED)
        else:
            self._sleeper(delay)

    @abstractmethod
    def _send_once(self, method, path, *, headers, body, cancel):
        """Return one :class:`TransportResponse`, or raise ``OSError``."""


class HttpTransport(Transport):
    """Standard-library HTTP transport over :mod:`http.client`."""

    def __init__(
        self,
        base_url,
        provider="adapter",
        *,
        timeout=DEFAULT_TIMEOUT,
        connector=None,
        **kwargs,
    ):
        super().__init__(provider, **kwargs)
        parts = urllib.parse.urlsplit(base_url)
        self.scheme = (parts.scheme or "https").lower()
        self.host = parts.hostname
        self.port = parts.port
        self.base_path = parts.path.rstrip("/")
        self.timeout = timeout
        self._connector = connector

    def _full_path(self, path):
        if not path.startswith("/"):
            path = "/" + path
        return self.base_path + path

    def _send_once(self, method, path, *, headers, body, cancel):
        if cancel is not None:
            cancel.raise_if_cancelled(self.provider)
        full_path = self._full_path(path)
        if self._connector is not None:
            return self._connector(method, full_path, headers, body)
        if self.scheme == "https":
            connection = http.client.HTTPSConnection(
                self.host, self.port, timeout=self.timeout
            )
        else:
            connection = http.client.HTTPConnection(
                self.host, self.port, timeout=self.timeout
            )
        try:
            connection.request(method, full_path, body=body, headers=headers)
            raw = connection.getresponse()
        except Exception:
            connection.close()
            raise
        status = raw.status
        response_headers = {k.lower(): v for k, v in raw.getheaders()}

        def chunks():
            try:
                while True:
                    if cancel is not None:
                        cancel.raise_if_cancelled(self.provider)
                    data = raw.read(8192)
                    if not data:
                        return
                    yield data
            finally:
                connection.close()

        return TransportResponse(
            status,
            response_headers,
            chunks(),
            provider=self.provider,
            close=connection.close,
        )


class FixtureTransport(Transport):
    """Replay recorded exchanges from JSON; never opens a socket.

    An exchange is::

        {
          "method": "POST",
          "path": "/chat/completions",        # matched as a path suffix
          "responses": [
            {"status": 200, "headers": {...},
             "chunks": ["data: ...\n\n", ...]}]
        }

    ``responses`` may also be a single response object. A response carries
    either UTF-8 ``chunks`` or one ``body_base64`` binary body. Successive
    requests to the same exchange consume successive responses; the last
    response repeats once the list is exhausted.

    ``on_chunk(chunk, index)`` is a per-chunk test hook invoked before each
    chunk is yielded, which is how tests drive mid-stream cancellation.
    """

    def __init__(self, exchanges, provider="fixture", *, on_chunk=None, **kwargs):
        super().__init__(provider, **kwargs)
        self._exchanges = [self._normalize(exchange) for exchange in exchanges]
        self._cursors = {}
        self.on_chunk = on_chunk

    @classmethod
    def from_directory(cls, directory, provider="fixture", **kwargs):
        path = Path(directory)
        if not path.is_dir():
            raise AdapterConfigError(
                provider, f"fixture directory not found: {path.name or directory}"
            )
        exchanges = []
        for file in sorted(path.glob("*.json")):
            with file.open("r", encoding="utf-8") as handle:
                exchanges.append(json.load(handle))
        return cls(exchanges, provider=provider, **kwargs)

    @staticmethod
    def _normalize(exchange):
        responses = exchange.get("responses")
        if responses is None:
            responses = [exchange]
        elif isinstance(responses, dict):
            responses = [responses]
        return {
            "id": exchange.get("method", "POST") + " " + exchange["path"],
            "method": str(exchange.get("method", "POST")).upper(),
            "path": str(exchange["path"]),
            "responses": list(responses),
        }

    @staticmethod
    def _decode_chunks(spec):
        if "body_base64" in spec:
            return [base64.b64decode(spec["body_base64"])]
        decoded = []
        for chunk in spec.get("chunks") or ():
            if isinstance(chunk, str):
                decoded.append(chunk.encode("utf-8"))
            else:
                decoded.append(bytes(chunk))
        return decoded

    def _match(self, method, path):
        method = str(method).upper()
        for exchange in self._exchanges:
            if exchange["method"] != method:
                continue
            if path.endswith(exchange["path"]) or exchange["path"].endswith(path):
                return exchange
        raise AdapterError(
            self.provider,
            f"no fixture for {method} {path}",
            ErrorClass.PROTOCOL,
        )

    def _send_once(self, method, path, *, headers, body, cancel):
        exchange = self._match(method, path)
        index = self._cursors.get(exchange["id"], 0)
        self._cursors[exchange["id"]] = index + 1
        responses = exchange["responses"]
        spec = responses[min(index, len(responses) - 1)]
        status = int(spec.get("status", 200))
        raw_chunks = self._decode_chunks(spec)

        def chunks():
            for chunk_index, chunk in enumerate(raw_chunks):
                if self.on_chunk is not None:
                    self.on_chunk(chunk, chunk_index)
                if cancel is not None:
                    cancel.raise_if_cancelled(self.provider)
                yield chunk

        return TransportResponse(
            status,
            spec.get("headers") or {},
            chunks(),
            provider=self.provider,
        )


def default_transport(base_url, environ, provider):
    """Pick the fixture transport when ``PARLEVOX_FIXTURE_DIR`` is set.

    The HTTP transport is never constructed in fixture mode, so a fixture run
    cannot open a socket even if the base URL is unreachable.
    """
    fixture_dir = environ.get("PARLEVOX_FIXTURE_DIR")
    if fixture_dir:
        return FixtureTransport.from_directory(fixture_dir, provider=provider)
    return HttpTransport(base_url, provider=provider)
