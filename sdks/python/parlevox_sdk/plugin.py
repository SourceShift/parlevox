"""The sidecar plugin loop.

One process, one framed stream. Control is JSON, audio is binary. A ``cancel``
is acknowledged and never terminates the process; only ``shutdown`` does.

In-flight ``transcribe``/``analyze`` audio is buffered in a bounded per-request
queue: the capacity is :data:`MAX_PENDING_AUDIO_FRAMES` frames and the explicit
:data:`PENDING_AUDIO_OVERFLOW_POLICY` is ``DropOldest``. Every evicted frame is
counted on a monotonic per-request counter and reported as ``dropped_frames`` in
the terminal ``done`` message, the same way the Rust ``AudioRing`` accounts for
drops. No request can grow its buffer without bound.

Cancelled request ids are kept as tombstones so a late ``audio_end`` cannot
resurrect suppressed work. That ledger is itself bounded: at most
:data:`MAX_CANCELLED` ids are retained in insertion order, the oldest is evicted
when full, and every eviction is counted (:class:`CancelledTombstones`). A late
``audio_end`` for a tombstoned id is a silent no-op; one for an evicted id is
rejected as an unknown request rather than executed.
"""

import inspect
import os
import sys
from collections import OrderedDict, deque

from .adapters.base import AdapterConfigError, AdapterError, redact
from .fakes import FakeLlm, FakeStt, FakeTts, FakeVad
from .protocol import (
    KIND_AUDIO,
    KIND_CLOSE,
    KIND_JSON,
    PROTOCOL_VERSION,
    SIDECAR_ABI,
    decode_audio,
    decode_json,
    encode_audio,
    encode_json,
    read_frame,
    write_frame,
)

#: Capacity of one in-flight request's pending-audio buffer, in frames.
MAX_PENDING_AUDIO_FRAMES = 64

#: Explicit OverflowPolicy for the pending-audio buffer: evict the oldest frame
#: to admit the new one (``DropOldest``). Mirrors ``parlevox_core::OverflowPolicy``.
PENDING_AUDIO_OVERFLOW_POLICY = "drop_oldest"

#: Capacity of the cancelled-request tombstone ledger, in ids. Oldest entries
#: are evicted first once the ledger is full; evictions are counted.
MAX_CANCELLED = 256


#: Supported ``PARLEVOX_ADAPTERS`` modalities and the adapter names each may
#: select. ``vad`` has no OpenAI-wire adapter in this slice.
_SUPPORTED_ADAPTERS = {
    "stt": {"fake", "openai"},
    "llm": {"fake", "openai"},
    "tts": {"fake", "openai"},
    "vad": {"fake"},
}

#: Plugin name used when at least one real adapter is selected.
_OPENAI_PLUGIN_NAME = "python-openai"


def _redact_tree(value):
    """Recursively redact every string that is about to reach the wire.

    Scrubbing at the single send choke point means an error branch that
    formats ``str(err)`` — including a ``ValueError`` or a defensive
    ``except Exception`` — cannot bypass the redaction every adapter already
    applies to its own messages. Audio-header fields pass through here too.
    """
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {
            redact(key) if isinstance(key, str) else key: _redact_tree(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_redact_tree(item) for item in value)
    if isinstance(value, list):
        return [_redact_tree(item) for item in value]
    return value


def _parse_adapter_selection(raw):
    """Parse ``modality:adapter,modality:adapter`` into a dict.

    A malformed or unsupported token raises :class:`AdapterConfigError` naming
    only the offending token (never a credential or environment value).
    Duplicates are allowed; the last one wins.
    """
    selection = {}
    text = (raw or "").strip()
    if not text:
        return selection
    for piece in text.split(","):
        token = piece.strip()
        if not token:
            continue
        if ":" not in token:
            raise AdapterConfigError(
                "adapters", f"malformed adapter token {token!r}"
            )
        modality, _, adapter = token.partition(":")
        modality = modality.strip().lower()
        adapter = adapter.strip().lower()
        if adapter not in _SUPPORTED_ADAPTERS.get(modality, set()):
            raise AdapterConfigError("adapters", f"unknown adapter token {token!r}")
        selection[modality] = adapter
    return selection


def _accepts_keywords(provider_method, names):
    """The subset of ``names`` a callable accepts as keyword arguments.

    Mirrors :func:`_accepts_history`: providers written against the older,
    narrower signature keep working, and a genuine error raised inside the
    provider is still reported rather than swallowed.
    """
    try:
        parameters = inspect.signature(provider_method).parameters
    except (TypeError, ValueError):
        return set()
    accepted = set()
    for name in names:
        parameter = parameters.get(name)
        if parameter is not None and parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            accepted.add(name)
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        accepted.update(names)
    return accepted


def _accepts_history(provider):
    """Whether ``provider.generate`` can accept a ``history`` keyword.

    Providers written before structured history implement only
    ``generate(self, prompt)``. Introspection is used instead of catching
    ``TypeError`` around the call so a genuine error raised *inside* the
    provider is still reported as an error rather than silently retried.
    A callable that cannot be inspected is treated as one-argument.
    """
    try:
        parameters = inspect.signature(provider.generate).parameters
    except (TypeError, ValueError):
        return False
    history = parameters.get("history")
    if history is not None and history.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    ):
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


class CancelledTombstones:
    """Bounded, insertion-ordered set of cancelled request ids.

    Inserts go through :meth:`remember`, removals through :meth:`discard`.
    Keeping insertion order makes eviction deterministic and oldest-first; the
    :attr:`evicted` counter is monotonic for the life of the plugin.
    """

    def __init__(self, capacity=MAX_CANCELLED):
        self.capacity = capacity
        self._ids = OrderedDict()
        self.evicted = 0

    def __len__(self):
        return len(self._ids)

    def __contains__(self, request_id):
        return request_id in self._ids

    def __iter__(self):
        return iter(self._ids)

    def remember(self, request_id):
        """Record `request_id`, evicting the oldest id if the ledger is full."""
        if request_id in self._ids:
            return
        if self.capacity <= 0:
            return
        while len(self._ids) >= self.capacity:
            oldest, _ = self._ids.popitem(last=False)
            self.evicted += 1
            print(
                "parlevox sidecar: evicted cancelled tombstone "
                f"{oldest!r} (total evicted={self.evicted})",
                file=sys.stderr,
            )
        self._ids[request_id] = True

    def discard(self, request_id):
        self._ids.pop(request_id, None)

    def clear(self):
        """Drop every tombstone; ``evicted`` stays monotonic for the process."""
        self._ids.clear()


class Plugin:
    def __init__(self, name="python-fake", stt=None, llm=None, tts=None, vad=None):
        self.name = name
        self.stt = stt or FakeStt()
        self.llm = llm or FakeLlm()
        self.tts = tts or FakeTts()
        self.vad = vad or FakeVad()

    @classmethod
    def from_env(cls, environ=None, name="python-fake"):
        """Build a plugin from ``PARLEVOX_ADAPTERS``.

        Unset, empty, or an all-``fake`` selection returns exactly
        ``cls(name=name)`` so the default fake path stays byte-identical. Real
        adapters are constructed without resolving their credential: the key is
        read at first use, so importing or starting the plugin needs no secret.
        """
        env = os.environ if environ is None else environ
        selection = _parse_adapter_selection(env.get("PARLEVOX_ADAPTERS", ""))
        if not selection or all(value == "fake" for value in selection.values()):
            return cls(name=name)

        stt = llm = tts = None
        if selection.get("stt") == "openai":
            from .adapters.openai_stt import OpenAISTT

            stt = OpenAISTT(environ=env)
        if selection.get("llm") == "openai":
            from .adapters.openai_llm import OpenAILlm

            llm = OpenAILlm(environ=env)
        if selection.get("tts") == "openai":
            from .adapters.openai_tts import OpenAITTS

            tts = OpenAITTS(environ=env)
        # ``vad`` only accepts ``fake``, so it stays at the default FakeVad.
        return cls(name=_OPENAI_PLUGIN_NAME, stt=stt, llm=llm, tts=tts)

    def manifest(self):
        return {
            "name": self.name,
            "modality": "tts",
            "capabilities": [
                "stt",
                "llm",
                "tts",
                "vad",
                "streaming",
                "binary_pcm",
                "cancel",
                "reset",
            ],
            "config_schema": {"type": "object", "properties": {}},
        }

    def _ready_payload(self):
        """The one `ready` shape used by both `hello` and `reset`.

        Keeping the acknowledgement in one helper means the initial handshake
        and a pool reset cannot drift: both carry the top-level capability list
        that tells the client this process is safe to reuse.
        """
        return {
            "type": "ready",
            "protocol_version": PROTOCOL_VERSION,
            "abi": SIDECAR_ABI,
            "plugin": self.name,
            "manifest": self.manifest(),
            "capabilities": ["reset"],
        }

    # -- wire helpers -------------------------------------------------
    @staticmethod
    def _send(out, message):
        write_frame(out, KIND_JSON, encode_json(_redact_tree(message)))

    @staticmethod
    def _send_audio(out, request_id, seq, sample_rate, channels, pcm):
        header = _redact_tree(
            {
                "request_id": request_id,
                "seq": seq,
                "sample_rate": sample_rate,
                "channels": channels,
            }
        )
        write_frame(out, KIND_AUDIO, encode_audio(header, pcm))

    @staticmethod
    def _send_error(out, request_id, cls, message):
        # Route through ``_send`` so the one choke point redacts the message,
        # whichever branch produced it.
        Plugin._send(
            out,
            {
                "type": "error",
                "request_id": request_id,
                "class": cls,
                "message": message,
            },
        )

    # -- bounded pending audio ----------------------------------------
    @staticmethod
    def _buffer_audio(
        request_id, pcm, pending_audio, pending_requests, pending_dropped, cancelled
    ):
        """Append one audio frame under the explicit capacity and policy.

        ``pending_audio`` maps a request id to a ``deque`` with
        ``maxlen=MAX_PENDING_AUDIO_FRAMES``; appending to a full deque evicts the
        oldest frame, which is counted on the request's monotonic
        ``pending_dropped`` counter.

        Frames are only buffered for a request that is currently open. Frames
        for a cancelled (tombstoned or evicted) or unknown request are ignored,
        so post-cancel audio can never rebuild state.
        """
        if request_id in cancelled:
            return
        if request_id not in pending_requests:
            return
        buffer = pending_audio.get(request_id)
        if buffer is None:
            buffer = deque(maxlen=MAX_PENDING_AUDIO_FRAMES)
            pending_audio[request_id] = buffer
            pending_dropped.setdefault(request_id, 0)
        if buffer.maxlen is not None and len(buffer) >= buffer.maxlen:
            pending_dropped[request_id] = pending_dropped.get(request_id, 0) + 1
        buffer.append(bytes(pcm))

    # -- loop ---------------------------------------------------------
    def run(self, stdin, stdout):
        """Serve frames until EOF, close, or shutdown."""
        pending_audio = {}
        pending_requests = {}
        pending_dropped = {}
        cancelled = CancelledTombstones()
        while True:
            try:
                kind, payload = read_frame(stdin)
            except EOFError:
                return
            if kind == KIND_CLOSE:
                return
            if kind == KIND_AUDIO:
                header, pcm = decode_audio(payload)
                self._buffer_audio(
                    header["request_id"],
                    pcm,
                    pending_audio,
                    pending_requests,
                    pending_dropped,
                    cancelled,
                )
                continue
            if kind != KIND_JSON:
                continue
            message = decode_json(payload)
            if not self.handle(
                message,
                stdout,
                pending_audio,
                pending_requests,
                pending_dropped,
                cancelled,
            ):
                return

    def handle(
        self,
        message,
        out,
        pending_audio=None,
        pending_requests=None,
        pending_dropped=None,
        cancelled=None,
    ):
        """Handle one control message. Returns False to stop the loop."""
        pending_audio = pending_audio if pending_audio is not None else {}
        pending_requests = pending_requests if pending_requests is not None else {}
        pending_dropped = pending_dropped if pending_dropped is not None else {}
        cancelled = cancelled if cancelled is not None else CancelledTombstones()
        kind = message.get("type")

        if kind == "hello":
            version = message.get("protocol_version")
            abi = message.get("abi")
            if version != PROTOCOL_VERSION:
                self._send_error(out, "", "protocol", f"unsupported protocol version {version}")
                return True
            if abi != SIDECAR_ABI:
                self._send_error(out, "", "protocol", f"unsupported abi {abi!r}")
                return True
            self._send(out, self._ready_payload())
            return True

        if kind == "reset":
            # A reset is a session boundary: every per-request container and
            # the cancellation ledger start empty, and the process keeps
            # serving. The acknowledgement has the same shape as the handshake
            # so the client can validate it identically.
            pending_audio.clear()
            pending_requests.clear()
            pending_dropped.clear()
            cancelled.clear()
            self._send(out, self._ready_payload())
            return True

        if kind == "synthesize":
            return self._synthesize(message, out)
        if kind == "generate":
            return self._generate(message, out)
        if kind in ("transcribe", "analyze"):
            request_id = message["request_id"]
            pending_requests[request_id] = message
            pending_audio.pop(request_id, None)
            pending_dropped[request_id] = 0
            cancelled.discard(request_id)
            return True
        if kind == "audio_end":
            return self._finish_audio_request(
                message["request_id"],
                out,
                pending_audio,
                pending_requests,
                pending_dropped,
                cancelled,
            )
        if kind == "cancel":
            request_id = message.get("request_id", "")
            # Teardown first: drop this request's buffered audio and queued
            # request before acknowledging, so the ack means "nothing of yours
            # is still buffered". An unknown id is a no-op (the pops miss) that
            # still gets an ack, and cancel is idempotent.
            known = request_id in pending_requests or request_id in pending_audio
            pending_audio.pop(request_id, None)
            pending_requests.pop(request_id, None)
            pending_dropped.pop(request_id, None)
            if known:
                cancelled.remember(request_id)
            self._send(
                out,
                {"type": "cancel_ack", "request_id": request_id},
            )
            return True
        if kind == "shutdown":
            return False

        self._send_error(out, message.get("request_id", ""), "protocol", f"unknown message {kind!r}")
        return True

    def _synthesize(self, message, out):
        request_id = message["request_id"]
        sample_rate = message.get("sample_rate", 16_000)
        channels = message.get("channels", 1)
        # A provider may accept the requested output format as keywords. Older
        # providers (including FakeTts) only take ``text``; introspecting keeps
        # their call shape — and therefore the demo output — unchanged.
        options = _accepts_keywords(self.tts.synthesize, ("sample_rate", "channels"))
        keywords = {}
        if "sample_rate" in options:
            keywords["sample_rate"] = sample_rate
        if "channels" in options:
            keywords["channels"] = channels
        try:
            chunks = list(self.tts.synthesize(message.get("text", ""), **keywords))
        except AdapterError as err:
            self._send_error(out, request_id, err.wire_class(), err.public_message())
            return True
        except ValueError as err:
            self._send_error(out, request_id, "protocol", str(err))
            return True
        except Exception as err:  # pragma: no cover - defensive
            self._send_error(out, request_id, "transient", str(err))
            return True
        self._send(
            out,
            {
                "type": "audio_start",
                "request_id": request_id,
                "sample_rate": sample_rate,
                "channels": channels,
            },
        )
        for seq, pcm in enumerate(chunks):
            self._send_audio(out, request_id, seq, sample_rate, channels, pcm)
        self._send(
            out,
            {"type": "done", "request_id": request_id, "dropped_frames": 0},
        )
        return True

    def _generate(self, message, out):
        request_id = message["request_id"]
        prompt = message.get("prompt", "")
        # Prior turns, oldest first, are already structured messages on the
        # wire. A provider that implements only ``generate(self, prompt)``
        # cannot receive them; introspect rather than raising a TypeError, so
        # such a plugin keeps working on later turns too.
        history = message.get("history") or []
        try:
            if history and _accepts_history(self.llm):
                deltas = list(self.llm.generate(prompt, history=history))
            else:
                deltas = list(self.llm.generate(prompt))
        except AdapterError as err:
            self._send_error(out, request_id, err.wire_class(), err.public_message())
            return True
        except ValueError as err:
            self._send_error(out, request_id, "protocol", str(err))
            return True
        except Exception as err:  # pragma: no cover - defensive
            self._send_error(out, request_id, "transient", str(err))
            return True
        for delta in deltas:
            self._send(out, {"type": "delta", "request_id": request_id, "text": delta})
        self._send(
            out,
            {"type": "done", "request_id": request_id, "dropped_frames": 0},
        )
        return True

    def _finish_audio_request(
        self,
        request_id,
        out,
        pending_audio,
        pending_requests,
        pending_dropped,
        cancelled,
    ):
        if request_id in cancelled:
            # The request was cancelled after its audio arrived: its work is
            # suppressed, so a late audio_end must not resurrect it. The
            # tombstone is intentionally kept so repeated late audio_end frames
            # stay silent no-ops. An evicted id is no longer tombstoned and
            # falls through to the unknown-request rejection below.
            pending_audio.pop(request_id, None)
            pending_requests.pop(request_id, None)
            pending_dropped.pop(request_id, None)
            return True
        request = pending_requests.pop(request_id, None)
        dropped_frames = pending_dropped.pop(request_id, 0)
        buffer = pending_audio.pop(request_id, None)
        if buffer is None:
            pcm = b""
        elif isinstance(buffer, (bytes, bytearray)):
            pcm = bytes(buffer)
        else:
            pcm = b"".join(buffer)
        if request is None:
            self._send_error(out, request_id, "protocol", "audio_end for unknown request")
            return True
        try:
            if request["type"] == "transcribe":
                # The core sends the utterance's own rate and channel count on
                # the ``Transcribe`` request (``PluginRequest::Transcribe``).
                # Thread them to a provider that accepts them so the upload
                # container declares the audio it actually carries; older
                # providers (including FakeStt) keep their ``transcribe(pcm)``
                # shape and therefore their output unchanged.
                sample_rate = request.get("sample_rate", 16_000)
                channels = request.get("channels", 1)
                options = _accepts_keywords(
                    self.stt.transcribe, ("sample_rate", "channels")
                )
                keywords = {}
                if "sample_rate" in options:
                    keywords["sample_rate"] = sample_rate
                if "channels" in options:
                    keywords["channels"] = channels
                for chunk in self.stt.transcribe(pcm, **keywords):
                    if chunk["final"]:
                        self._send(out, {"type": "final", "request_id": request_id, "text": chunk["text"]})
                    else:
                        self._send(out, {"type": "partial", "request_id": request_id, "text": chunk["text"]})
            else:
                for transition in self.vad.analyze(pcm):
                    self._send(
                        out,
                        {
                            "type": "vad",
                            "request_id": request_id,
                            "event": transition["event"],
                            "timestamp_ms": transition["timestamp_ms"],
                        },
                    )
        except AdapterError as err:
            self._send_error(out, request_id, err.wire_class(), err.public_message())
            return True
        except ValueError as err:
            self._send_error(out, request_id, "protocol", str(err))
            return True
        except Exception as err:  # pragma: no cover - defensive
            self._send_error(out, request_id, "transient", str(err))
            return True
        self._send(
            out,
            {
                "type": "done",
                "request_id": request_id,
                "dropped_frames": dropped_frames,
            },
        )
        return True
