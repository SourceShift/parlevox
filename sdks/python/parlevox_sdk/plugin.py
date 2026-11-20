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
import sys
from collections import OrderedDict, deque

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


class Plugin:
    def __init__(self, name="python-fake", stt=None, llm=None, tts=None, vad=None):
        self.name = name
        self.stt = stt or FakeStt()
        self.llm = llm or FakeLlm()
        self.tts = tts or FakeTts()
        self.vad = vad or FakeVad()

    def manifest(self):
        return {
            "name": self.name,
            "modality": "tts",
            "capabilities": ["stt", "llm", "tts", "vad", "streaming", "binary_pcm", "cancel"],
            "config_schema": {"type": "object", "properties": {}},
        }

    # -- wire helpers -------------------------------------------------
    @staticmethod
    def _send(out, message):
        write_frame(out, KIND_JSON, encode_json(message))

    @staticmethod
    def _send_audio(out, request_id, seq, sample_rate, channels, pcm):
        header = {
            "request_id": request_id,
            "seq": seq,
            "sample_rate": sample_rate,
            "channels": channels,
        }
        write_frame(out, KIND_AUDIO, encode_audio(header, pcm))

    @staticmethod
    def _send_error(out, request_id, cls, message):
        write_frame(
            out,
            KIND_JSON,
            encode_json(
                {
                    "type": "error",
                    "request_id": request_id,
                    "class": cls,
                    "message": message,
                }
            ),
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
            self._send(
                out,
                {
                    "type": "ready",
                    "protocol_version": PROTOCOL_VERSION,
                    "abi": SIDECAR_ABI,
                    "plugin": self.name,
                    "manifest": self.manifest(),
                },
            )
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
        try:
            chunks = list(self.tts.synthesize(message.get("text", "")))
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
                for chunk in self.stt.transcribe(pcm):
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
