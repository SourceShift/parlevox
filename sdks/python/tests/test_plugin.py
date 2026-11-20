"""Plugin dispatch: bounded pending audio and honest cancel semantics."""

import io
from collections import deque

from parlevox_sdk.plugin import (
    MAX_CANCELLED,
    MAX_PENDING_AUDIO_FRAMES,
    CancelledTombstones,
    Plugin,
    _accepts_history,
)
from parlevox_sdk.protocol import (
    KIND_AUDIO,
    KIND_JSON,
    PROTOCOL_VERSION,
    SIDECAR_ABI,
    decode_audio,
    decode_json,
    encode_audio,
    encode_frame,
    encode_json,
    read_frame,
)


def json_frame(message):
    return encode_frame(KIND_JSON, encode_json(message))


def audio_frame(request_id, seq=0, pcm=b"\x00\x00\x00\x00"):
    header = {
        "request_id": request_id,
        "seq": seq,
        "sample_rate": 16_000,
        "channels": 1,
    }
    return encode_frame(KIND_AUDIO, encode_audio(header, pcm))


def json_messages(blob):
    return [decode_json(payload) for kind, payload in read_all(blob) if kind == KIND_JSON]


def run_frames(plugin, frames):
    out = io.BytesIO()
    plugin.run(io.BytesIO(b"".join(frames)), out)
    return out.getvalue()


def read_all(blob):
    stream = io.BytesIO(blob)
    frames = []
    while True:
        try:
            frames.append(read_frame(stream))
        except EOFError:
            return frames


def test_cancel_is_acknowledged_and_plugin_keeps_serving():
    plugin = Plugin()
    out = io.BytesIO()
    assert plugin.handle({"type": "cancel", "request_id": "r1"}, out) is True

    frames = read_all(out.getvalue())
    assert len(frames) == 1
    kind, payload = frames[0]
    assert kind == KIND_JSON
    assert decode_json(payload) == {"type": "cancel_ack", "request_id": "r1"}

    # Still able to synthesize afterwards: cancel did not kill the process.
    out = io.BytesIO()
    plugin.handle(
        {
            "type": "synthesize",
            "request_id": "r2",
            "text": "hi",
            "sample_rate": 16000,
            "channels": 1,
        },
        out,
    )
    frames = read_all(out.getvalue())
    messages = [decode_json(p) for k, p in frames if k == KIND_JSON]
    audio = [p for k, p in frames if k == KIND_AUDIO]
    assert messages[0]["type"] == "audio_start"
    assert messages[-1] == {
        "type": "done",
        "request_id": "r2",
        "dropped_frames": 0,
    }
    assert len(audio) == 2
    for payload in audio:
        header, pcm = decode_audio(payload)
        assert header["request_id"] == "r2"
        assert len(pcm) % 2 == 0


class RecordingLlm:
    """Records how the plugin called it, including structured history."""

    def __init__(self):
        self.calls = []

    def generate(self, prompt, history=()):
        self.calls.append((prompt, list(history)))
        return ["ok"]


class OneArgLlm:
    """A third-party provider that predates the optional history argument."""

    def __init__(self):
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return ["ok"]


class KwargsLlm:
    """Accepts history only through ``**kwargs``."""

    def __init__(self):
        self.kwargs = []

    def generate(self, prompt, **kwargs):
        self.kwargs.append(kwargs)
        return ["ok"]


class OpaqueLlm:
    """A provider whose callable cannot be introspected."""

    generate = object()


def test_history_acceptance_follows_the_callable_signature():
    assert _accepts_history(RecordingLlm()) is True
    assert _accepts_history(KwargsLlm()) is True
    assert _accepts_history(OneArgLlm()) is False
    # Uninspectable callables are treated as one-argument rather than crashing.
    assert _accepts_history(OpaqueLlm()) is False

    class PositionalStar:
        def generate(self, prompt, *history):
            return ["ok"]

    # A variadic positional parameter cannot be supplied by keyword.
    assert _accepts_history(PositionalStar()) is False


def generate_messages(plugin, message):
    out = io.BytesIO()
    plugin.handle(message, out)
    return [decode_json(payload) for kind, payload in read_all(out.getvalue()) if kind == KIND_JSON]


def test_generate_forwards_structured_history_when_present():
    llm = RecordingLlm()
    plugin = Plugin(llm=llm)
    history = [
        {"role": "user", "text": "first"},
        {"role": "assistant", "text": "reply"},
    ]

    messages = generate_messages(
        plugin,
        {"type": "generate", "request_id": "r1", "prompt": "next", "history": history},
    )

    assert llm.calls == [("next", history)]
    assert [m["text"] for m in messages if m["type"] == "delta"] == ["ok"]
    assert messages[-1] == {"type": "done", "request_id": "r1", "dropped_frames": 0}


def test_generate_keeps_one_argument_providers_working():
    llm = OneArgLlm()
    plugin = Plugin(llm=llm)

    # No history key at all: the legacy wire shape.
    messages = generate_messages(
        plugin, {"type": "generate", "request_id": "r1", "prompt": "next"}
    )
    assert [m["text"] for m in messages if m["type"] == "delta"] == ["ok"]

    # An explicit empty history is treated the same way.
    messages = generate_messages(
        plugin,
        {"type": "generate", "request_id": "r2", "prompt": "next", "history": []},
    )
    assert [m["text"] for m in messages if m["type"] == "delta"] == ["ok"]
    assert messages[-1] == {"type": "done", "request_id": "r2", "dropped_frames": 0}

    # Turn two: history is non-empty, but this provider cannot accept it.
    # It must still serve the turn instead of raising a TypeError that the
    # plugin would misreport as a transient failure.
    messages = generate_messages(
        plugin,
        {
            "type": "generate",
            "request_id": "r3",
            "prompt": "next",
            "history": [{"role": "user", "text": "first"}],
        },
    )
    assert [m["text"] for m in messages if m["type"] == "delta"] == ["ok"]
    assert messages[-1] == {"type": "done", "request_id": "r3", "dropped_frames": 0}
    assert not any(message["type"] == "error" for message in messages)
    assert llm.prompts == ["next", "next", "next"]


def test_unknown_message_is_a_protocol_error_event():
    plugin = Plugin()
    out = io.BytesIO()
    plugin.handle({"type": "nonsense", "request_id": "r3"}, out)
    frames = read_all(out.getvalue())
    kind, payload = frames[0]
    message = decode_json(payload)
    assert message["type"] == "error"
    assert message["class"] == "protocol"


def test_shutdown_stops_the_loop():
    plugin = Plugin()
    assert plugin.handle({"type": "shutdown"}, io.BytesIO()) is False


def test_pending_audio_is_capped_and_counts_deterministic_drops():
    plugin = Plugin()
    overflow = 3
    total = MAX_PENDING_AUDIO_FRAMES + overflow
    frames = [json_frame({"type": "transcribe", "request_id": "r1", "sample_rate": 16000, "channels": 1})]
    frames.extend(audio_frame("r1", seq=i) for i in range(total))
    frames.append(json_frame({"type": "audio_end", "request_id": "r1"}))

    messages = json_messages(run_frames(plugin, frames))
    done = messages[-1]
    assert done == {"type": "done", "request_id": "r1", "dropped_frames": overflow}
    # The bounded buffer still contained exactly the retained frames, so the
    # request completes normally instead of failing or growing unbounded.
    assert any(message["type"] == "final" for message in messages)

    # Identical input gives an identical count: no timing or hash dependence.
    again = json_messages(run_frames(Plugin(), frames))
    assert again[-1] == done


def test_pending_audio_at_capacity_drops_nothing():
    plugin = Plugin()
    frames = [json_frame({"type": "transcribe", "request_id": "r1", "sample_rate": 16000, "channels": 1})]
    frames.extend(audio_frame("r1", seq=i) for i in range(MAX_PENDING_AUDIO_FRAMES))
    frames.append(json_frame({"type": "audio_end", "request_id": "r1"}))

    done = json_messages(run_frames(plugin, frames))[-1]
    assert done == {"type": "done", "request_id": "r1", "dropped_frames": 0}


def test_cancel_purges_state_before_acking():
    plugin = Plugin()
    out = io.BytesIO()
    pending_audio = {"r1": deque([b"\x00\x00"], maxlen=MAX_PENDING_AUDIO_FRAMES)}
    pending_requests = {"r1": {"type": "transcribe", "request_id": "r1"}}
    pending_dropped = {"r1": 0}
    cancelled = CancelledTombstones()

    assert (
        plugin.handle(
            {"type": "cancel", "request_id": "r1"},
            out,
            pending_audio,
            pending_requests,
            pending_dropped,
            cancelled,
        )
        is True
    )

    # Teardown completed before the ack was written.
    assert pending_audio == {}
    assert pending_requests == {}
    assert pending_dropped == {}
    assert "r1" in cancelled
    assert json_messages(out.getvalue()) == [{"type": "cancel_ack", "request_id": "r1"}]


def test_cancel_suppresses_pending_work_and_plugin_keeps_serving():
    plugin = Plugin()
    frames = [json_frame({"type": "transcribe", "request_id": "r1", "sample_rate": 16000, "channels": 1})]
    frames.extend(audio_frame("r1", seq=i) for i in range(3))
    frames.append(json_frame({"type": "cancel", "request_id": "r1"}))
    # A late audio_end for the cancelled id must not resurrect the request.
    frames.append(json_frame({"type": "audio_end", "request_id": "r1"}))
    # The same process serves the next request normally.
    frames.append(
        json_frame(
            {
                "type": "synthesize",
                "request_id": "r2",
                "text": "hi",
                "sample_rate": 16000,
                "channels": 1,
            }
        )
    )

    messages = json_messages(run_frames(plugin, frames))
    assert messages[0] == {"type": "cancel_ack", "request_id": "r1"}
    assert not any(message.get("request_id") == "r1" and message["type"] != "cancel_ack" for message in messages)
    assert not any(message["type"] in ("partial", "final", "error") for message in messages)
    assert messages[-1] == {
        "type": "done",
        "request_id": "r2",
        "dropped_frames": 0,
    }


def test_cancel_for_unknown_id_is_a_noop_ack():
    plugin = Plugin()
    out = io.BytesIO()
    pending_audio = {}
    pending_requests = {}
    pending_dropped = {}
    cancelled = CancelledTombstones()

    assert (
        plugin.handle(
            {"type": "cancel", "request_id": "ghost"},
            out,
            pending_audio,
            pending_requests,
            pending_dropped,
            cancelled,
        )
        is True
    )
    assert len(cancelled) == 0
    assert json_messages(out.getvalue()) == [{"type": "cancel_ack", "request_id": "ghost"}]

    # Idempotent: a second cancel for the same unknown id behaves identically.
    out = io.BytesIO()
    plugin.handle({"type": "cancel", "request_id": "ghost"}, out)
    assert json_messages(out.getvalue()) == [{"type": "cancel_ack", "request_id": "ghost"}]


def test_cancelled_tombstones_never_exceed_capacity():
    tombstones = CancelledTombstones()
    overflow = 5
    for i in range(MAX_CANCELLED + overflow):
        tombstones.remember(f"r{i}")
    assert len(tombstones) == MAX_CANCELLED
    assert tombstones.evicted == overflow


def test_eviction_is_oldest_first_and_counted():
    tombstones = CancelledTombstones(capacity=3)
    for i in range(5):
        tombstones.remember(f"r{i}")
    assert list(tombstones) == ["r2", "r3", "r4"]
    assert tombstones.evicted == 2
    assert "r0" not in tombstones and "r1" not in tombstones

    # Re-remembering a live id neither grows the ledger nor evicts anything.
    tombstones.remember("r4")
    assert len(tombstones) == 3
    assert tombstones.evicted == 2


def test_repeated_late_audio_end_for_live_tombstone_is_a_silent_noop():
    plugin = Plugin()
    frames = [json_frame({"type": "transcribe", "request_id": "r1", "sample_rate": 16000, "channels": 1})]
    frames.extend(audio_frame("r1", seq=i) for i in range(3))
    frames.append(json_frame({"type": "cancel", "request_id": "r1"}))
    # Two late audio_end frames for the same tombstoned id: both stay silent.
    frames.append(json_frame({"type": "audio_end", "request_id": "r1"}))
    frames.append(json_frame({"type": "audio_end", "request_id": "r1"}))
    frames.append(
        json_frame(
            {
                "type": "synthesize",
                "request_id": "r2",
                "text": "hi",
                "sample_rate": 16000,
                "channels": 1,
            }
        )
    )

    messages = json_messages(run_frames(plugin, frames))
    assert messages[0] == {"type": "cancel_ack", "request_id": "r1"}
    assert not any(
        message.get("request_id") == "r1" and message["type"] != "cancel_ack"
        for message in messages
    )
    assert not any(message["type"] in ("partial", "final", "error") for message in messages)
    assert messages[-1]["type"] == "done" and messages[-1]["request_id"] == "r2"


def test_late_audio_end_for_evicted_tombstone_is_rejected_not_executed():
    plugin = Plugin()
    out = io.BytesIO()
    pending_audio = {}
    pending_requests = {}
    pending_dropped = {}
    cancelled = CancelledTombstones(capacity=2)

    # Cancel three known requests; the oldest tombstone (r1) is evicted.
    for request_id in ("r1", "r2", "r3"):
        pending_requests[request_id] = {"type": "transcribe", "request_id": request_id}
        plugin.handle(
            {"type": "cancel", "request_id": request_id},
            out,
            pending_audio,
            pending_requests,
            pending_dropped,
            cancelled,
        )
    assert cancelled.evicted == 1
    assert "r1" not in cancelled

    out = io.BytesIO()
    assert (
        plugin.handle(
            {"type": "audio_end", "request_id": "r1"},
            out,
            pending_audio,
            pending_requests,
            pending_dropped,
            cancelled,
        )
        is True
    )
    messages = json_messages(out.getvalue())
    assert messages == [
        {
            "type": "error",
            "request_id": "r1",
            "class": "protocol",
            "message": "audio_end for unknown request",
        }
    ]
    assert not any(message["type"] in ("partial", "final") for message in messages)


def test_reset_clears_pending_state_and_readvertises_capabilities():
    plugin = Plugin()
    out = io.BytesIO()
    pending_audio = {"r1": deque([b"\x00\x00"], maxlen=MAX_PENDING_AUDIO_FRAMES)}
    pending_requests = {"r1": {"type": "transcribe", "request_id": "r1"}}
    pending_dropped = {"r1": 4}
    cancelled = CancelledTombstones()
    cancelled.remember("r0")

    assert (
        plugin.handle(
            {"type": "reset"},
            out,
            pending_audio,
            pending_requests,
            pending_dropped,
            cancelled,
        )
        is True
    )

    # Every per-session container, including the tombstone ledger, is empty.
    assert pending_audio == {}
    assert pending_requests == {}
    assert pending_dropped == {}
    assert len(cancelled) == 0

    messages = json_messages(out.getvalue())
    assert len(messages) == 1
    ready = messages[0]
    assert ready["type"] == "ready"
    assert ready["protocol_version"] == PROTOCOL_VERSION
    assert ready["abi"] == SIDECAR_ABI
    assert "reset" in ready["capabilities"]
    # The provider manifest still advertises every capability, including reset.
    assert "reset" in ready["manifest"]["capabilities"]

    # The process keeps serving after a reset.
    out = io.BytesIO()
    plugin.handle(
        {
            "type": "synthesize",
            "request_id": "r2",
            "text": "hi",
            "sample_rate": 16000,
            "channels": 1,
        },
        out,
    )
    messages = json_messages(out.getvalue())
    assert messages[0]["type"] == "audio_start"
    assert messages[-1] == {"type": "done", "request_id": "r2", "dropped_frames": 0}


def test_post_cancel_audio_is_not_rebuffered():
    plugin = Plugin()
    out = io.BytesIO()
    pending_audio = {}
    pending_requests = {"r1": {"type": "transcribe", "request_id": "r1"}}
    pending_dropped = {"r1": 0}
    cancelled = CancelledTombstones()

    plugin.handle(
        {"type": "cancel", "request_id": "r1"},
        out,
        pending_audio,
        pending_requests,
        pending_dropped,
        cancelled,
    )
    assert pending_audio == {} and pending_requests == {}

    Plugin._buffer_audio(
        "r1", b"\x00\x00", pending_audio, pending_requests, pending_dropped, cancelled
    )
    assert pending_audio == {}
    assert pending_dropped == {}

    # An evicted tombstone is not an open request either, so audio is ignored.
    tombstones = CancelledTombstones(capacity=1)
    tombstones.remember("old")
    tombstones.remember("new")
    assert "old" not in tombstones
    Plugin._buffer_audio(
        "old", b"\x00\x00", pending_audio, pending_requests, pending_dropped, tombstones
    )
    assert pending_audio == {}
