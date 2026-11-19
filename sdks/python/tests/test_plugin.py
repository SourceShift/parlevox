"""Plugin dispatch: bounded pending audio and honest cancel semantics."""

import io
from collections import deque

from voicekit_sdk.plugin import MAX_PENDING_AUDIO_FRAMES, Plugin
from voicekit_sdk.protocol import (
    KIND_AUDIO,
    KIND_JSON,
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
    assert messages[-1] == {"type": "done", "request_id": "r2"}
    assert len(audio) == 2
    for payload in audio:
        header, pcm = decode_audio(payload)
        assert header["request_id"] == "r2"
        assert len(pcm) % 2 == 0


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
    cancelled = set()

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
    assert messages[-1] == {"type": "done", "request_id": "r2"}


def test_cancel_for_unknown_id_is_a_noop_ack():
    plugin = Plugin()
    out = io.BytesIO()
    pending_audio = {}
    pending_requests = {}
    pending_dropped = {}
    cancelled = set()

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
    assert cancelled == set()
    assert json_messages(out.getvalue()) == [{"type": "cancel_ack", "request_id": "ghost"}]

    # Idempotent: a second cancel for the same unknown id behaves identically.
    out = io.BytesIO()
    plugin.handle({"type": "cancel", "request_id": "ghost"}, out)
    assert json_messages(out.getvalue()) == [{"type": "cancel_ack", "request_id": "ghost"}]
