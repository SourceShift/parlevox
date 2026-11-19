"""Plugin dispatch: cancel is acknowledged and does not stop the process."""

import io

from voicekit_sdk.plugin import Plugin
from voicekit_sdk.protocol import KIND_AUDIO, KIND_JSON, decode_audio, decode_json, read_frame


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
