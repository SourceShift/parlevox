"""End-to-end sidecar test: spawn the plugin, handshake, audio, cancel, re-use.

No network, no install: the child runs from ``PYTHONPATH=sdks/python``.
"""

import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from parlevox_sdk.fakes import FAKE_LLM_DELTAS, FAKE_TTS_FRAMES
from parlevox_sdk.protocol import (
    KIND_AUDIO,
    KIND_JSON,
    PROTOCOL_VERSION,
    SIDECAR_ABI,
    decode_audio,
    decode_json,
    encode_json,
    read_frame,
    write_frame,
)

SDK_DIR = Path(__file__).resolve().parents[1]


def _env():
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SDK_DIR) + (os.pathsep + existing if existing else "")
    return env


@pytest.fixture
def plugin_process():
    process = subprocess.Popen(
        [sys.executable, "-u", "-m", "parlevox_sdk.sidecar_main"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_env(),
    )
    yield process
    if process.poll() is None:
        process.kill()
        process.wait(timeout=5)
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            stream.close()


def send(process, kind, payload):
    write_frame(process.stdin, kind, payload)


def frame_reader(stream):
    """Read frames on a thread; a buffered stream makes select() unreliable."""
    frames = queue.Queue()

    def worker():
        while True:
            try:
                frames.put(read_frame(stream))
            except Exception as err:  # noqa: BLE001 - surfaced to the test
                frames.put(err)
                return

    threading.Thread(target=worker, daemon=True).start()
    return frames


def recv(frames, timeout=10):
    item = frames.get(timeout=timeout)
    if isinstance(item, Exception):
        raise item
    return item


def recv_json_until(frames, wanted):
    while True:
        kind, payload = recv(frames)
        if kind == KIND_JSON:
            message = decode_json(payload)
            if message["type"] == wanted:
                return message


def test_hello_synthesize_cancel_and_reuse(plugin_process):
    process = plugin_process
    frames = frame_reader(process.stdout)

    send(
        process,
        KIND_JSON,
        encode_json(
            {
                "type": "hello",
                "protocol_version": PROTOCOL_VERSION,
                "abi": SIDECAR_ABI,
                "modalities": ["tts"],
            }
        ),
    )
    ready = recv_json_until(frames, "ready")
    assert ready["protocol_version"] == PROTOCOL_VERSION
    assert ready["abi"] == SIDECAR_ABI
    assert ready["manifest"]["name"]

    # TTS: audio_start, binary PCM frames, done.
    send(
        process,
        KIND_JSON,
        encode_json(
            {
                "type": "synthesize",
                "request_id": "tts-1",
                "text": "hi",
                "sample_rate": 16000,
                "channels": 1,
            }
        ),
    )
    audio_frames = []
    seen_start = False
    while True:
        kind, payload = recv(frames)
        if kind == KIND_AUDIO:
            header, pcm = decode_audio(payload)
            assert header["request_id"] == "tts-1"
            assert len(pcm) % 2 == 0 and pcm
            audio_frames.append(pcm)
            continue
        message = decode_json(payload)
        if message["type"] == "audio_start":
            seen_start = True
        if message["type"] == "done":
            break
    assert seen_start
    assert len(audio_frames) == FAKE_TTS_FRAMES

    # Cancel is acknowledged and leaves the process running.
    send(process, KIND_JSON, encode_json({"type": "cancel", "request_id": "tts-1"}))
    ack = recv_json_until(frames, "cancel_ack")
    assert ack["request_id"] == "tts-1"

    # The same process still serves a completion request.
    send(
        process,
        KIND_JSON,
        encode_json({"type": "generate", "request_id": "llm-1", "prompt": "hello"}),
    )
    deltas = []
    while True:
        _, payload = recv(frames)
        message = decode_json(payload)
        if message["type"] == "delta":
            deltas.append(message["text"])
        if message["type"] == "done":
            break
    assert tuple(deltas) == FAKE_LLM_DELTAS

    send(process, KIND_JSON, encode_json({"type": "shutdown"}))
    assert process.wait(timeout=10) == 0
