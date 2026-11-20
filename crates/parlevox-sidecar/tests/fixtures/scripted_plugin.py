"""A tiny scripted plugin for reset and pool tests.

It speaks the same framed protocol as ``parlevox_sdk`` but depends only on the
standard library, so pool tests do not need the bundled SDK on ``PYTHONPATH``.
The mode is the first positional argument:

``ack``
    Advertise ``reset`` and acknowledge every reset with ``ready``.
``no_reset_capability``
    A plugin built before the additive field: ``ready`` omits ``capabilities``.
``fail_all_resets``
    Advertise ``reset`` but answer every reset with a session-level error.
``fail_second_reset``
    Acknowledge the first reset, reject every later one. This exercises the
    acquire-time probe: release succeeds, the next acquire must replace the
    process rather than hand it out.
``never_ack``
    Advertise ``reset`` and never answer it, so the client's timeout fires.
``hold_requests``
    Advertise ``reset`` and never complete a request, leaving a route open.
"""

import json
import struct
import sys

PROTOCOL_VERSION = 1
SIDECAR_ABI = "parlevox-sidecar/1"
KIND_JSON = 1


def read_frame(stream):
    header = stream.read(4)
    if len(header) < 4:
        return None
    (body_len,) = struct.unpack(">I", header)
    body = stream.read(body_len)
    if len(body) < body_len:
        return None
    return body[0], body[1:]


def write_json(stream, message):
    payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
    body = bytes([KIND_JSON]) + payload
    stream.write(struct.pack(">I", len(body)) + body)
    stream.flush()


def ready_payload(advertise_reset):
    payload = {
        "type": "ready",
        "protocol_version": PROTOCOL_VERSION,
        "abi": SIDECAR_ABI,
        "plugin": "scripted",
        "manifest": {
            "name": "scripted",
            "modality": "tts",
            "capabilities": ["reset"] if advertise_reset else [],
            "config_schema": {"type": "object", "properties": {}},
        },
    }
    if advertise_reset:
        payload["capabilities"] = ["reset"]
    return payload


def main(mode):
    advertise_reset = mode != "no_reset_capability"
    resets = 0
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    while True:
        frame = read_frame(stdin)
        if frame is None:
            return
        kind, payload = frame
        if kind != KIND_JSON:
            continue
        try:
            message = json.loads(payload.decode("utf-8"))
        except ValueError:
            return
        message_type = message.get("type")
        if message_type == "shutdown":
            return
        if message_type == "hello":
            write_json(stdout, ready_payload(advertise_reset))
        elif message_type == "reset":
            resets += 1
            if mode == "never_ack":
                continue
            if mode == "fail_all_resets" or (
                mode == "fail_second_reset" and resets > 1
            ):
                write_json(
                    stdout,
                    {
                        "type": "error",
                        "request_id": "",
                        "class": "protocol",
                        "message": "reset refused by scripted plugin",
                    },
                )
                continue
            write_json(stdout, ready_payload(advertise_reset))
        elif message_type in ("synthesize", "generate", "transcribe", "analyze"):
            if mode == "hold_requests":
                continue
            # Nothing else is needed for lifecycle tests.
            continue


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ack")
