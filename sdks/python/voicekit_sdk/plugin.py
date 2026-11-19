"""The sidecar plugin loop.

One process, one framed stream. Control is JSON, audio is binary. A ``cancel``
is acknowledged and never terminates the process; only ``shutdown`` does.
"""

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

    # -- loop ---------------------------------------------------------
    def run(self, stdin, stdout):
        """Serve frames until EOF, close, or shutdown."""
        pending_audio = {}
        pending_requests = {}
        while True:
            try:
                kind, payload = read_frame(stdin)
            except EOFError:
                return
            if kind == KIND_CLOSE:
                return
            if kind == KIND_AUDIO:
                header, pcm = decode_audio(payload)
                pending_audio.setdefault(header["request_id"], bytearray()).extend(pcm)
                continue
            if kind != KIND_JSON:
                continue
            message = decode_json(payload)
            if not self.handle(message, stdout, pending_audio, pending_requests):
                return

    def handle(self, message, out, pending_audio=None, pending_requests=None):
        """Handle one control message. Returns False to stop the loop."""
        pending_audio = pending_audio if pending_audio is not None else {}
        pending_requests = pending_requests if pending_requests is not None else {}
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
            pending_requests[message["request_id"]] = message
            return True
        if kind == "audio_end":
            return self._finish_audio_request(message["request_id"], out, pending_audio, pending_requests)
        if kind == "cancel":
            self._send(
                out,
                {"type": "cancel_ack", "request_id": message.get("request_id", "")},
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
        self._send(out, {"type": "done", "request_id": request_id})
        return True

    def _generate(self, message, out):
        request_id = message["request_id"]
        try:
            deltas = list(self.llm.generate(message.get("prompt", "")))
        except ValueError as err:
            self._send_error(out, request_id, "protocol", str(err))
            return True
        except Exception as err:  # pragma: no cover - defensive
            self._send_error(out, request_id, "transient", str(err))
            return True
        for delta in deltas:
            self._send(out, {"type": "delta", "request_id": request_id, "text": delta})
        self._send(out, {"type": "done", "request_id": request_id})
        return True

    def _finish_audio_request(self, request_id, out, pending_audio, pending_requests):
        request = pending_requests.pop(request_id, None)
        pcm = bytes(pending_audio.pop(request_id, b""))
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
        self._send(out, {"type": "done", "request_id": request_id})
        return True
