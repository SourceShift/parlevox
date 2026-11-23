"""Deliberately broken provider: emits a frame even when the body is empty.

Invariant broken: ``tts-no-unrequested-audio``. The extra frame is appended
only after the real adapter's generator finishes, so a blank-text request still
raises inside the real adapter and ``tts-empty-text-refused`` stays green.
"""

from parlevox_sdk.adapters.elevenlabs_tts import ElevenLabsTTS
from parlevox_sdk.fakes import FAKE_SAMPLE_RATE

#: Vendor cassette directory this provider replays.
FIXTURES = "elevenlabs"


class BrokenUnrequestedAudio(ElevenLabsTTS):
    """Append one unsolicited frame after the real response body."""

    def synthesize(self, text, sample_rate=FAKE_SAMPLE_RATE, channels=1):
        yield from super().synthesize(
            text, sample_rate=sample_rate, channels=channels
        )
        yield b"\x00\x00" * 160


PROVIDER = BrokenUnrequestedAudio
