"""Deliberately broken provider: blank text yields audio instead of raising.

Invariant broken: ``tts-empty-text-refused``. Non-blank text is delegated to
the real OpenAI TTS adapter, so every other invariant still holds.
"""

from parlevox_sdk.adapters.openai_tts import NATIVE_SAMPLE_RATE, OpenAITTS

#: Vendor cassette directory this provider replays.
FIXTURES = "openai"


class BrokenEmptyTextAudio(OpenAITTS):
    """Speak blank text with a canned frame instead of refusing it."""

    def synthesize(self, text, sample_rate=NATIVE_SAMPLE_RATE, channels=1):
        if not text.strip():
            yield b"\x00\x00" * 160
            return
        yield from super().synthesize(
            text, sample_rate=sample_rate, channels=channels
        )


PROVIDER = BrokenEmptyTextAudio
