"""Deliberately broken provider: two final transcripts instead of one.

Invariant broken: ``stt-partials-then-one-final``. Everything else is the real
OpenAI STT adapter, so the conformance kit must flag exactly this one red line.
"""

from parlevox_sdk.adapters.openai_stt import OpenAISTT

#: Vendor cassette directory this provider replays.
FIXTURES = "openai"


class BrokenTwoFinals(OpenAISTT):
    """Yield the real final chunk, then a second final chunk."""

    def transcribe(self, pcm, *, sample_rate=None, channels=None):
        yield from super().transcribe(pcm, sample_rate=sample_rate, channels=channels)
        yield {"text": "and a second final", "final": True}


PROVIDER = BrokenTwoFinals
