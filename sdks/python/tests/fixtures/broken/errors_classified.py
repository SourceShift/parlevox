"""Deliberately broken provider: swallows a classified failure as success.

Invariant broken: ``errors-classified``. A non-cancelled ``AdapterError`` is
turned into a normal final chunk, so a 500 looks like a successful
transcription. Cancellation is deliberately re-raised so
``cancellation-propagates`` stays green.
"""

from parlevox_sdk.adapters.base import AdapterError, ErrorClass
from parlevox_sdk.adapters.deepgram_stt import DeepgramSTT

#: Vendor cassette directory this provider replays.
FIXTURES = "deepgram"


class BrokenSwallowsErrors(DeepgramSTT):
    """Yield a final chunk for any classified failure except cancellation."""

    def transcribe(self, pcm, *, sample_rate=None, channels=None):
        try:
            yield from super().transcribe(
                pcm, sample_rate=sample_rate, channels=channels
            )
        except AdapterError as err:
            if err.classification is ErrorClass.CANCELLED:
                raise
            yield {"text": "", "final": True}


PROVIDER = BrokenSwallowsErrors
