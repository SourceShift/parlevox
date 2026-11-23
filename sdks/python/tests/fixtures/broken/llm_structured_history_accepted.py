"""Deliberately broken provider: drops structured history from its signature.

Invariant broken: ``llm-structured-history-accepted``. ``generate`` accepts
only ``prompt``, so the host cannot pass prior turns and the conformance kit's
signature pre-check fails exactly this red line.
"""

from parlevox_sdk.adapters.openai_llm import OpenAILlm

#: Vendor cassette directory this provider replays.
FIXTURES = "openai"


class BrokenNoHistory(OpenAILlm):
    """A one-argument ``generate`` that silently loses every prior turn."""

    def generate(self, prompt):
        yield from super().generate(prompt)


PROVIDER = BrokenNoHistory
