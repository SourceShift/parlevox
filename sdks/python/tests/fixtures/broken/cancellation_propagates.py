"""Deliberately broken provider: advertises a cancel token, ignores it.

Invariant broken: ``cancellation-propagates``. The constructor accepts
``cancel`` so the conformance kit takes the constructor-token branch, but the
token is dropped before reaching the real adapter, so cancelled work runs to
completion. The real adapter's transport still classifies a 500 correctly, so
``errors-classified`` stays green.
"""

from parlevox_sdk.adapters.openai_llm import OpenAILlm

#: Vendor cassette directory this provider replays.
FIXTURES = "openai"


class BrokenIgnoresCancel(OpenAILlm):
    """Accept ``cancel`` in the signature, then pass ``None`` down."""

    def __init__(
        self,
        transport=None,
        *,
        base_url=None,
        model=None,
        environ=None,
        cancel=None,
    ):
        super().__init__(
            transport=transport,
            base_url=base_url,
            model=model,
            environ=environ,
            cancel=None,
        )


PROVIDER = BrokenIgnoresCancel
