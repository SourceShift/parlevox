"""Deliberately broken host: reports zero dropped frames no matter what.

Invariant broken: ``drop-counters-honest``. The pending-audio counter lives in
the plugin, not in an adapter, so this defect is a ``Plugin`` subclass. Every
adapter-level invariant still holds: the slots are the bundled fakes.
"""

from parlevox_sdk.plugin import Plugin

#: Vendor cassette directory the kit is handed; the fakes never read it.
FIXTURES = "openai"


class BrokenDropCounters(Plugin):
    """Under-report ``dropped_frames`` on every terminal ``done`` message."""

    @staticmethod
    def _send(out, message):
        if isinstance(message, dict) and message.get("type") == "done":
            message = dict(message, dropped_frames=0)
        Plugin._send(out, message)


PROVIDER = BrokenDropCounters
