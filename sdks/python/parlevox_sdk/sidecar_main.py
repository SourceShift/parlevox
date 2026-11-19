"""Entry point for ``python -m parlevox_sdk.sidecar_main``.

Reads framed requests on stdin and writes framed events on stdout. Diagnostics
go to stderr; stdout belongs to the protocol.
"""

import sys

from .plugin import Plugin


def main(argv=None):
    del argv
    plugin = Plugin()
    try:
        plugin.run(sys.stdin.buffer, sys.stdout.buffer)
    except BrokenPipeError:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
