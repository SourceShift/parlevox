"""Make the bundled SDK importable regardless of how pytest is invoked.

A caller may run e.g. ``python -m pytest sdks/python/tests/test_plugin.py`` from
the repository root without setting ``PYTHONPATH=sdks/python``. Anchor the SDK
directory on ``sys.path`` here so collection never depends on the caller's
environment.
"""

import sys
from pathlib import Path

_SDK_ROOT = Path(__file__).resolve().parents[1]
if str(_SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(_SDK_ROOT))
