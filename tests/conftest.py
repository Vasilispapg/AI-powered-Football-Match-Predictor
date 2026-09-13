"""Make the package importable when the tests run from a source checkout.

``pip install -e .`` is the supported setup, but the suite should also run on a
bare clone so that a failing install does not look like a failing test.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
