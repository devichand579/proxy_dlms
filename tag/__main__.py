"""Entry point for python -m tag."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from run_tag import run_from_argv

if __name__ == "__main__":
    raise SystemExit(run_from_argv(sys.argv[1:]))
