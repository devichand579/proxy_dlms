"""Ensure repo and llada-pcfg roots are importable before other package imports."""

from __future__ import annotations

import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent.parent
_REPO = _PKG.parent

for path in (_REPO, _PKG):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)
