"""
Repo layout and shared path helpers for llada-pcfg.

All generated artifacts (checkpoints, corpora, eval outputs) live under the
Strace repo root — not inside ``llada-pcfg/``.

Typical layout::

    Strace/
      checkpoints/llada-{variant}-pcfg/
      data/pcfg/
      eval_outputs/pcfg/
      llada-pcfg/
        scripts/
        pcfg_utils/
        utils.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_PACKAGE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_ROOT.parent

CHECKPOINTS_DIRNAME = "checkpoints"
DATA_DIRNAME = "data"
PCFG_DATA_SUBDIR = "pcfg"
EVAL_OUTPUTS_DIRNAME = "eval_outputs"
PCFG_EVAL_SUBDIR = "pcfg"


def package_root() -> Path:
    return _PACKAGE_ROOT


def repo_root() -> Path:
    return _REPO_ROOT


def checkpoints_dir() -> Path:
    return _REPO_ROOT / CHECKPOINTS_DIRNAME


def data_dir() -> Path:
    return _REPO_ROOT / DATA_DIRNAME


def pcfg_corpus_dir() -> Path:
    return data_dir() / PCFG_DATA_SUBDIR


def eval_outputs_dir() -> Path:
    return _REPO_ROOT / EVAL_OUTPUTS_DIRNAME / PCFG_EVAL_SUBDIR


def default_checkpoint_dir(variant: str) -> Path:
    key = str(variant).lower().replace("llada-", "")
    return checkpoints_dir() / f"llada-{key}-pcfg"


def resolve_repo_path(path: str | Path | None, *, base: Path | None = None) -> Path | None:
    """Resolve a path relative to repo root (or ``base`` when provided)."""
    if path is None:
        return None
    text = str(path).strip()
    if not text or text.lower() in ("null", "none"):
        return None
    resolved = Path(text).expanduser()
    if not resolved.is_absolute():
        resolved = (base or _REPO_ROOT) / resolved
    return resolved.resolve()


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=indent) + "\n")
    return target


def ensure_import_paths() -> Path:
    """Add repo and package roots to ``sys.path``; return package root."""
    for path in (_REPO_ROOT, _PACKAGE_ROOT):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    return _PACKAGE_ROOT
