"""Path tokens for CST train/eval: model, scheduler, k/w, task (staged hiddens and aggregates)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


def fs_token(s: str, *, max_len: int = 80) -> str:
    """Single path component: alnum, dot, dash, underscore; other chars -> underscore."""
    t = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(s).strip())
    t = re.sub(r"_+", "_", t).strip("._-")
    if not t:
        t = "x"
    if len(t) > max_len:
        t = t[:max_len].rstrip("._-") or "x"
    return t


def model_label_from_path(model_path: str) -> str:
    """Last path component (e.g. ``llada`` from ``./models/llada``)."""
    p = Path(str(model_path).strip())
    name = p.name
    if not name or name in (".", ".."):
        return "model"
    return fs_token(name, max_len=64)


def cst_k_w_suffix(k: int, w: int) -> str:
    """Compact ``k``/``w`` segment for path components (e.g. ``k4096_w4``)."""
    return f"k{int(k)}_w{int(w)}"


def cst_train_name_token(
    model_label: str, scheduler_mode_short: str, k: int, w: int
) -> str:
    """Token for training staging files and ``cst_*_train_*.pt`` (includes ``k``, ``w``, and ``_train``)."""
    kw = cst_k_w_suffix(k, w)
    return f"{fs_token(model_label)}_scheduler_{fs_token(scheduler_mode_short)}_{kw}_train"


def cst_train_staging_prefix(
    model_label: str,
    scheduler_mode_short: str,
    k: int,
    w: int,
    epoch: int,
    segment: int,
) -> str:
    """Prefix for collect staging: one chunk file is ``{prefix}_{batch:06d}.pt``."""
    return f"{cst_train_name_token(model_label, scheduler_mode_short, k, w)}_e{int(epoch):03d}_seg{int(segment):03d}"


def cst_eval_task_name_token(
    model_label: str, scheduler_tag: str, task_name: str, k: int, w: int
) -> str:
    """Token for eval hidden files (includes ``k``, ``w``, ``_task_``, and lm-eval task name)."""
    kw = cst_k_w_suffix(k, w)
    return (
        f"{fs_token(model_label)}_scheduler_{fs_token(scheduler_tag)}_{kw}_task_{fs_token(str(task_name))}"
    )


# Alternate eval-aggregate basename (one file per run dir); tried if the named file is absent.
LEGACY_CST_EVAL_AGG_BASENAME = "cst_eval_agg.json"


def cst_resumable_eval_agg_basename(cst_eval_name_token: str) -> str:
    """``cst_<token>_eval_agg.json`` (resumable scalar aggregates; same token as hiddens)."""
    t = str(cst_eval_name_token).strip() or "eval"
    return f"cst_{fs_token(t, max_len=200)}_eval_agg.json"


def cst_resumable_eval_agg_path(output_dir: str, cst_eval_name_token: str) -> str:
    return str(Path(str(output_dir)) / cst_resumable_eval_agg_basename(cst_eval_name_token))


# Inverse of collect ``_mode_map``: (mode, strategy) -> short code for filenames.
_MODE_STRATEGY_TO_SHORT: dict[tuple[str, str], str] = {
    ("pure_diffusion", "confidence"): "pd",
    ("block_diffusion", "confidence"): "bd",
    ("eb_sampler", "confidence"): "eb",
    ("fdm", "confidence"): "fdm",
    ("fhs", "confidence"): "fhs",
    ("top_k", "confidence"): "top_k_c",
    ("top_k", "top_margin"): "top_k_m",
    ("dus", "confidence"): "dus",
}


def scheduler_short_from_eval_mode(mode: str, strategy: str) -> str:
    """Map lm-eval ``mode``/``strategy`` to the same short tags as training ``scheduler_mode``."""
    m = str(mode).strip().lower()
    s = str(strategy).strip().lower()
    if s == "none":
        s = "confidence"
    key = (m, s)
    if key in _MODE_STRATEGY_TO_SHORT:
        return _MODE_STRATEGY_TO_SHORT[key]
    if (m, "confidence") in _MODE_STRATEGY_TO_SHORT and s in ("", "none"):
        return _MODE_STRATEGY_TO_SHORT[(m, "confidence")]
    return fs_token(f"{m}_{s}", max_len=32)


def train_checkpoint_names(name_token: Optional[str]) -> tuple[str, str, str]:
    """Return ``(latest_pt, best_pt, state_json)``; long names with ``name_token``, or short fixed basenames."""
    if name_token:
        base = f"cst_{name_token}"
        return f"{base}_latest.pt", f"{base}_best.pt", f"{base}_state.json"
    return "cst_latest.pt", "cst_best.pt", "cst_train_state.json"


def resolve_train_resume_latest(ckpt_dir: Path, name_token: str) -> Optional[Path]:
    """Prefer ``cst_{name_token}_latest.pt``; else ``cst_latest.pt`` if present."""
    n = Path(ckpt_dir) / f"cst_{name_token}_latest.pt"
    o = Path(ckpt_dir) / "cst_latest.pt"
    if n.is_file():
        return n
    if o.is_file():
        return o
    return None


def resolve_train_best_in_dir(ckpt_dir: Path) -> Optional[Path]:
    """Prefer ``cst_best.pt``; else newest ``cst_*_train_best.pt`` by mtime (offline eval fallback)."""
    d = Path(ckpt_dir)
    leg = d / "cst_best.pt"
    if leg.is_file():
        return leg
    cands = sorted(d.glob("cst_*_train_best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if cands:
        return cands[0]
    return None
