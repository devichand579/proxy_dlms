"""Path tokens for CST train/eval: model, scheduler, k/w/lambda_sparsity, task (staged hiddens and aggregates)."""

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


def lambda_sparsity_token(lambda_sparsity: float) -> str:
    """Filesystem-safe ``ls…`` tag (e.g. ``ls0.001``, ``ls3e-06``)."""
    v = float(lambda_sparsity)
    if v == 0.0:
        return "ls0"
    return "ls" + fs_token(f"{v:.6g}", max_len=16)


def cst_hyper_suffix(k: int, w: int, lambda_sparsity: float) -> str:
    """Compact hyperparam segment (e.g. ``k4096_w4_ls0.001``)."""
    return f"k{int(k)}_w{int(w)}_{lambda_sparsity_token(lambda_sparsity)}"


def cst_train_name_token(
    model_label: str,
    scheduler_mode_short: str,
    k: int,
    w: int,
    lambda_sparsity: float,
) -> str:
    """Token for training staging files and ``cst_*_train_*.pt``."""
    hyp = cst_hyper_suffix(k, w, lambda_sparsity)
    return f"{fs_token(model_label)}_scheduler_{fs_token(scheduler_mode_short)}_{hyp}_train"


def cst_hidden_segment_prefix(name_token: str, segment: int) -> str:
    """Segment prefix shared by train and eval hidden chunks."""
    return f"{fs_token(name_token, max_len=200)}_seg{int(segment):03d}"


def cst_hidden_chunk_basename(name_token: str, segment: int, batch_idx: int) -> str:
    """Unified hidden chunk filename: ``{name_token}_seg{seg}_{batch:06d}.pt``."""
    return f"{cst_hidden_segment_prefix(name_token, segment)}_{int(batch_idx):06d}.pt"


_HIDDEN_CHUNK_BATCH_SUFFIX = re.compile(r"_(\d{6})\.pt$")


def parse_hidden_chunk_batch_idx(filename: str) -> Optional[int]:
    """Parse global ``batch_idx`` from a hidden chunk basename."""
    m = _HIDDEN_CHUNK_BATCH_SUFFIX.search(str(filename))
    return int(m.group(1)) if m else None


def cst_eval_task_name_token(
    model_label: str,
    scheduler_tag: str,
    task_name: str,
    k: int,
    w: int,
    lambda_sparsity: float,
) -> str:
    """Token for eval hidden files (includes ``k``, ``w``, ``lambda_sparsity``, task name)."""
    hyp = cst_hyper_suffix(k, w, lambda_sparsity)
    return (
        f"{fs_token(model_label)}_scheduler_{fs_token(scheduler_tag)}_{hyp}"
        f"_task_{fs_token(str(task_name))}"
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
    """Return ``(latest, best, state_json)``; long names with ``name_token``, or short fixed basenames.

    ``latest`` / ``best`` name *directories* (sharded checkpoints, see
    :func:`cst.train.trainer.save_cst_checkpoint`). The resolvers below still
    accept the legacy single-``.pt`` form.
    """
    if name_token:
        base = f"cst_{name_token}"
        return f"{base}_latest", f"{base}_best", f"{base}_state.json"
    return "cst_latest", "cst_best", "cst_train_state.json"


def _first_existing(*cands: Path) -> Optional[Path]:
    for c in cands:
        if c.exists():
            return c
    return None


def resolve_train_resume_latest(ckpt_dir: Path, name_token: str) -> Optional[Path]:
    """Prefer ``cst_{name_token}_latest`` (dir or legacy ``.pt``); else the short ``cst_latest`` form."""
    d = Path(ckpt_dir)
    return _first_existing(
        d / f"cst_{name_token}_latest",
        d / f"cst_{name_token}_latest.pt",
        d / "cst_latest",
        d / "cst_latest.pt",
    )


def resolve_train_best_in_dir(ckpt_dir: Path) -> Optional[Path]:
    """Prefer ``cst_best``; else newest ``cst_*_train_best`` by mtime (offline eval fallback).

    Matches both sharded directories and legacy ``.pt`` files.
    """
    d = Path(ckpt_dir)
    leg = _first_existing(d / "cst_best", d / "cst_best.pt")
    if leg is not None:
        return leg
    cands = sorted(
        list(d.glob("cst_*_train_best")) + list(d.glob("cst_*_train_best.pt")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if cands:
        return cands[0]
    return None
