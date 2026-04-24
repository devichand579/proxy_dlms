"""Persist resumable CST eval scalars; loads named JSON or ``cst_eval_agg.json`` if the former is missing."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from cst.eval.run_eval_cst import new_eval_aggregator
from cst.naming import LEGACY_CST_EVAL_AGG_BASENAME, cst_resumable_eval_agg_path

logger = logging.getLogger("cst.eval.agg_persist")

AGG_VERSION = 1


def cst_eval_agg_path(output_dir: str, cst_eval_name_token: str) -> str:
    """Alias of :func:`cst.naming.cst_resumable_eval_agg_path`."""
    return cst_resumable_eval_agg_path(output_dir, cst_eval_name_token)


def _agg_paths_to_try(output_dir: str, cst_eval_name_token: str) -> list[str]:
    d = str(output_dir)
    return [
        cst_resumable_eval_agg_path(d, cst_eval_name_token),
        os.path.join(d, LEGACY_CST_EVAL_AGG_BASENAME),
    ]


def save_cst_eval_agg(
    output_dir: str,
    cst_eval_name_token: str,
    agg: Dict[str, List[float]],
    *,
    cst_checkpoint: str,
    cst_name_token: str,
    task_name: str,
    segment_idx: int,
    n_files_merged: int,
) -> None:
    """Write JSON after a CST segment (metadata + per-metric lists)."""
    p = cst_resumable_eval_agg_path(str(output_dir), cst_eval_name_token)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    payload: Dict[str, Any] = {
        "version": AGG_VERSION,
        "cst_checkpoint": str(cst_checkpoint),
        "cst_name_token": str(cst_name_token),
        "task_name": str(task_name),
        "segment_idx": int(segment_idx),
        "n_files_merged": int(n_files_merged),
        "agg": agg,
    }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=0)


def _load_one_file(
    p: str,
    *,
    expect_checkpoint: str,
    expect_name_token: str,
) -> Optional[Tuple[Dict[str, List[float]], int, int, str]]:
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("could not read CST eval agg %s: %s", p, e)
        return None
    v = raw.get("version")
    if v != AGG_VERSION:
        logger.warning("CST eval agg %s: unknown version %r — not loading", p, v)
        return None
    if str(raw.get("cst_checkpoint", "")) != str(expect_checkpoint):
        logger.warning(
            "CST eval agg %s: checkpoint path changed (file=%r vs current=%r) — not loading",
            p,
            raw.get("cst_checkpoint"),
            expect_checkpoint,
        )
        return None
    if str(raw.get("cst_name_token", "")) != str(expect_name_token):
        logger.warning(
            "CST eval agg %s: name token mismatch — not loading prior aggregate", p
        )
        return None
    agg = raw.get("agg")
    if not isinstance(agg, dict):
        return None
    base = new_eval_aggregator()
    for k, default_list in base.items():
        vlist = agg.get(k)
        if isinstance(vlist, list):
            base[k] = [float(x) for x in vlist]
        else:
            base[k] = list(default_list)
    seg = int(raw.get("segment_idx", 0))
    nfm = int(raw.get("n_files_merged", len(base.get("nmse", []))))
    tname = str(raw.get("task_name", "unknown"))
    return base, seg, nfm, tname


def load_cst_eval_agg(
    output_dir: str,
    cst_eval_name_token: str,
    *,
    expect_checkpoint: str,
    expect_name_token: str,
) -> Optional[Tuple[Dict[str, List[float]], int, int, str]]:
    """Try the named aggregate file, then ``cst_eval_agg.json``; require checkpoint and name token to match."""
    for p in _agg_paths_to_try(output_dir, cst_eval_name_token):
        if not os.path.isfile(p):
            continue
        out = _load_one_file(
            p, expect_checkpoint=expect_checkpoint, expect_name_token=expect_name_token
        )
        if out is not None:
            return out
    return None
