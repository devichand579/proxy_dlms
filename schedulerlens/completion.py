"""Completion diagnostics + strace eval join for SchedulerLens CSV rows.

Proxy caps (hard-gated M̂ / star Ĵ) live in ``proxies.py``.
This module adds commit / EOS / mask rates, run-level conf/ent means,
and accuracy / NFE from ``fetch_evals/strace_metrics.csv``.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from schedulerlens.proxies import TraceProxyError, validate_trace_for_metrics
from traces.trace_npz import iter_npz_traces
from traces.utils import squeeze_to_1d

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STRACE_METRICS = _REPO_ROOT / "fetch_evals" / "strace_metrics.csv"

# Model → (eos_id, mask_id). Fallback is LLaDA defaults.
_EOS_MASK_BY_MODEL = {
    "llada": (126081, 126336),
    "llada1_5": (126081, 126336),
    "dream": (151643, 151666),
}

# Map short scheduler keys used by SchedulerLens → strace ``mode`` substrings.
_SCHED_TO_MODE_HINTS = {
    "pd": ("pure_diffusion",),
    "bd": ("block_diffusion",),
    "eb": ("eb_sampler",),
    "fdm": ("fdm",),
    "fhs": ("fhs",),
    "dus": ("dus",),
    "top_k": ("top_k",),
    "top_k_c": ("top_k_confidence", "top_k"),
    "top_k_m": ("top_k_top_margin", "top_margin"),
}

COMPLETION_CSV_COLUMNS = (
    "mean_commits_per_step",
    "max_commits_per_step",
    "total_commits",
    "n_parallel_steps",
    "n_steps_mean",
    "final_masks_mean",
    "final_masks_frac",
    "early_stop_rate",
    "eos_in_output_rate",
    "output_len_mean",
    "gen_length",
    "conf_mean",
    "ent_mean",
    "accuracy",
    "avg_nfe_per_sample",
    "strace_name",
    "strace_metric_key",
)


def _vec(val, n: int, dtype) -> Optional[np.ndarray]:
    if val is None:
        return None
    a = squeeze_to_1d(val).astype(dtype, copy=False)
    if a.shape[0] != n:
        return None
    return a


def _sel(steps: list, t: int, n: int) -> np.ndarray:
    st = steps[t]
    sel = _vec(st.get("selected_positions"), n, np.bool_)
    if sel is not None and int(sel.sum()) > 0:
        return sel
    if t + 1 < len(steps):
        m0 = _vec(st.get("mask_index"), n, np.bool_)
        m1 = _vec(steps[t + 1].get("mask_index"), n, np.bool_)
        if m0 is not None and m1 is not None:
            return m0 & ~m1
    return np.zeros(n, dtype=bool)


def _final_gen_ids(trace: Mapping[str, Any], gen_length: int) -> Optional[np.ndarray]:
    final = trace.get("final_sequence")
    if final is None:
        return None
    a = np.asarray(final)
    if a.ndim == 2:
        a = a[0]
    a = squeeze_to_1d(a).astype(np.int64, copy=False)
    pl = int(trace.get("prompt_len") or 0)
    if pl <= 0 and a.size > gen_length:
        pl = int(a.size - gen_length)
    if pl >= 0 and a.size >= pl + gen_length:
        return a[pl : pl + gen_length]
    if a.size == gen_length:
        return a
    return a[-gen_length:] if a.size >= gen_length else a


def _eos_mask_ids(model: str) -> tuple[int, int]:
    key = str(model or "").strip().lower()
    for name, ids in _EOS_MASK_BY_MODEL.items():
        if key == name or key.startswith(name):
            return ids
    return _EOS_MASK_BY_MODEL["llada"]


def _safe_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def sample_completion_metrics(
    trace: Mapping[str, Any],
    *,
    model: str,
    steps_budget: int,
    trace_name: Optional[Union[str, Path]] = None,
) -> Dict[str, float]:
    """Per-sample commit / EOS / mask diagnostics + conf/ent means."""
    validate_trace_for_metrics(trace, trace_name=trace_name)
    steps = trace.get("step_traces") or []
    if not steps:
        return {}
    n = int(trace.get("gen_length") or 0)
    if n <= 0:
        m0 = _vec(steps[0].get("mask_index"), 0, np.bool_)
        n = int(m0.shape[0]) if m0 is not None else 0
    if n <= 0:
        return {}

    eos_id, mask_id = _eos_mask_ids(model)
    n_tok = 0.0
    n_par = 0.0
    commits: list[float] = []
    conf_vals: list[float] = []
    ent_vals: list[float] = []
    for t, st in enumerate(steps):
        sel = _sel(steps, t, n)
        k = int(sel.sum())
        commits.append(float(k))
        conf = _vec(st.get("confidences"), n, np.float64)
        ent = _vec(st.get("entropies"), n, np.float64)
        if conf is not None:
            conf_vals.extend(conf.tolist())
        if ent is not None:
            ent_vals.extend(ent.tolist())
        if k == 0:
            continue
        n_tok += float(k)
        if k >= 2:
            n_par += 1.0

    last_mask = _vec(steps[-1].get("mask_index"), n, np.bool_)
    final_masks = float(last_mask.sum()) if last_mask is not None else float("nan")

    gen = _final_gen_ids(trace, n)
    eos_hit = 0.0
    out_len = float(n - final_masks) if np.isfinite(final_masks) else float("nan")
    if gen is not None and gen.size > 0:
        eos_pos = np.where(gen == eos_id)[0]
        if eos_pos.size > 0:
            eos_hit = 1.0
            out_len = float(eos_pos[0])
        else:
            out_len = float(np.sum(gen != mask_id))

    n_steps = float(len(steps))
    early = 1.0 if (steps_budget > 0 and n_steps + 1e-9 < float(steps_budget)) else 0.0
    commits_arr = np.asarray(commits, dtype=np.float64)
    return {
        "mean_commits_per_step": float(np.mean(commits_arr)) if commits_arr.size else 0.0,
        "max_commits_per_step": float(np.max(commits_arr)) if commits_arr.size else 0.0,
        "total_commits": n_tok,
        "n_parallel_steps": n_par,
        "n_steps": n_steps,
        "final_masks": final_masks,
        "final_masks_frac": (
            (final_masks / float(n)) if np.isfinite(final_masks) and n else float("nan")
        ),
        "early_stop": early,
        "eos_in_output": eos_hit,
        "output_len": out_len,
        "gen_length": float(n),
        "conf_mean": float(np.mean(conf_vals)) if conf_vals else float("nan"),
        "ent_mean": float(np.mean(ent_vals)) if ent_vals else float("nan"),
    }


def compute_completion_metrics(
    trace_path: Union[str, Path],
    *,
    model: str,
    steps_budget: int,
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    """Mean completion diagnostics over samples in one NPZ."""
    bags: Dict[str, list] = {}
    n = 0
    path = Path(trace_path)
    for i, tr in enumerate(iter_npz_traces(path)):
        if max_samples is not None and i >= max_samples:
            break
        try:
            rec = sample_completion_metrics(
                tr,
                model=model,
                steps_budget=steps_budget,
                trace_name=path,
            )
        except TraceProxyError:
            raise
        if not rec:
            continue
        n += 1
        for k, v in rec.items():
            bags.setdefault(k, []).append(v)
    if n == 0:
        return {
            k: float("nan")
            for k in (
                "mean_commits_per_step",
                "max_commits_per_step",
                "total_commits",
                "n_parallel_steps",
                "n_steps_mean",
                "final_masks_mean",
                "final_masks_frac",
                "early_stop_rate",
                "eos_in_output_rate",
                "output_len_mean",
                "gen_length",
                "conf_mean",
                "ent_mean",
            )
        }
    out = {k: float(np.nanmean(v)) for k, v in bags.items()}
    out["early_stop_rate"] = out.pop("early_stop")
    out["eos_in_output_rate"] = out.pop("eos_in_output")
    out["final_masks_mean"] = out.pop("final_masks")
    out["n_steps_mean"] = out.pop("n_steps")
    out["output_len_mean"] = out.pop("output_len")
    return out


def load_strace_metrics(
    path: Optional[Union[str, Path]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Index strace_metrics.csv by run ``name`` (filename stem)."""
    path = Path(path) if path else DEFAULT_STRACE_METRICS
    if not path.exists():
        logger.warning("strace_metrics.csv not found: %s", path)
        return {}
    by_name: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            name = str(row.get("name") or "").strip()
            if not name:
                continue
            by_name[name] = {
                "name": name,
                "model": row.get("model"),
                "task": row.get("task"),
                "mode": row.get("mode"),
                "steps": row.get("steps"),
                "metric": _safe_float(row.get("metric")),
                "avg_nfe_per_sample": _safe_float(row.get("avg_nfe_per_sample")),
                "metric_key": row.get("metric_key"),
            }
    return by_name


def _stem_from_row(row: Mapping[str, Any]) -> str:
    filename = str(row.get("filename") or "")
    if filename.endswith(".npz"):
        return filename[:-4]
    if filename:
        return Path(filename).stem
    return ""


def _fallback_match(
    row: Mapping[str, Any],
    catalog: Mapping[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    model = str(row.get("model") or "")
    task = str(row.get("task") or "")
    steps = str(row.get("steps") or "")
    sched = str(row.get("scheduler") or "")
    hints = _SCHED_TO_MODE_HINTS.get(sched, (sched,))
    candidates = []
    for entry in catalog.values():
        if str(entry.get("model")) != model:
            continue
        if str(entry.get("task")) != task:
            continue
        if str(entry.get("steps")) != steps:
            continue
        mode = str(entry.get("mode") or "")
        if any(h in mode for h in hints):
            candidates.append(entry)
    if len(candidates) == 1:
        return candidates[0]
    return None


def join_rows_with_strace(
    rows: Sequence[Mapping[str, Any]],
    *,
    strace_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """Attach accuracy / NFE from strace_metrics onto SchedulerLens rows.

    Primary join: filename stem == strace ``name``.
    Fallback: match model + task + steps + mode hint from scheduler.
    """
    catalog = load_strace_metrics(strace_path)
    joined: List[Dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        stem = _stem_from_row(row)
        hit = catalog.get(stem)
        if hit is None:
            hit = _fallback_match(row, catalog)
        if hit is not None:
            out["accuracy"] = hit["metric"]
            out["avg_nfe_per_sample"] = hit["avg_nfe_per_sample"]
            out["strace_name"] = hit["name"]
            out["strace_metric_key"] = hit.get("metric_key")
        else:
            out["accuracy"] = float("nan")
            out["avg_nfe_per_sample"] = float("nan")
            out["strace_name"] = ""
            out["strace_metric_key"] = ""
        joined.append(out)
    return joined
