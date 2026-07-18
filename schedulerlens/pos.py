"""POS-stratified unmasking analysis from generation traces."""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .metrics import _iter_traces
from .tokenizer_utils import diffusion_skip_token_ids, load_trace_tokenizer

logger = logging.getLogger(__name__)

_NLTK_READY = False


def _ensure_nltk() -> None:
    global _NLTK_READY
    if _NLTK_READY:
        return
    import nltk

    for resource in ("punkt", "punkt_tab", "averaged_perceptron_tagger_eng"):
        try:
            nltk.data.find(
                "tokenizers/punkt"
                if resource == "punkt"
                else (
                    "tokenizers/punkt_tab"
                    if resource == "punkt_tab"
                    else "taggers/averaged_perceptron_tagger_eng"
                )
            )
        except LookupError:
            pkg = resource
            if resource == "punkt_tab":
                pkg = "punkt_tab"
            nltk.download(pkg, quiet=True)
    _NLTK_READY = True


def pos_tag_token(tokenizer, token_id: int) -> Tuple[str, str]:
    """Decode one token id and return (surface, Penn-Treebank POS tag)."""
    _ensure_nltk()
    from nltk import pos_tag

    text = tokenizer.decode([int(token_id)], skip_special_tokens=False).strip()
    if not text:
        return "", "EMPTY"
    tag = pos_tag([text])[0][1]
    return text, tag


def _squeeze_batch(arr) -> Optional[np.ndarray]:
    if arr is None:
        return None
    out = np.asarray(arr)
    while out.ndim > 1:
        out = out[0]
    return out


def reconstruct_unmasking_order(trace_dict: dict) -> Dict[int, int]:
    """Gen position -> first unmask step (mask_index diff, same as visualization_json)."""
    stored = trace_dict.get("unmasking_order") or {}
    if stored:
        return {int(k): int(v) for k, v in stored.items()}

    step_traces = trace_dict.get("step_traces", [])
    if not step_traces:
        return {}

    order: Dict[int, int] = {}
    prev = _squeeze_batch(step_traces[0].get("mask_index"))
    if prev is None:
        return {}

    for step_idx, step_trace in enumerate(step_traces):
        current = _squeeze_batch(step_trace.get("mask_index"))
        if current is None:
            continue
        n = min(len(prev), len(current))
        for pos in range(n):
            if prev[pos] and not current[pos] and pos not in order:
                order[int(pos)] = int(step_idx)
        prev = current
    return order


def _unmask_events_from_selected(
    trace_dict: dict,
    tokenizer,
    skip_ids: set,
) -> List[Tuple[int, int, int, str, str]]:
    """Prefer explicit selected_positions when present."""
    events: List[Tuple[int, int, int, str, str]] = []
    for step_trace in trace_dict.get("step_traces", []):
        step = int(step_trace.get("step", 0))
        sel = _squeeze_batch(step_trace.get("selected_positions"))
        tok = _squeeze_batch(step_trace.get("selected_tokens"))
        if sel is None or tok is None:
            continue
        for pos in np.nonzero(sel)[0].tolist():
            tid = int(tok[int(pos)])
            if tid in skip_ids:
                continue
            text, tag = pos_tag_token(tokenizer, tid)
            if not text:
                continue
            events.append((step, int(pos), tid, text, tag))
    return events


def _unmask_events_from_masks(
    trace_dict: dict,
    tokenizer,
    skip_ids: set,
) -> List[Tuple[int, int, int, str, str]]:
    """Infer unmask events from consecutive mask_index tensors."""
    step_traces = trace_dict.get("step_traces", [])
    if len(step_traces) < 2:
        return []

    events: List[Tuple[int, int, int, str, str]] = []
    for step in range(len(step_traces) - 1):
        mask_t = _squeeze_batch(step_traces[step].get("mask_index"))
        mask_next = _squeeze_batch(step_traces[step + 1].get("mask_index"))
        preds = _squeeze_batch(step_traces[step].get("predictions"))
        if mask_t is None or mask_next is None:
            continue
        n = min(len(mask_t), len(mask_next))
        for pos in range(n):
            if not (mask_t[pos] and not mask_next[pos]):
                continue
            tid: Optional[int] = None
            if preds is not None and pos < len(preds):
                tid = int(preds[pos])
            if tid is None or tid in skip_ids:
                continue
            text, tag = pos_tag_token(tokenizer, tid)
            if not text:
                continue
            events.append((step, int(pos), tid, text, tag))
    return events


def extract_unmask_events(
    trace_dict: dict,
    tokenizer,
    skip_ids: set,
) -> List[Tuple[int, int, int, str, str]]:
    """Return list of (step, gen_pos, token_id, token_str, pos_tag)."""
    selected = _unmask_events_from_selected(trace_dict, tokenizer, skip_ids)
    if selected:
        return selected
    return _unmask_events_from_masks(trace_dict, tokenizer, skip_ids)


def _bucket_indices(t_max: int, bucket_size: int) -> List[Tuple[int, int]]:
    n_buckets = (t_max + bucket_size - 1) // bucket_size
    return [
        (b * bucket_size, min((b + 1) * bucket_size, t_max))
        for b in range(n_buckets)
    ]


@dataclass
class PosTaskMetrics:
    """Task-level POS unmasking statistics."""

    n_samples: int
    n_steps: int
    model_dir: str
    model_name: str
    mean_unmask_step_by_pos: Dict[str, float]
    pos_counts: Dict[str, int]
    pos_unmask_count_trend: Dict[str, List[float]]
    top_k_tokens_per_step: List[List[dict]]
    top_k: int
    bucket_size: Optional[int] = None
    n_buckets: int = 0
    pos_unmask_bucket_trend: Dict[str, List[float]] = field(default_factory=dict)
    mean_unmask_step_by_pos_bucket: Dict[str, List[float]] = field(default_factory=dict)
    pos_tags: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Samples   : {self.n_samples}",
            f"Steps     : {self.n_steps}",
            f"Model     : {self.model_name} ({self.model_dir})",
            f"Top-k/step: {self.top_k}",
            "",
            "Mean normalized unmask step by POS (t/T, lower = earlier):",
        ]
        ranked = sorted(
            self.mean_unmask_step_by_pos.items(),
            key=lambda kv: kv[1],
        )
        for tag, val in ranked[:20]:
            count = self.pos_counts.get(tag, 0)
            lines.append(f"  {tag:8s}  t/T={val:.4f}  n={count}")
        if len(ranked) > 20:
            lines.append(f"  ... ({len(ranked) - 20} more POS tags)")
        if self.bucket_size is not None and self.n_buckets > 0:
            lines.extend(self._bucket_summary_lines())
        return "\n".join(lines)

    def _bucket_step_range(self, bucket_index: int) -> str:
        start = bucket_index * int(self.bucket_size)
        end = min(start + int(self.bucket_size), self.n_steps) - 1
        return f"{start}-{end}"

    def _bucket_summary_lines(self) -> List[str]:
        lines = [
            "",
            f"POS unmask counts by bucket (size={self.bucket_size}, top tags):",
        ]
        top_tags = sorted(
            self.pos_counts.keys(),
            key=lambda t: self.pos_counts[t],
            reverse=True,
        )[:6]
        header = f"  {'b':>3s}  {'steps':>7s}  " + "  ".join(f"{t:>6s}" for t in top_tags)
        lines.append(header)
        for b in range(self.n_buckets):
            vals = []
            for tag in top_tags:
                series = self.pos_unmask_bucket_trend.get(tag, [])
                v = float(series[b]) if b < len(series) else float("nan")
                vals.append(f"{v:.2f}" if np.isfinite(v) else "nan")
            lines.append(
                f"  {b:3d}  {self._bucket_step_range(b):>7s}  " + "  ".join(f"{v:>6s}" for v in vals)
            )
        return lines

    def to_dict(self) -> dict:
        out = {
            "n_samples": self.n_samples,
            "n_steps": self.n_steps,
            "model_dir": self.model_dir,
            "model_name": self.model_name,
            "top_k": self.top_k,
            "mean_unmask_step_by_pos": self.mean_unmask_step_by_pos,
            "pos_counts": self.pos_counts,
            "pos_unmask_count_trend": self.pos_unmask_count_trend,
            "top_k_tokens_per_step": self.top_k_tokens_per_step,
            "pos_tags": self.pos_tags,
        }
        if self.bucket_size is not None:
            out["bucket_size"] = self.bucket_size
            out["n_buckets"] = self.n_buckets
            out["pos_unmask_bucket_trend"] = self.pos_unmask_bucket_trend
            out["mean_unmask_step_by_pos_bucket"] = self.mean_unmask_step_by_pos_bucket
        return out


def _aggregate_top_k(
    counters: Sequence[Counter],
    top_k: int,
) -> List[List[dict]]:
    out: List[List[dict]] = []
    for counter in counters:
        rows = []
        for (token, tag), count in counter.most_common(top_k):
            rows.append({"token": token, "pos": tag, "count": float(count)})
        out.append(rows)
    return out


def compute_pos_metrics(
    trace_path: Union[str, Path],
    *,
    max_samples: Optional[int] = None,
    bucket_size: Optional[int] = None,
    top_k: int = 10,
    model_path: Optional[Union[str, Path]] = None,
    models_root: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> PosTaskMetrics:
    """POS-stratified unmasking stats from a trace file."""
    trace_path = Path(trace_path)
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")

    tokenizer, model_dir = load_trace_tokenizer(
        trace_path, model_path=model_path, models_root=models_root
    )
    skip_ids = diffusion_skip_token_ids(model_dir)
    model_name = model_dir.name

    per_sample_events: List[List[Tuple[int, int, int, str, str]]] = []
    t_max = 0

    for i, trace_dict in enumerate(_iter_traces(trace_path)):
        if max_samples is not None and i >= max_samples:
            break
        try:
            events = extract_unmask_events(trace_dict, tokenizer, skip_ids)
            per_sample_events.append(events)
            n_steps = len(trace_dict.get("step_traces", []))
            if n_steps:
                t_max = max(t_max, n_steps)
        except Exception as exc:
            logger.warning("Sample %d failed: %s", i, exc)
            continue
        if verbose and (i + 1) % 50 == 0:
            logger.info("Processed %d samples ...", i + 1)

    n_samples = len(per_sample_events)
    if n_samples == 0 or t_max == 0:
        return PosTaskMetrics(
            n_samples=0,
            n_steps=0,
            model_dir=str(model_dir),
            model_name=model_name,
            mean_unmask_step_by_pos={},
            pos_counts={},
            pos_unmask_count_trend={},
            top_k_tokens_per_step=[],
            top_k=top_k,
        )

    if verbose:
        logger.info("Processed %d samples total.", n_samples)

    norm_denom = float(t_max)
    pos_norm_steps: Dict[str, List[float]] = defaultdict(list)
    pos_counts: Counter = Counter()
    step_counters: List[Counter] = [Counter() for _ in range(t_max)]

    for events in per_sample_events:
        for step, _pos, _tid, text, tag in events:
            pos_counts[tag] += 1
            pos_norm_steps[tag].append(float(step) / norm_denom)
            if 0 <= step < t_max:
                step_counters[step][(text, tag)] += 1

    mean_by_pos = {
        tag: float(np.mean(vals)) for tag, vals in pos_norm_steps.items() if vals
    }
    all_tags = sorted(pos_counts.keys(), key=lambda t: pos_counts[t], reverse=True)

    sample_tag_trends = _per_sample_tag_trends(per_sample_events, all_tags, t_max)
    mean_trend_by_tag = {
        tag: [
            float(np.mean([sample[t] for sample in sample_tag_trends[tag]]))
            for t in range(t_max)
        ]
        for tag in all_tags
    }

    top_k_per_step = _aggregate_top_k(step_counters, top_k)

    bucket_trends: Dict[str, List[float]] = {}
    bucket_mean_by_pos: Dict[str, List[float]] = {}
    n_buckets = 0
    if bucket_size is not None:
        if bucket_size <= 0:
            raise ValueError(f"bucket_size must be positive, got {bucket_size}")
        buckets = _bucket_indices(t_max, bucket_size)
        n_buckets = len(buckets)
        bucket_norm_steps: Dict[Tuple[str, int], List[float]] = defaultdict(list)
        for events in per_sample_events:
            for step, _, _, _, tag in events:
                for b, (start, end) in enumerate(buckets):
                    if start <= step < end:
                        span = end - start
                        norm = float(step - start) / float(span) if span > 0 else 0.0
                        bucket_norm_steps[(tag, b)].append(norm)
        for tag in all_tags:
            count_vals: List[float] = []
            mean_vals: List[float] = []
            for b, (start, end) in enumerate(buckets):
                vals = [
                    sum(1 for step, _, _, _, tg in ev if start <= step < end and tg == tag)
                    for ev in per_sample_events
                ]
                count_vals.append(float(np.mean(vals)) if vals else float("nan"))
                norms = bucket_norm_steps.get((tag, b), [])
                mean_vals.append(float(np.mean(norms)) if norms else float("nan"))
            bucket_trends[tag] = count_vals
            bucket_mean_by_pos[tag] = mean_vals

    return PosTaskMetrics(
        n_samples=n_samples,
        n_steps=t_max,
        model_dir=str(model_dir),
        model_name=model_name,
        mean_unmask_step_by_pos=mean_by_pos,
        pos_counts=dict(pos_counts),
        pos_unmask_count_trend=mean_trend_by_tag,
        top_k_tokens_per_step=top_k_per_step,
        top_k=top_k,
        bucket_size=bucket_size,
        n_buckets=n_buckets,
        pos_unmask_bucket_trend=bucket_trends,
        mean_unmask_step_by_pos_bucket=bucket_mean_by_pos,
        pos_tags=all_tags,
    )


def _per_sample_tag_trends(
    per_sample_events: List[List[Tuple[int, int, int, str, str]]],
    tags: Iterable[str],
    t_max: int,
) -> Dict[str, List[List[float]]]:
    tag_set = set(tags)
    out: Dict[str, List[List[float]]] = {tag: [] for tag in tag_set}
    for events in per_sample_events:
        counters = [Counter() for _ in range(t_max)]
        for step, _, _, _, tag in events:
            if tag in tag_set and 0 <= step < t_max:
                counters[step][tag] += 1
        for tag in tag_set:
            out[tag].append([float(counters[t][tag]) for t in range(t_max)])
    return out

# --- CLI ---

def run_pos_analysis(
    trace_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    max_samples: Optional[int] = None,
    bucket_size: Optional[int] = None,
    top_k: int = 10,
    model_path: Optional[Union[str, Path]] = None,
    models_root: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Compute POS metrics, write pos_metrics.json and plots."""
    from schedulerlens.plots import save_pos_plots

    trace_path = Path(trace_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_pos_metrics(
        trace_path,
        max_samples=max_samples,
        bucket_size=bucket_size,
        top_k=top_k,
        model_path=model_path,
        models_root=models_root,
        verbose=verbose,
    )

    metrics_path = output_dir / "pos_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "trace_path": str(trace_path.resolve()),
                **metrics.to_dict(),
            },
            f,
            indent=2,
        )

    result: Dict[str, Any] = {
        "n_samples": metrics.n_samples,
        "n_steps": metrics.n_steps,
        "model_name": metrics.model_name,
        "model_dir": metrics.model_dir,
        "metrics_path": str(metrics_path.resolve()),
        "metrics": metrics,
        "pos_mean_unmask_plot": None,
        "pos_unmask_trend_plot": None,
        "pos_bucket_trend_plot": None,
        "pos_bucket_mean_unmask_plot": None,
    }

    if metrics.n_steps > 0 and metrics.mean_unmask_step_by_pos:
        prefix = trace_path.stem
        mean_plot, trend_plot, bucket_plot, bucket_mean_plot = save_pos_plots(
            metrics, output_dir, prefix=prefix
        )
        result["pos_mean_unmask_plot"] = str(mean_plot.resolve())
        result["pos_unmask_trend_plot"] = str(trend_plot.resolve())
        result["pos_bucket_trend_plot"] = (
            str(bucket_plot.resolve()) if bucket_plot else None
        )
        result["pos_bucket_mean_unmask_plot"] = (
            str(bucket_mean_plot.resolve()) if bucket_mean_plot else None
        )
    else:
        logger.warning("Skipped POS plots: no unmask events found.")

    return result

