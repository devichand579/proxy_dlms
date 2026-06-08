"""Task-level aggregation of SchedulerLens proxies over all samples.

Reads a trace file produced by the eval harness (JSONL or JSON format)
and computes:
  - M̂, Ĵ: scalars averaged over all samples
  - M̂_t trend: per-step M̂_t averaged over all samples (aligned by step index)
  - Ĵ_t trend: per-step Ĵ_t averaged over all samples
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from .proxies import SampleProxies, compute_sample_proxies

logger = logging.getLogger(__name__)


@dataclass
class TaskMetrics:
    """Population-level M̂/Ĵ estimates for one (model, scheduler, task, budget) config."""
    M_hat: float                         # scalar, averaged over samples
    J_hat: float                         # scalar, averaged over samples
    M_hat_trend: np.ndarray              # shape [T_max], per-step average
    J_hat_trend: np.ndarray              # shape [T_max], per-step average
    M_hat_trend_std: np.ndarray          # per-step std
    J_hat_trend_std: np.ndarray          # per-step std
    n_samples: int
    n_steps: int                         # number of steps (T_max)
    sample_M_hats: List[float] = field(default_factory=list)
    sample_J_hats: List[float] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Samples : {self.n_samples}",
            f"Steps   : {self.n_steps}",
            f"M̂       : {self.M_hat:.6f}  (ordering quality proxy, lower = better)",
            f"Ĵ       : {self.J_hat:.6f}  (parallelism error proxy, lower = better)",
        ]
        return "\n".join(lines)


def _iter_traces(path: Union[str, Path]):
    """Yield individual trace dicts from a JSONL or JSON trace file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return

    # JSONL: each non-comment line is a JSON object
    lines = [l for l in content.splitlines() if l.strip() and not l.strip().startswith("#")]

    # Heuristic: if first non-comment line starts with '{' and file has multiple lines → JSONL
    if lines and lines[0].strip().startswith("{"):
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                # Each line in the eval harness JSONL is one sample trace
                yield obj
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line")
        return

    # JSON: {"task": ..., "samples": [...]}
    try:
        doc = json.loads(content)
    except json.JSONDecodeError:
        logger.error("Cannot parse trace file: %s", path)
        return

    if isinstance(doc, list):
        for item in doc:
            yield item
    elif isinstance(doc, dict):
        samples = doc.get("samples", [])
        if samples:
            for item in samples:
                yield item
        else:
            # Single-sample JSON (e.g. from GenerationTrace.save_to_json)
            yield doc


def compute_task_metrics(
    trace_path: Union[str, Path],
    max_samples: Optional[int] = None,
    verbose: bool = True,
) -> TaskMetrics:
    """Compute M̂, Ĵ, and per-step trends from a trace file.

    Args:
        trace_path: Path to a .json or .jsonl trace file from the eval harness.
        max_samples: If set, only process the first N samples (useful for debugging).
        verbose: Log progress.

    Returns:
        TaskMetrics with M̂, Ĵ scalars and per-step trend arrays.
    """
    trace_path = Path(trace_path)
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")

    all_sample_proxies: List[SampleProxies] = []

    for i, trace_dict in enumerate(_iter_traces(trace_path)):
        if max_samples is not None and i >= max_samples:
            break
        try:
            sp = compute_sample_proxies(trace_dict)
            all_sample_proxies.append(sp)
        except Exception as e:
            logger.warning("Sample %d failed: %s", i, e)
            continue

        if verbose and (i + 1) % 50 == 0:
            logger.info("Processed %d samples ...", i + 1)

    n_samples = len(all_sample_proxies)
    if n_samples == 0:
        logger.warning("No valid samples found in %s", trace_path)
        empty = np.array([])
        return TaskMetrics(
            M_hat=0.0, J_hat=0.0,
            M_hat_trend=empty, J_hat_trend=empty,
            M_hat_trend_std=empty, J_hat_trend_std=empty,
            n_samples=0, n_steps=0,
        )

    if verbose:
        logger.info("Processed %d samples total.", n_samples)

    # ---- Scalar M̂ and Ĵ averaged over samples ---- #
    sample_M = [sp.M_hat for sp in all_sample_proxies]
    sample_J = [sp.J_hat for sp in all_sample_proxies]
    M_hat = float(np.mean(sample_M))
    J_hat = float(np.mean(sample_J))

    # ---- Per-step trends averaged over samples ---- #
    # Different samples may have different number of steps.
    # Align by step index; only average over samples that have that step.
    T_max = max(len(sp.step_proxies) for sp in all_sample_proxies)

    M_by_step: Dict[int, List[float]] = {t: [] for t in range(T_max)}
    J_by_step: Dict[int, List[float]] = {t: [] for t in range(T_max)}

    for sp in all_sample_proxies:
        for proxy in sp.step_proxies:
            t = proxy.step
            if t < T_max:
                M_by_step[t].append(proxy.M_hat_t)
                J_by_step[t].append(proxy.J_hat_t)

    M_trend = np.array([
        float(np.mean(M_by_step[t])) if M_by_step[t] else float("nan")
        for t in range(T_max)
    ])
    J_trend = np.array([
        float(np.mean(J_by_step[t])) if J_by_step[t] else float("nan")
        for t in range(T_max)
    ])
    M_trend_std = np.array([
        float(np.std(M_by_step[t])) if len(M_by_step[t]) > 1 else 0.0
        for t in range(T_max)
    ])
    J_trend_std = np.array([
        float(np.std(J_by_step[t])) if len(J_by_step[t]) > 1 else 0.0
        for t in range(T_max)
    ])

    return TaskMetrics(
        M_hat=M_hat,
        J_hat=J_hat,
        M_hat_trend=M_trend,
        J_hat_trend=J_trend,
        M_hat_trend_std=M_trend_std,
        J_hat_trend_std=J_trend_std,
        n_samples=n_samples,
        n_steps=T_max,
        sample_M_hats=sample_M,
        sample_J_hats=sample_J,
    )
