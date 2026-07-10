"""Aggregate sample proxies from a trace file."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from .proxies import PROXY_ATTRS, SampleProxies, compute_sample_proxies

logger = logging.getLogger(__name__)


@dataclass
class ProxyReport:
    entropy: float = 0.0
    uniform: float = 0.0
    trend: np.ndarray = field(default_factory=lambda: np.array([]))
    bucket_trend: np.ndarray = field(default_factory=lambda: np.array([]))
    sample_entropy: List[float] = field(default_factory=list)
    sample_uniform: List[float] = field(default_factory=list)


@dataclass
class TaskMetrics:
    """Task-level caps and per-step trends for M̂/Ĵ and sub-proxies."""
    M_hat: float
    J_hat: float
    M_hat_uniform: float
    J_hat_uniform: float
    proxies: Dict[str, ProxyReport]
    n_unmasked_trend: np.ndarray
    n_samples: int
    n_steps: int
    n_unmasked_bucket_trend: np.ndarray = field(default_factory=lambda: np.array([]))
    bucket_size: Optional[int] = None
    n_buckets: int = 0

    @property
    def M_hat_trend(self) -> np.ndarray:
        return self.proxies["M_hat_t"].trend

    @property
    def J_hat_trend(self) -> np.ndarray:
        return self.proxies["J_hat_t"].trend

    @property
    def sample_M_hats(self) -> List[float]:
        return self.proxies["M_hat_t"].sample_entropy

    @property
    def sample_J_hats(self) -> List[float]:
        return self.proxies["J_hat_t"].sample_entropy

    @property
    def sample_M_hats_uniform(self) -> List[float]:
        return self.proxies["M_hat_t"].sample_uniform

    @property
    def sample_J_hats_uniform(self) -> List[float]:
        return self.proxies["J_hat_t"].sample_uniform

    def summary(self) -> str:
        lines = [
            f"Samples : {self.n_samples}",
            f"Steps   : {self.n_steps}",
            "",
            "M-side (lower is better for M̂_t and regret):",
        ]
        for name in ("M_hat_t", "mean_regret", "cascade_gain"):
            rep = self.proxies[name]
            lines.append(
                f"  {name:14s}  entropy={rep.entropy:.6f}  uniform={rep.uniform:.6f}"
            )
        lines.extend(["", "J-side (lower is better for Ĵ_t, instability, rank_volatility):"])
        for name in ("J_hat_t", "instability", "rank_volatility"):
            rep = self.proxies[name]
            lines.append(
                f"  {name:14s}  entropy={rep.entropy:.6f}  uniform={rep.uniform:.6f}"
            )
        if self.bucket_size is not None and self.n_buckets > 0:
            lines.extend(self._bucket_summary_lines())
        return "\n".join(lines)

    def _bucket_step_range(self, bucket_index: int) -> str:
        start = bucket_index * int(self.bucket_size)
        end = min(start + int(self.bucket_size), self.n_steps) - 1
        return f"{start}-{end}"

    def _bucket_summary_lines(self) -> List[str]:
        bs = int(self.bucket_size)
        lines = [
            "",
            f"Bucket trends (size={bs}, {self.n_buckets} buckets):",
            (
                f"  {'b':>3s}  {'steps':>7s}  {'M_hat_t':>8s}  {'regret':>8s}  "
                f"{'cascade':>8s}  {'J_hat_t':>8s}  {'instab':>8s}  "
                f"{'rank_vol':>8s}  {'unmask':>6s}"
            ),
        ]
        for b in range(self.n_buckets):
            vals = []
            for name in PROXY_ATTRS:
                v = float(self.proxies[name].bucket_trend[b])
                vals.append(f"{v:.6f}" if np.isfinite(v) else "nan")
            u = float(self.n_unmasked_bucket_trend[b])
            unmask = f"{u:.2f}" if np.isfinite(u) else "nan"
            lines.append(
                f"  {b:3d}  {self._bucket_step_range(b):>7s}  "
                f"{vals[0]:>8s}  {vals[1]:>8s}  {vals[2]:>8s}  {vals[3]:>8s}  "
                f"{vals[4]:>8s}  {vals[5]:>8s}  {unmask:>6s}"
            )
        return lines

    def to_dict(self) -> dict:
        """Return JSON-serializable metrics."""
        out = {
            "M_hat": self.M_hat,
            "J_hat": self.J_hat,
            "M_hat_uniform": self.M_hat_uniform,
            "J_hat_uniform": self.J_hat_uniform,
            "n_samples": self.n_samples,
            "n_steps": self.n_steps,
            "n_unmasked_trend": self.n_unmasked_trend.tolist(),
            "mean_unmasked_per_step": float(np.nanmean(self.n_unmasked_trend))
            if len(self.n_unmasked_trend) else 0.0,
            "proxies": {},
        }
        for name, rep in self.proxies.items():
            entry = {
                "entropy": rep.entropy,
                "uniform": rep.uniform,
                "trend": rep.trend.tolist(),
                "sample_entropy": rep.sample_entropy,
                "sample_uniform": rep.sample_uniform,
            }
            if self.bucket_size is not None:
                entry["bucket_trend"] = rep.bucket_trend.tolist()
            out["proxies"][name] = entry
        if self.bucket_size is not None:
            out["bucket_size"] = self.bucket_size
            out["n_buckets"] = self.n_buckets
            out["n_unmasked_bucket_trend"] = self.n_unmasked_bucket_trend.tolist()
        return out


def _empty_proxy_reports() -> Dict[str, ProxyReport]:
    return {name: ProxyReport() for name in PROXY_ATTRS}


def _step_trends(
    all_sample_proxies: List[SampleProxies],
    attr: str,
    t_max: int,
) -> np.ndarray:
    by_step: Dict[int, List[float]] = {t: [] for t in range(t_max)}
    for sp in all_sample_proxies:
        for proxy in sp.step_proxies:
            t = proxy.step
            if t < t_max:
                by_step[t].append(float(getattr(proxy, attr)))

    return np.array([
        float(np.mean(by_step[t])) if by_step[t] else float("nan")
        for t in range(t_max)
    ])


def _bucket_trends_from_samples(
    all_sample_proxies: List[SampleProxies],
    attr: str,
    t_max: int,
    bucket_size: int,
) -> np.ndarray:
    """Mean per-sample bucket averages, aligned across samples."""
    if bucket_size <= 0:
        raise ValueError(f"bucket_size must be positive, got {bucket_size}")
    n_buckets = (t_max + bucket_size - 1) // bucket_size
    by_bucket: List[List[float]] = [[] for _ in range(n_buckets)]

    for sp in all_sample_proxies:
        step_map = {p.step: float(getattr(p, attr)) for p in sp.step_proxies}
        for b in range(n_buckets):
            start = b * bucket_size
            end = min(start + bucket_size, t_max)
            vals = [step_map[t] for t in range(start, end) if t in step_map]
            if vals:
                by_bucket[b].append(float(np.mean(vals)))

    return np.array([
        float(np.mean(by_bucket[b])) if by_bucket[b] else float("nan")
        for b in range(n_buckets)
    ])


def _apply_bucket_trends(
    all_sample_proxies: List[SampleProxies],
    proxies: Dict[str, ProxyReport],
    t_max: int,
    bucket_size: int,
) -> tuple[int, np.ndarray]:
    n_buckets = (t_max + bucket_size - 1) // bucket_size
    for name in PROXY_ATTRS:
        proxies[name].bucket_trend = _bucket_trends_from_samples(
            all_sample_proxies, name, t_max, bucket_size
        )
    n_unmasked_bucket = _bucket_trends_from_samples(
        all_sample_proxies, "n_selected", t_max, bucket_size
    )
    return n_buckets, n_unmasked_bucket


def _iter_traces(path: Union[str, Path]):
    """Yield one trace dict per sample from JSONL, JSON, or NPZ."""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        from traces.trace_npz import iter_npz_traces

        yield from iter_npz_traces(path)
        return

    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return

    # JSONL: each non-comment line is a JSON object
    lines = [l for l in content.splitlines() if l.strip() and not l.strip().startswith("#")]

    if lines and lines[0].strip().startswith("{"):
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line")
        return

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
            yield doc


def compute_task_metrics(
    trace_path: Union[str, Path],
    max_samples: Optional[int] = None,
    bucket_size: Optional[int] = None,
    verbose: bool = True,
) -> TaskMetrics:
    """Compute task-level caps and per-step trends from a trace file."""
    trace_path = Path(trace_path)
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")

    all_sample_proxies: List[SampleProxies] = []

    for i, trace_dict in enumerate(_iter_traces(trace_path)):
        if max_samples is not None and i >= max_samples:
            break
        try:
            all_sample_proxies.append(compute_sample_proxies(trace_dict))
        except Exception as e:
            logger.warning("Sample %d failed: %s", i, e)
            continue

        if verbose and (i + 1) % 50 == 0:
            logger.info("Processed %d samples ...", i + 1)

    n_samples = len(all_sample_proxies)
    if n_samples == 0:
        logger.warning("No valid samples found in %s", trace_path)
        empty = _empty_proxy_reports()
        return TaskMetrics(
            M_hat=0.0,
            J_hat=0.0,
            M_hat_uniform=0.0,
            J_hat_uniform=0.0,
            proxies=empty,
            n_unmasked_trend=np.array([]),
            n_samples=0,
            n_steps=0,
        )

    if verbose:
        logger.info("Processed %d samples total.", n_samples)

    t_max = max(len(sp.step_proxies) for sp in all_sample_proxies)
    proxies = _empty_proxy_reports()

    for name in PROXY_ATTRS:
        rep = proxies[name]
        rep.sample_entropy = [sp.caps[name].entropy for sp in all_sample_proxies]
        rep.sample_uniform = [sp.caps[name].uniform for sp in all_sample_proxies]
        rep.entropy = float(np.mean(rep.sample_entropy))
        rep.uniform = float(np.mean(rep.sample_uniform))
        rep.trend = _step_trends(all_sample_proxies, name, t_max)

    n_unmasked_trend = _step_trends(all_sample_proxies, "n_selected", t_max)

    n_buckets = 0
    n_unmasked_bucket_trend = np.array([])
    if bucket_size is not None:
        if bucket_size <= 0:
            raise ValueError(f"bucket_size must be positive, got {bucket_size}")
        n_buckets, n_unmasked_bucket_trend = _apply_bucket_trends(
            all_sample_proxies, proxies, t_max, bucket_size
        )

    return TaskMetrics(
        M_hat=proxies["M_hat_t"].entropy,
        J_hat=proxies["J_hat_t"].entropy,
        M_hat_uniform=proxies["M_hat_t"].uniform,
        J_hat_uniform=proxies["J_hat_t"].uniform,
        proxies=proxies,
        n_unmasked_trend=n_unmasked_trend,
        n_unmasked_bucket_trend=n_unmasked_bucket_trend,
        n_samples=n_samples,
        n_steps=t_max,
        bucket_size=bucket_size,
        n_buckets=n_buckets,
    )
