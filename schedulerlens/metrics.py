"""Aggregate sample proxies from a trace file."""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .proxies import PROXY_ATTRS, SampleProxies, TraceProxyError, compute_sample_proxies

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
    """Yield one trace dict per sample from an NPZ trace archive."""
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError(
            f"SchedulerLens expects an NPZ trace archive, got {path.name!r}. "
            "Convert JSONL traces with: python traces/trace_audit.py convert <file.jsonl>"
        )
    from traces.trace_npz import iter_npz_traces

    yield from iter_npz_traces(path)


def _ensure_online_len(sums: List[float], counts: List[int], t_max: int) -> None:
    if t_max <= len(sums):
        return
    pad = t_max - len(sums)
    sums.extend([0.0] * pad)
    counts.extend([0] * pad)


def compute_task_metrics(
    trace_path: Union[str, Path],
    max_samples: Optional[int] = None,
    bucket_size: Optional[int] = None,
    verbose: bool = True,
) -> TaskMetrics:
    """Compute task-level caps and per-step trends from a trace file.

    Streams samples when ``bucket_size`` is None (compare default) so large NPZs
    do not retain every ``SampleProxies`` in RAM before oracle replay.
    """
    trace_path = Path(trace_path)
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")

    need_buckets = bucket_size is not None
    if need_buckets and bucket_size <= 0:
        raise ValueError(f"bucket_size must be positive, got {bucket_size}")

    # Online path (no buckets): O(1) sample retention.
    sample_entropy = {name: [] for name in PROXY_ATTRS}
    sample_uniform = {name: [] for name in PROXY_ATTRS}
    trend_sum = {name: [] for name in PROXY_ATTRS}
    trend_count = {name: [] for name in PROXY_ATTRS}
    n_unmasked_sum: List[float] = []
    n_unmasked_count: List[int] = []

    # Bucket path still materializes samples (uncommon for compare).
    all_sample_proxies: List[SampleProxies] = []
    n_samples = 0
    t_max = 0

    for i, trace_dict in enumerate(_iter_traces(trace_path)):
        if max_samples is not None and i >= max_samples:
            break
        try:
            sp = compute_sample_proxies(trace_dict, trace_name=trace_path)
        except TraceProxyError as e:
            raise TraceProxyError(f"sample {i}: {e}") from e

        n_samples += 1
        t_max = max(t_max, len(sp.step_proxies))

        if need_buckets:
            all_sample_proxies.append(sp)
        else:
            for name in PROXY_ATTRS:
                sample_entropy[name].append(float(sp.caps[name].entropy))
                sample_uniform[name].append(float(sp.caps[name].uniform))
                _ensure_online_len(trend_sum[name], trend_count[name], t_max)
            _ensure_online_len(n_unmasked_sum, n_unmasked_count, t_max)
            for proxy in sp.step_proxies:
                t = int(proxy.step)
                if t >= t_max:
                    continue
                for name in PROXY_ATTRS:
                    trend_sum[name][t] += float(getattr(proxy, name))
                    trend_count[name][t] += 1
                n_unmasked_sum[t] += float(proxy.n_selected)
                n_unmasked_count[t] += 1

        if verbose and (i + 1) % 50 == 0:
            logger.info("Processed %d samples ...", i + 1)

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

    proxies = _empty_proxy_reports()
    n_buckets = 0
    n_unmasked_bucket_trend = np.array([])

    if need_buckets:
        t_max = max(len(sp.step_proxies) for sp in all_sample_proxies)
        for name in PROXY_ATTRS:
            rep = proxies[name]
            rep.sample_entropy = [sp.caps[name].entropy for sp in all_sample_proxies]
            rep.sample_uniform = [sp.caps[name].uniform for sp in all_sample_proxies]
            rep.entropy = float(np.mean(rep.sample_entropy))
            rep.uniform = float(np.mean(rep.sample_uniform))
            rep.trend = _step_trends(all_sample_proxies, name, t_max)
        n_unmasked_trend = _step_trends(all_sample_proxies, "n_selected", t_max)
        n_buckets, n_unmasked_bucket_trend = _apply_bucket_trends(
            all_sample_proxies, proxies, t_max, int(bucket_size)
        )
        all_sample_proxies.clear()
    else:
        for name in PROXY_ATTRS:
            rep = proxies[name]
            rep.sample_entropy = sample_entropy[name]
            rep.sample_uniform = sample_uniform[name]
            rep.entropy = float(np.mean(rep.sample_entropy)) if rep.sample_entropy else 0.0
            rep.uniform = float(np.mean(rep.sample_uniform)) if rep.sample_uniform else 0.0
            sums, counts = trend_sum[name], trend_count[name]
            rep.trend = np.array(
                [
                    (sums[t] / counts[t]) if counts[t] else float("nan")
                    for t in range(t_max)
                ],
                dtype=np.float64,
            )
        n_unmasked_trend = np.array(
            [
                (n_unmasked_sum[t] / n_unmasked_count[t])
                if n_unmasked_count[t]
                else float("nan")
                for t in range(t_max)
            ],
            dtype=np.float64,
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

# --- from run_meta.py ---
_KNOWN_MODELS = ("llada", "llada1_5", "llada-300m-pcfg","llada-300m-pcfg-1B", "llada-200m-pcfg", "llada-100m-pcfg", "dream")

# Longer mode names first.
_KNOWN_MODES = (
    "pure_diffusion",
    "block_diffusion",
    "eb_sampler",
    "top_k_c",
    "top_k_m",
    "fdm",
    "fhs",
    "dus",
)

_MODE_TO_SCHEDULER = {
    "pure_diffusion": "pd",
    "block_diffusion": "bd",
    "eb_sampler": "eb",
    "fdm": "fdm",
    "fhs": "fhs",
    "top_k_c": "top_k_c",
    "top_k_m": "top_k_m",
    "dus": "dus",
}

# PCFG stems: {hf_model_short}-pcfg_{scheduler}_s{steps}
# e.g. llada-100m-pcfg-pcfg_bd_s128 → model=llada-100m-pcfg, scheduler=bd, steps=128
# The HF short name is everything before the *last* "-pcfg_" (scheduler marker).
_PCFG_SCHED_RE = re.compile(
    r"^(?P<model>.+)-pcfg_(?P<scheduler>[a-z0-9_]+)_s(?P<steps>\d+)$",
    re.IGNORECASE,
)
_STEPS_RE = re.compile(r"_s(?P<steps>\d+)$", re.IGNORECASE)


def _is_pcfg_name(stem: str) -> bool:
    """Filename-based PCFG detection (mirrors traces.trace_metrics.is_pcfg_trace)."""
    name = str(stem)
    return name.startswith("pcfg_") or "-pcfg_" in name


def parse_pcfg_stem(stem: str) -> Optional[tuple[str, str, int]]:
    """Return (hf_model_short, scheduler, steps) or None.

    For ``llada-100m-pcfg-pcfg_bd_s128`` → ``("llada-100m-pcfg", "bd", 128)``.
    The model id is the substring before the last ``-pcfg_`` marker.
    """
    m = _PCFG_SCHED_RE.match(stem)
    if not m:
        return None
    return m.group("model"), m.group("scheduler").lower(), int(m.group("steps"))


@dataclass
class RunMeta:
    filename: str
    stem: str
    model: str
    scheduler: str
    task: str
    steps: Optional[int]
    is_pcfg: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _strip_steps(stem: str) -> tuple[str, Optional[int]]:
    m = _STEPS_RE.search(stem)
    if not m:
        return stem, None
    return stem[: m.start()], int(m.group("steps"))


def _match_prefix(stem: str, candidates: tuple[str, ...]) -> Optional[str]:
    lower = stem.lower()
    for name in candidates:
        n = name.lower()
        if lower == n or lower.startswith(n + "_") or lower.startswith(n + "-"):
            return name
    return None


def _match_suffix(body: str, candidates: tuple[str, ...]) -> Optional[str]:
    lower = body.lower()
    for name in candidates:
        n = name.lower()
        if lower == n or lower.endswith("_" + n):
            return name
    return None


def parse_run_stem(path: Union[str, Path]) -> RunMeta:
    """Parse identity fields from an NPZ / run filename stem."""
    path = Path(path)
    stem = path.stem
    filename = path.name
    pcfg = _is_pcfg_name(stem)

    pcfg_parts = parse_pcfg_stem(stem)
    if pcfg_parts is not None:
        model_short, scheduler, steps = pcfg_parts
        return RunMeta(
            filename=filename,
            stem=stem,
            model=model_short,
            scheduler=scheduler,
            task="pcfg",
            steps=steps,
            is_pcfg=True,
        )

    body, steps = _strip_steps(stem)
    model = _match_prefix(body, _KNOWN_MODELS)
    if model is None:
        model = body.split("_")[0] if body else stem

    rest = body
    ml = model.lower()
    if rest.lower().startswith(ml + "_"):
        rest = rest[len(model) + 1 :]
    elif rest.lower().startswith(ml + "-"):
        rest = rest[len(model) + 1 :]
    elif rest.lower() == ml:
        rest = ""

    mode = _match_suffix(rest, _KNOWN_MODES)
    if mode is not None:
        task = rest[: -(len(mode) + 1)] if rest.lower().endswith("_" + mode.lower()) else ""
        if not task and rest.lower() != mode.lower():
            task = rest
        scheduler = _MODE_TO_SCHEDULER.get(mode, mode)
    else:
        parts = rest.split("_") if rest else []
        if len(parts) >= 2:
            task = "_".join(parts[:-1])
            mode = parts[-1]
            scheduler = _MODE_TO_SCHEDULER.get(mode, mode)
        elif parts:
            task = parts[0]
            scheduler = "unknown"
        else:
            task = "unknown"
            scheduler = "unknown"

    return RunMeta(
        filename=filename,
        stem=stem,
        model=model,
        scheduler=scheduler,
        task=task or "unknown",
        steps=steps,
        is_pcfg=pcfg,
    )


# --- from export_csv.py ---
CSV_COLUMNS: List[str] = [
    "repo",
    "filename",
    "model",
    "scheduler",
    "task",
    "steps",
    "is_pcfg",
    "n_samples",
    "n_steps",
    "M_hat",
    "J_hat",
    "M_hat_uniform",
    "J_hat_uniform",
    "mean_regret_entropy",
    "mean_regret_uniform",
    "cascade_gain_entropy",
    "cascade_gain_uniform",
    "instability_entropy",
    "instability_uniform",
    "rank_volatility_entropy",
    "rank_volatility_uniform",
    "model_commit_ppl",
    "oracle_ppl",
    "Mt_total",
    "Jt_total",
    "corr_M_full",
    "corr_M_early",
    "corr_M_mid",
    "corr_M_late",
    "corr_J_full",
    "corr_J_early",
    "corr_J_mid",
    "corr_J_late",
    "corr_ppl",
]


def _fmt(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        return value
    return value


def empty_metrics_row(**overrides: Any) -> Dict[str, Any]:
    """Return a CSV row dict with NaN/empty defaults, then apply overrides."""
    row: Dict[str, Any] = {col: "" for col in CSV_COLUMNS}
    for key in (
        "M_hat",
        "J_hat",
        "M_hat_uniform",
        "J_hat_uniform",
        "mean_regret_entropy",
        "mean_regret_uniform",
        "cascade_gain_entropy",
        "cascade_gain_uniform",
        "instability_entropy",
        "instability_uniform",
        "rank_volatility_entropy",
        "rank_volatility_uniform",
        "model_commit_ppl",
        "oracle_ppl",
        "Mt_total",
        "Jt_total",
        "corr_M_full",
        "corr_M_early",
        "corr_M_mid",
        "corr_M_late",
        "corr_J_full",
        "corr_J_early",
        "corr_J_mid",
        "corr_J_late",
        "corr_ppl",
    ):
        row[key] = float("nan")
    row.update(overrides)
    return row


def write_metrics_csv(
    path: Union[str, Path],
    rows: Sequence[Mapping[str, Any]],
    *,
    append: bool = False,
) -> Path:
    """Write or append unified SchedulerLens metrics rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not append or not path.exists() or path.stat().st_size == 0
    mode = "a" if append and path.exists() else "w"
    with path.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({col: _fmt(row.get(col, "")) for col in CSV_COLUMNS})
    return path


def read_metrics_csv(path: Union[str, Path]) -> List[Dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# --- from correlation.py ---
BLOCK_NAMES = ("full", "early", "mid", "late")


@dataclass
class BlockCorrelations:
    corr_M_full: float = float("nan")
    corr_M_early: float = float("nan")
    corr_M_mid: float = float("nan")
    corr_M_late: float = float("nan")
    corr_J_full: float = float("nan")
    corr_J_early: float = float("nan")
    corr_J_mid: float = float("nan")
    corr_J_late: float = float("nan")
    corr_ppl: float = float("nan")

    def to_dict(self) -> dict:
        return asdict(self)


def trajectory_blocks(n_steps: int) -> Dict[str, Tuple[int, int]]:
    """Equal thirds of [0, n_steps): early / mid / late, plus full.

    For ``n_steps < 3``, early/mid/late collapse to the full span (avoid
    overlapping partial thirds that look like independent blocks).
    """
    if n_steps <= 0:
        return {name: (0, 0) for name in BLOCK_NAMES}
    if n_steps < 3:
        full = (0, n_steps)
        return {name: full for name in BLOCK_NAMES}
    third = n_steps // 3
    early_end = third
    mid_end = 2 * third
    # Give remainder to late so union covers [0, n_steps).
    return {
        "full": (0, n_steps),
        "early": (0, early_end),
        "mid": (early_end, mid_end),
        "late": (mid_end, n_steps),
    }


def pearson_corr(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation; NaN if fewer than 2 finite paired points or zero variance."""
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    if xs.shape != ys.shape:
        raise ValueError(f"shape mismatch: {xs.shape} vs {ys.shape}")
    mask = np.isfinite(xs) & np.isfinite(ys)
    if int(mask.sum()) < 2:
        return float("nan")
    a = xs[mask]
    b = ys[mask]
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def block_correlations(
    oracle: Sequence[float],
    proxy: Sequence[float],
    *,
    n_steps: Optional[int] = None,
) -> Dict[str, float]:
    """Pearson ρ of oracle vs proxy over full / early / mid / late."""
    n = n_steps if n_steps is not None else len(oracle)
    blocks = trajectory_blocks(n)
    out: Dict[str, float] = {}
    for name, (start, end) in blocks.items():
        out[name] = pearson_corr(oracle[start:end], proxy[start:end])
    return out


def mean_step_series(
    per_sample_series: Iterable[Sequence[float]],
    n_steps: int,
) -> np.ndarray:
    """Mean across samples of aligned per-step series (pad missing with nan)."""
    by_step: List[List[float]] = [[] for _ in range(n_steps)]
    for series in per_sample_series:
        for t, val in enumerate(series):
            if t < n_steps and np.isfinite(val):
                by_step[t].append(float(val))
    return np.array([
        float(np.mean(vals)) if vals else float("nan") for vals in by_step
    ])


def _pad_to(series: Sequence[float], n: int) -> np.ndarray:
    out = np.full(n, np.nan, dtype=np.float64)
    arr = np.asarray(series, dtype=np.float64).ravel()
    m = min(n, arr.size)
    if m > 0:
        out[:m] = arr[:m]
    return out


def compute_oracle_proxy_correlations(
    *,
    mt_trend: Sequence[float],
    m_hat_trend: Sequence[float],
    jt_trend: Sequence[float],
    j_hat_trend: Sequence[float],
    sample_oracle_ppl: Sequence[float] | None = None,
    sample_model_ppl: Sequence[float] | None = None,
) -> BlockCorrelations:
    """Build BlockCorrelations from task-level trends and optional sample PPL lists."""
    n = min(len(mt_trend), len(m_hat_trend), len(jt_trend), len(j_hat_trend))
    if n <= 0:
        n = max(len(mt_trend), len(m_hat_trend), len(jt_trend), len(j_hat_trend), 0)
    mt = _pad_to(mt_trend, n)
    mh = _pad_to(m_hat_trend, n)
    jt = _pad_to(jt_trend, n)
    jh = _pad_to(j_hat_trend, n)
    m_blocks = block_correlations(mt, mh, n_steps=n)
    j_blocks = block_correlations(jt, jh, n_steps=n)
    corr_ppl = float("nan")
    if sample_oracle_ppl is not None and sample_model_ppl is not None:
        corr_ppl = pearson_corr(sample_oracle_ppl, sample_model_ppl)
    return BlockCorrelations(
        corr_M_full=m_blocks["full"],
        corr_M_early=m_blocks["early"],
        corr_M_mid=m_blocks["mid"],
        corr_M_late=m_blocks["late"],
        corr_J_full=j_blocks["full"],
        corr_J_early=j_blocks["early"],
        corr_J_mid=j_blocks["mid"],
        corr_J_late=j_blocks["late"],
        corr_ppl=corr_ppl,
    )


# --- from analyze.py ---

def build_metrics_row(
    trace_path: Union[str, Path],
    *,
    metrics: Optional[TaskMetrics] = None,
    perplexity: Optional[Any] = None,
    correlations: Optional[BlockCorrelations] = None,
    meta: Optional[RunMeta] = None,
    repo: str = "",
    max_samples: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Assemble one CSV-ready row for a trace NPZ."""
    trace_path = Path(trace_path)
    meta = meta or parse_run_stem(trace_path)
    if metrics is None:
        metrics = compute_task_metrics(
            trace_path, max_samples=max_samples, verbose=verbose
        )

    row = empty_metrics_row(
        repo=repo,
        filename=meta.filename,
        model=meta.model,
        scheduler=meta.scheduler,
        task=meta.task,
        steps=meta.steps if meta.steps is not None else "",
        is_pcfg=meta.is_pcfg,
        n_samples=metrics.n_samples,
        n_steps=metrics.n_steps,
        M_hat=metrics.M_hat,
        J_hat=metrics.J_hat,
        M_hat_uniform=metrics.M_hat_uniform,
        J_hat_uniform=metrics.J_hat_uniform,
    )
    for name in ("mean_regret", "cascade_gain", "instability", "rank_volatility"):
        rep = metrics.proxies[name]
        row[f"{name}_entropy"] = rep.entropy
        row[f"{name}_uniform"] = rep.uniform

    if perplexity is not None:
        row["model_commit_ppl"] = perplexity.model_commit_ppl
        row["oracle_ppl"] = perplexity.oracle_ppl
        row["Mt_total"] = perplexity.Mt_total
        row["Jt_total"] = perplexity.Jt_total

    if correlations is not None:
        row.update(correlations.to_dict())

    return row


def collect_sample_proxy_trends(
    trace_path: Union[str, Path],
    *,
    max_samples: Optional[int] = None,
) -> tuple[List[SampleProxies], int]:
    """Return per-sample proxies and max step count (fail-hard)."""
    from schedulerlens.proxies import SampleProxies, compute_sample_proxies
    samples: List = []
    for i, tr in enumerate(_iter_traces(trace_path)):
        if max_samples is not None and i >= max_samples:
            break
        samples.append(compute_sample_proxies(tr, trace_name=trace_path))
    t_max = max((len(sp.step_proxies) for sp in samples), default=0)
    return samples, t_max


def correlations_from_trends(
    *,
    mt_trend: Sequence[float],
    m_hat_trend: Sequence[float],
    jt_trend: Sequence[float],
    j_hat_trend: Sequence[float],
    sample_oracle_nll: Sequence[float] | None = None,
    sample_model_nll: Sequence[float] | None = None,
) -> BlockCorrelations:
    from schedulerlens.oracle import nll_to_ppl as _ppl
    sample_oracle_ppl = (
        [_ppl(x) for x in sample_oracle_nll] if sample_oracle_nll is not None else None
    )
    sample_model_ppl = (
        [_ppl(x) for x in sample_model_nll] if sample_model_nll is not None else None
    )
    return compute_oracle_proxy_correlations(
        mt_trend=mt_trend,
        m_hat_trend=m_hat_trend,
        jt_trend=jt_trend,
        j_hat_trend=j_hat_trend,
        sample_oracle_ppl=sample_oracle_ppl,
        sample_model_ppl=sample_model_ppl,
    )

