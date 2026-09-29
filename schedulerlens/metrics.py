"""Aggregate sample proxies from a trace file."""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .proxies import (
    PROXY_ATTRS,
    SampleProxies,
    TraceProxyError,
    compute_sample_proxies,
    cumulative_caps,
    prefix_commit_fraction,
)

logger = logging.getLogger(__name__)


@dataclass
class ProxyReport:
    cap: float = 0.0
    trend: np.ndarray = field(default_factory=lambda: np.array([]))
    bucket_trend: np.ndarray = field(default_factory=lambda: np.array([]))
    sample_caps: List[float] = field(default_factory=list)


@dataclass
class TaskMetrics:
    """Task-level caps and per-step trends for M̂ (γ = 1) and Ĵ."""
    M_hat: float
    J_hat: float  # full-trajectory j_hat sum
    proxies: Dict[str, ProxyReport]
    n_unmasked_trend: np.ndarray
    n_samples: int
    n_steps: int
    n_unmasked_bucket_trend: np.ndarray = field(default_factory=lambda: np.array([]))
    bucket_size: Optional[int] = None
    n_buckets: int = 0
    # K -> (M_hat, J_hat, G_hat) sample means over steps with step < K
    prefix_caps: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)
    # K -> per-sample lists of cumulative M̂ and Ĵ over steps with step < K
    prefix_samples: Dict[int, Dict[str, List[float]]] = field(default_factory=dict)
    # K -> sample-mean fraction of the gen region unmasked after step < K
    prefix_commit_frac: Dict[int, float] = field(default_factory=dict)
    # Sample-mean fraction unmasked at the end of the trajectory
    commit_frac: float = float("nan")

    @property
    def M_hat_trend(self) -> np.ndarray:
        return self.proxies["M_hat_t"].trend

    @property
    def J_hat_trend(self) -> np.ndarray:
        return self.proxies["J_hat_t"].trend

    @property
    def sample_M_hats(self) -> List[float]:
        return self.proxies["M_hat_t"].sample_caps

    @property
    def sample_J_hats(self) -> List[float]:
        return self.proxies["J_hat_t"].sample_caps

    def summary(self) -> str:
        lines = [
            f"Samples : {self.n_samples}",
            f"Steps   : {self.n_steps}",
            "",
            "M̂ / Ĵ (sample means of sum caps):",
            f"  M_hat        {self.M_hat:.6f}",
            f"  J_hat        {self.J_hat:.6f}",
            f"  G_hat        {self.M_hat + self.J_hat:.6f}",
        ]
        if self.prefix_caps:
            lines.append("Prefix caps (sum over step < K, sample means):")
            for k in sorted(self.prefix_caps):
                m, j, g = self.prefix_caps[k]
                frac = self.prefix_commit_frac.get(k, float("nan"))
                lines.append(
                    f"  K={k:<4d}  M_hat={m:.6f}  J_hat={j:.6f}  G_hat={g:.6f}  "
                    f"commit_frac={frac:.4f}"
                )
        for name in PROXY_ATTRS:
            rep = self.proxies[name]
            lines.append(
                f"  {name:14s}  cap={rep.cap:.6f}"
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
                f"  {'b':>3s}  {'steps':>7s}  {'M_hat_t':>8s}  "
                f"{'J_hat_t':>8s}  {'unmask':>6s}"
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
                f"{vals[0]:>8s}  {vals[1]:>8s}  {unmask:>6s}"
            )
        return lines

    def to_dict(self) -> dict:
        """Return JSON-serializable metrics."""
        out = {
            "M_hat": self.M_hat,
            "J_hat": self.J_hat,
            "G_hat": float(self.M_hat) + float(self.J_hat),
            "n_samples": self.n_samples,
            "n_steps": self.n_steps,
            "n_unmasked_trend": self.n_unmasked_trend.tolist(),
            "mean_unmasked_per_step": float(np.nanmean(self.n_unmasked_trend))
            if len(self.n_unmasked_trend) else 0.0,
            "proxies": {},
        }
        for name, rep in self.proxies.items():
            entry = {
                "cap": rep.cap,
                "trend": rep.trend.tolist(),
                "sample_caps": rep.sample_caps,
            }
            if self.bucket_size is not None:
                entry["bucket_trend"] = rep.bucket_trend.tolist()
            out["proxies"][name] = entry
        if self.bucket_size is not None:
            out["bucket_size"] = self.bucket_size
            out["n_buckets"] = self.n_buckets
            out["n_unmasked_bucket_trend"] = self.n_unmasked_bucket_trend.tolist()
        if self.prefix_caps:
            out["prefix_caps"] = {
                str(k): {"M_hat": m, "J_hat": j, "G_hat": g}
                for k, (m, j, g) in sorted(self.prefix_caps.items())
            }
        return out


def parse_prefix_steps(spec: Optional[Union[str, Sequence[int]]]) -> List[int]:
    """Parse ``8,16,32,64`` or an int sequence into sorted unique positive Ks."""
    if spec is None:
        return []
    if isinstance(spec, str):
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        values = [int(p) for p in parts]
    else:
        values = [int(k) for k in spec]
    out: List[int] = []
    seen: set[int] = set()
    for k in values:
        if k <= 0:
            raise ValueError(f"prefix_steps must be positive, got {k}")
        if k not in seen:
            seen.add(k)
            out.append(k)
    return sorted(out)


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
    prefix_steps: Optional[Sequence[int]] = None,
) -> TaskMetrics:
    """Compute task-level caps and per-step trends from a trace file.

    Streams samples when ``bucket_size`` is None (compare default) so large NPZs
    do not retain every ``SampleProxies`` in RAM before oracle replay.

    ``prefix_steps`` (e.g. ``[8, 16, 32, 64]``) also accumulates sample-mean
    cumulative M̂/Ĵ/Ĝ over steps with ``step < K`` into ``TaskMetrics.prefix_caps``.
    """
    trace_path = Path(trace_path)
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")

    need_buckets = bucket_size is not None
    if need_buckets and bucket_size <= 0:
        raise ValueError(f"bucket_size must be positive, got {bucket_size}")

    prefix_ks = parse_prefix_steps(prefix_steps)
    prefix_sample: Dict[int, Dict[str, List[float]]] = {
        k: {"M": [], "J": [], "C": []} for k in prefix_ks
    }
    end_commit_frac: List[float] = []

    # Online path (no buckets): O(1) sample retention.
    sample_caps = {name: [] for name in PROXY_ATTRS}
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

        gen_length = int(trace_dict.get("gen_length") or 0)
        if gen_length <= 0 and sp.step_proxies:
            head = sp.step_proxies[0]
            gen_length = (
                int(head.n_committed) + int(head.n_selected) + int(head.n_remaining)
            )
        for k in prefix_ks:
            m_k, j_k, _ = cumulative_caps(sp.step_proxies, k)
            prefix_sample[k]["M"].append(m_k)
            prefix_sample[k]["J"].append(j_k)
            prefix_sample[k]["C"].append(
                prefix_commit_fraction(sp.step_proxies, k, gen_length)
            )
        end_commit_frac.append(
            prefix_commit_fraction(sp.step_proxies, t_max + 1, gen_length)
        )

        if need_buckets:
            all_sample_proxies.append(sp)
        else:
            for name in PROXY_ATTRS:
                sample_caps[name].append(float(sp.caps[name]))
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

    def _mean_prefix_caps() -> Dict[int, Tuple[float, float, float]]:
        out: Dict[int, Tuple[float, float, float]] = {}
        for k in prefix_ks:
            ms = prefix_sample[k]["M"]
            js = prefix_sample[k]["J"]
            m_mean = float(np.mean(ms)) if ms else 0.0
            j_mean = float(np.mean(js)) if js else 0.0
            out[k] = (m_mean, j_mean, m_mean + j_mean)
        return out

    def _mean_prefix_commit_frac() -> Dict[int, float]:
        out: Dict[int, float] = {}
        for k in prefix_ks:
            cs = prefix_sample[k]["C"]
            out[k] = float(np.mean(cs)) if cs else 0.0
        return out

    if n_samples == 0:
        logger.warning("No valid samples found in %s", trace_path)
        empty = _empty_proxy_reports()
        return TaskMetrics(
            M_hat=0.0,
            J_hat=0.0,
            proxies=empty,
            n_unmasked_trend=np.array([]),
            n_samples=0,
            n_steps=0,
            prefix_caps={},
            prefix_commit_frac={},
            commit_frac=float("nan"),
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
            rep.sample_caps = [float(sp.caps[name]) for sp in all_sample_proxies]
            rep.cap = float(np.mean(rep.sample_caps))
            rep.trend = _step_trends(all_sample_proxies, name, t_max)
        n_unmasked_trend = _step_trends(all_sample_proxies, "n_selected", t_max)
        n_buckets, n_unmasked_bucket_trend = _apply_bucket_trends(
            all_sample_proxies, proxies, t_max, int(bucket_size)
        )
        all_sample_proxies.clear()
    else:
        for name in PROXY_ATTRS:
            rep = proxies[name]
            rep.sample_caps = sample_caps[name]
            rep.cap = float(np.mean(rep.sample_caps)) if rep.sample_caps else 0.0
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

    j_full = proxies["J_hat_t"].cap
    return TaskMetrics(
        M_hat=proxies["M_hat_t"].cap,
        J_hat=j_full,
        proxies=proxies,
        n_unmasked_trend=n_unmasked_trend,
        n_unmasked_bucket_trend=n_unmasked_bucket_trend,
        n_samples=n_samples,
        n_steps=t_max,
        bucket_size=bucket_size,
        n_buckets=n_buckets,
        prefix_caps=_mean_prefix_caps(),
        prefix_samples=prefix_sample,
        prefix_commit_frac=_mean_prefix_commit_frac(),
        commit_frac=(
            float(np.mean(end_commit_frac)) if end_commit_frac else float("nan")
        ),
    )

# --- from run_meta.py ---
_KNOWN_MODELS = ("llada", "llada1_5", "llada-300m-pcfg","llada-300m-pcfg-1B", "llada-200m-pcfg", "llada-100m-pcfg", "dream")

# Longer mode names first (Hub files use top_k_confidence / top_k_top_margin).
_KNOWN_MODES = (
    "top_k_confidence",
    "top_k_top_margin",
    "top_k_margin",
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
    "top_k_confidence": "top_k_c",
    "top_k_top_margin": "top_k_m",
    "top_k_margin": "top_k_m",
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
_IDENTITY_COLUMNS: List[str] = [
    "repo",
    "filename",
    "model",
    "scheduler",
    "task",
    "steps",
    "prefix_steps",
    "is_pcfg",
    "n_samples",
    "n_steps",
]
_CAP_COLUMNS: List[str] = [
    "M_hat",
    "J_hat",
    "G_hat",
    "commit_frac",
]
_COMPLETION_COLUMNS: List[str] = [
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
]
_EVAL_COLUMNS: List[str] = [
    "accuracy",
    "avg_nfe_per_sample",
    "strace_name",
    "strace_metric_key",
]
_ORACLE_COLUMNS: List[str] = [
    "model_commit_ppl",
    "oracle_ppl",
    "Mt_total",
    "Jt_total",
    "pearson_M",
    "pearson_J",
    "pearson_G",
]
NORMAL_CSV_COLUMNS: List[str] = (
    _IDENTITY_COLUMNS + _CAP_COLUMNS + _COMPLETION_COLUMNS + _EVAL_COLUMNS
)
PCFG_CSV_COLUMNS: List[str] = (
    _IDENTITY_COLUMNS + _CAP_COLUMNS + _COMPLETION_COLUMNS + _ORACLE_COLUMNS
)
CSV_COLUMNS: List[str] = NORMAL_CSV_COLUMNS

# Cleared on prefix-K rows (full-trajectory oracle totals).
_PREFIX_ROW_CLEAR_KEYS: Tuple[str, ...] = (
    "model_commit_ppl",
    "oracle_ppl",
    "Mt_total",
    "Jt_total",
)


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


def _row_is_pcfg(row: Mapping[str, Any]) -> bool:
    value = row.get("is_pcfg")
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def empty_metrics_row(**overrides: Any) -> Dict[str, Any]:
    """Return a CSV row dict with NaN/empty defaults, then apply overrides."""
    columns = list(dict.fromkeys(NORMAL_CSV_COLUMNS + PCFG_CSV_COLUMNS))
    row: Dict[str, Any] = {col: "" for col in columns}
    for key in (
        "M_hat",
        "J_hat",
        "G_hat",
        "commit_frac",
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
        *_ORACLE_COLUMNS,
    ):
        row[key] = float("nan")
    row.update(overrides)
    return row


def write_metrics_csv(
    path: Union[str, Path],
    rows: Sequence[Mapping[str, Any]],
    *,
    append: bool = False,
    pcfg: Optional[bool] = None,
) -> Path:
    """Write or append SchedulerLens metrics rows.

    A normal sheet has the eval columns. A PCFG sheet has the oracle and
    correlation columns instead.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if pcfg is None:
        flags = [_row_is_pcfg(row) for row in rows]
        pcfg = bool(flags) and all(flags)
    columns = PCFG_CSV_COLUMNS if pcfg else NORMAL_CSV_COLUMNS
    write_header = not append or not path.exists() or path.stat().st_size == 0
    mode = "a" if append and path.exists() else "w"
    with path.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({col: _fmt(row.get(col, "")) for col in columns})
    return path


def read_metrics_csv(path: Union[str, Path]) -> List[Dict[str, str]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# --- sample-level prefix correlations (PCFG) ---
CORR_PREFIXES: Tuple[int, ...] = (8, 16, 32, 64)


def pearson_corr(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation; NaN if fewer than 2 finite paired points or zero variance."""
    xs = np.asarray(x, dtype=np.float64).reshape(-1)
    ys = np.asarray(y, dtype=np.float64).reshape(-1)
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


def per_sample_prefix_sums(
    sample_idx: Sequence[int],
    step: Sequence[int],
    values: Sequence[float],
    n_samples: int,
    prefixes: Sequence[int],
) -> Dict[str, np.ndarray]:
    """Sum ``values`` per sample over steps before each prefix, plus the full sum."""
    idx = np.asarray(sample_idx, dtype=np.int64).reshape(-1)
    st = np.asarray(step, dtype=np.int64).reshape(-1)
    val = np.asarray(values, dtype=np.float64).reshape(-1)
    if not (idx.shape == st.shape == val.shape):
        raise ValueError("oracle step arrays have different lengths")
    n = max(int(n_samples), int(idx.max()) + 1 if idx.size else 0)
    finite = np.isfinite(val)
    out: Dict[str, np.ndarray] = {}
    cuts: List[Tuple[str, Optional[int]]] = [("full", None)]
    cuts.extend((str(int(k)), int(k)) for k in prefixes)
    for label, limit in cuts:
        mask = finite if limit is None else finite & (st < limit)
        totals = np.zeros(n, dtype=np.float64)
        if int(mask.sum()):
            np.add.at(totals, idx[mask], val[mask])
        out[label] = totals
    return out


def prefix_pearsons(
    metrics: TaskMetrics,
    *,
    step_sample_idx: Sequence[int],
    step: Sequence[int],
    step_mt: Sequence[float],
    step_jt: Sequence[float],
) -> Dict[str, Tuple[float, float, float]]:
    """Sample-level Pearson of M̂/Mt, Ĵ/Jt, and Ĝ/G at each prefix and full.

    Ĝ is M̂ + Ĵ and G is Mt + Jt, summed over the same steps as the pair.
    """
    prefixes = tuple(sorted(int(k) for k in metrics.prefix_samples))
    n = int(metrics.n_samples)
    mt = per_sample_prefix_sums(step_sample_idx, step, step_mt, n, prefixes)
    jt = per_sample_prefix_sums(step_sample_idx, step, step_jt, n, prefixes)
    m_hat: Dict[str, Sequence[float]] = {
        "full": metrics.proxies["M_hat_t"].sample_caps,
    }
    j_hat: Dict[str, Sequence[float]] = {
        "full": metrics.proxies["J_hat_t"].sample_caps,
    }
    for k, bucket in metrics.prefix_samples.items():
        label = str(int(k))
        m_hat[label] = bucket["M"]
        j_hat[label] = bucket["J"]
    out: Dict[str, Tuple[float, float, float]] = {}
    for label, mh in m_hat.items():
        if label not in mt or label not in jt or label not in j_hat:
            continue
        mh_arr = np.asarray(mh, dtype=np.float64).reshape(-1)
        jh_arr = np.asarray(j_hat[label], dtype=np.float64).reshape(-1)
        mt_arr = np.asarray(mt[label][:n], dtype=np.float64).reshape(-1)
        jt_arr = np.asarray(jt[label][:n], dtype=np.float64).reshape(-1)
        out[label] = (
            pearson_corr(mh_arr, mt_arr),
            pearson_corr(jh_arr, jt_arr),
            pearson_corr(mh_arr + jh_arr, mt_arr + jt_arr),
        )
    return out


# --- from analyze.py ---

def build_metrics_row(
    trace_path: Union[str, Path],
    *,
    metrics: Optional[TaskMetrics] = None,
    perplexity: Optional[Any] = None,
    prefix_pearson: Optional[Mapping[str, Tuple[float, float, float]]] = None,
    meta: Optional[RunMeta] = None,
    repo: str = "",
    max_samples: Optional[int] = None,
    verbose: bool = False,
    completion: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble one CSV-ready row for a trace NPZ (full trajectory)."""
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
        prefix_steps="",
        is_pcfg=meta.is_pcfg,
        n_samples=metrics.n_samples,
        n_steps=metrics.n_steps,
        M_hat=metrics.M_hat,
        J_hat=metrics.J_hat,
        G_hat=float(metrics.M_hat) + float(metrics.J_hat),
        commit_frac=metrics.commit_frac,
    )

    if completion:
        for key in (
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
        ):
            if key in completion:
                row[key] = completion[key]

    if perplexity is not None:
        row["model_commit_ppl"] = perplexity.model_commit_ppl
        row["oracle_ppl"] = perplexity.oracle_ppl
        row["Mt_total"] = perplexity.Mt_total
        row["Jt_total"] = perplexity.Jt_total

    full_m, full_j, full_g = (prefix_pearson or {}).get(
        "full", (float("nan"), float("nan"), float("nan"))
    )
    row["pearson_M"] = full_m
    row["pearson_J"] = full_j
    row["pearson_G"] = full_g

    return row


def expand_prefix_metric_rows(
    base_row: Mapping[str, Any],
    metrics: TaskMetrics,
    prefix_pearson: Optional[Mapping[str, Tuple[float, float, float]]] = None,
) -> List[Dict[str, Any]]:
    """Full-traj row plus one row per K in ``metrics.prefix_caps``.

    Prefix rows overwrite ``M_hat`` / ``J_hat`` / ``G_hat`` with cumulative caps
    over ``step < K``. ``pearson_M``, ``pearson_J``, and ``pearson_G`` are the
    sample-level Pearson values for that same prefix.
    """
    pearson = prefix_pearson or {}
    rows: List[Dict[str, Any]] = [dict(base_row)]
    if not metrics.prefix_caps:
        return rows
    for k in sorted(metrics.prefix_caps):
        m, j, g = metrics.prefix_caps[k]
        row = dict(base_row)
        row["prefix_steps"] = int(k)
        row["M_hat"] = float(m)
        row["J_hat"] = float(j)
        row["G_hat"] = float(g)
        row["commit_frac"] = float(metrics.prefix_commit_frac.get(k, float("nan")))
        pair = pearson.get(str(int(k)))
        if pair is not None:
            row["pearson_M"], row["pearson_J"], row["pearson_G"] = pair
        for key in _PREFIX_ROW_CLEAR_KEYS:
            row[key] = float("nan")
        rows.append(row)
    return rows


def collect_sample_proxy_trends(
    trace_path: Union[str, Path],
    *,
    max_samples: Optional[int] = None,
) -> tuple[List[SampleProxies], int]:
    """Return per-sample proxies and max step count (fail-hard)."""
    samples: List = []
    for i, tr in enumerate(_iter_traces(trace_path)):
        if max_samples is not None and i >= max_samples:
            break
        samples.append(compute_sample_proxies(tr, trace_name=trace_path))
    t_max = max((len(sp.step_proxies) for sp in samples), default=0)
    return samples, t_max

