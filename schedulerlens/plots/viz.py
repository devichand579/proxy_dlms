"""All SchedulerLens matplotlib plots (trends, surfaces, compare, quad, eval-join)."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from schedulerlens.metrics import TaskMetrics, _iter_traces, trajectory_blocks
from schedulerlens.proxies import J_PROXY_ATTRS, M_PROXY_ATTRS, PROXY_ATTRS

# --- util ---
def finite_values(rows: Sequence[Mapping[str, Any]], key: str) -> bool:
    for row in rows:
        try:
            v = float(row.get(key, float("nan")))
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            return True
    return False


def row_float(row: Mapping[str, Any], key: str) -> float:
    try:
        return float(row.get(key, float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def unique_sorted(rows: Sequence[Mapping[str, Any]], key: str) -> list[str]:
    return sorted({str(r.get(key, "")) for r in rows if str(r.get(key, ""))})

# --- surfaces ---
@dataclass
class TraceSurfaces:
    """Mean step×token matrices across samples (shape [n_steps, gen_length])."""
    confidence: np.ndarray
    entropy: np.ndarray
    mt_at_commit: np.ndarray = field(default_factory=lambda: np.array([]))
    n_samples: int = 0
    n_steps: int = 0
    gen_length: int = 0

    def to_dict(self) -> dict:
        out = {
            "n_samples": self.n_samples,
            "n_steps": self.n_steps,
            "gen_length": self.gen_length,
            "confidence": self.confidence.tolist(),
            "entropy": self.entropy.tolist(),
        }
        if self.mt_at_commit.size:
            out["mt_at_commit"] = self.mt_at_commit.tolist()
        return out


def _squeeze(arr) -> np.ndarray:
    out = np.asarray(arr)
    while out.ndim > 1 and out.shape[0] == 1:
        out = out[0]
    return out


def compute_trace_surfaces(
    trace_path: Union[str, Path],
    *,
    max_samples: Optional[int] = None,
    mt_commit_grids: Optional[List[np.ndarray]] = None,
) -> TraceSurfaces:
    """Average confidences / entropies over samples into step×token matrices.

    ``mt_commit_grids`` (optional): per-sample arrays of shape [n_steps, gen_length]
    with Mt contribution at commit positions (zeros elsewhere), averaged the same way.
    """
    conf_acc: Optional[np.ndarray] = None
    ent_acc: Optional[np.ndarray] = None
    conf_count: Optional[np.ndarray] = None
    ent_count: Optional[np.ndarray] = None
    n_samples = 0
    n_steps = 0
    gen_length = 0

    for i, tr in enumerate(_iter_traces(trace_path)):
        if max_samples is not None and i >= max_samples:
            break
        steps = tr.get("step_traces") or []
        if not steps:
            continue
        g = int(tr.get("gen_length") or 0)
        t_len = len(steps)
        if g <= 0:
            first = _squeeze(steps[0].get("confidences"))
            g = int(first.shape[0]) if first is not None else 0
        if g <= 0:
            continue

        if conf_acc is None:
            n_steps = t_len
            gen_length = g
            conf_acc = np.zeros((n_steps, gen_length), dtype=np.float64)
            ent_acc = np.zeros((n_steps, gen_length), dtype=np.float64)
            conf_count = np.zeros((n_steps, gen_length), dtype=np.float64)
            ent_count = np.zeros((n_steps, gen_length), dtype=np.float64)
        else:
            n_steps = max(n_steps, t_len)
            if t_len > conf_acc.shape[0] or g > conf_acc.shape[1]:
                new_conf = np.zeros((n_steps, max(gen_length, g)), dtype=np.float64)
                new_ent = np.zeros_like(new_conf)
                new_cc = np.zeros_like(new_conf)
                new_ec = np.zeros_like(new_conf)
                new_conf[: conf_acc.shape[0], : conf_acc.shape[1]] = conf_acc
                new_ent[: ent_acc.shape[0], : ent_acc.shape[1]] = ent_acc
                new_cc[: conf_count.shape[0], : conf_count.shape[1]] = conf_count
                new_ec[: ent_count.shape[0], : ent_count.shape[1]] = ent_count
                conf_acc, ent_acc = new_conf, new_ent
                conf_count, ent_count = new_cc, new_ec
                gen_length = max(gen_length, g)

        for t, st in enumerate(steps):
            if t >= conf_acc.shape[0]:
                break
            conf = st.get("confidences")
            ent = st.get("entropies")
            if conf is not None:
                row = _squeeze(conf).astype(np.float64)
                n = min(len(row), conf_acc.shape[1])
                conf_acc[t, :n] += row[:n]
                conf_count[t, :n] += 1.0
            if ent is not None:
                row = _squeeze(ent).astype(np.float64)
                n = min(len(row), ent_acc.shape[1])
                ent_acc[t, :n] += row[:n]
                ent_count[t, :n] += 1.0
        n_samples += 1

    if conf_acc is None or n_samples == 0:
        return TraceSurfaces(
            confidence=np.array([]),
            entropy=np.array([]),
            n_samples=0,
            n_steps=0,
            gen_length=0,
        )

    conf_mean = np.divide(
        conf_acc, conf_count, out=np.full_like(conf_acc, np.nan), where=conf_count > 0
    )
    ent_mean = np.divide(
        ent_acc, ent_count, out=np.full_like(ent_acc, np.nan), where=ent_count > 0
    )

    mt_mean = np.array([])
    if mt_commit_grids:
        mt_acc = np.zeros_like(conf_mean)
        mt_count = np.zeros_like(conf_mean)
        for grid in mt_commit_grids:
            g = np.asarray(grid, dtype=np.float64)
            ts = min(g.shape[0], mt_acc.shape[0])
            gs = min(g.shape[1], mt_acc.shape[1]) if g.ndim == 2 else 0
            if gs == 0:
                continue
            mt_acc[:ts, :gs] += g[:ts, :gs]
            mt_count[:ts, :gs] += 1.0
        mt_mean = np.divide(
            mt_acc, mt_count, out=np.full_like(mt_acc, np.nan), where=mt_count > 0
        )

    return TraceSurfaces(
        confidence=conf_mean,
        entropy=ent_mean,
        mt_at_commit=mt_mean,
        n_samples=n_samples,
        n_steps=int(conf_mean.shape[0]),
        gen_length=int(conf_mean.shape[1]),
    )

# --- trends ---
matplotlib.use("Agg")


_PROXY_LABELS = {
    "M_hat_t": "M̂_t (main)",
    "mean_regret": "mean regret",
    "cascade_gain": "cascade gain",
    "J_hat_t": "Ĵ_t (main)",
    "instability": "instability",
    "rank_volatility": "rank volatility",
}
_MAIN_PROXIES = frozenset({"M_hat_t", "J_hat_t"})

_SIDE_CONFIG = {
    "M": {
        "attrs": M_PROXY_ATTRS,
        "filename": "M_trends.png",
        "suptitle": "M-side: ordering quality",
        "labels": {
            "M_hat_t": "M̂_t (main)",
            "mean_regret": "mean regret",
            "cascade_gain": "cascade gain",
        },
    },
    "J": {
        "attrs": J_PROXY_ATTRS,
        "filename": "J_trends.png",
        "suptitle": "J-side: parallelism error",
        "labels": {
            "J_hat_t": "Ĵ_t (main)",
            "instability": "instability",
            "rank_volatility": "rank volatility",
        },
    },
}


def save_trend_plots(
    metrics: TaskMetrics,
    output_dir: Union[str, Path],
    prefix: Optional[str] = None,
    dpi: int = 150,
) -> Tuple[Path, Path, Path]:
    """Write M/J trend panels and unmasked-count trend; return (M, J, unmasked)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = prefix or "task"

    if metrics.n_steps == 0:
        raise ValueError("Cannot plot trends: metrics have n_steps=0")

    steps = np.arange(metrics.n_steps)
    m_path = _save_side_panel(metrics, steps, stem, output_dir, "M", dpi)
    j_path = _save_side_panel(metrics, steps, stem, output_dir, "J", dpi)
    unmasked_path = _save_unmasked_trend(metrics, steps, stem, output_dir, dpi)
    return m_path, j_path, unmasked_path


def _save_side_panel(
    metrics: TaskMetrics,
    steps: np.ndarray,
    stem: str,
    output_dir: Path,
    side: str,
    dpi: int,
) -> Path:
    cfg = _SIDE_CONFIG[side]
    attrs = cfg["attrs"]
    out_path = output_dir / f"{stem}_{cfg['filename']}"

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)

    for idx, (ax, name) in enumerate(zip(axes, attrs)):
        rep = metrics.proxies[name]
        mean = np.asarray(rep.trend, dtype=np.float64)
        valid = np.isfinite(mean)
        is_main = idx == 0
        color = "#1d4ed8" if is_main else "#64748b"
        lw = 2.0 if is_main else 1.2

        ax.plot(steps[valid], mean[valid], color=color, linewidth=lw)
        ax.set_ylabel(cfg["labels"][name])
        ax.set_title(
            cfg["labels"][name],
            fontweight="bold" if is_main else "normal",
            loc="left",
        )
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Step")
    fig.suptitle(cfg["suptitle"], fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def save_bucket_plots(
    metrics: TaskMetrics,
    output_dir: Union[str, Path],
    prefix: Optional[str] = None,
    dpi: int = 150,
) -> Tuple[Path, Path]:
    """Write bucketed 6-proxy panel and unmasked panel; return both paths."""
    if metrics.bucket_size is None or metrics.n_buckets == 0:
        raise ValueError("bucket_size must be set on metrics before plotting buckets")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = prefix or "task"
    bs = metrics.bucket_size
    buckets = np.arange(metrics.n_buckets)

    all_path = output_dir / f"{stem}_bucket{bs}_all_trends.png"
    unmasked_path = output_dir / f"{stem}_bucket{bs}_unmasked.png"

    fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
    for ax, name in zip(axes, PROXY_ATTRS):
        trend = np.asarray(metrics.proxies[name].bucket_trend, dtype=np.float64)
        valid = np.isfinite(trend)
        is_main = name in _MAIN_PROXIES
        color = "#1d4ed8" if is_main else "#64748b"
        lw = 2.0 if is_main else 1.2
        ax.plot(buckets[valid], trend[valid], color=color, linewidth=lw)
        ax.set_ylabel(_PROXY_LABELS[name])
        ax.set_title(
            _PROXY_LABELS[name],
            fontweight="bold" if is_main else "normal",
            loc="left",
        )
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(f"Step bucket (size={bs})")
    fig.suptitle(
        "Bucketed proxy trends (mean over steps in each bucket)", fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(all_path, dpi=dpi)
    plt.close(fig)

    unmasked = np.asarray(metrics.n_unmasked_bucket_trend, dtype=np.float64)
    valid = np.isfinite(unmasked)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(buckets[valid], unmasked[valid], color="#059669", linewidth=2.0)
    ax.set_xlabel(f"Step bucket (size={bs})")
    ax.set_ylabel("Tokens unmasked (mean)")
    ax.set_title(
        f"Average tokens unmasked per bucket (size={bs})",
        loc="left",
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(unmasked_path, dpi=dpi)
    plt.close(fig)

    return all_path, unmasked_path


def _save_unmasked_trend(
    metrics: TaskMetrics,
    steps: np.ndarray,
    stem: str,
    output_dir: Path,
    dpi: int,
) -> Path:
    out_path = output_dir / f"{stem}_unmasked_count_trend.png"
    mean = np.asarray(metrics.n_unmasked_trend, dtype=np.float64)
    valid = np.isfinite(mean)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps[valid], mean[valid], color="#059669", linewidth=2.0)
    ax.set_xlabel("Step")
    ax.set_ylabel("Tokens unmasked (mean)")
    ax.set_title("Average tokens unmasked per step", loc="left", fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path

# --- quad ---
def _row_key(row: Mapping[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(row.get("model", "")),
        str(row.get("task", "")),
        str(row.get("scheduler", "")),
        str(row.get("steps", "")),
    )


def _index_rows(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[str, str, str, str], Mapping[str, Any]]:
    out: Dict[Tuple[str, str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        out[_row_key(row)] = row
    return out


def save_quad_plots(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    dpi: int = 150,
) -> Dict[str, Path]:
    """Write faceted bars + heatmaps for M_hat / J_hat over the four axes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return {}

    paths: Dict[str, Path] = {}
    paths["quad_bars"] = _save_quad_bars(rows, output_dir, dpi, plt)
    heat = _save_quad_heatmaps(rows, output_dir, dpi, plt)
    paths.update(heat)
    return paths


def _save_quad_bars(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    dpi: int,
    plt,
) -> Path:
    """One row per (model, task); grouped bars of M_hat / J_hat by scheduler, hue=steps."""
    out = output_dir / "compare_quad_bars.png"
    models = unique_sorted(rows, "model")
    tasks = unique_sorted(rows, "task")
    schedulers = unique_sorted(rows, "scheduler")
    step_vals = unique_sorted(rows, "steps")
    pairs = [(m, t) for m in models for t in tasks]
    # Keep only pairs that appear.
    pairs = [
        (m, t)
        for m, t in pairs
        if any(str(r.get("model")) == m and str(r.get("task")) == t for r in rows)
    ]
    if not pairs:
        pairs = [(models[0] if models else "", tasks[0] if tasks else "")]

    n_rows = len(pairs)
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(max(8, 0.55 * len(schedulers) * max(len(step_vals), 1) + 2), 2.8 * n_rows),
        squeeze=False,
    )
    indexed = _index_rows(rows)
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(step_vals), 1)))
    x = np.arange(len(schedulers))
    width = 0.8 / max(len(step_vals), 1)

    for ri, (model, task) in enumerate(pairs):
        for ci, metric in enumerate(("M_hat", "J_hat")):
            ax = axes[ri][ci]
            for si, steps in enumerate(step_vals):
                vals = []
                for sched in schedulers:
                    row = indexed.get((model, task, sched, steps), {})
                    vals.append(row_float(row, metric))
                offsets = x - 0.4 + width / 2 + si * width
                plot_vals = [v if np.isfinite(v) else 0.0 for v in vals]
                ax.bar(
                    offsets,
                    plot_vals,
                    width=width * 0.9,
                    color=cmap[si],
                    label=f"s={steps}" if ri == 0 and ci == 0 else None,
                )
            ax.set_xticks(x)
            ax.set_xticklabels(schedulers, rotation=45, ha="right", fontsize=7)
            ax.set_title(f"{model} · {task} · {metric}", loc="left", fontsize=9, fontweight="bold")
            ax.grid(True, axis="y", alpha=0.3)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=8, title="step budget")
    fig.suptitle("M̂ / Ĵ by model × task × scheduler × steps", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 0.92, 0.96))
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def _save_quad_heatmaps(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    dpi: int,
    plt,
) -> Dict[str, Path]:
    """scheduler × steps heatmaps of M_hat and J_hat, one figure per metric, faceted by model×task."""
    paths: Dict[str, Path] = {}
    models = unique_sorted(rows, "model")
    tasks = unique_sorted(rows, "task")
    schedulers = unique_sorted(rows, "scheduler")
    step_vals = unique_sorted(rows, "steps")
    pairs = [
        (m, t)
        for m in models
        for t in tasks
        if any(str(r.get("model")) == m and str(r.get("task")) == t for r in rows)
    ]
    if not pairs or not schedulers or not step_vals:
        return paths

    indexed = _index_rows(rows)
    for metric, key in (("M_hat", "quad_heat_M"), ("J_hat", "quad_heat_J")):
        out = output_dir / f"compare_{key}.png"
        n = len(pairs)
        n_cols = min(3, n)
        n_rows = (n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3.8 * n_cols, 2.8 * n_rows),
            squeeze=False,
        )
        for idx, (model, task) in enumerate(pairs):
            ax = axes[idx // n_cols][idx % n_cols]
            mat = np.full((len(schedulers), len(step_vals)), np.nan, dtype=np.float64)
            for i, sched in enumerate(schedulers):
                for j, steps in enumerate(step_vals):
                    mat[i, j] = row_float(
                        indexed.get((model, task, sched, steps), {}), metric
                    )
            im = ax.imshow(mat, aspect="auto", cmap="viridis")
            ax.set_xticks(range(len(step_vals)))
            ax.set_xticklabels(step_vals, fontsize=7)
            ax.set_yticks(range(len(schedulers)))
            ax.set_yticklabels(schedulers, fontsize=7)
            ax.set_xlabel("steps")
            ax.set_title(f"{model} · {task}", loc="left", fontsize=8, fontweight="bold")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for idx in range(len(pairs), n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].axis("off")
        fig.suptitle(f"{metric}: scheduler × step budget", fontweight="bold")
        fig.tight_layout()
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        paths[key] = out
    return paths

# --- eval_join ---
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_STRACE_METRICS = _REPO_ROOT / "fetch_evals" / "strace_metrics.csv"

# Map short scheduler keys used by SchedulerLens → strace `mode` substrings.
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


def _safe_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _stem_from_row(row: Mapping[str, Any]) -> str:
    filename = str(row.get("filename") or "")
    if filename.endswith(".npz"):
        return filename[:-4]
    if filename:
        return Path(filename).stem
    return ""


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


def save_eval_join_plots(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    strace_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
) -> Dict[str, Path]:
    """Scatter accuracy vs proxies and NFE Pareto; skip if no joins."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joined = join_rows_with_strace(rows, strace_path=strace_path)
    n_hit = sum(1 for r in joined if np.isfinite(row_float(r, "accuracy")))
    if n_hit == 0:
        logger.info("No strace_metrics joins; skipping accuracy/NFE plots")
        return {}

    paths: Dict[str, Path] = {}
    paths["acc_vs_proxy"] = _save_accuracy_vs_proxy(joined, output_dir, dpi, plt)
    paths["nfe_pareto"] = _save_nfe_pareto(joined, output_dir, dpi, plt)
    paths["budget_curves"] = _save_budget_curves(joined, output_dir, dpi, plt)
    return paths


def _save_accuracy_vs_proxy(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    dpi: int,
    plt,
) -> Path:
    out = output_dir / "compare_accuracy_vs_proxy.png"
    schedulers = unique_sorted(rows, "scheduler")
    colors = {
        s: plt.cm.tab10(i / max(len(schedulers), 1)) for i, s in enumerate(schedulers)
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, proxy in zip(axes, ("M_hat", "J_hat")):
        for row in rows:
            acc = row_float(row, "accuracy")
            pval = row_float(row, proxy)
            if not (np.isfinite(acc) and np.isfinite(pval)):
                continue
            sched = str(row.get("scheduler", ""))
            steps = row_float(row, "steps")
            size = 40 + (0 if not np.isfinite(steps) else min(steps, 512) / 512 * 80)
            ax.scatter(
                pval,
                acc,
                s=size,
                color=colors.get(sched, "#64748b"),
                alpha=0.75,
                label=sched,
            )
        ax.set_xlabel(proxy)
        ax.set_ylabel("accuracy (strace metric)")
        ax.set_title(f"Accuracy vs {proxy}", loc="left", fontweight="bold")
        ax.grid(True, alpha=0.3)
        # Dedupe legend labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc="best")
    fig.suptitle("Accuracy vs M̂ / Ĵ (point size ∝ step budget)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def _save_nfe_pareto(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    dpi: int,
    plt,
) -> Path:
    out = output_dir / "compare_nfe_pareto.png"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    schedulers = unique_sorted(rows, "scheduler")
    colors = {
        s: plt.cm.tab10(i / max(len(schedulers), 1)) for i, s in enumerate(schedulers)
    }
    for ax, color_key, title in (
        (axes[0], "M_hat", "NFE vs accuracy (color = M̂)"),
        (axes[1], "J_hat", "NFE vs accuracy (color = Ĵ)"),
    ):
        xs, ys, cs, labs = [], [], [], []
        for row in rows:
            nfe = row_float(row, "avg_nfe_per_sample")
            acc = row_float(row, "accuracy")
            cval = row_float(row, color_key)
            if not (np.isfinite(nfe) and np.isfinite(acc)):
                continue
            xs.append(nfe)
            ys.append(acc)
            cs.append(cval if np.isfinite(cval) else 0.0)
            labs.append(str(row.get("scheduler", "")))
        if not xs:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue
        sc = ax.scatter(xs, ys, c=cs, cmap="viridis", s=55, alpha=0.85)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=color_key)
        # Light scheduler markers via edge annotation for a few points
        for x, y, lab in zip(xs, ys, labs):
            ax.annotate(lab, (x, y), fontsize=6, alpha=0.7)
        ax.set_xlabel("avg NFE / sample")
        ax.set_ylabel("accuracy")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(True, alpha=0.3)
    fig.suptitle("NFE–accuracy Pareto with proxy coloring", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def _save_budget_curves(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    dpi: int,
    plt,
) -> Path:
    """For each (model, task, scheduler) with ≥2 step budgets, plot M_hat / accuracy vs steps."""
    out = output_dir / "compare_budget_curves.png"
    groups: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("model", "")),
            str(row.get("task", "")),
            str(row.get("scheduler", "")),
        )
        groups.setdefault(key, []).append(row)

    multi = {k: v for k, v in groups.items() if len(v) >= 2}
    if not multi:
        # Still write a small placeholder explaining why.
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Need ≥2 step budgets per (model, task, scheduler) for budget curves",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        return out

    keys = sorted(multi.keys())[:8]
    n = len(keys)
    fig, axes = plt.subplots(n, 2, figsize=(10, 2.4 * n), squeeze=False)
    for i, key in enumerate(keys):
        group = sorted(multi[key], key=lambda r: row_float(r, "steps"))
        steps = [row_float(r, "steps") for r in group]
        mhat = [row_float(r, "M_hat") for r in group]
        jhat = [row_float(r, "J_hat") for r in group]
        acc = [row_float(r, "accuracy") for r in group]
        model, task, sched = key

        ax0 = axes[i][0]
        ax0.plot(steps, mhat, "o-", color="#1d4ed8", label="M̂")
        ax0.plot(steps, jhat, "s--", color="#b45309", label="Ĵ")
        ax0.set_xlabel("steps")
        ax0.set_title(f"{model} · {task} · {sched}: proxies", loc="left", fontsize=8)
        ax0.legend(fontsize=7)
        ax0.grid(True, alpha=0.3)

        ax1 = axes[i][1]
        if any(np.isfinite(a) for a in acc):
            ax1.plot(steps, acc, "o-", color="#059669", label="accuracy")
        nfe = [row_float(r, "avg_nfe_per_sample") for r in group]
        if any(np.isfinite(v) for v in nfe):
            ax1b = ax1.twinx()
            ax1b.plot(steps, nfe, "s--", color="#7c3aed", label="NFE")
            ax1b.set_ylabel("NFE", color="#7c3aed")
        ax1.set_xlabel("steps")
        ax1.set_ylabel("accuracy")
        ax1.set_title(f"{model} · {task} · {sched}: accuracy / NFE", loc="left", fontsize=8)
        ax1.grid(True, alpha=0.3)

    fig.suptitle("Step-budget curves", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out

# --- compare orchestration ---
matplotlib.use("Agg")




def save_compare_plots(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Union[str, Path],
    *,
    surfaces_by_stem: Optional[Dict[str, TraceSurfaces]] = None,
    fidelity_points: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    strace_metrics_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
) -> Dict[str, Path]:
    """Write overview / quad / fidelity / PPL / surface / accuracy-NFE plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    if rows:
        paths["overview"] = _save_overview(rows, output_dir, dpi)
        paths.update(save_quad_plots(rows, output_dir, dpi=dpi))
        if finite_values(rows, "oracle_ppl"):
            paths["ppl"] = _save_ppl_compare(rows, output_dir, dpi)
        if finite_values(rows, "corr_M_full") or finite_values(rows, "corr_J_full"):
            paths["corr_bars"] = _save_corr_bars(rows, output_dir, dpi)
        paths.update(
            save_eval_join_plots(
                rows, output_dir, strace_path=strace_metrics_path, dpi=dpi
            )
        )

    if fidelity_points:
        paths["fidelity"] = _save_fidelity_scatter(fidelity_points, output_dir, dpi)

    if surfaces_by_stem:
        paths["surfaces"] = _save_surface_panel(surfaces_by_stem, output_dir, dpi)

    return paths


def _save_overview(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    dpi: int,
) -> Path:
    out = output_dir / "compare_overview.png"
    metrics = ["M_hat", "J_hat", "model_commit_ppl"]
    if finite_values(rows, "oracle_ppl"):
        metrics.append("oracle_ppl")

    models = unique_sorted(rows, "model")
    schedulers = unique_sorted(rows, "scheduler")
    n_metrics = len(metrics)
    n_models = max(len(models), 1)
    fig, axes = plt.subplots(
        n_models, n_metrics, figsize=(3.6 * n_metrics, 2.8 * n_models), squeeze=False
    )

    x = np.arange(len(schedulers))
    for mi, model in enumerate(models):
        model_rows = [r for r in rows if str(r.get("model", "")) == model]
        by_sched = {str(r.get("scheduler", "")): r for r in model_rows}
        for mj, metric in enumerate(metrics):
            ax = axes[mi][mj]
            vals = [row_float(by_sched.get(s, {}), metric) for s in schedulers]
            colors = ["#1d4ed8" if np.isfinite(v) else "#cbd5e1" for v in vals]
            plot_vals = [v if np.isfinite(v) else 0.0 for v in vals]
            ax.bar(x, plot_vals, color=colors, width=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(schedulers, rotation=45, ha="right", fontsize=7)
            ax.set_title(f"{model} · {metric}", loc="left", fontsize=9, fontweight="bold")
            ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Scheduler × model overview", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def _save_ppl_compare(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    dpi: int,
) -> Path:
    out = output_dir / "compare_ppl.png"
    models = unique_sorted(rows, "model")
    fig, axes = plt.subplots(
        1, max(len(models), 1), figsize=(4.5 * max(len(models), 1), 4), squeeze=False
    )
    for i, model in enumerate(models):
        ax = axes[0][i]
        subset = [r for r in rows if str(r.get("model", "")) == model]
        sched = [str(r.get("scheduler", "")) for r in subset]
        model_ppl = [row_float(r, "model_commit_ppl") for r in subset]
        oracle_ppl = [row_float(r, "oracle_ppl") for r in subset]
        x = np.arange(len(sched))
        w = 0.35
        ax.bar(
            x - w / 2,
            [v if np.isfinite(v) else 0 for v in model_ppl],
            w,
            label="model",
            color="#2563eb",
        )
        ax.bar(
            x + w / 2,
            [v if np.isfinite(v) else 0 for v in oracle_ppl],
            w,
            label="oracle",
            color="#059669",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(sched, rotation=45, ha="right", fontsize=7)
        ax.set_title(
            f"{model}: model vs oracle PPL", loc="left", fontweight="bold", fontsize=9
        )
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def _save_corr_bars(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    dpi: int,
) -> Path:
    out = output_dir / "compare_proxy_corr.png"
    blocks = ["full", "early", "mid", "late"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, side, prefix in (
        (axes[0], "M", "corr_M_"),
        (axes[1], "J", "corr_J_"),
    ):
        means = []
        for b in blocks:
            vals = [row_float(r, f"{prefix}{b}") for r in rows]
            finite = [v for v in vals if np.isfinite(v)]
            means.append(float(np.mean(finite)) if finite else float("nan"))
        heights = [m if np.isfinite(m) else float("nan") for m in means]
        ax.bar(blocks, [0.0 if not np.isfinite(h) else h for h in heights], color="#1d4ed8", alpha=0.85)
        for i, m in enumerate(means):
            if np.isfinite(m):
                ax.text(i, m, f"{m:.2f}", ha="center", va="bottom", fontsize=8)
            else:
                ax.text(i, 0.0, "n/a", ha="center", va="bottom", fontsize=8, color="#64748b")
        ax.set_ylim(-1.05, 1.05)
        ax.axhline(0, color="#94a3b8", linewidth=0.8)
        ax.set_title(
            f"{side} vs {side}̂ correlation by trajectory block",
            loc="left",
            fontweight="bold",
        )
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def _save_fidelity_scatter(
    fidelity_points: Dict[str, Dict[str, np.ndarray]],
    output_dir: Path,
    dpi: int,
) -> Path:
    out = output_dir / "compare_proxy_fidelity.png"
    mt_all: List[float] = []
    mhat_all: List[float] = []
    jt_all: List[float] = []
    jhat_all: List[float] = []
    block_ids: List[int] = []

    for _stem, pts in fidelity_points.items():
        mt = np.asarray(pts.get("mt", []), dtype=np.float64)
        mhat = np.asarray(pts.get("m_hat", []), dtype=np.float64)
        jt = np.asarray(pts.get("jt", []), dtype=np.float64)
        jhat = np.asarray(pts.get("j_hat", []), dtype=np.float64)
        n = min(len(mt), len(mhat), len(jt), len(jhat))
        blocks = trajectory_blocks(n)
        for t in range(n):
            if not (np.isfinite(mt[t]) and np.isfinite(mhat[t])):
                continue
            if t < blocks["early"][1]:
                bid = 0
            elif t < blocks["mid"][1]:
                bid = 1
            else:
                bid = 2
            mt_all.append(float(mt[t]))
            mhat_all.append(float(mhat[t]))
            jt_all.append(float(jt[t]) if np.isfinite(jt[t]) else float("nan"))
            jhat_all.append(float(jhat[t]) if np.isfinite(jhat[t]) else float("nan"))
            block_ids.append(bid)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {0: "#2563eb", 1: "#d97706", 2: "#059669"}
    labels = {0: "early", 1: "mid", 2: "late"}
    block_arr = np.asarray(block_ids)

    for ax, xs, ys, title in (
        (axes[0], mhat_all, mt_all, "Mt vs M̂_t"),
        (axes[1], jhat_all, jt_all, "Jt vs Ĵ_t"),
    ):
        xs_a = np.asarray(xs, dtype=np.float64)
        ys_a = np.asarray(ys, dtype=np.float64)
        for bid in (0, 1, 2):
            mask = (block_arr == bid) & np.isfinite(xs_a) & np.isfinite(ys_a)
            if not np.any(mask):
                continue
            ax.scatter(
                xs_a[mask],
                ys_a[mask],
                s=12,
                alpha=0.55,
                color=colors[bid],
                label=labels[bid],
            )
        finite = np.isfinite(xs_a) & np.isfinite(ys_a)
        if int(finite.sum()) >= 2:
            rho = float(np.corrcoef(xs_a[finite], ys_a[finite])[0, 1])
            ax.set_title(f"{title}  (ρ={rho:.3f})", loc="left", fontweight="bold")
        else:
            ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("proxy")
        ax.set_ylabel("oracle")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Proxy fidelity (oracle vs proxy per step)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def _save_surface_panel(
    surfaces_by_stem: Dict[str, TraceSurfaces],
    output_dir: Path,
    dpi: int,
) -> Path:
    out = output_dir / "compare_surfaces.png"
    stems = list(surfaces_by_stem.keys())[:6]
    n = len(stems)
    if n == 0:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "no surfaces", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        return out

    has_mt = any(surfaces_by_stem[s].mt_at_commit.size for s in stems)
    n_cols = 3 if has_mt else 2
    fig, axes = plt.subplots(n, n_cols, figsize=(4.2 * n_cols, 2.6 * n), squeeze=False)

    for i, stem in enumerate(stems):
        surf = surfaces_by_stem[stem]
        panels = [
            (surf.confidence, "confidence"),
            (surf.entropy, "entropy"),
        ]
        if has_mt:
            panels.append((surf.mt_at_commit, "Mt@commit"))
        for j, (mat, title) in enumerate(panels):
            ax = axes[i][j]
            if mat is None or (isinstance(mat, np.ndarray) and mat.size == 0):
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{stem}\n{title}", fontsize=8, loc="left")
            if i == n - 1:
                ax.set_xlabel("token pos")
            if j == 0:
                ax.set_ylabel("step")

    fig.suptitle("Step × token surfaces (mean across samples)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out
