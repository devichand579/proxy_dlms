"""Save M-side and J-side trend panels (main proxy + two sub-proxies each)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .metrics import TaskMetrics
from .proxies import J_PROXY_ATTRS, M_PROXY_ATTRS, PROXY_ATTRS

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
        ax.set_title(cfg["labels"][name], fontweight="bold" if is_main else "normal", loc="left")
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
        ax.set_title(_PROXY_LABELS[name], fontweight="bold" if is_main else "normal", loc="left")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(f"Step bucket (size={bs})")
    fig.suptitle("Bucketed proxy trends (mean over steps in each bucket)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(all_path, dpi=dpi)
    plt.close(fig)

    unmasked = np.asarray(metrics.n_unmasked_bucket_trend, dtype=np.float64)
    valid = np.isfinite(unmasked)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(buckets[valid], unmasked[valid], color="#059669", linewidth=2.0)
    ax.set_xlabel(f"Step bucket (size={bs})")
    ax.set_ylabel("Tokens unmasked (mean)")
    ax.set_title(f"Average tokens unmasked per bucket (size={bs})", loc="left", fontweight="bold")
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
