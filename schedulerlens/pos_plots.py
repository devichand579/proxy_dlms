"""Plots for POS-stratified unmasking analysis."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .pos_analysis import PosTaskMetrics

PosPlotPaths = Tuple[Path, Path, Optional[Path], Optional[Path]]


def save_pos_plots(
    metrics: PosTaskMetrics,
    output_dir: Union[str, Path],
    prefix: Optional[str] = None,
    *,
    max_pos_lines: int = 8,
    dpi: int = 150,
) -> PosPlotPaths:
    """Write POS plots; return (mean_step, step_trend, bucket_count, bucket_mean)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = prefix or "task"

    if metrics.n_steps == 0 or not metrics.mean_unmask_step_by_pos:
        raise ValueError("Cannot plot POS metrics: no unmask events found")

    mean_path = _save_mean_unmask_by_pos(metrics, output_dir, stem, dpi)
    trend_path = _save_pos_unmask_trend(metrics, output_dir, stem, max_pos_lines, dpi)
    bucket_count_path: Optional[Path] = None
    bucket_mean_path: Optional[Path] = None
    if metrics.bucket_size is not None and metrics.n_buckets > 0:
        bucket_count_path = _save_pos_bucket_trend(
            metrics, output_dir, stem, max_pos_lines, dpi
        )
        bucket_mean_path = _save_bucket_mean_unmask_by_pos(
            metrics, output_dir, stem, max_pos_lines, dpi
        )
    return mean_path, trend_path, bucket_count_path, bucket_mean_path


def _top_pos_tags(metrics: PosTaskMetrics, n: int) -> List[str]:
    return sorted(
        metrics.pos_counts.keys(),
        key=lambda t: metrics.pos_counts[t],
        reverse=True,
    )[:n]


def _save_mean_unmask_by_pos(
    metrics: PosTaskMetrics,
    output_dir: Path,
    stem: str,
    dpi: int,
) -> Path:
    out_path = output_dir / f"{stem}_pos_mean_unmask_step.png"
    ranked = sorted(
        metrics.mean_unmask_step_by_pos.items(),
        key=lambda kv: kv[1],
    )
    tags = [t for t, _ in ranked]
    vals = [v for _, v in ranked]

    height = max(4.0, 0.28 * len(tags))
    fig, ax = plt.subplots(figsize=(9, height))
    y = np.arange(len(tags))
    ax.barh(y, vals, color="#2563eb")
    ax.set_yticks(y)
    ax.set_yticklabels(tags)
    ax.invert_yaxis()
    ax.set_xlabel("Mean normalized unmask step (t/T)")
    ax.set_title(
        "When each POS tag is committed (lower = earlier)",
        loc="left",
        fontweight="bold",
    )
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _save_pos_unmask_trend(
    metrics: PosTaskMetrics,
    output_dir: Path,
    stem: str,
    max_pos_lines: int,
    dpi: int,
) -> Path:
    out_path = output_dir / f"{stem}_pos_unmask_count_trend.png"
    steps = np.arange(metrics.n_steps)
    tags = _top_pos_tags(metrics, max_pos_lines)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(tags), 1)))

    fig, ax = plt.subplots(figsize=(10, 5))
    for color, tag in zip(colors, tags):
        trend = np.asarray(metrics.pos_unmask_count_trend.get(tag, []), dtype=np.float64)
        if len(trend) != metrics.n_steps:
            continue
        valid = np.isfinite(trend)
        ax.plot(steps[valid], trend[valid], label=tag, color=color, linewidth=1.5)

    ax.set_xlabel("Step")
    ax.set_ylabel("Tokens unmasked (mean across samples)")
    ax.set_title(
        "POS unmask rate over steps (top tags by frequency)",
        loc="left",
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _save_pos_bucket_trend(
    metrics: PosTaskMetrics,
    output_dir: Path,
    stem: str,
    max_pos_lines: int,
    dpi: int,
) -> Path:
    bs = int(metrics.bucket_size)
    out_path = output_dir / f"{stem}_pos_bucket{bs}_unmask_trend.png"
    buckets = np.arange(metrics.n_buckets)
    tags = _top_pos_tags(metrics, max_pos_lines)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(tags), 1)))

    fig, ax = plt.subplots(figsize=(10, 5))
    for color, tag in zip(colors, tags):
        trend = np.asarray(
            metrics.pos_unmask_bucket_trend.get(tag, []), dtype=np.float64
        )
        if len(trend) != metrics.n_buckets:
            continue
        valid = np.isfinite(trend)
        ax.plot(buckets[valid], trend[valid], label=tag, color=color, linewidth=1.8)

    ax.set_xlabel(f"Step bucket (size={bs})")
    ax.set_ylabel("Tokens unmasked (mean across samples)")
    ax.set_title(
        f"POS unmask counts by bucket (size={bs})",
        loc="left",
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _bucket_step_range(metrics: PosTaskMetrics, bucket_index: int) -> str:
    bs = int(metrics.bucket_size)
    start = bucket_index * bs
    end = min(start + bs, metrics.n_steps) - 1
    return f"{start}-{end}"


def _save_bucket_mean_unmask_by_pos(
    metrics: PosTaskMetrics,
    output_dir: Path,
    stem: str,
    max_pos_lines: int,
    dpi: int,
) -> Path:
    """One horizontal bar panel per bucket: earliest POS tags within the bucket."""
    bs = int(metrics.bucket_size)
    out_path = output_dir / f"{stem}_pos_bucket{bs}_mean_unmask_step.png"
    n_buckets = metrics.n_buckets
    n_cols = min(4, n_buckets)
    n_rows = (n_buckets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.6 * n_cols, 2.4 * n_rows),
        squeeze=False,
        sharex=True,
    )

    for b in range(n_buckets):
        ax = axes[b // n_cols][b % n_cols]
        candidates: List[Tuple[str, float]] = []
        for tag, series in metrics.mean_unmask_step_by_pos_bucket.items():
            if b >= len(series) or not np.isfinite(series[b]):
                continue
            candidates.append((tag, float(series[b])))

        if not candidates:
            ax.text(0.5, 0.5, "no events", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlim(0.0, 1.0)
        else:
            candidates.sort(key=lambda row: row[1])
            candidates = candidates[:max_pos_lines]
            tags = [row[0] for row in candidates]
            vals = [row[1] for row in candidates]
            y = np.arange(len(tags))
            ax.barh(y, vals, color="#2563eb", height=0.7)
            ax.set_yticks(y)
            ax.set_yticklabels(tags, fontsize=7)
            ax.invert_yaxis()
            ax.set_xlim(0.0, 1.0)

        ax.set_title(f"b={b}  steps {_bucket_step_range(metrics, b)}", fontsize=8, loc="left")
        ax.grid(True, axis="x", alpha=0.25)
        if b // n_cols == n_rows - 1:
            ax.set_xlabel("Mean step within bucket (0=start, 1=end)", fontsize=8)

    for idx in range(n_buckets, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    fig.suptitle(
        f"Earliest POS tags within each bucket (size={bs}; top {max_pos_lines} by mean step)",
        fontweight="bold",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
