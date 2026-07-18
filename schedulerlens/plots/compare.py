"""Multi-run comparison plots (overview, fidelity, PPL, surfaces) + orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from schedulerlens.correlation import trajectory_blocks

from ._util import finite_values, row_float, unique_sorted
from .eval_join import save_eval_join_plots
from .quad import save_quad_plots
from .surfaces import TraceSurfaces


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
        ax.bar(blocks, [m if np.isfinite(m) else 0 for m in means], color="#1d4ed8")
        for i, m in enumerate(means):
            if np.isfinite(m):
                ax.text(i, m, f"{m:.2f}", ha="center", va="bottom", fontsize=8)
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
