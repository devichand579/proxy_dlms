#!/usr/bin/env python3
"""SchedulerLens completion-normalized paper figures (PDF only, no CSV outputs).

"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("atlas_plots")

DEFAULT_CSV = ROOT / "fetch_evals" / "schedulerlens_metrics.csv"
DEFAULT_OUT = ROOT / "outputs" / "atlas_plots"

SCHEDULERS = ["pd", "bd", "eb", "top_k_c", "top_k_m", "dus", "fdm"]
FIT_SCHEDULERS = ["pd", "bd", "eb", "top_k_c", "top_k_m", "dus"]  # no FDM
SCHEDULER_LABELS = {
    "pd": "PD",
    "bd": "BD",
    "eb": "EB",
    "top_k_c": "Top-k-C",
    "top_k_m": "Top-k-M",
    "dus": "DUS",
    "fdm": "FDM",
}
COLORS = {
    "pd": "#4C78A8",
    "bd": "#F28E2B",
    "eb": "#59A14F",
    "top_k_c": "#B279A2",
    "top_k_m": "#8F8F8F",
    "dus": "#E15759",
    "fdm": "#9C755F",
}
MARKERS = {
    "pd": "o",
    "bd": "s",
    "eb": "^",
    "top_k_c": "v",
    "top_k_m": "P",
    "dus": "X",
    "fdm": "D",
}
MODELS = ["llada", "llada1_5", "dream"]
MODEL_LABELS = {"llada": "LLaDA", "llada1_5": "LLaDA-1.5", "dream": "Dream"}
MODEL_COLORS = {"llada": "#6F8FBD", "llada1_5": "#75B184", "dream": "#C96E72"}
TASKS = ["arc_challenge_chat", "gsm8k", "humaneval", "mbpp", "minerva_math500"]
TASK_LABELS = {
    "arc_challenge_chat": "ARC-C",
    "gsm8k": "GSM8K",
    "humaneval": "HumanEval",
    "mbpp": "MBPP",
    "minerva_math500": "Math500",
}
# Row order in atlas heatmaps (task, steps).
ATLAS_ROWS = [
    ("arc_challenge_chat", 32),
    ("arc_challenge_chat", 64),
    ("arc_challenge_chat", 128),
    ("gsm8k", 128),
    ("gsm8k", 256),
    ("gsm8k", 512),
    ("humaneval", 128),
    ("humaneval", 256),
    ("humaneval", 512),
    ("mbpp", 128),
    ("mbpp", 256),
    ("mbpp", 512),
    ("minerva_math500", 128),
    ("minerva_math500", 256),
    ("minerva_math500", 512),
]
TRANSITIONS = [
    ("llada", "llada1_5", r"LLaDA $\rightarrow$ LLaDA-1.5"),
    ("llada1_5", "dream", r"LLaDA-1.5 $\rightarrow$ Dream"),
    ("llada", "dream", r"LLaDA $\rightarrow$ Dream"),
]
BUDGET_STEPS = (128, 256, 512)


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_pdf(fig: plt.Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("wrote %s", path)
    return path


def load_data(csv_path: Path) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    data = data[data.scheduler.isin(SCHEDULERS)].copy()
    expected = len(MODELS) * len(TASKS) * len(SCHEDULERS) * 3
    if len(data) < expected:
        raise ValueError(
            f"Expected ≥{expected} non-FHS rows in {csv_path}, found {len(data)}"
        )

    data["commit_fraction"] = 1.0 - data.final_masks_frac.astype(float)
    if (data.commit_fraction <= 0).any():
        raise ValueError("Non-positive committed fractions present")

    data["M_comp"] = data.M_hat / data.commit_fraction
    data["J_early_comp"] = data.J_hat_early / data.commit_fraction
    data["J_full_comp"] = data.J_hat / data.commit_fraction
    data["G_comp"] = (data.M_hat + data.J_hat) / data.commit_fraction
    return data


# ---------------------------------------------------------------------------
# Atlas: score heatmaps + G/f × NFE × score surfaces
# ---------------------------------------------------------------------------


def _score_matrix(model_data: pd.DataFrame) -> np.ndarray:
    mat = np.full((len(ATLAS_ROWS), len(SCHEDULERS)), np.nan, dtype=np.float64)
    for i, (task, steps) in enumerate(ATLAS_ROWS):
        for j, sch in enumerate(SCHEDULERS):
            hit = model_data[
                (model_data.task == task)
                & (model_data.steps == steps)
                & (model_data.scheduler == sch)
            ]
            if len(hit) == 1:
                mat[i, j] = float(hit.accuracy.iloc[0])
            elif len(hit) > 1:
                mat[i, j] = float(hit.accuracy.mean())
    return mat


def _row_labels() -> list[str]:
    return [f"{TASK_LABELS[t]}  {s}" for t, s in ATLAS_ROWS]


def fit_g_surface(task_data: pd.DataFrame):
    x = np.log10(np.maximum(task_data.G_comp.to_numpy(float), 1e-12))
    y = np.log2(np.maximum(task_data.avg_nfe_per_sample.to_numpy(float), 1e-6))
    z = task_data.accuracy.to_numpy(float)
    design = np.column_stack([np.ones_like(x), x, y])
    coef, _, _, _ = np.linalg.lstsq(design, z, rcond=None)
    prediction = design @ coef
    residual = float(np.sum((z - prediction) ** 2))
    total = float(np.sum((z - z.mean()) ** 2))
    r2 = 1.0 - residual / total if total > 0 else float("nan")
    return x, y, z, coef, r2


def make_atlas(data: pd.DataFrame, out_dir: Path) -> None:
    fig = plt.figure(figsize=(16.0, 9.6))
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.15, 1.0],
        hspace=0.28,
        left=0.06,
        right=0.94,
        top=0.96,
        bottom=0.08,
    )
    gs_heat = gs[0].subgridspec(1, 3, wspace=0.18)
    gs_surf = gs[1].subgridspec(1, 5, wspace=0.08)

    # --- heatmaps (raw score) ---
    cmap = mpl.colormaps["viridis"]
    ylabels = _row_labels()
    last_im = None
    for col, model in enumerate(MODELS):
        ax = fig.add_subplot(gs_heat[0, col])
        mat = _score_matrix(data[data.model == model])
        last_im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
        # annotate every cell; outline best scheduler per row
        for i in range(mat.shape[0]):
            row = mat[i]
            j_best = int(np.nanargmax(row)) if np.any(np.isfinite(row)) else -1
            for j in range(mat.shape[1]):
                v = row[j]
                if not np.isfinite(v):
                    ax.text(j, i, "—", ha="center", va="center", fontsize=5.5, color="#666")
                    continue
                # readable text on viridis
                color = "white" if v < 0.55 else "#111"
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color=color,
                    fontweight="bold" if j == j_best else "normal",
                )
            if j_best >= 0:
                ax.add_patch(
                    Rectangle(
                        (j_best - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="white",
                        linewidth=1.4,
                    )
                )
        ax.set_xticks(range(len(SCHEDULERS)))
        ax.set_xticklabels(
            [SCHEDULER_LABELS[s] for s in SCHEDULERS], rotation=45, ha="right", fontsize=7
        )
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels if col == 0 else [], fontsize=6.5)
        ax.set_title(MODEL_LABELS[model], pad=6)
        ax.tick_params(length=0)
        # light separators between task blocks
        for cut in (2.5, 5.5, 8.5, 11.5):
            ax.axhline(cut, color="white", linewidth=0.6, alpha=0.7)

    cax = fig.add_axes([0.955, 0.58, 0.012, 0.34])
    cb = fig.colorbar(last_im, cax=cax)
    cb.set_label("score", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # --- interaction surfaces: show FDM, but fit / ρ exclude FDM ---
    for idx, task in enumerate(TASKS):
        ax = fig.add_subplot(gs_surf[0, idx], projection="3d")
        task_data = data[data.task == task].copy()
        fit_data = task_data[task_data.scheduler.isin(FIT_SCHEDULERS)].copy()
        x, y, z, coef, r2 = fit_g_surface(fit_data)

        x_grid = np.linspace(np.quantile(x, 0.03), np.quantile(x, 0.97), 18)
        y_grid = np.linspace(np.quantile(y, 0.03), np.quantile(y, 0.97), 18)
        xx, yy = np.meshgrid(x_grid, y_grid)
        zz = np.clip(coef[0] + coef[1] * xx + coef[2] * yy, 0, 1)
        ax.plot_surface(
            xx,
            yy,
            zz,
            color="#7EA3B5",
            alpha=0.26,
            linewidth=0.35,
            edgecolor="#4B6570",
            antialiased=True,
            shade=False,
        )
        for scheduler in FIT_SCHEDULERS:
            subset = fit_data[fit_data.scheduler == scheduler]
            ax.scatter(
                np.log10(np.maximum(subset.G_comp, 1e-12)),
                np.log2(np.maximum(subset.avg_nfe_per_sample, 1e-6)),
                subset.accuracy,
                s=27,
                marker="o",
                color=COLORS[scheduler],
                edgecolor="white",
                linewidth=0.45,
                alpha=0.9,
                depthshade=False,
            )
        fdm = task_data[task_data.scheduler == "fdm"]
        if len(fdm):
            ax.scatter(
                np.log10(np.maximum(fdm.G_comp, 1e-12)),
                np.log2(np.maximum(fdm.avg_nfe_per_sample, 1e-6)),
                fdm.accuracy,
                s=34,
                marker="D",
                facecolor="none",
                edgecolor=COLORS["fdm"],
                linewidth=1.0,
                alpha=0.9,
                depthshade=False,
            )

        rho_g = spearmanr(fit_data.G_comp, fit_data.accuracy).statistic
        rho_nfe = spearmanr(fit_data.avg_nfe_per_sample, fit_data.accuracy).statistic
        ax.set_title(TASK_LABELS[task], fontsize=10, pad=6)
        ax.set_xlabel(r"$\log_{10}(\widehat{G}/f)$", labelpad=2)
        ax.set_ylabel(r"$\log_2(\mathrm{NFE})$", labelpad=2)
        ax.set_zlabel("score", labelpad=2)
        ax.tick_params(labelsize=6, pad=0)
        ax.view_init(elev=22, azim=-58)
        ax.set_box_aspect((1.05, 1.0, 0.85))
        rho_nfe_display = 0.0 if abs(rho_nfe) < 0.005 else rho_nfe
        ax.text2D(
            0.5,
            -0.20,
            rf"$\rho_{{\widehat{{G}}/f}}={rho_g:+.2f}$"
            rf"   $\rho_{{\mathrm{{NFE}}}}={rho_nfe_display:.2f}$"
            rf"   $R^2={r2:.2f}$",
            transform=ax.transAxes,
            ha="center",
            fontsize=6.8,
        )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=6,
            markerfacecolor=COLORS[s],
            markeredgecolor="white",
            label=SCHEDULER_LABELS[s],
        )
        for s in FIT_SCHEDULERS
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="",
            markersize=6,
            markerfacecolor="none",
            markeredgecolor=COLORS["fdm"],
            label="FDM",
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=7,
        frameon=False,
        columnspacing=1.1,
        handletextpad=0.35,
        fontsize=7.8,
    )
    _save_pdf(fig, out_dir / "atlas.pdf")


# ---------------------------------------------------------------------------
# Task scatters: M̂/f and Ĵ_early/f (single figure, log x)
# ---------------------------------------------------------------------------


def _draw_task_row(
    axes: Sequence[plt.Axes],
    data: pd.DataFrame,
    column: str,
    symbol: str,
    *,
    show_ylabel: bool,
    add_one: bool,
) -> None:
    """One row of per-task scatters; x = log10(proxy) or log10(1+proxy)."""
    x_plot_column = f"_{column}_plot"
    data = data.copy()
    raw = data[column].astype(float)
    data[x_plot_column] = (
        np.log10(1.0 + raw) if add_one else np.log10(np.maximum(raw, 1e-12))
    )
    xlabel = (
        rf"$\log_{{10}}(1+{symbol})$" if add_one else rf"$\log_{{10}}({symbol})$"
    )

    for index, (ax, task) in enumerate(zip(axes, TASKS)):
        task_data = data[data.task == task].copy()
        rho = spearmanr(task_data[column], task_data.accuracy).statistic

        x = task_data[x_plot_column].to_numpy(float)
        slope, intercept = np.polyfit(
            task_data[x_plot_column].to_numpy(float),
            task_data.accuracy.to_numpy(float),
            1,
        )
        pad = 0.05 * max(float(x.max() - x.min()), 1e-6)
        line = np.linspace(x.min() - pad, x.max() + pad, 100)
        ax.plot(
            line,
            intercept + slope * line,
            color="#526D7A",
            linewidth=1.25,
            alpha=0.9,
            zorder=1,
        )
        for scheduler in SCHEDULERS:
            subset = task_data[task_data.scheduler == scheduler]
            is_fdm = scheduler == "fdm"
            ax.scatter(
                subset[x_plot_column],
                subset.accuracy,
                s=38 if is_fdm else 28,
                marker="D" if is_fdm else "o",
                facecolors="none" if is_fdm else COLORS[scheduler],
                edgecolors=COLORS[scheduler] if is_fdm else "white",
                linewidths=0.55 if is_fdm else 0.45,
                alpha=0.9,
                zorder=3 if is_fdm else 2,
            )
        ax.set_title(TASK_LABELS[task], pad=5)
        ax.set_xlabel(xlabel)
        if show_ylabel and index == 0:
            ax.set_ylabel("score")
        ax.grid(True, color="#D7D7D7", linewidth=0.5, alpha=0.72)
        ax.set_axisbelow(True)
        ax.set_xlim(x.min() - pad, x.max() + pad)
        ax.text(
            0.97,
            0.96,
            rf"$\rho={rho:+.2f}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.0,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.76,
                "pad": 1.5,
            },
        )


def make_proxies_by_task(data: pd.DataFrame, out_dir: Path) -> None:
    """Two-row figure: M̂/f (log10) and Ĵ_early/f (log10(1+·))."""
    fig, axes = plt.subplots(2, 5, figsize=(15.6, 6.2), sharey="row")
    fig.subplots_adjust(left=0.055, right=0.992, top=0.92, bottom=0.14, wspace=0.18, hspace=0.38)

    _draw_task_row(
        axes[0],
        data,
        "M_comp",
        r"\widehat{M}^{\tau=1.25}/f",
        show_ylabel=True,
        add_one=False,
    )
    _draw_task_row(
        axes[1],
        data,
        "J_early_comp",
        r"\widehat{J}_{\mathrm{early}}/f",
        show_ylabel=True,
        add_one=True,
    )

    handles = [
        Line2D(
            [0],
            [0],
            marker="D" if s == "fdm" else "o",
            linestyle="",
            markersize=6,
            markerfacecolor="none" if s == "fdm" else COLORS[s],
            markeredgecolor=COLORS[s] if s == "fdm" else "white",
            label=SCHEDULER_LABELS[s],
        )
        for s in SCHEDULERS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=7,
        frameon=False,
        columnspacing=1.15,
        handletextpad=0.35,
        fontsize=7.8,
    )
    _save_pdf(fig, out_dir / "proxies_by_task.pdf")


# ---------------------------------------------------------------------------
# Transitions (Δ = new − old)
# ---------------------------------------------------------------------------


def matched_transition(data: pd.DataFrame, old: str, new: str) -> pd.DataFrame:
    keys = ["task", "scheduler", "steps"]
    left = data[data.model == old].set_index(keys)
    right = data[data.model == new].set_index(keys)
    pair = left.join(right, lsuffix="_old", rsuffix="_new", how="inner")
    pair["delta_M"] = pair.M_comp_new - pair.M_comp_old
    pair["delta_J_early"] = pair.J_early_comp_new - pair.J_early_comp_old
    pair["score_change_pp"] = 100 * (pair.accuracy_new - pair.accuracy_old)
    return pair.reset_index()


def make_transition_map(data: pd.DataFrame, out_dir: Path) -> None:
    use = data[data.scheduler.isin(SCHEDULERS)].copy()
    pairs = [matched_transition(use, old, new) for old, new, _ in TRANSITIONS]
    x_limit = 1.08 * max(abs(p.delta_M).max() for p in pairs)
    y_limit = 1.08 * max(abs(p.delta_J_early).max() for p in pairs)
    score_limit = max(abs(p.score_change_pp).max() for p in pairs)
    norm = mpl.colors.TwoSlopeNorm(vmin=-score_limit, vcenter=0, vmax=score_limit)
    cmap = mpl.colormaps["RdBu"]

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.0), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.91, top=0.88, bottom=0.23, wspace=0.12)

    for ax, pair, (_, _, title) in zip(axes, pairs, TRANSITIONS):
        # lower proxies → better: "both improve" = negative ΔM and ΔJ
        ax.add_patch(
            Rectangle(
                (-x_limit, -y_limit),
                x_limit,
                y_limit,
                facecolor="#EDF5ED",
                edgecolor="none",
                zorder=0,
            )
        )
        ax.add_patch(
            Rectangle(
                (0, 0),
                x_limit,
                y_limit,
                facecolor="#FAF0EF",
                edgecolor="none",
                zorder=0,
            )
        )
        ax.axhline(0, color="#777777", linewidth=0.75, zorder=1)
        ax.axvline(0, color="#777777", linewidth=0.75, zorder=1)
        for scheduler in SCHEDULERS:
            subset = pair[pair.scheduler == scheduler]
            ax.scatter(
                subset.delta_M,
                subset.delta_J_early,
                c=cmap(norm(subset.score_change_pp.to_numpy())),
                s=38 if scheduler == "fdm" else 33,
                marker=MARKERS[scheduler],
                edgecolor="white",
                linewidth=0.35,
                alpha=0.62,
                zorder=2,
            )
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_yscale("symlog", linthresh=2.0)
        ax.set_xlim(-x_limit, x_limit)
        ax.set_ylim(-y_limit, y_limit)
        ax.set_title(title, pad=6)
        ax.grid(True, color="#D8D8D8", linewidth=0.45, alpha=0.6)
        ax.text(
            0.02,
            0.03,
            "both improve",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.1,
            color="#486A4B",
        )
        ax.text(
            0.98,
            0.97,
            "both worsen",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.1,
            color="#7A5555",
        )
        ax.set_xlabel(r"$\Delta(\widehat{M}^{\tau=1.25}/f)$")
    axes[0].set_ylabel(r"$\Delta(\widehat{J}_{\mathrm{early}}/f)$")

    handles = [
        Line2D(
            [0],
            [0],
            marker=MARKERS[s],
            linestyle="",
            markersize=6.5,
            markerfacecolor="#A7A7A7",
            markeredgecolor="#222222",
            label=SCHEDULER_LABELS[s],
        )
        for s in SCHEDULERS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.455, 0.045),
        ncol=7,
        frameon=False,
        columnspacing=0.9,
        handletextpad=0.35,
        fontsize=7.4,
    )
    cax = fig.add_axes([0.925, 0.25, 0.012, 0.62])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label(r"$\Delta$ score (pp)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    _save_pdf(fig, out_dir / "model_transition.pdf")


# ---------------------------------------------------------------------------
# Model signatures
# ---------------------------------------------------------------------------


def draw_signature_panel(
    ax: plt.Axes, data: pd.DataFrame, column: str, title: str, symlog: bool = False
) -> None:
    values = [
        data.loc[data.model == model, column].dropna().to_numpy() for model in MODELS
    ]
    bp = ax.boxplot(
        values,
        positions=np.arange(1, 4),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#202020", "linewidth": 1.15},
        whiskerprops={"color": "#303030", "linewidth": 0.9},
        capprops={"color": "#303030", "linewidth": 0.9},
        boxprops={"edgecolor": "#444444", "linewidth": 0.9},
    )
    for patch, model in zip(bp["boxes"], MODELS):
        patch.set_facecolor(MODEL_COLORS[model])
        patch.set_alpha(0.78)
    rng = np.random.default_rng(7)
    for x, series, model in zip(range(1, 4), values, MODELS):
        ax.scatter(
            x + rng.normal(0, 0.035, len(series)),
            series,
            s=5,
            color=MODEL_COLORS[model],
            edgecolors="none",
            alpha=0.16,
            zorder=1,
        )
    if symlog:
        ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xticks([1, 2, 3], [MODEL_LABELS[m] for m in MODELS], rotation=12)
    ax.set_title(title, pad=6)
    ax.grid(axis="y", color="#D7D7D7", linewidth=0.5, alpha=0.75)
    ax.set_axisbelow(True)
    ax.margins(y=0.06)


def make_model_signatures(data: pd.DataFrame, out_dir: Path) -> None:
    use = data[data.scheduler.isin(SCHEDULERS)].copy()
    use["M_comp_log"] = np.log10(np.maximum(use.M_comp.astype(float), 1e-12))
    use["J_early_comp_log"] = np.log10(1.0 + use.J_early_comp.astype(float))
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.25))
    fig.subplots_adjust(left=0.055, right=0.992, top=0.88, bottom=0.22, wspace=0.24)
    panels = [
        (axes[0], "conf_mean", r"confidence ($c$)", "(a)"),
        (axes[1], "ent_mean", r"entropy ($H$)", "(b)"),
        (axes[2], "M_comp_log", r"$\log_{10}(\widehat{M}^{\tau=1.25}/f)$", "(c)"),
        (
            axes[3],
            "J_early_comp_log",
            r"$\log_{10}(1+\widehat{J}_{\mathrm{early}}/f)$",
            "(d)",
        ),
    ]
    for ax, col, title, tag in panels:
        draw_signature_panel(ax, use, col, title, symlog=False)
        ax.text(
            0.5,
            -0.28,
            tag,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
        )
    _save_pdf(fig, out_dir / "model_signatures.pdf")


# ---------------------------------------------------------------------------
# Execution budget (NFE / commits / f_commit / EOS vs nominal steps)
# ---------------------------------------------------------------------------


def make_execution_budget(data: pd.DataFrame, out_dir: Path) -> None:
    # Average over models + non-ARC tasks on the 128/256/512 grid (ARC only has ≤128).
    sub = data[
        data.steps.isin(BUDGET_STEPS) & (data.task != "arc_challenge_chat")
    ].copy()
    sub["commit_fraction"] = sub.commit_fraction.astype(float)
    agg = (
        sub.groupby(["scheduler", "steps"], as_index=False)
        .agg(
            nfe=("avg_nfe_per_sample", "mean"),
            commits=("mean_commits_per_step", "mean"),
            f_commit=("commit_fraction", "mean"),
            eos=("eos_in_output_rate", "mean"),
        )
    )

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.94, bottom=0.14, wspace=0.28, hspace=0.32)
    panels = [
        (axes[0, 0], "nfe", "Measured NFE", "NFE per sample", True),
        (axes[0, 1], "commits", "Commits per step", "Tokens", False),
        (axes[1, 0], "f_commit", "Committed fraction", "Fraction", False),
        (axes[1, 1], "eos", "EOS rate", "Fraction", False),
    ]
    xs = list(BUDGET_STEPS)
    for ax, col, title, ylabel, logy in panels:
        for sch in SCHEDULERS:
            rows = agg[agg.scheduler == sch].set_index("steps").reindex(xs)
            ys = rows[col].to_numpy(float)
            style_kw = {
                "color": COLORS[sch],
                "marker": MARKERS[sch] if sch != "fdm" else "D",
                "markersize": 5.5,
                "linewidth": 1.35,
                "linestyle": "--" if sch == "fdm" else "-",
                "markerfacecolor": "none" if sch == "fdm" else COLORS[sch],
                "markeredgecolor": COLORS[sch],
                "markeredgewidth": 1.0 if sch == "fdm" else 0.6,
            }
            ax.plot(xs, ys, **style_kw)
        ax.set_title(title, pad=6)
        ax.set_xlabel("Steps")
        ax.set_ylabel(ylabel)
        ax.set_xticks(xs)
        if logy:
            ax.set_yscale("log")
        ax.grid(True, color="#D7D7D7", linewidth=0.5, alpha=0.75)
        ax.set_axisbelow(True)

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[s],
            marker=MARKERS[s] if s != "fdm" else "D",
            linestyle="--" if s == "fdm" else "-",
            markersize=6,
            markerfacecolor="none" if s == "fdm" else COLORS[s],
            markeredgecolor=COLORS[s],
            label=SCHEDULER_LABELS[s],
        )
        for s in SCHEDULERS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=7,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.35,
        fontsize=7.6,
    )
    _save_pdf(fig, out_dir / "execution_budget.pdf")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUT)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_argparser().parse_args(argv)
    style()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(Path(args.csv))
    logger.info("loaded %d non-FHS rows from %s", len(data), args.csv)

    make_atlas(data, out_dir)
    make_proxies_by_task(data, out_dir)
    make_transition_map(data, out_dir)
    make_model_signatures(data, out_dir)
    make_execution_budget(data, out_dir)

    logger.info("wrote figures under %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
