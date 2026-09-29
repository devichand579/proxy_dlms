#!/usr/bin/env python3
"""PCFG proxy↔oracle correlation figures.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from repo_env_defaults import apply_repo_env_defaults  # noqa: E402

logger = logging.getLogger("pcfg_corr_plots")

# Paper order (no FHS).
SCHED_ORDER = ("pd", "bd", "eb", "dus", "fdm", "top_k_c", "top_k_m")
PREFIX_ORDER = ("8", "16", "32", "64", "full")


def lab(scheduler: str) -> str:
    return {
        "top_k_c": "top-k-c",
        "top_k_m": "top-k-m",
    }.get(scheduler, scheduler)


def _finite(x: Any) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(v))


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _prefix_label(row: Mapping[str, Any]) -> str:
    raw = str(row.get("prefix") or row.get("prefix_steps") or "").strip()
    return raw if raw else "full"


def _official_tau(row: Mapping[str, Any]) -> bool:
    tau = row.get("tau")
    if tau is None or str(tau).strip() == "":
        return True
    try:
        return abs(float(tau) - 1.0) < 1e-9
    except (TypeError, ValueError):
        return False


def rows_for_heatmap(
    rows: Sequence[Mapping[str, Any]],
    *,
    exclude: Sequence[str] = ("fhs",),
) -> List[Dict[str, Any]]:
    """One row per scheduler at γ = 1, with pearson_M / pearson_J / pearson_G by prefix."""
    exclude_set = {s.lower() for s in exclude}
    chosen: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in rows:
        sch = str(r.get("scheduler") or "").lower()
        if not sch or sch in exclude_set or not _official_tau(r):
            continue
        label = _prefix_label(r)
        if label not in PREFIX_ORDER:
            continue
        if not (
            _finite(r.get("pearson_M"))
            or _finite(r.get("pearson_J"))
            or _finite(r.get("pearson_G"))
        ):
            continue
        key = (sch, label)
        prev = chosen.get(key)
        steps = int(float(r.get("steps") or 0) or 0)
        if prev is None or steps >= int(float(prev.get("steps") or 0) or 0):
            chosen[key] = dict(r)
    by_sch: Dict[str, Dict[str, Any]] = {}
    for (sch, label), r in chosen.items():
        slot = by_sch.setdefault(sch, {"scheduler": sch})
        slot[f"pearson_M_{label}"] = r.get("pearson_M")
        slot[f"pearson_J_{label}"] = r.get("pearson_J")
        slot[f"pearson_G_{label}"] = r.get("pearson_G")
        prev_steps = int(float(slot.get("steps") or 0) or 0)
        steps = int(float(r.get("steps") or 0) or 0)
        if steps >= prev_steps:
            slot["steps"] = steps
    ordered = []
    for sch in SCHED_ORDER:
        if sch in by_sch:
            ordered.append(by_sch[sch])
    for sch, r in by_sch.items():
        if sch not in SCHED_ORDER:
            ordered.append(r)
    return ordered


def _corr_matrix(
    rows: Sequence[Mapping[str, Any]],
    prefix: str,
) -> Tuple[np.ndarray, List[str]]:
    labels = [lab(str(r["scheduler"])) for r in rows]
    mat = np.full((len(rows), len(PREFIX_ORDER)), np.nan, dtype=np.float64)
    for i, r in enumerate(rows):
        for j, blk in enumerate(PREFIX_ORDER):
            key = f"{prefix}_{blk}"
            if _finite(r.get(key)):
                mat[i, j] = float(r[key])
    return mat, labels


def plot_corr_heatmap(
    rows: Sequence[Mapping[str, Any]],
    out_stem: Path,
) -> Path:
    """Three-panel RdBu heatmap: sample Pearson of M̂, Ĵ, and Ĝ with Mt, Jt, and G by prefix."""
    use = rows_for_heatmap(rows)
    if not use:
        raise SystemExit("No rows with finite pearson_M, pearson_J, or pearson_G to plot")

    m_mat, ylabels = _corr_matrix(use, "pearson_M")
    j_mat, _ = _corr_matrix(use, "pearson_J")
    g_mat, _ = _corr_matrix(use, "pearson_G")

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.4), constrained_layout=True)
    panels = [
        (axes[0], m_mat, r"$\hat{M}$"),
        (axes[1], j_mat, r"$\hat{J}$"),
        (axes[2], g_mat, r"$\hat{G}$"),
    ]
    im = None
    for ax, mat, panel_title in panels:
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(PREFIX_ORDER)))
        ax.set_xticklabels(list(PREFIX_ORDER), fontsize=9)
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels, fontsize=9)
        ax.set_title(panel_title, fontsize=12, pad=6)
        ax.tick_params(length=0)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if not np.isfinite(v):
                    txt = "—"
                    color = "#666"
                else:
                    txt = f"{v:.2f}"
                    color = "white" if abs(v) >= 0.55 else "#111"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

    fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04, label=r"Pearson $\rho$")

    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf = out_stem.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", pdf)
    return pdf



def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    p_h = sub.add_parser("heatmap", help="Plot corr heatmap from PCFG CSV")
    p_h.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "fetch_evals" / "schedulerlens_metrics_pcfg.csv",
    )
    p_h.add_argument("--exclude", default="fhs", help="Comma-separated schedulers to drop")
    p_h.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "pcfg_plots",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    apply_repo_env_defaults()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_argparser().parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exclude = tuple(s.strip() for s in str(args.exclude).split(",") if s.strip())
    rows = load_csv_rows(Path(args.csv))
    filtered = rows_for_heatmap(rows, exclude=exclude)
    plot_corr_heatmap(filtered, out_dir / "pcfg_corr_heatmap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
