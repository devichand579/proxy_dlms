#!/usr/bin/env python3
"""PCFG proxy↔oracle correlation figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from repo_env_defaults import apply_repo_env_defaults  # noqa: E402
from schedulerlens.hub import (  # noqa: E402
    download_file_to_temp,
    remove_temp_download,
)
from schedulerlens.metrics import (  # noqa: E402
    compute_task_metrics,
    parse_run_stem,
    pearson_corr,
)
from schedulerlens.oracle import (  # noqa: E402
    index_oracle_artifacts,
    load_oracle_artifacts,
    resolve_oracle_npz,
)
from schedulerlens.proxies import GATED_TAU  # noqa: E402

logger = logging.getLogger("pcfg_corr_plots")

# Paper order (no FHS).
SCHED_ORDER = ("pd", "bd", "eb", "dus", "fdm", "top_k_c", "top_k_m")
BLOCK_ORDER = ("full", "early", "mid", "late")
DEFAULT_TAUS = [round(x, 2) for x in np.arange(2.0, 0.24, -0.25)]


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


def rows_for_heatmap(
    rows: Sequence[Mapping[str, Any]],
    *,
    exclude: Sequence[str] = ("fhs",),
) -> List[Dict[str, Any]]:
    """One row per scheduler; prefer stems with finite corr_M_full."""
    exclude_set = {s.lower() for s in exclude}
    by_sch: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        sch = str(r.get("scheduler") or "").lower()
        if not sch or sch in exclude_set:
            continue
        if not _finite(r.get("corr_M_full")):
            continue
        # Prefer larger step budgets when duplicates exist.
        prev = by_sch.get(sch)
        steps = int(float(r.get("steps") or 0) or 0)
        if prev is None or steps >= int(float(prev.get("steps") or 0) or 0):
            by_sch[sch] = dict(r)
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
    mat = np.full((len(rows), len(BLOCK_ORDER)), np.nan, dtype=np.float64)
    for i, r in enumerate(rows):
        for j, blk in enumerate(BLOCK_ORDER):
            key = f"{prefix}_{blk}"
            if _finite(r.get(key)):
                mat[i, j] = float(r[key])
    return mat, labels


def plot_corr_heatmap(
    rows: Sequence[Mapping[str, Any]],
    out_stem: Path,
) -> Path:
    """Two-panel RdBu heatmap: ρ(ˆM^{τ=1.25}, M) and ρ(ˆJ, J) by block."""
    use = rows_for_heatmap(rows)
    if not use:
        raise SystemExit("No rows with finite corr_M_full to plot")

    m_mat, ylabels = _corr_matrix(use, "corr_M")
    j_mat, _ = _corr_matrix(use, "corr_J")

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), constrained_layout=True)
    panels = [
        (axes[0], m_mat, rf"$\hat{{M}}^{{\tau={GATED_TAU:g}}}$"),
        (axes[1], j_mat, r"$\hat{J}$"),
    ]
    im = None
    for ax, mat, panel_title in panels:
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(BLOCK_ORDER)))
        ax.set_xticklabels(list(BLOCK_ORDER), fontsize=9)
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


def _pooled_block_rho(
    pairs: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, float]:
    """Concatenate scheduler step curves then Pearson per trajectory block."""
    from schedulerlens.metrics import trajectory_blocks

    if not pairs:
        return {b: float("nan") for b in BLOCK_ORDER}
    mt_parts: List[np.ndarray] = []
    mh_parts: List[np.ndarray] = []
    for mt, mh in pairs:
        n = min(len(mt), len(mh))
        if n <= 0:
            continue
        mt_parts.append(np.asarray(mt[:n], dtype=np.float64))
        mh_parts.append(np.asarray(mh[:n], dtype=np.float64))
    if not mt_parts:
        return {b: float("nan") for b in BLOCK_ORDER}
    out: Dict[str, float] = {}
    for blk in BLOCK_ORDER:
        xs: List[float] = []
        ys: List[float] = []
        for mt, mh in zip(mt_parts, mh_parts):
            n = len(mt)
            blocks = trajectory_blocks(n)
            lo, hi = blocks[blk]
            xs.extend(mt[lo:hi].tolist())
            ys.extend(mh[lo:hi].tolist())
        out[blk] = pearson_corr(xs, ys)
    return out


def _cap_rho(
    mt_totals: Sequence[float],
    m_hats: Sequence[float],
) -> float:
    return pearson_corr(mt_totals, m_hats)


def sweep_tau(
    *,
    oracle_dir: Path,
    repo_id: Optional[str],
    namespace: Optional[str],
    local_dir: Optional[Path],
    taus: Sequence[float],
    max_samples: Optional[int],
    exclude: Sequence[str] = ("fhs",),
    revision: Optional[str] = None,
) -> Dict[str, Any]:
    """Recompute M̂(τ) trends vs oracle Mt; return pooled ρ table + best τ."""
    exclude_set = {s.lower() for s in exclude}
    oracle_index = index_oracle_artifacts(oracle_dir)
    if not oracle_index:
        raise SystemExit(f"No oracle.npz under {oracle_dir}")

    # Discover stems from oracle index (unique paths).
    stems: List[str] = []
    seen_paths = set()
    for stem, path in sorted(oracle_index.items()):
        key = str(path.resolve())
        if key in seen_paths:
            continue
        meta = parse_run_stem(stem)
        if meta.scheduler and meta.scheduler.lower() in exclude_set:
            logger.info("skip %s (excluded scheduler)", stem)
            continue
        seen_paths.add(key)
        stems.append(stem)

    if local_dir is not None:
        local_dir = Path(local_dir)
        local_map = {p.stem: p for p in local_dir.glob("*.npz")}
    else:
        local_map = {}
        if not repo_id:
            raise SystemExit("Provide --repo or --local-dir for traces")

    # Per-stem: load mt_trend + Mt_total once; for each tau compute m_hat trend.
    per_stem: Dict[str, Dict[str, Any]] = {}
    for stem in stems:
        oracle_path = resolve_oracle_npz(None, stem, index=oracle_index)
        assert oracle_path is not None
        arts = load_oracle_artifacts(oracle_path)
        mt_trend = np.asarray(arts.get("mt_trend", []), dtype=np.float64).reshape(-1)
        mt_total = float(np.asarray(arts.get("Mt_total", np.nan)).reshape(-1)[0])
        if mt_trend.size == 0:
            logger.warning("no mt_trend in %s; skip", oracle_path)
            continue

        temp_dir = None
        try:
            if stem in local_map:
                trace_path = local_map[stem]
            elif local_dir is not None:
                logger.warning("no local NPZ for %s; skip", stem)
                continue
            else:
                hub_name = f"{stem}.npz"
                logger.info("download %s", hub_name)
                trace_path, temp_dir = download_file_to_temp(
                    hub_name,
                    repo_id=repo_id,
                    namespace=namespace,
                    revision=revision,
                )

            trends_by_tau: Dict[str, np.ndarray] = {}
            caps_by_tau: Dict[str, float] = {}
            for tau in taus:
                logger.info("%s  τ=%.2f", stem, tau)
                metrics = compute_task_metrics(
                    Path(trace_path),
                    max_samples=max_samples,
                    verbose=False,
                    tau=float(tau),
                )
                mh = np.asarray(metrics.proxies["M_hat_t"].trend, dtype=np.float64)
                trends_by_tau[f"{tau:.2f}"] = mh
                caps_by_tau[f"{tau:.2f}"] = float(metrics.M_hat)

            # Ungated reference: τ → +inf (all commits).
            logger.info("%s  τ=inf (ungated)", stem)
            metrics_inf = compute_task_metrics(
                Path(trace_path),
                max_samples=max_samples,
                verbose=False,
                tau=1e9,
            )
            trends_by_tau["inf"] = np.asarray(
                metrics_inf.proxies["M_hat_t"].trend, dtype=np.float64
            )
            caps_by_tau["inf"] = float(metrics_inf.M_hat)

            sch = (parse_run_stem(stem).scheduler or "").lower()
            per_stem[stem] = {
                "scheduler": sch,
                "mt_trend": mt_trend.tolist(),
                "mt_total": mt_total,
                "m_hat_trends": {k: v.tolist() for k, v in trends_by_tau.items()},
                "m_hat_caps": caps_by_tau,
            }
        finally:
            if temp_dir is not None:
                remove_temp_download(temp_dir)

    if not per_stem:
        raise SystemExit("No stems completed for τ sweep")

    tau_keys = ["inf"] + [f"{t:.2f}" for t in taus]
    table: Dict[str, Dict[str, float]] = {}
    for key in tau_keys:
        pairs = []
        caps_m = []
        caps_o = []
        for stem, info in per_stem.items():
            mt = np.asarray(info["mt_trend"], dtype=np.float64)
            mh = np.asarray(info["m_hat_trends"][key], dtype=np.float64)
            pairs.append((mt, mh))
            caps_m.append(float(info["m_hat_caps"][key]))
            caps_o.append(float(info["mt_total"]))
        pooled = _pooled_block_rho(pairs)
        pooled["cap"] = _cap_rho(caps_o, caps_m)
        table[key] = pooled

    # Best τ: maximize pooled full ρ among finite taus (exclude inf for pick).
    best_tau = None
    best_full = float("-inf")
    for t in taus:
        key = f"{t:.2f}"
        full = float(table[key]["full"])
        if np.isfinite(full) and full > best_full:
            best_full = full
            best_tau = float(t)

    return {
        "taus": [float(t) for t in taus],
        "exclude": list(exclude),
        "n_schedulers": len(per_stem),
        "stems": sorted(per_stem.keys()),
        "table": table,
        "best_tau": best_tau,
        "best_full": best_full if np.isfinite(best_full) else None,
        "default_tau": float(GATED_TAU),
        "per_stem": per_stem,
    }


def plot_tau_sweep(result: Mapping[str, Any], out_stem: Path) -> Path:
    table = result["table"]
    taus = list(result["taus"])
    keys = [f"{t:.2f}" for t in taus]
    best = result.get("best_tau")

    fig, ax = plt.subplots(figsize=(5.4, 3.4), constrained_layout=True)
    for blk, style in zip(
        BLOCK_ORDER,
        [("-", 2.0), ("--", 1.4), ("-.", 1.4), (":", 1.6)],
    ):
        ys = [table[k][blk] for k in keys]
        ax.plot(taus, ys, style[0], lw=style[1], label=blk, marker="o", ms=4)
    if best is not None:
        ax.axvline(best, color="#111", lw=1.0, ls="--", alpha=0.7, label=rf"$\tau={best:g}$")
    ax.axhline(0.0, color="#ccc", lw=0.7)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\rho(\hat{M}_t, M_t)$")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.set_xlim(max(taus) + 0.05, min(taus) - 0.05)

    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf = out_stem.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", pdf)
    return pdf


def _print_tau_table(result: Mapping[str, Any]) -> None:
    table = result["table"]
    print(
        f"{'τ':>6s}  {'full':>7s}  {'early':>7s}  {'mid':>7s}  {'late':>7s}  {'cap':>7s}"
    )
    for key in ["inf"] + [f"{t:.2f}" for t in result["taus"]]:
        row = table[key]
        label = "none" if key == "inf" else key
        print(
            f"{label:>6s}  "
            f"{row['full']:7.3f}  {row['early']:7.3f}  "
            f"{row['mid']:7.3f}  {row['late']:7.3f}  {row['cap']:7.3f}"
        )
    print(
        f"\nbest τ (max full step-curve ρ, no FHS): "
        f"{result.get('best_tau')}  "
        f"(full={result.get('best_full'):.3f})"
        if result.get("best_tau") is not None
        else "\nbest τ: n/a"
    )
    print(f"current GATED_TAU default: {result.get('default_tau')}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def add_out(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "-o",
            "--output-dir",
            type=Path,
            default=ROOT / "outputs" / "pcfg_plots",
        )

    p_h = sub.add_parser("heatmap", help="Plot corr heatmap from PCFG CSV")
    p_h.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "fetch_evals" / "schedulerlens_metrics_pcfg.csv",
    )
    p_h.add_argument("--exclude", default="fhs", help="Comma-separated schedulers to drop")
    add_out(p_h)

    p_t = sub.add_parser("tau-sweep", help="Sweep hard-gate τ vs oracle Mt")
    p_t.add_argument("--oracle", type=Path, required=True, help="oracle.npz file or folder")
    src = p_t.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo", dest="repo_id", default=None)
    src.add_argument("--local-dir", type=Path, default=None)
    p_t.add_argument("--namespace", default=None)
    p_t.add_argument("--revision", default=None)
    p_t.add_argument("--max-samples", type=int, default=None)
    p_t.add_argument("--exclude", default="fhs")
    p_t.add_argument(
        "--taus",
        default=",".join(str(t) for t in DEFAULT_TAUS),
        help="Comma-separated τ grid (default: 2.0 … 0.25 step 0.25)",
    )
    add_out(p_t)

    p_a = sub.add_parser("all", help="Heatmap + τ sweep")
    p_a.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "fetch_evals" / "schedulerlens_metrics_pcfg.csv",
    )
    p_a.add_argument("--oracle", type=Path, required=True)
    src2 = p_a.add_mutually_exclusive_group(required=True)
    src2.add_argument("--repo", dest="repo_id", default=None)
    src2.add_argument("--local-dir", type=Path, default=None)
    p_a.add_argument("--namespace", default=None)
    p_a.add_argument("--revision", default=None)
    p_a.add_argument("--max-samples", type=int, default=None)
    p_a.add_argument("--exclude", default="fhs")
    p_a.add_argument("--taus", default=",".join(str(t) for t in DEFAULT_TAUS))
    add_out(p_a)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    apply_repo_env_defaults()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_argparser().parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exclude = tuple(s.strip() for s in str(args.exclude).split(",") if s.strip())

    if args.command in ("heatmap", "all"):
        rows = load_csv_rows(Path(args.csv))
        filtered = rows_for_heatmap(rows, exclude=exclude)
        plot_corr_heatmap(filtered, out_dir / "pcfg_corr_heatmap")

    if args.command in ("tau-sweep", "all"):
        taus = [float(x) for x in str(args.taus).split(",") if x.strip()]
        result = sweep_tau(
            oracle_dir=Path(args.oracle),
            repo_id=getattr(args, "repo_id", None),
            namespace=getattr(args, "namespace", None),
            local_dir=getattr(args, "local_dir", None),
            taus=taus,
            max_samples=getattr(args, "max_samples", None),
            exclude=exclude,
            revision=getattr(args, "revision", None),
        )
        # Drop bulky per-stem arrays from the printed JSON summary file.
        slim = {k: v for k, v in result.items() if k != "per_stem"}
        slim["stems"] = result.get("stems")
        json_path = out_dir / "pcfg_tau_sweep.json"
        json_path.write_text(json.dumps(slim, indent=2))
        # Keep full dump separately.
        (out_dir / "pcfg_tau_sweep_full.json").write_text(json.dumps(result, indent=2))
        plot_tau_sweep(result, out_dir / "pcfg_tau_sweep")
        _print_tau_table(result)
        logger.info("wrote %s", json_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
