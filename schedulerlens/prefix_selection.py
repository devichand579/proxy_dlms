"""Prefix selection: the scheduler with the lowest Ĝ/f after a short prefix.

One model, one task, one step budget: the schedulers that actually ran at that
budget are scored by mean(Ĝ) / mean(commit fraction). The lowest one is the
scheduler that would be finished. Accuracy is the finished-run score already
stored on the row, and the best score is the best among those schedulers.

``compute_saved`` is how many times less NFE that costs. The pick runs to its
full NFE. Every other scheduler stops after ``prefix`` steps. The column is
(sum of full NFEs) / (NFE actually spent), so 5 means a 5× saving.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("schedulerlens.prefix_selection")

DEFAULT_RUNS = Path(__file__).resolve().parents[1] / "outputs/schedulerlens/prefix_sweep/prefix_sweep.csv"
SCHEDULERS = ("pd", "bd", "eb", "top_k_c", "top_k_m", "dus")
TASKS = ("gsm8k", "humaneval", "mbpp", "minerva_math500")
BUDGETS = (128, 256, 512)
MODELS = ("dream", "llada", "llada1_5")


def _compute_saved(nfes: list[float], pick_index: int, prefix: int) -> float:
    """Full-schedule NFE divided by the NFE spent when only the pick is finished."""
    full = 0.0
    spent = 0.0
    for index, nfe in enumerate(nfes):
        if nfe != nfe:
            return float("nan")
        full += nfe
        if index == pick_index:
            spent += nfe
        else:
            spent += min(nfe, float(prefix))
    if spent == 0.0:
        return float("nan")
    return full / spent


def _excluded(spec) -> set[str]:
    if not spec:
        return set()
    if isinstance(spec, str):
        parts = spec.split(",")
    else:
        parts = spec
    return {str(part).strip() for part in parts if str(part).strip()}


def selection_table(
    frame: pd.DataFrame,
    *,
    tau: str = "1.0",
    prefixes: tuple[str, ...] = ("8", "16", "32", "64"),
    exclude_task=None,
    exclude_scheduler=None,
) -> pd.DataFrame:
    """One row per model, task, step budget, and prefix.

    Every scheduler with a run at that budget is a choice. The pick is the
    lowest Ĝ/f among those choices.
    """
    tasks = tuple(name for name in TASKS if name not in _excluded(exclude_task))
    schedulers = tuple(name for name in SCHEDULERS if name not in _excluded(exclude_scheduler))
    sub = frame[
        (frame.tau.astype(str) == str(tau))
        & frame.scheduler.isin(schedulers)
        & frame.task.isin(tasks)
        & frame.steps.isin(BUDGETS)
        & frame.prefix.astype(str).isin(prefixes)
    ].copy()
    sub["ratio"] = sub["G_hat"].astype(float) / sub["commit_frac"].astype(float)
    rows = []
    for (task, steps, model), group in sub.groupby(["task", "steps", "model"]):
        acc = group[group.prefix.astype(str) == "full"]
        if acc.empty:
            acc = group.drop_duplicates("scheduler").set_index("scheduler")["accuracy"]
        else:
            acc = acc.set_index("scheduler")["accuracy"]
        for prefix in prefixes:
            chunk = group[group.prefix.astype(str) == prefix].set_index("scheduler")
            present = [name for name in schedulers if name in chunk.index and name in acc.index]
            if len(present) < 2:
                continue
            chunk = chunk.loc[present]
            scores = acc.loc[present]
            pick = chunk["ratio"].idxmin()
            best = scores.idxmax()
            if "avg_nfe_per_sample" in chunk.columns:
                nfe = [float(chunk.loc[name, "avg_nfe_per_sample"]) for name in present]
            else:
                nfe = [float("nan")] * len(present)
            pick_index = present.index(pick)
            pick_nfe = nfe[pick_index]
            rows.append(
                {
                    "task": task,
                    "steps": int(steps),
                    "model": model,
                    "prefix": prefix,
                    "schedulers": ",".join(present),
                    "nfe": ",".join(f"{value:g}" for value in nfe),
                    "n_schedulers": len(present),
                    "pick": pick,
                    "pick_nfe": pick_nfe,
                    "best": best,
                    "pick_acc": float(scores[pick]),
                    "mean_acc": float(scores.mean()),
                    "best_acc": float(scores.max()),
                    "lift": float(scores[pick] - scores.mean()),
                    "gap": float(scores.max() - scores[pick]),
                    "compute_saved": _compute_saved(nfe, pick_index, int(prefix)),
                }
            )
    return pd.DataFrame(rows)


def main(
    runs_path: Path | None = None,
    tau: str = "1.0",
    exclude_task=None,
    exclude_scheduler=None,
    quiet: bool = False,
    output: Path | None = None,
) -> int:
    logging.basicConfig(
        level=logging.WARNING if quiet else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    path = Path(runs_path) if runs_path is not None else DEFAULT_RUNS
    if not path.is_file():
        logger.error("missing runs csv: %s", path)
        return 1
    table = selection_table(
        pd.read_csv(path),
        tau=tau,
        exclude_task=exclude_task,
        exclude_scheduler=exclude_scheduler,
    )
    if table.empty:
        logger.error("no rows for tau %s in %s", tau, path)
        return 1
    out = Path(output) if output is not None else path.parent / f"prefix_selection_tau_{tau}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    for prefix, group in table.groupby("prefix"):
        logger.info(
            "prefix %s  pick %.3f  average %.3f  best %.3f  lift %+.3f  exact best %.0f%%  compute saved %.1fx",
            prefix,
            group.pick_acc.mean(),
            group.mean_acc.mean(),
            group.best_acc.mean(),
            group.lift.mean(),
            100.0 * (group.pick == group.best).mean(),
            group.compute_saved.mean(),
        )
    logger.info("wrote %s (%d cells)", out, len(table))
    return 0
