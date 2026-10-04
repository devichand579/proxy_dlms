"""Prefix selection, one scheduler per sample.

For one model, task, and step budget, every sample is scored on its own.
At a prefix, each scheduler has that sample's Ĝ / commit fraction. The lowest
ratio is the scheduler finished for that sample. Its finished-run score is the
rule's score. The average score is the mean of the schedulers' accuracies.
The best score is the accuracy of the scheduler with the highest accuracy.

A sample counts as right when the chosen scheduler is that highest-accuracy
scheduler. Schedulers whose accuracies match are all counted as best.

The scheduler chosen on the most samples is the majority pick. ``majority_acc``
is that scheduler's accuracy on every sample. Ties keep the earliest scheduler
in the scheduler list.

``compute_saved`` is full-schedule NFE divided by the NFE this rule spends.
Every scheduler runs for ``prefix`` steps. Only the sample's pick is finished.
"""

from __future__ import annotations

import csv
import logging
import math
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from repo_env_defaults import apply_repo_env_defaults
from schedulerlens.prefix_sweep import TAU_NAMES, sample_rows
from traces.hf_trace_upload import download_file_to_temp, remove_temp_download
from traces.trace_metrics import TraceScorer, _is_code_task
from traces.trace_npz import iter_npz_traces

logger = logging.getLogger("schedulerlens.prefix_selection")

DEFAULT_RUNS = Path(__file__).resolve().parents[1] / "outputs/schedulerlens/prefix_sweep/prefix_sweep.csv"
SCHEDULERS = ("pd", "bd", "eb", "top_k_c", "top_k_m", "dus")
TASKS = ("gsm8k", "humaneval", "mbpp", "minerva_math500")
BUDGETS = (128, 256, 512)
SAMPLE_FIELDS = (
    "filename",
    "model",
    "scheduler",
    "task",
    "steps",
    "avg_nfe_per_sample",
    "doc_id",
    "prefix",
    "G_hat",
    "commit_frac",
    "score",
)
CODE_WORKERS = 8


def _spent(nfes: list[float], pick_index: int, prefix: int) -> float:
    """NFE to run every scheduler to ``prefix`` and finish ``pick_index``."""
    spent = 0.0
    for index, nfe in enumerate(nfes):
        if nfe != nfe:
            return float("nan")
        if index == pick_index:
            spent += nfe
        else:
            spent += min(nfe, float(prefix))
    return spent


def _compute_saved(nfes: list[float], pick_indices: list[int], prefix: int) -> float:
    """Full-schedule NFE divided by the NFE spent under per-sample picks."""
    full = 0.0
    for nfe in nfes:
        if nfe != nfe:
            return float("nan")
        full += nfe
    spent = 0.0
    for pick_index in pick_indices:
        one = _spent(nfes, pick_index, prefix)
        if one != one:
            return float("nan")
        spent += one
    if spent == 0.0:
        return float("nan")
    return (len(pick_indices) * full) / spent


def _excluded(spec) -> set[str]:
    if not spec:
        return set()
    if isinstance(spec, str):
        parts = spec.split(",")
    else:
        parts = spec
    return {str(part).strip() for part in parts if str(part).strip()}


def _tau_name(tau: str) -> str:
    if tau in TAU_NAMES:
        return tau
    value = float(tau)
    for name in TAU_NAMES:
        if float(name) == value:
            return name
    raise ValueError(f"tau {tau} is not one of {', '.join(TAU_NAMES)}")


def _same_score(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)


def selection_table(
    frame: pd.DataFrame,
    *,
    prefixes: tuple[str, ...] = ("8", "16", "32", "64"),
    exclude_task=None,
    exclude_scheduler=None,
) -> pd.DataFrame:
    """One row per model, task, step budget, and prefix, aggregated over samples."""
    tasks = tuple(name for name in TASKS if name not in _excluded(exclude_task))
    schedulers = tuple(name for name in SCHEDULERS if name not in _excluded(exclude_scheduler))
    sub = frame[
        frame.scheduler.isin(schedulers)
        & frame.task.isin(tasks)
        & frame.steps.isin(BUDGETS)
        & frame.prefix.astype(str).isin(prefixes)
    ].copy()
    sub["doc_id"] = sub["doc_id"].astype(str)
    sub = sub.drop_duplicates(
        ["task", "steps", "model", "scheduler", "doc_id", "prefix"],
        keep="last",
    )
    rows = []
    for (task, steps, model), group in sub.groupby(["task", "steps", "model"], sort=False):
        for prefix in prefixes:
            chunk = group[group.prefix.astype(str) == prefix]
            present = [name for name in schedulers if name in set(chunk.scheduler.astype(str))]
            if len(present) < 2:
                continue
            by_sched = {
                name: chunk[chunk.scheduler.astype(str) == name].set_index("doc_id")
                for name in present
            }
            docs = set(by_sched[present[0]].index)
            for name in present[1:]:
                docs &= set(by_sched[name].index)
            if not docs:
                continue
            nfes = [float(by_sched[name]["avg_nfe_per_sample"].iloc[0]) for name in present]
            pick_scores: list[float] = []
            mean_scores: list[float] = []
            picks: list[str] = []
            pick_indices: list[int] = []
            sched_sums = {name: 0.0 for name in present}
            for doc in docs:
                ratios: list[float] = []
                scores: list[float] = []
                for name in present:
                    row = by_sched[name].loc[doc]
                    gap = float(row.G_hat)
                    frac = float(row.commit_frac)
                    score = float(row.score)
                    if gap != gap or frac != frac or frac <= 0 or score != score:
                        raise ValueError(
                            f"non-finite prefix score for {model} {task} s{steps} {name} doc {doc}"
                        )
                    ratios.append(gap / frac)
                    scores.append(score)
                pick_index = min(range(len(present)), key=lambda i: (ratios[i], i))
                for name, score in zip(present, scores):
                    sched_sums[name] += score
                pick_scores.append(scores[pick_index])
                mean_scores.append(sum(scores) / len(scores))
                picks.append(present[pick_index])
                pick_indices.append(pick_index)
            counts = Counter(picks)
            modal = min(present, key=lambda name: (-counts[name], present.index(name)))
            n_samples = len(docs)
            pick_acc = sum(pick_scores) / n_samples
            mean_acc = sum(mean_scores) / n_samples
            sched_acc = {name: total / n_samples for name, total in sched_sums.items()}
            majority_acc = sched_acc[modal]
            best_acc = max(sched_acc.values())
            best_names = [name for name in present if _same_score(sched_acc[name], best_acc)]
            n_right = sum(pick in best_names for pick in picks)
            rows.append(
                {
                    "task": task,
                    "steps": int(steps),
                    "model": model,
                    "prefix": prefix,
                    "schedulers": ",".join(present),
                    "nfe": ",".join(f"{value:g}" for value in nfes),
                    "n_schedulers": len(present),
                    "n_samples": n_samples,
                    "pick": modal,
                    "pick_counts": ",".join(f"{name}:{counts[name]}" for name in present if counts[name]),
                    "best": ",".join(best_names),
                    "n_right": n_right,
                    "right_frac": n_right / n_samples,
                    "pick_acc": pick_acc,
                    "majority_acc": majority_acc,
                    "mean_acc": mean_acc,
                    "best_acc": best_acc,
                    "lift": pick_acc - mean_acc,
                    "gap": best_acc - pick_acc,
                    "compute_saved": _compute_saved(nfes, pick_indices, int(prefix)),
                }
            )
    return pd.DataFrame(rows)


def _run_list(frame: pd.DataFrame, tau: str, tasks: tuple[str, ...], schedulers: tuple[str, ...]) -> pd.DataFrame:
    frame = frame.copy()
    frame["steps"] = frame["steps"].astype(int)
    sub = frame[
        (frame.tau.astype(str) == tau)
        & frame.scheduler.isin(schedulers)
        & frame.task.isin(tasks)
        & frame.steps.isin(BUDGETS)
    ]
    keep = ["filename", "model", "scheduler", "task", "steps", "avg_nfe_per_sample"]
    runs = sub.drop_duplicates("filename")[keep].copy()
    runs["steps"] = runs["steps"].astype(int)
    return runs.reset_index(drop=True)


def _read_done(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def _prefix_costs(sample: dict, tau: str, prefixes: tuple[str, ...]) -> dict[str, tuple[float, float]] | None:
    parsed = sample_rows(sample)
    if parsed is None:
        return None
    out: dict[str, tuple[float, float]] = {}
    for item in parsed:
        label = str(item["prefix"])
        if label not in prefixes:
            continue
        frac = float(item["commit_frac"])
        gap = float(item[tau]) + float(item["j"])
        if gap != gap or frac != frac:
            raise ValueError(f"non-finite G or commit fraction at prefix {label}")
        if frac <= 0:
            continue
        out[label] = (gap, frac)
    return out


def _rows_for_trace(
    path: Path,
    meta,
    scorer: TraceScorer,
    tau: str,
    prefixes: tuple[str, ...],
) -> list[dict]:
    packed: list[tuple[dict[str, tuple[float, float]], dict]] = []
    n_unusable = 0
    for sample in iter_npz_traces(path):
        costs = _prefix_costs(sample, tau, prefixes)
        sample.pop("step_traces", None)
        if costs is None:
            n_unusable += 1
            continue
        packed.append((costs, sample))
    if n_unusable:
        logger.warning("%s skipped %d samples with no step costs", meta.filename, n_unusable)

    matched: list[tuple[str, object, str] | None] = []
    for _, sample in packed:
        matched.append(scorer.match(sample))

    def _score(hit: tuple[str, object, str] | None) -> tuple[str, float] | None:
        if hit is None:
            return None
        doc_id, inst, text = hit
        return doc_id, scorer.score_text(inst, text)

    scored: list[tuple[str, float] | None] = [None] * len(matched)
    first = next((index for index, hit in enumerate(matched) if hit is not None), None)
    if first is not None:
        scored[first] = _score(matched[first])
    rest = [] if first is None else [index for index in range(len(matched)) if index != first]
    if _is_code_task(scorer.task_name) and len(rest) > 1:
        workers = min(CODE_WORKERS, len(rest))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, value in zip(rest, pool.map(_score, [matched[index] for index in rest])):
                scored[index] = value
    else:
        for index in rest:
            scored[index] = _score(matched[index])

    rows: list[dict] = []
    seen: set[str] = set()
    n_unmatched = 0
    for (costs, _), hit in zip(packed, scored):
        if hit is None:
            n_unmatched += 1
            continue
        doc_id, score = hit
        if doc_id in seen:
            raise ValueError(f"duplicate doc_id {doc_id} in {meta.filename}")
        seen.add(doc_id)
        for label, (gap, frac) in costs.items():
            rows.append(
                {
                    "filename": meta.filename,
                    "model": meta.model,
                    "scheduler": meta.scheduler,
                    "task": meta.task,
                    "steps": int(meta.steps),
                    "avg_nfe_per_sample": float(meta.avg_nfe_per_sample),
                    "doc_id": doc_id,
                    "prefix": label,
                    "G_hat": gap,
                    "commit_frac": frac,
                    "score": score,
                }
            )
    if n_unmatched:
        logger.warning("%s unmatched %d / %d samples", meta.filename, n_unmatched, len(packed))
    if not seen:
        raise RuntimeError(f"{meta.filename}: no scored samples")
    logger.info(
        "%s scored %d samples metric=%s",
        meta.filename,
        len(seen),
        scorer.metric_key,
    )
    return rows


def collect_samples(
    runs: pd.DataFrame,
    cache_path: Path,
    manifest_path: Path,
    *,
    tau: str,
    prefixes: tuple[str, ...] = ("8", "16", "32", "64"),
) -> None:
    """Download each trace once and append per-sample prefix rows."""
    os.environ.setdefault("TRACE_NPZ_MMAP", "1")
    done = _read_done(manifest_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not cache_path.is_file()
    scorers: dict[tuple[str, str], TraceScorer] = {}
    with cache_path.open("a", newline="") as handle, manifest_path.open("a") as manifest:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS)
        if new_file:
            writer.writeheader()
        for meta in runs.itertuples(index=False):
            filename = str(meta.filename)
            if filename in done:
                continue
            key = (str(meta.model), str(meta.task))
            scorer = scorers.get(key)
            if scorer is None:
                scorer = TraceScorer(str(meta.task), str(meta.model))
                scorers[key] = scorer
            logger.info("downloading %s", filename)
            path, temp_dir = download_file_to_temp(
                filename,
                repo_id="schedulerlens-traces",
                namespace="devichand",
                token=os.environ["HF_TOKEN"],
            )
            try:
                written = _rows_for_trace(path, meta, scorer, tau, prefixes)
            finally:
                remove_temp_download(temp_dir)
            for item in written:
                writer.writerow(item)
            handle.flush()
            manifest.write(filename + "\n")
            manifest.flush()
            done.add(filename)


def _sample_paths(output: Path, tau: str) -> tuple[Path, Path]:
    return (
        output.with_name(f"prefix_selection_samples_tau_{tau}.csv"),
        output.with_name(f"prefix_selection_done_tau_{tau}.txt"),
    )


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
        format="%(asctime)s %(levelname)s %(message)s",
    )
    apply_repo_env_defaults()
    tau_name = _tau_name(str(tau))
    tasks = tuple(name for name in TASKS if name not in _excluded(exclude_task))
    schedulers = tuple(name for name in SCHEDULERS if name not in _excluded(exclude_scheduler))
    path = Path(runs_path) if runs_path is not None else DEFAULT_RUNS
    if not path.is_file():
        logger.error("missing runs csv: %s", path)
        return 1
    runs = _run_list(pd.read_csv(path), tau_name, tasks, schedulers)
    if runs.empty:
        logger.error("no runs for tau %s in %s", tau_name, path)
        return 1
    out = Path(output) if output is not None else path.parent / f"prefix_selection_tau_{tau_name}.csv"
    cache_path, manifest_path = _sample_paths(out, tau_name)
    logger.info("%d traces, cache %s", len(runs), cache_path)
    collect_samples(runs, cache_path, manifest_path, tau=tau_name)
    samples = pd.read_csv(cache_path, dtype={"doc_id": str})
    samples["steps"] = samples["steps"].astype(int)
    table = selection_table(
        samples,
        exclude_task=exclude_task,
        exclude_scheduler=exclude_scheduler,
    )
    if table.empty:
        logger.error("no selection rows from %s", cache_path)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    for prefix, group in table.groupby("prefix"):
        logger.info(
            "prefix %s  pick %.3f  majority %.3f  average %.3f  best %.3f  lift %+.3f  right %d/%d  compute saved %.2fx",
            prefix,
            group.pick_acc.mean(),
            group.majority_acc.mean(),
            group.mean_acc.mean(),
            group.best_acc.mean(),
            group.lift.mean(),
            int(group.n_right.sum()),
            int(group.n_samples.sum()),
            group.compute_saved.mean(),
        )
    logger.info("wrote %s (%d cells)", out, len(table))
    return 0
