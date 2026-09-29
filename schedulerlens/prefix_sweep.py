"""Prefix sweep of M̂.

The unmasked set is whatever the scheduler already committed. Tokens in that set
are ordered from lowest entropy to highest. A token adds its −log c only while
the sum of their entropies stays within γ. Ĵ is j_hat on the whole set.

``main`` writes M̂, Ĵ, and Ĝ/f at each prefix. ``main_pcfg`` writes sample-level
Pearson of M̂ with Mt, Ĵ with Jt, and Ĝ with G (Mt + Jt) at prefixes 8, 16, 32, 64 and full.
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_env_defaults import apply_repo_env_defaults

apply_repo_env_defaults()

from schedulerlens.completion import duplicate_run_stems, load_strace_metrics
from schedulerlens.metrics import _MODE_TO_SCHEDULER, parse_pcfg_stem, pearson_corr
from schedulerlens.oracle import load_oracle_artifacts
from schedulerlens.proxies import j_hat, m_hat
from traces.hf_trace_upload import download_file_to_temp, remove_temp_download
from traces.trace_npz import iter_npz_traces

logger = logging.getLogger("schedulerlens.prefix_sweep")

STRACE_METRICS = REPO_ROOT / "fetch_evals/strace_metrics.csv"
OUT_DIR = REPO_ROOT / "outputs/schedulerlens/prefix_sweep"
SWEEP_CSV = OUT_DIR / "prefix_sweep.csv"
PCFG_ORACLE_DIR = REPO_ROOT / "oracle_pcfg"
PCFG_OUT_DIR = REPO_ROOT / "outputs/schedulerlens/prefix_sweep_pcfg"
PCFG_REPO_ID = "devichand/llada-pcfg-traces"
TASKS = ("gsm8k", "humaneval", "mbpp", "minerva_math500", "arc_challenge_chat")
BUDGETS = (128, 256, 512)
ARC_BUDGETS = (32, 64, 128)
PREFIXES = (8, 16, 32, 64)
GAMMAS = (0.1, 0.25, 0.5, 1.0, 1.25, 1.5, 2.0)
TAU_NAMES = tuple(str(g) for g in GAMMAS)


def _required_sweep_keys() -> set[tuple[str, str]]:
    prefixes = {*(str(k) for k in PREFIXES), "full"}
    return {(prefix, tau) for prefix in prefixes for tau in TAU_NAMES}


def finished_filenames(path: Path) -> set[str]:
    """Filenames whose prefix and tau rows are all present.

    A filename with only some of those rows is removed from the CSV so the
    next pass writes the full set once.
    """
    if not path.is_file():
        return set()
    prev = pd.read_csv(path, dtype=str)
    if prev.empty or "filename" not in prev.columns:
        return set()
    required = _required_sweep_keys()
    complete: set[str] = set()
    for filename, group in prev.groupby("filename", sort=False):
        have = set(zip(group["prefix"].astype(str), group["tau"].astype(str)))
        if required <= have:
            complete.add(str(filename))
    partial = int(prev["filename"].nunique()) - len(complete)
    if partial:
        prev.loc[prev["filename"].astype(str).isin(complete)].to_csv(path, index=False)
        logger.info(
            "dropped %d runs missing a prefix or tau; %d runs already complete",
            partial,
            len(complete),
        )
    return complete


def _as1d(value, dtype):
    if value is None:
        return None
    arr = np.asarray(value)
    while arr.ndim > 1 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 1:
        return None
    return np.asarray(arr, dtype=dtype)


def _zero_costs() -> dict[str, float]:
    return {name: 0.0 for name in TAU_NAMES}


def _step_costs(sample: dict) -> list[dict] | None:
    steps = sample.get("step_traces") or []
    out = []
    for t, st in enumerate(steps):
        ent = _as1d(st.get("entropies"), np.float64)
        conf = _as1d(st.get("confidences"), np.float64)
        sel = _as1d(st.get("selected_positions"), np.bool_)
        mask = _as1d(st.get("mask_index"), np.bool_)
        if ent is None or conf is None or mask is None:
            return None
        if sel is None:
            # Finished row in a batch: model was called, nothing was unmasked.
            if not (ent.shape == conf.shape == mask.shape):
                return None
            out.append(
                {
                    "step": t,
                    "n_committed": int(np.sum(~mask)),
                    "n_selected": 0,
                    "j": 0.0,
                    "m": _zero_costs(),
                }
            )
            continue
        if not (ent.shape == conf.shape == sel.shape == mask.shape):
            return None
        h = ent[sel]
        c = conf[sel]
        finite = np.isfinite(h) & np.isfinite(c)
        h = h[finite]
        c = c[finite]
        costs = _zero_costs()
        for gamma in GAMMAS:
            costs[str(gamma)] = m_hat(c, h, gamma)
        out.append(
            {
                "step": t,
                "n_committed": int(np.sum(~mask)),
                "n_selected": int(h.size),
                "j": j_hat(h.astype(np.float64)),
                "m": costs,
            }
        )
    return out


def _prefix_values(steps: list[dict], gen_length: int, prefix: int) -> dict | None:
    length = int(gen_length)
    if length <= 0 or not steps:
        return None
    later = [sp for sp in steps if int(sp["step"]) >= prefix]
    if later:
        committed = int(min(later, key=lambda sp: int(sp["step"]))["n_committed"])
        used = [sp for sp in steps if int(sp["step"]) < prefix]
    else:
        last = max(steps, key=lambda sp: int(sp["step"]))
        committed = int(last["n_committed"]) + int(last["n_selected"])
        used = steps
    frac = committed / float(length)
    if frac <= 0:
        return None
    m = {name: 0.0 for name in TAU_NAMES}
    j = 0.0
    for sp in used:
        j += float(sp["j"])
        for name, value in sp["m"].items():
            m[name] += float(value)
    m["j"] = j
    m["commit_frac"] = frac
    return m


def sample_rows(sample: dict) -> list[dict] | None:
    steps = _step_costs(sample)
    if steps is None:
        return None
    gen_length = int(sample.get("gen_length") or 0)
    if gen_length <= 0 and steps:
        head = steps[0]
        gen_length = int(head["n_committed"]) + int(head["n_selected"])
    rows = []
    full_k = max(int(sp["step"]) for sp in steps) + 1
    for prefix, label in [*( (k, str(k)) for k in PREFIXES ), (full_k, "full")]:
        vals = _prefix_values(steps, gen_length, prefix)
        if vals is None:
            continue
        vals["prefix"] = label
        rows.append(vals)
    return rows


def _excluded(spec) -> set[str]:
    if not spec:
        return set()
    if isinstance(spec, str):
        parts = spec.split(",")
    else:
        parts = spec
    return {str(part).strip() for part in parts if str(part).strip()}


def run_table(exclude_task=None, exclude_scheduler=None) -> pd.DataFrame:
    """Runs to sweep, taken from ``fetch_evals/strace_metrics.csv``.

    The Hub filename is ``{name}.npz``. ``mode`` is mapped to the short
    scheduler key. A repeated NFE for the same model, mode, and task is dropped.
    """
    drop_task = _excluded(exclude_task)
    drop_sched = _excluded(exclude_scheduler)
    rows = []
    for name, row in load_strace_metrics(STRACE_METRICS).items():
        task = str(row.get("task") or "")
        scheduler = _MODE_TO_SCHEDULER.get(str(row.get("mode") or ""), str(row.get("mode") or ""))
        steps = int(float(row["steps"]))
        if task in drop_task or scheduler in drop_sched:
            continue
        if task == "arc_challenge_chat":
            if steps not in ARC_BUDGETS or scheduler in {"fdm", "fhs"}:
                continue
        elif task not in TASKS[:-1] or steps not in BUDGETS or scheduler == "fdm":
            continue
        metric = row.get("metric")
        rows.append(
            {
                "filename": f"{name}.npz",
                "model": row.get("model"),
                "scheduler": scheduler,
                "task": task,
                "steps": steps,
                "avg_nfe_per_sample": float(row["avg_nfe_per_sample"]),
                "accuracy": float(metric) if metric == metric else float("nan"),
            }
        )
    both = pd.DataFrame(rows)
    if both.empty:
        logger.info("prefix sweep keeps 0 runs")
        return both
    dropped = duplicate_run_stems(STRACE_METRICS)
    stems = both.filename.astype(str).map(lambda fname: Path(fname).stem)
    before = len(both)
    both = both.loc[~stems.isin(dropped)].copy()
    logger.info(
        "prefix sweep drops %d runs whose NFE repeats a lower step budget; keeps %d",
        before - len(both),
        len(both),
    )
    return both.sort_values(["task", "steps", "model", "scheduler"]).reset_index(drop=True)


def process_file(row) -> list[dict]:
    filename = str(row.filename)
    path, temp_dir = download_file_to_temp(
        filename,
        repo_id="schedulerlens-traces",
        namespace="devichand",
        token=os.environ["HF_TOKEN"],
    )
    acc = {label: [] for label in ("full", *(str(k) for k in PREFIXES))}
    names = TAU_NAMES
    sums = {
        label: {name: [] for name in ("j", "commit_frac", *names)}
        for label in acc
    }
    n = 0
    try:
        for sample in iter_npz_traces(path):
            parsed = sample_rows(sample)
            if parsed is None:
                continue
            n += 1
            for item in parsed:
                label = item["prefix"]
                if label not in sums:
                    continue
                sums[label]["j"].append(item["j"])
                sums[label]["commit_frac"].append(item["commit_frac"])
                for name in names:
                    sums[label][name].append(item[name])
    finally:
        remove_temp_download(temp_dir)
    rows = []
    for label, bucket in sums.items():
        if not bucket["j"]:
            continue
        j = float(np.mean(bucket["j"]))
        frac = float(np.mean(bucket["commit_frac"]))
        for name in names:
            m = float(np.mean(bucket[name]))
            rows.append(
                {
                    "filename": filename,
                    "model": row.model,
                    "scheduler": row.scheduler,
                    "task": row.task,
                    "steps": int(row.steps),
                    "avg_nfe_per_sample": float(row.avg_nfe_per_sample),
                    "prefix": label,
                    "tau": name,
                    "M_hat": m,
                    "J_hat": j,
                    "G_hat": m + j,
                    "commit_frac": frac,
                    "accuracy": float(row.accuracy),
                    "n_samples": n,
                }
            )
    return rows


def main(exclude_task=None, exclude_scheduler=None, quiet: bool = False, output: Path | None = None) -> int:
    logging.basicConfig(
        level=logging.WARNING if quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    os.environ.setdefault("TRACE_NPZ_MMAP", "1")
    run_path = Path(output) if output is not None else SWEEP_CSV
    run_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "filename", "model", "scheduler", "task", "steps", "avg_nfe_per_sample",
        "prefix", "tau",
        "M_hat", "J_hat", "G_hat", "commit_frac", "accuracy", "n_samples",
    ]
    finished = finished_filenames(run_path)
    frame = run_table(exclude_task=exclude_task, exclude_scheduler=exclude_scheduler)
    logger.info("%d runs, %d already done", len(frame), len(finished))
    new_file = not run_path.is_file()
    with run_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new_file:
            writer.writeheader()
        for row in frame.itertuples(index=False):
            if str(row.filename) in finished:
                continue
            t0 = time.time()
            written = process_file(row)
            for item in written:
                writer.writerow(item)
            handle.flush()
            n_samples = written[0]["n_samples"] if written else 0
            logger.info("%s samples=%s in %.0fs", row.filename, n_samples, time.time() - t0)
    logger.info("wrote %s", run_path)
    return 0


def _oracle_step_mats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arts = load_oracle_artifacts(path)
    n = int(np.asarray(arts["n_samples"]).reshape(-1)[0])
    step = np.asarray(arts["step"], dtype=np.int64)
    width = int(step.max()) + 1 if step.size else 0
    mt = np.full((n, width), np.nan, dtype=np.float64)
    jt = np.full((n, width), np.nan, dtype=np.float64)
    if step.size:
        si = np.asarray(arts["step_sample_idx"], dtype=np.int64)
        mt[si, step] = np.asarray(arts["step_Mt"], dtype=np.float64)
        jt[si, step] = np.asarray(arts["step_Jt"], dtype=np.float64)
    return mt, jt


def _pcfg_prefix_sums(
    steps: list[dict],
    mt_row: np.ndarray,
    jt_row: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Per-sample sums of M̂, Ĵ, Mt, and Jt over steps with index < K."""
    full_k = max(int(sp["step"]) for sp in steps) + 1
    labels = [(k, str(k)) for k in PREFIXES]
    labels.append((full_k, "full"))
    out: dict[str, dict[str, float]] = {}
    width = int(mt_row.shape[0])
    names = TAU_NAMES
    for prefix, label in labels:
        used = [sp for sp in steps if int(sp["step"]) < prefix]
        bucket = {name: 0.0 for name in names}
        j = 0.0
        mt = 0.0
        jt = 0.0
        for sp in used:
            t = int(sp["step"])
            if t < width and np.isfinite(mt_row[t]):
                mt += float(mt_row[t])
            if t < width and np.isfinite(jt_row[t]):
                jt += float(jt_row[t])
            j += float(sp["j"])
            for name in names:
                bucket[name] += float(sp["m"][name])
        bucket["J_hat"] = j
        bucket["Mt"] = mt
        bucket["Jt"] = jt
        out[label] = bucket
    return out


def process_pcfg_file(row) -> list[dict]:
    filename = str(row.filename)
    oracle_path = PCFG_ORACLE_DIR / f"{Path(filename).stem}_oracle.npz"
    if not oracle_path.is_file():
        raise FileNotFoundError(oracle_path)
    mt_mat, jt_mat = _oracle_step_mats(oracle_path)
    path, temp_dir = download_file_to_temp(
        filename,
        repo_id=PCFG_REPO_ID,
        token=os.environ["HF_TOKEN"],
    )
    names = TAU_NAMES
    labels = ("full", *(str(k) for k in PREFIXES))
    keys = (*names, "J_hat", "Mt", "Jt")
    sample_m = {label: {key: [] for key in keys} for label in labels}
    try:
        for i, sample in enumerate(iter_npz_traces(path)):
            if i >= mt_mat.shape[0]:
                break
            steps = _step_costs(sample)
            if not steps:
                continue
            parsed = _pcfg_prefix_sums(steps, mt_mat[i], jt_mat[i])
            for label, bucket in parsed.items():
                if label not in sample_m:
                    continue
                for key in keys:
                    sample_m[label][key].append(bucket[key])
    finally:
        remove_temp_download(temp_dir)

    sample_rows = []
    for label, bucket in sample_m.items():
        mt = np.asarray(bucket["Mt"], dtype=np.float64)
        jt = np.asarray(bucket["Jt"], dtype=np.float64)
        jh = np.asarray(bucket["J_hat"], dtype=np.float64)
        if mt.size == 0:
            continue
        rho_j = pearson_corr(jh, jt)
        gt = mt + jt
        for name in names:
            mh = np.asarray(bucket[name], dtype=np.float64)
            sample_rows.append(
                {
                    "filename": filename,
                    "model": row.model,
                    "scheduler": row.scheduler,
                    "task": row.task,
                    "steps": int(row.steps),
                    "prefix": label,
                    "tau": name,
                    "M_hat": float(np.mean(mh)),
                    "J_hat": float(np.mean(jh)),
                    "G_hat": float(np.mean(mh + jh)),
                    "Mt": float(np.mean(mt)),
                    "Jt": float(np.mean(jt)),
                    "Gt": float(np.mean(gt)),
                    "pearson_M": pearson_corr(mh, mt),
                    "pearson_J": rho_j,
                    "pearson_G": pearson_corr(mh + jh, gt),
                    "n_samples": int(mt.size),
                }
            )
    return sample_rows


def _append(path: Path, rows: list[dict], fields: list[str]) -> None:
    new_file = not path.is_file()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new_file:
            writer.writeheader()
        for item in rows:
            writer.writerow(item)
        handle.flush()


def pcfg_run_table(exclude_task=None, exclude_scheduler=None) -> pd.DataFrame:
    """PCFG runs that already have an oracle NPZ in ``oracle_pcfg``.

    The Hub trace is ``{stem}.npz`` on ``devichand/llada-pcfg-traces``, where
    ``stem`` is the oracle filename with ``_oracle`` removed.
    """
    drop_task = _excluded(exclude_task)
    drop_sched = _excluded(exclude_scheduler)
    rows = []
    for path in sorted(PCFG_ORACLE_DIR.glob("*_oracle.npz")):
        stem = path.name[: -len("_oracle.npz")]
        parsed = parse_pcfg_stem(stem)
        if parsed is None:
            logger.warning("skipping unparsed oracle %s", path.name)
            continue
        model, scheduler, steps = parsed
        if "pcfg" in drop_task or scheduler in drop_sched:
            continue
        rows.append(
            {
                "filename": f"{stem}.npz",
                "model": model,
                "scheduler": scheduler,
                "task": "pcfg",
                "steps": steps,
            }
        )
    frame = pd.DataFrame(rows)
    logger.info("prefix-sweep-pcfg keeps %d runs with a local oracle", len(frame))
    if frame.empty:
        return frame
    return frame.sort_values(["steps", "scheduler"]).reset_index(drop=True)


def main_pcfg(
    exclude_task=None,
    exclude_scheduler=None,
    quiet: bool = False,
    output: Path | None = None,
) -> int:
    logging.basicConfig(
        level=logging.WARNING if quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    os.environ.setdefault("TRACE_NPZ_MMAP", "1")
    corr_path = Path(output) if output is not None else PCFG_OUT_DIR / "prefix_sweep_pcfg.csv"
    corr_path.parent.mkdir(parents=True, exist_ok=True)
    corr_fields = [
        "filename", "model", "scheduler", "task", "steps", "prefix", "tau",
        "M_hat", "J_hat", "G_hat", "Mt", "Jt", "Gt",
        "pearson_M", "pearson_J", "pearson_G", "n_samples",
    ]
    frame = pcfg_run_table(exclude_task=exclude_task, exclude_scheduler=exclude_scheduler)
    finished = finished_filenames(corr_path)
    logger.info("%d pcfg runs, %d already done", len(frame), len(finished))
    for row in frame.itertuples(index=False):
        if str(row.filename) in finished:
            continue
        t0 = time.time()
        sample_rows = process_pcfg_file(row)
        _append(corr_path, sample_rows, corr_fields)
        n_samples = sample_rows[0]["n_samples"] if sample_rows else 0
        logger.info("%s samples=%s in %.0fs", row.filename, n_samples, time.time() - t0)
    logger.info("wrote %s", corr_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
