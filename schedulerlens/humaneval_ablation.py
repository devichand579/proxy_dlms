"""HumanEval proxy diagnostics and inference-time reranking ablations."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Optional

from schedulerlens.proxies import DEFAULT_ENTROPY_TAU, compute_sample_proxies
from traces.trace_checks import iter_repaired_jsonl
from traces.trace_metrics import _load_scoring_task, _score_code_instance


PROXY_NAMES = ("M_hat", "J_hat_early", "G_hat")


def scheduler_from_run_name(run_name: str) -> str:
    """Canonical scheduler key shared by two-node shard run names."""
    match = re.match(r"^.+?_humaneval_(.+?)_s\d+", run_name)
    return match.group(1) if match else run_name


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased HumanEval pass@k estimator."""
    if n < 1 or k < 1 or k > n:
        raise ValueError(f"Expected 1 <= k <= n, got n={n}, k={k}")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pearson_corr(xs: Iterable[float], ys: Iterable[float]) -> Optional[float]:
    x = [float(v) for v in xs]
    y = [float(v) for v in ys]
    if len(x) != len(y) or len(x) < 2:
        return None
    mx, my = sum(x) / len(x), sum(y) / len(y)
    dx, dy = [v - mx for v in x], [v - my for v in y]
    denom = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    return None if denom == 0 else sum(a * b for a, b in zip(dx, dy)) / denom


def roc_auc_for_lower_score(rows: list[dict], score: str) -> Optional[float]:
    """Pairwise AUROC where lower proxy score predicts passing code."""
    positives = [row for row in rows if row["passed"]]
    negatives = [row for row in rows if not row["passed"]]
    if not positives or not negatives:
        return None
    wins = 0.0
    for pos in positives:
        for neg in negatives:
            if pos[score] < neg[score]:
                wins += 1.0
            elif pos[score] == neg[score]:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _latest_checkpoint_records(path: Path) -> dict[int, dict]:
    latest: dict[int, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            record = json.loads(line)
            latest[int(record["doc_id"])] = record
    return latest


def _trace_candidates(path: Path, tau: float) -> dict[tuple[int, int], dict]:
    candidates: dict[tuple[int, int], dict] = {}
    for kind, payload in iter_repaired_jsonl(path):
        if kind != "sample":
            continue
        sample, _repairs = payload
        metadata = sample.get("metadata") or {}
        doc_id = int(metadata["doc_id"])
        draw_idx = int(metadata.get("draw_idx", metadata["sample_idx"]))
        proxy = compute_sample_proxies(sample, trace_name=path, tau=tau)
        candidates[(doc_id, draw_idx)] = {
            "doc_id": doc_id,
            "source_doc_id": int(metadata.get("source_doc_id", doc_id)),
            "draw_idx": draw_idx,
            "M_hat": proxy.M_hat,
            "J_hat_early": proxy.J_hat_early,
            "G_hat": proxy.G_hat,
        }
    return candidates


def _task_instances() -> dict[int, object]:
    task = _load_scoring_task("humaneval", 0)
    return {int(instance.doc_id): instance for instance in task.instances}


def _score_candidates(
    rows: list[dict],
    instances: dict[int, object],
    *,
    timeout: float,
    workers: int,
) -> None:
    def score(row: dict) -> bool:
        instance = instances[row["source_doc_id"]]
        metrics = _score_code_instance(
            row["response"], instance, "humaneval", timeout=timeout
        )
        return bool(metrics.get("pass@1,none", 0.0))

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        passed = list(pool.map(score, rows))
    for row, value in zip(rows, passed):
        row["passed"] = value


def _selection_summary(
    by_doc: dict[int, list[dict]],
    *,
    score: str,
    subset_sizes: list[int],
) -> dict:
    docs = sorted(by_doc)
    top1 = []
    subsets = {}
    for doc_id in docs:
        ranked = sorted(by_doc[doc_id], key=lambda row: (row[score], row["draw_idx"]))
        top1.append(float(ranked[0]["passed"]))
    for size in subset_sizes:
        pool_accuracy = []
        selected_pass = []
        random_pass = []
        for doc_id in docs:
            rows = by_doc[doc_id]
            if size > len(rows):
                continue
            selected = sorted(
                rows, key=lambda row: (row[score], row["draw_idx"])
            )[:size]
            correct = sum(bool(row["passed"]) for row in rows)
            pool_accuracy.append(
                sum(bool(row["passed"]) for row in selected) / len(selected)
            )
            selected_pass.append(float(any(row["passed"] for row in selected)))
            random_pass.append(pass_at_k(len(rows), correct, size))
        subsets[str(size)] = {
            "top_pool_expected_pass_at_1": sum(pool_accuracy) / len(pool_accuracy),
            "top_pool_pass_at_m": sum(selected_pass) / len(selected_pass),
            "random_pool_expected_pass_at_1": sum(
                sum(bool(row["passed"]) for row in by_doc[d]) / len(by_doc[d])
                for d in docs
                if size <= len(by_doc[d])
            )
            / len(pool_accuracy),
            "random_pool_pass_at_m": sum(random_pass) / len(random_pass),
            "n_questions": len(pool_accuracy),
        }
    return {
        "top1_pass_at_1": sum(top1) / len(top1),
        "subset_sizes": subsets,
    }


def _scheduler_summary(
    scheduler: str,
    rows: list[dict],
    subset_sizes: list[int],
) -> dict:
    by_doc: dict[int, list[dict]] = {}
    for row in rows:
        by_doc.setdefault(row["source_doc_id"], []).append(row)
    pass1 = sum(bool(row["passed"]) for row in rows) / len(rows)
    pass_k = {}
    for size in sorted(set([1, *subset_sizes])):
        values = []
        for candidates in by_doc.values():
            if size <= len(candidates):
                values.append(
                    pass_at_k(
                        len(candidates),
                        sum(bool(row["passed"]) for row in candidates),
                        size,
                    )
                )
        pass_k[str(size)] = sum(values) / len(values) if values else None
    return {
        "scheduler": scheduler,
        "n_questions": len(by_doc),
        "n_candidates": len(rows),
        "pass_at_1": pass1,
        "pass_at_k": pass_k,
        "proxy_diagnostics": {
            score: {
                "pearson_with_correctness": pearson_corr(
                    [row[score] for row in rows],
                    [float(row["passed"]) for row in rows],
                ),
                "auc_lower_is_better": roc_auc_for_lower_score(rows, score),
                "reranking": _selection_summary(
                    by_doc, score=score, subset_sizes=subset_sizes
                ),
            }
            for score in PROXY_NAMES
        },
    }


def _zscore_by_scheduler(rows: list[dict], score: str) -> None:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["scheduler"], []).append(row)
    for scheduler_rows in grouped.values():
        values = [row[score] for row in scheduler_rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(variance)
        for row in scheduler_rows:
            row[f"{score}_z"] = 0.0 if std == 0 else (row[score] - mean) / std


def analyze(
    *,
    outputs_root: Path,
    traces_root: Path,
    pattern: str,
    subset_sizes: list[int],
    tau: float,
    timeout: float,
    workers: int,
    schedulers: Optional[set[str]] = None,
) -> dict:
    instances = _task_instances()
    all_rows: list[dict] = []
    scheduler_rows: dict[str, list[dict]] = {}
    run_dirs = sorted(path for path in outputs_root.glob(pattern) if path.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"No run directories match {outputs_root / pattern}")
    for run_dir in run_dirs:
        scheduler = scheduler_from_run_name(run_dir.name)
        if schedulers is not None and scheduler not in schedulers:
            continue
        checkpoint = run_dir / "checkpoint.jsonl"
        trace_path = traces_root / f"{run_dir.name}.jsonl"
        if not checkpoint.is_file() or not trace_path.is_file():
            continue
        records = _latest_checkpoint_records(checkpoint)
        proxies = _trace_candidates(trace_path, tau)
        rows = []
        for (doc_id, draw_idx), proxy in proxies.items():
            record = records.get(doc_id)
            if record is None or draw_idx >= len(record.get("resps", [])):
                continue
            rows.append(
                {
                    **proxy,
                    "scheduler": run_dir.name,
                    "response": record["resps"][draw_idx],
                }
            )
        if not rows:
            continue
        _score_candidates(rows, instances, timeout=timeout, workers=workers)
        for row in rows:
            row["scheduler"] = scheduler
            row["source_run"] = run_dir.name
        scheduler_rows.setdefault(scheduler, []).extend(rows)
        all_rows.extend(rows)
    summaries = [
        _scheduler_summary(name, rows, subset_sizes)
        for name, rows in sorted(scheduler_rows.items())
    ]
    for score in PROXY_NAMES:
        _zscore_by_scheduler(all_rows, score)
    cross_by_doc: dict[int, list[dict]] = {}
    for row in all_rows:
        cross_by_doc.setdefault(row["source_doc_id"], []).append(row)
    cross_scheduler = {
        score: _selection_summary(
            cross_by_doc, score=f"{score}_z", subset_sizes=subset_sizes
        )
        for score in PROXY_NAMES
    }
    return {
        "task": "humaneval",
        "subset_sizes": subset_sizes,
        "tau": tau,
        "n_schedulers": len(summaries),
        "n_candidates": len(all_rows),
        "scheduler_filter": sorted(schedulers) if schedulers is not None else None,
        "by_scheduler": summaries,
        "cross_scheduler_zscore_reranking": cross_scheduler,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--traces-root", type=Path, required=True)
    parser.add_argument("--pattern", default="*humaneval*")
    parser.add_argument("--subset-sizes", default="4,6,12")
    parser.add_argument("--tau", type=float, default=DEFAULT_ENTROPY_TAU)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--schedulers",
        default=None,
        help="Optional comma-separated canonical scheduler names",
    )
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    subset_sizes = sorted(
        {int(value) for value in args.subset_sizes.split(",") if value.strip()}
    )
    result = analyze(
        outputs_root=args.outputs_root,
        traces_root=args.traces_root,
        pattern=args.pattern,
        subset_sizes=subset_sizes,
        tau=args.tau,
        timeout=args.timeout,
        workers=args.workers,
        schedulers=(
            {value.strip() for value in args.schedulers.split(",") if value.strip()}
            if args.schedulers
            else None
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(args.output)


if __name__ == "__main__":
    main()
