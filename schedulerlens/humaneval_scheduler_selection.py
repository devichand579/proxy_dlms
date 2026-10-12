"""Proxy-only HumanEval scheduler and hierarchical candidate selection."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from schedulerlens.humaneval_ablation import (
    PROXY_NAMES,
    _latest_checkpoint_records,
    _score_candidates,
    _task_instances,
    _trace_candidates,
    scheduler_from_run_name,
)
from schedulerlens.proxies import DEFAULT_ENTROPY_TAU


def _mean(values):
    return sum(values) / len(values)


def analyze_selection(
    *,
    outputs_root: Path,
    traces_root: Path,
    pattern: str,
    tau: float,
    timeout: float,
    workers: int,
) -> dict:
    instances = _task_instances()
    candidate_rows = []
    run_dirs = sorted(path for path in outputs_root.glob(pattern) if path.is_dir())
    for run_dir in run_dirs:
        checkpoint = run_dir / "checkpoint.jsonl"
        trace = traces_root / f"{run_dir.name}.jsonl"
        if not checkpoint.is_file() or not trace.is_file():
            continue
        records = _latest_checkpoint_records(checkpoint)
        proxies = _trace_candidates(trace, tau)
        scheduler = scheduler_from_run_name(run_dir.name)
        for (doc_id, draw_idx), proxy in proxies.items():
            record = records.get(doc_id)
            if record is None or draw_idx >= len(record.get("resps", [])):
                continue
            candidate_rows.append(
                {
                    **proxy,
                    "scheduler": scheduler,
                    "source_run": run_dir.name,
                    "response": record["resps"][draw_idx],
                }
            )
    if not candidate_rows:
        raise RuntimeError("No aligned HumanEval checkpoint/trace candidates found")
    _score_candidates(
        candidate_rows, instances, timeout=timeout, workers=workers
    )

    grouped = {}
    for row in candidate_rows:
        key = (row["source_doc_id"], row["scheduler"])
        grouped.setdefault(key, []).append(row)
    schedulers = sorted({scheduler for _, scheduler in grouped})
    questions = sorted({doc_id for doc_id, _ in grouped})

    summaries = {}
    for (doc_id, scheduler), rows in grouped.items():
        summaries[(doc_id, scheduler)] = {
            "pass_at_1": _mean([float(row["passed"]) for row in rows]),
            "candidates": rows,
            **{
                f"mean_{proxy}": _mean([row[proxy] for row in rows])
                for proxy in PROXY_NAMES
            },
        }

    for proxy in PROXY_NAMES:
        for scheduler in schedulers:
            values = [
                summaries[(doc_id, scheduler)][f"mean_{proxy}"]
                for doc_id in questions
            ]
            mean = _mean(values)
            std = math.sqrt(_mean([(value - mean) ** 2 for value in values]))
            for doc_id in questions:
                value = summaries[(doc_id, scheduler)][f"mean_{proxy}"]
                summaries[(doc_id, scheduler)][f"z_{proxy}"] = (
                    0.0 if std == 0 else (value - mean) / std
                )

    results = {}
    per_question = {}
    for proxy in PROXY_NAMES:
        scheduler_pass = []
        hierarchical_pass = []
        decisions = {}
        for doc_id in questions:
            selected_scheduler = min(
                schedulers,
                key=lambda scheduler: (
                    summaries[(doc_id, scheduler)][f"z_{proxy}"],
                    scheduler,
                ),
            )
            selected = summaries[(doc_id, selected_scheduler)]
            scheduler_pass.append(selected["pass_at_1"])
            best_candidate = min(
                selected["candidates"],
                key=lambda row: (row[proxy], row["draw_idx"]),
            )
            hierarchical_pass.append(float(best_candidate["passed"]))
            decisions[str(doc_id)] = {
                "scheduler": selected_scheduler,
                "normalized_scheduler_score": selected[f"z_{proxy}"],
                "selected_scheduler_pass_at_1": selected["pass_at_1"],
                "candidate_draw_idx": best_candidate["draw_idx"],
                "candidate_proxy": best_candidate[proxy],
                "candidate_passed": bool(best_candidate["passed"]),
            }
        results[proxy] = {
            "proxy_selected_scheduler_expected_pass_at_1": _mean(
                scheduler_pass
            ),
            "hierarchical_top1_pass_at_1": _mean(hierarchical_pass),
            "decisions": decisions,
        }

    fixed = {
        scheduler: _mean(
            [summaries[(doc_id, scheduler)]["pass_at_1"] for doc_id in questions]
        )
        for scheduler in schedulers
    }
    oracle = _mean(
        [
            max(
                summaries[(doc_id, scheduler)]["pass_at_1"]
                for scheduler in schedulers
            )
            for doc_id in questions
        ]
    )
    random_scheduler = _mean(list(fixed.values()))
    return {
        "task": "humaneval",
        "n_questions": len(questions),
        "n_schedulers": len(schedulers),
        "k": len(next(iter(summaries.values()))["candidates"]),
        "normalization": "per-scheduler z-score of mean trajectory proxy across questions",
        "fixed_scheduler_pass_at_1": fixed,
        "random_scheduler_expected_pass_at_1": random_scheduler,
        "oracle_best_scheduler_pass_at_1": oracle,
        "proxy_selection": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--traces-root", type=Path, required=True)
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--tau", type=float, default=DEFAULT_ENTROPY_TAU)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
    result = analyze_selection(
        outputs_root=args.outputs_root,
        traces_root=args.traces_root,
        pattern=args.pattern,
        tau=args.tau,
        timeout=args.timeout,
        workers=args.workers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(args.output)
    for proxy, row in result["proxy_selection"].items():
        print(
            proxy,
            row["proxy_selected_scheduler_expected_pass_at_1"],
            row["hierarchical_top1_pass_at_1"],
        )


if __name__ == "__main__":
    main()
