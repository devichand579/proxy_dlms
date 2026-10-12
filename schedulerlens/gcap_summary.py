"""Score persisted G-cap candidate pools with the task's native metric."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

from traces.trace_metrics import (
    TASK_SAMPLE_COUNTS,
    _is_code_task,
    _load_scoring_task,
    _pick_primary_metric,
    _score_code_instance,
    _score_instance,
)

SUPPORTED_SCORES = ("raw", "per_commit")


def _checkpoint_records(run_dirs: Iterable[Path]) -> list[dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    sources: dict[tuple[str, str], Path] = {}
    for run_dir in run_dirs:
        checkpoint = Path(run_dir) / "checkpoint.jsonl"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
        for raw in checkpoint.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            record = json.loads(line)
            key = (str(record.get("task_name", "")), str(record.get("doc_id")))
            if key in records and sources[key] != checkpoint:
                raise ValueError(
                    f"duplicate document {key} in {sources[key]} and {checkpoint}"
                )
            records[key] = record
            sources[key] = checkpoint
    return sorted(records.values(), key=lambda row: int(row["doc_id"]))


def _task_instances(task) -> dict[str, Any]:
    instances: dict[str, Any] = {}
    for instance in task.instances:
        key = str(getattr(instance, "doc_id", ""))
        instances.setdefault(key, instance)
    return instances


def _metric_value(task, instance, task_name: str, response: str) -> float:
    current = copy.deepcopy(instance)
    if _is_code_task(task_name):
        metrics = _score_code_instance(response, current, task_name)
    else:
        metrics = _score_instance(task, current, response)
    value, _key = _pick_primary_metric(metrics, task_name)
    if value is None:
        raise ValueError(
            f"no primary metric for task={task_name!r}, doc_id={instance.doc_id}"
        )
    return float(value)


def _candidate_selection_score(
    candidate: dict[str, Any],
    *,
    prefix_steps: int | None,
    score_mode: str,
) -> float:
    """Recompute a candidate score from persisted steps without regenerating it."""
    if score_mode not in SUPPORTED_SCORES:
        raise ValueError(
            f"unknown G-cap score mode {score_mode!r}; expected {SUPPORTED_SCORES}"
        )
    steps = candidate.get("step_proxies")
    if not isinstance(steps, list):
        raise ValueError("candidate is missing persisted step_proxies")
    selected_steps = (
        steps
        if prefix_steps is None
        else [step for step in steps if int(step["step"]) < int(prefix_steps)]
    )
    g_cap = sum(
        float(step["M_hat_t"]) + float(step["J_hat_t"])
        for step in selected_steps
        if int(step.get("n_selected", 0)) > 0
    )
    if score_mode == "raw":
        return g_cap
    commits = sum(int(step.get("n_selected", 0)) for step in selected_steps)
    return g_cap / float(max(commits, 1))


def summarize(
    *,
    task_name: str,
    scheduler: str,
    run_dirs: Iterable[Path],
    prefix_steps: int | None = 32,
    score_mode: str = "per_commit",
    k: int = 8,
    num_fewshot: int = 0,
    code_workers: int = 8,
    allow_partial: bool = False,
) -> dict[str, Any]:
    records = _checkpoint_records(run_dirs)
    expected = TASK_SAMPLE_COUNTS.get(task_name)
    if expected is not None and not allow_partial and len(records) != expected:
        raise ValueError(
            f"{task_name} coverage is incomplete: {len(records)}/{expected} documents"
        )

    task = _load_scoring_task(task_name, num_fewshot)
    instances = _task_instances(task)
    jobs: list[tuple[Any, str]] = []
    selections: list[int] = []
    for record in records:
        doc_id = str(record["doc_id"])
        if doc_id not in instances:
            raise KeyError(f"task instance not found for doc_id={doc_id}")
        rerank = record.get("gcap_rerank") or {}
        candidates = rerank.get("candidates") or []
        if len(candidates) != int(k):
            raise ValueError(
                f"doc_id={doc_id} has {len(candidates)} candidates; expected {k}"
            )
        selections.append(
            min(
                range(int(k)),
                key=lambda index: _candidate_selection_score(
                    candidates[index],
                    prefix_steps=prefix_steps,
                    score_mode=score_mode,
                ),
            )
        )
        jobs.extend(
            (instances[doc_id], str(candidate.get("response", "")))
            for candidate in candidates
        )

    def score(job: tuple[Any, str]) -> float:
        instance, response = job
        return _metric_value(task, instance, task_name, response)

    if _is_code_task(task_name) and int(code_workers) > 1:
        with ThreadPoolExecutor(max_workers=int(code_workers)) as pool:
            flat_scores = list(pool.map(score, jobs))
    else:
        flat_scores = [score(job) for job in jobs]

    per_question = [
        flat_scores[offset : offset + int(k)]
        for offset in range(0, len(flat_scores), int(k))
    ]
    candidate_total = sum(len(scores) for scores in per_question)
    pass_at_1 = (
        sum(sum(scores) for scores in per_question) / candidate_total
        if candidate_total
        else 0.0
    )
    pass_at_8 = (
        sum(1.0 if any(score > 0.0 for score in scores) else 0.0 for scores in per_question)
        / len(per_question)
        if per_question
        else 0.0
    )
    selected_at_1 = (
        sum(scores[index] for scores, index in zip(per_question, selections))
        / len(per_question)
        if per_question
        else 0.0
    )
    return {
        "task": task_name,
        "scheduler": scheduler,
        "questions": len(per_question),
        "k": int(k),
        "selection_scope": (
            "full_trajectory" if prefix_steps is None else f"first_{int(prefix_steps)}_steps"
        ),
        "prefix_steps": None if prefix_steps is None else int(prefix_steps),
        "score_mode": score_mode,
        "pass_at_1": float(pass_at_1),
        "pass_at_8": float(pass_at_8),
        "min_g_selected_at_1": float(selected_at_1),
        "min_g_prefix_selected_at_1": float(selected_at_1),
    }


def _upsert_csv(path: Path, row: dict[str, Any]) -> None:
    fields = list(row)
    existing: list[dict[str, str]] = []
    if path.is_file():
        with path.open(newline="") as handle:
            existing = list(csv.DictReader(handle))
    key = (
        str(row["task"]),
        str(row["scheduler"]),
        str(row["selection_scope"]),
        str(row["score_mode"]),
    )
    existing = [
        item
        for item in existing
        if (
            str(item.get("task")),
            str(item.get("scheduler")),
            str(item.get("selection_scope")),
            str(item.get("score_mode")),
        )
        != key
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing)
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--scheduler", required=True)
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--prefix-steps", type=int, default=32)
    parser.add_argument(
        "--full-trajectory",
        action="store_true",
        help="Select using all persisted diffusion steps instead of a prefix.",
    )
    parser.add_argument("--score-mode", choices=SUPPORTED_SCORES, default="per_commit")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--code-workers", type=int, default=8)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary-csv", type=Path)
    args = parser.parse_args()

    result = summarize(
        task_name=args.task,
        scheduler=args.scheduler,
        run_dirs=args.run_dir,
        prefix_steps=None if args.full_trajectory else args.prefix_steps,
        score_mode=args.score_mode,
        k=args.k,
        num_fewshot=args.num_fewshot,
        code_workers=args.code_workers,
        allow_partial=args.allow_partial,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    if args.summary_csv:
        _upsert_csv(args.summary_csv, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
