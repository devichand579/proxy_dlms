"""Per-step adaptive choices, summarized across samples."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional


def _choice_stem(task: Optional[str], steps: Optional[int], trace_name: Optional[str]) -> str:
    if task:
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(task))
        if steps is not None:
            return f"{safe}_s{int(steps)}"
        return safe
    return trace_name or "pc_adaptive"


def adaptive_choice_paths(
    output_dir: Optional[str],
    trace_name: Optional[str],
    *,
    task: Optional[str] = None,
    steps: Optional[int] = None,
) -> tuple[Path, Path]:
    """JSONL of per-sample choices and the summary CSV next to the eval output.

    Both files are ``{task}_s{steps}_adaptive_choices.*``.
    """
    stem = _choice_stem(task, steps, trace_name)
    if output_dir:
        base = Path(output_dir)
    else:
        base = Path("outputs")
    base.mkdir(parents=True, exist_ok=True)
    return (
        base / f"{stem}_adaptive_choices.jsonl",
        base / f"{stem}_adaptive_choices.csv",
    )


def group_active_choices(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """Batch-row index -> steps where that row still had masked tokens."""
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row.get("active"):
            continue
        choice = {
            "step": int(row["step"]),
            "scheduler": row["rule"],
            "predictor": row["pred_model"],
            "controller": row["ctrl_model"],
        }
        for key in ("g_hat", "m_hat", "m_hat_other", "j_hat", "j_hat_other"):
            if key in row:
                choice[key] = float(row[key])
        grouped[int(row["batch"])].append(choice)
    for choices in grouped.values():
        choices.sort(key=lambda item: item["step"])
    return grouped


def append_adaptive_sample(
    jsonl_path: Path,
    *,
    task: str,
    doc_id: Any,
    choices: list[dict[str, Any]],
) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"task": str(task), "doc_id": doc_id, "choices": choices}
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _rank(counts: Counter) -> list[tuple[str, int]]:
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def _share(count: int, n_samples: int) -> float:
    if n_samples <= 0:
        return 0.0
    return count / n_samples


def write_adaptive_summary(jsonl_path: Path, csv_path: Path) -> int:
    """One CSV row per step. Later lines for the same sample replace earlier ones.

    Returns the number of samples in the summary.
    """
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    if jsonl_path.is_file():
        with jsonl_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                key = (str(record.get("task", "")), str(record.get("doc_id")))
                latest[key] = record

    schedulers: dict[int, Counter] = defaultdict(Counter)
    predictors: dict[int, Counter] = defaultdict(Counter)
    controllers: dict[int, Counter] = defaultdict(Counter)
    score_sums: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    score_ns: dict[int, Counter] = defaultdict(Counter)
    n_at_step: Counter = Counter()
    score_keys = ("g_hat", "m_hat", "m_hat_other", "j_hat", "j_hat_other")
    for record in latest.values():
        for choice in record.get("choices") or []:
            step = int(choice["step"])
            n_at_step[step] += 1
            schedulers[step][str(choice["scheduler"])] += 1
            predictors[step][str(choice["predictor"])] += 1
            controllers[step][str(choice["controller"])] += 1
            for key in score_keys:
                if key in choice:
                    score_sums[step][key] += float(choice[key])
                    score_ns[step][key] += 1

    fieldnames = ["step", "n_samples"]
    for rank in (1, 2, 3):
        fieldnames.extend(
            [f"scheduler_{rank}", f"scheduler_{rank}_n", f"scheduler_{rank}_share"]
        )
    fieldnames.extend(
        ["predictor", "predictor_n", "predictor_share", "controller", "controller_n", "controller_share"]
    )
    fieldnames.extend(["g_hat", "m_hat", "m_hat_other", "j_hat", "j_hat_other"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for step in sorted(n_at_step):
            n_samples = int(n_at_step[step])
            row: dict[str, Any] = {"step": step, "n_samples": n_samples}
            ranked = _rank(schedulers[step])
            for rank in (1, 2, 3):
                if rank <= len(ranked):
                    name, count = ranked[rank - 1]
                    row[f"scheduler_{rank}"] = name
                    row[f"scheduler_{rank}_n"] = count
                    row[f"scheduler_{rank}_share"] = f"{_share(count, n_samples):.4f}"
                else:
                    row[f"scheduler_{rank}"] = ""
                    row[f"scheduler_{rank}_n"] = ""
                    row[f"scheduler_{rank}_share"] = ""
            pred_name, pred_n = _rank(predictors[step])[0]
            ctrl_name, ctrl_n = _rank(controllers[step])[0]
            row["predictor"] = pred_name
            row["predictor_n"] = pred_n
            row["predictor_share"] = f"{_share(pred_n, n_samples):.4f}"
            row["controller"] = ctrl_name
            row["controller_n"] = ctrl_n
            row["controller_share"] = f"{_share(ctrl_n, n_samples):.4f}"
            for key in score_keys:
                n_scored = int(score_ns[step][key])
                if n_scored:
                    row[key] = f"{score_sums[step][key] / n_scored:.6f}"
                else:
                    row[key] = ""
            writer.writerow(row)
    return len(latest)
