"""Build batched eval prompts from lm-eval tasks (same few-shot defaults as ``launch_eval``)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from cst.eval_data.lmeval_task import (
    CST_EVAL_TASKS,
    eval_num_fewshot,
    expand_eval_task_names,
    load_lmeval_task,
)

__all__ = [
    "CST_EVAL_TASKS",
    "eval_num_fewshot",
    "expand_eval_task_names",
    "load_eval_row_batches",
]


def load_eval_row_batches(
    task_name: str,
    batch_size: int,
    *,
    limit: Optional[int] = None,
    num_fewshot: Optional[int] = None,
    seed: int = 0,
) -> List[List[Dict[str, Any]]]:
    """All eval instances for ``task_name``, batched for hidden capture."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    task, ns = load_lmeval_task(task_name, num_fewshot=num_fewshot, seed=int(seed))
    task.build_all_requests(limit=limit, rank=0, world_size=1)
    instances = task.instances
    rows: List[Dict[str, Any]] = []
    for inst in instances:
        if not inst.args:
            raise ValueError(f"empty instance args for task {task_name!r}")
        text = str(inst.args[0])
        doc_id = getattr(inst, "doc_id", len(rows))
        rows.append(
            {
                "text": text,
                "meta": {
                    "task": str(task_name),
                    "doc_id": doc_id,
                    "num_fewshot": ns,
                },
            }
        )
    if not rows:
        raise RuntimeError(f"lm-eval task {task_name!r} produced no instances")
    return [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]
