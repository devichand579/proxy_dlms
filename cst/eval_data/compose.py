"""Build batched eval prompts from lm-eval tasks (same few-shot defaults as ``launch_eval``)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def eval_num_fewshot(task_name: str) -> int:
    """Match ``launch_eval.get_num_fewshot`` for consistent eval prompts."""
    t = str(task_name).strip().lower()
    if "humaneval" in t:
        return 0
    if "gsm8k" in t:
        return 4
    if "math500" in t or "minerva_math500" in t:
        return 4
    if "mbpp3" in t:
        return 3
    if "mbpp" in t:
        return 3
    if "arc_challenge_chat" in t:
        return 5
    return 0


def _resolve_task(task_dict: Dict[str, Any], task_name: str):
    if task_name in task_dict:
        return task_dict[task_name]
    if len(task_dict) == 1:
        return next(iter(task_dict.values()))
    keys = ", ".join(sorted(task_dict.keys()))
    raise ValueError(f"lm-eval task {task_name!r} not found (available: {keys})")


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
    from lm_eval.tasks import TaskManager

    ns = int(num_fewshot) if num_fewshot is not None else eval_num_fewshot(task_name)
    tm = TaskManager()
    task_dict = tm.load_task_or_group(str(task_name).strip())
    task = _resolve_task(task_dict, str(task_name).strip())
    if hasattr(task, "set_fewshot_seed"):
        task.set_fewshot_seed(int(seed))
    if hasattr(task, "config") and hasattr(task.config, "num_fewshot"):
        task.config.num_fewshot = ns

    instances = task.build_all_requests(limit=limit, rank=0, world_size=1)
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
