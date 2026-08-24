"""lm-eval task load + few-shot context (shared by eval store and CST train mix)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from lm_eval.tasks import TaskManager


def eval_num_fewshot(task_name: str) -> int:
    """Per-task few-shot used by ``launch_eval`` / ``config/lmeval.yaml`` (null = these defaults)."""
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


def resolve_lmeval_task(task_dict: Dict[str, Any], task_name: str):
    if task_name in task_dict:
        return task_dict[task_name]
    if len(task_dict) == 1:
        return next(iter(task_dict.values()))
    keys = ", ".join(sorted(task_dict.keys()))
    raise ValueError(f"lm-eval task {task_name!r} not found (available: {keys})")


def load_lmeval_task(
    task_name: str,
    *,
    num_fewshot: Optional[int] = None,
    seed: int = 0,
):
    """Load a harness task and set few-shot count + sampler seed (same as eval store)."""
    ns = int(num_fewshot) if num_fewshot is not None else eval_num_fewshot(task_name)
    tm = TaskManager()
    task_dict = tm.load_task_or_group(str(task_name).strip())
    task = resolve_lmeval_task(task_dict, str(task_name).strip())
    if hasattr(task, "set_fewshot_seed"):
        task.set_fewshot_seed(int(seed))
    if hasattr(task, "config") and hasattr(task.config, "num_fewshot"):
        task.config.num_fewshot = ns
    return task, ns


def lmeval_fewshot_text(task, doc: Dict[str, Any], num_fewshot: int) -> str:
    """Prompt string from the harness sampler (same kwargs as ``build_all_requests``)."""
    gen_prefix = task.doc_to_prefix(doc) if hasattr(task, "doc_to_prefix") else None
    ctx = task.fewshot_context(doc, int(num_fewshot), gen_prefix=gen_prefix)
    if isinstance(ctx, list):
        if not ctx:
            raise ValueError("fewshot_context returned an empty list")
        ctx = ctx[0]
    text = str(ctx)
    if not text.strip():
        raise ValueError("fewshot_context returned empty text")
    return text
