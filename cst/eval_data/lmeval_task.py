"""lm-eval task load + few-shot context (shared by eval store and CST train mix)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from lm_eval.tasks import TaskManager

# lm-eval tasks used by ``launch_eval`` / ``scripts/eval.sh``.
# ``store.task_name=all`` stores each into ``{whoami}/hiddens-eval`` as its own folder.
CST_EVAL_TASKS: tuple[str, ...] = (
    "gsm8k",
    "minerva_math500",
    "arc_challenge_chat",
    "humaneval",
    "mbpp",
)


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


def expand_eval_task_names(task_name: Any) -> List[str]:
    """Expand ``all`` / comma-separated names / Hydra lists into lm-eval task ids."""
    if task_name is None:
        raise ValueError("task_name is empty")
    if isinstance(task_name, str):
        parts = [p.strip() for p in task_name.split(",") if p.strip()]
    else:
        try:
            parts = [str(p).strip() for p in task_name if str(p).strip()]
        except TypeError:
            parts = [str(task_name).strip()]
        if not parts:
            parts = [str(task_name).strip()]
    if not parts:
        raise ValueError("task_name is empty")
    out: List[str] = []
    seen: set[str] = set()
    for raw in parts:
        if raw.lower() in ("all", "*"):
            extra = list(CST_EVAL_TASKS)
        else:
            extra = [raw]
        for t in extra:
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


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
