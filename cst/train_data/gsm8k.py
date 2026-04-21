"""GSM8K load paths and prompt / few-shot text."""

from __future__ import annotations

import random
from typing import Any, Dict, List

from datasets import Dataset, load_dataset


def _load_gsm8k_train() -> Dataset:
    return load_dataset("openai/gsm8k", "main", split="train")


def _load_gsm8k_test_rows_for_fewshot() -> List[Dict[str, Any]]:
    """Test split rows for few-shot demos."""
    ds = load_dataset("openai/gsm8k", "main", split="test")
    return [dict(r) for r in ds]


def _gsm8k_text(row: Dict[str, Any]) -> str:
    q = str(row.get("question", "")).strip()
    return f"Question: {q}\nAnswer:"


def _gsm8k_format_lm_eval_fewshot(
    row_train: Dict[str, Any],
    test_pool: List[Dict[str, Any]],
    rng: random.Random,
    num_fewshot: int,
) -> str:
    """Train question plus optional few-shot blocks from ``test_pool``."""
    if num_fewshot <= 0 or not test_pool:
        return _gsm8k_text(row_train)
    n = min(num_fewshot, len(test_pool))
    demos = rng.sample(test_pool, n)
    parts: List[str] = []
    for d in demos:
        q = str(d.get("question", "")).strip()
        a = str(d.get("answer", "")).strip()
        parts.append(f"Question: {q}\nAnswer: {a}")
    q_main = str(row_train.get("question", "")).strip()
    parts.append(f"Question: {q_main}\nAnswer:")
    return "\n\n".join(parts)
