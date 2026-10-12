"""GSM8K load paths and 0-shot prompt text."""

from __future__ import annotations

from typing import Any, Dict

from datasets import Dataset, load_dataset


def _load_gsm8k_train() -> Dataset:
    return load_dataset("openai/gsm8k", "main", split="train")


def _gsm8k_text(row: Dict[str, Any]) -> str:
    q = str(row.get("question", "")).strip()
    return f"Question: {q}\nAnswer:"
