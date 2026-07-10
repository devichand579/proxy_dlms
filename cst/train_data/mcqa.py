"""ARC / CommonsenseQA loaders and MCQA prompt strings."""

from __future__ import annotations

import random
from typing import Any, Dict, List

from datasets import Dataset, load_dataset


def _one_of_letters(letters: List[str]) -> str:
    """Format option letters for the instruction line."""
    if not letters:
        return ""
    if len(letters) == 1:
        return letters[0]
    if len(letters) == 2:
        return f"{letters[0]} or {letters[1]}"
    return ", ".join(letters[:-1]) + f" or {letters[-1]}"


def _mcqa_answer_letter(row: Dict[str, Any]) -> str:
    """Correct answer letter from ``answerKey`` / ``choices``."""
    ch = row["choices"]
    labels = ch["label"]
    texts = ch["text"]
    n = len(texts)
    ak = row.get("answerKey")
    if isinstance(ak, int):
        if 0 <= ak < len(labels):
            return str(labels[ak]).strip()
        if 0 <= ak < n:
            return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[ak]
        return ""
    s = str(ak).strip() if ak is not None else ""
    if s.isdigit():
        i = int(s)
        # ARC 4-way: answerKey is usually 1–4 (1-based).
        if n == 4 and 1 <= i <= 4:
            return "ABCD"[i - 1]
        # CommonsenseQA and others: 0-based index 0 … n-1 (e.g. answerKey "0" -> A).
        if 0 <= i < n:
            if i < len(labels):
                lab = str(labels[i]).strip()
                if len(lab) == 1 and lab.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    return lab.upper()
            return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i]
        j = i - 1
        if 0 <= j < len(labels):
            return str(labels[j]).strip()
        if 0 <= j < n:
            return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[j]
    if len(s) == 1 and s.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        return s.upper()
    return s


def _format_mcqa_challenge_chat_row(row: Dict[str, Any]) -> str:
    """ARC/CSQA single-turn prompt (lm-eval ``doc_to_text`` style)."""
    q = str(row.get("question", "")).strip()
    texts = row["choices"]["text"]
    n = len(texts)
    letters = [chr(ord("A") + i) for i in range(n)]

    if n == 4:
        head = (
            "Given the following question and four candidate answers (A, B, C and D), "
            "choose the best answer."
        )
    elif n == 5:
        head = (
            "Given the following question and five candidate answers (A, B, C, D, and E), "
            "choose the best answer."
        )
    elif n == 3:
        head = (
            "Given the following question and three candidate answers (A, B, and C), "
            "choose the best answer."
        )
    else:
        opt_list = _one_of_letters(letters)
        head = (
            f"Given the following question and {n} candidate answers ({opt_list}), "
            "choose the best answer."
        )

    lines = [head, f"Question: {q}"]
    for i, t in enumerate(texts):
        lines.append(f"{letters[i]}. {str(t).strip()}")
    allowed = _one_of_letters(letters)
    lines.append(
        'Your response should end with "The best answer is [the_answer_letter]" '
        f"where the [the_answer_letter] is one of {allowed}."
    )
    return "\n".join(lines)


def _format_mcqa_chat_fewshot_demo(row: Dict[str, Any]) -> str:
    """One few-shot line: prompt + ``The best answer is <letter>``."""
    letter = _mcqa_answer_letter(row)
    return _format_mcqa_challenge_chat_row(row) + "\nThe best answer is " + letter


def _format_mcqa_chat_fewshot(
    row_train: Dict[str, Any],
    fewshot_pool: List[Dict[str, Any]],
    rng: random.Random,
    num_fewshot: int = 5,
) -> str:
    """Few-shot blocks from ``fewshot_pool``, then train question."""
    n = min(num_fewshot, len(fewshot_pool))
    demos = rng.sample(fewshot_pool, n) if n else []
    parts = [_format_mcqa_chat_fewshot_demo(dict(d)) for d in demos]
    parts.append(_format_mcqa_challenge_chat_row(dict(row_train)))
    return "\n\n".join(parts)


def _load_arc_easy_train() -> Dataset:
    return load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")


def _load_arc_easy_test_rows_for_fewshot() -> List[Dict[str, Any]]:
    """ARC-Easy test rows for few-shot."""
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    return [dict(r) for r in ds]


def _load_arc_challenge_train() -> Dataset:
    return load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")


def _load_arc_challenge_test_rows_for_fewshot() -> List[Dict[str, Any]]:
    """ARC-Challenge test rows for few-shot."""
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    return [dict(r) for r in ds]


def _load_commonsense_qa_train() -> Dataset:
    return load_dataset("tau/commonsense_qa", split="train")


def _load_commonsense_qa_test_rows_for_fewshot() -> List[Dict[str, Any]]:
    """Rows with gold labels for CommonsenseQA few-shot.

    The Hugging Face ``test`` split has empty ``answerKey`` (labels withheld for
    the benchmark). Use ``validation`` so ``answerKey`` (A–E) is present.
    """
    ds = load_dataset("tau/commonsense_qa", split="validation")
    return [dict(r) for r in ds]


def _mcqa_hf_id(row: Dict[str, Any], i: int) -> str:
    """``id`` field or fallback index."""
    return str(row.get("id", i))
