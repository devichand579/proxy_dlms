"""Match trace samples to lm-eval docs via decoded prompt text (few-shot safe)."""

from __future__ import annotations

import re
from typing import Any, Optional

_MBPP_TASK_RE = re.compile(r"here is your task:\s*", re.IGNORECASE)
_MBPP_TESTS_RE = re.compile(r"your code should pass these tests:", re.IGNORECASE)
_QUESTION_RE = re.compile(r"Question:\s*")
_GSM8K_ANSWER_RE = re.compile(r"\nAnswer:\s*")
_ARC_CHOICE_RE = re.compile(r"\nA\.\s")


def decode_prompt_text(sample: dict, tokenizer) -> Optional[str]:
    """Decode the eval prompt stored in a trace sample."""
    text: Optional[str] = None
    prompt = sample.get("prompt")
    if prompt is not None:
        ids = prompt[0] if isinstance(prompt[0], list) else prompt
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is not None:
            ids = [int(t) for t in ids if int(t) != int(pad_id)]
        else:
            ids = [int(t) for t in ids]
        if ids:
            text = tokenizer.decode(ids, skip_special_tokens=True)

    if text is None:
        final = sample.get("final_sequence")
        prompt_len = int(sample.get("prompt_len") or 0)
        if final and prompt_len > 0:
            seq = final[0] if final and isinstance(final[0], list) else final
            if seq:
                text = tokenizer.decode(list(seq[:prompt_len]), skip_special_tokens=True)
    return _normalize_prompt_text(text) if text else None


def _normalize_key(text: str) -> str:
    return " ".join(str(text).strip().split())


def doc_lookup_key(task_name: str, doc: dict[str, Any]) -> str:
    """Stable lookup key from an lm-eval document row."""
    t = str(task_name).strip().lower()
    if "mbpp" in t:
        return _normalize_key(doc.get("text", ""))
    if "humaneval" in t:
        return _normalize_key(doc.get("prompt", ""))
    if "gsm8k" in t:
        return _normalize_key(doc.get("question", ""))
    if "arc_challenge" in t:
        return _normalize_key(doc.get("question", ""))
    if "math500" in t or "minerva" in t:
        return _normalize_key(doc.get("problem", doc.get("question", "")))
    return ""


def _normalize_prompt_text(text: str) -> str:
    """lm-eval chat templates often store literal ``\\n`` in decoded prompts."""
    return str(text).replace("\\n", "\n")


def extract_doc_key_from_prompt(task_name: str, prompt_text: str) -> Optional[str]:
    """Extract the current-sample doc key from a (possibly few-shot) prompt."""
    t = str(task_name).strip().lower()
    if not prompt_text:
        return None
    prompt_text = _normalize_prompt_text(prompt_text)

    if "mbpp" in t:
        return _extract_mbpp_problem(prompt_text)
    if "humaneval" in t:
        return _extract_humaneval_prompt(prompt_text)
    if "gsm8k" in t:
        return _extract_gsm8k_question(prompt_text)
    if "arc_challenge" in t:
        return _extract_arc_question(prompt_text)
    if "math500" in t or "minerva" in t:
        return _extract_last_question(prompt_text)
    return None


def _extract_mbpp_problem(prompt_text: str) -> Optional[str]:
    matches = list(_MBPP_TASK_RE.finditer(prompt_text))
    if not matches:
        return None
    start = matches[-1].end()
    rest = prompt_text[start:]
    test_match = _MBPP_TESTS_RE.search(rest)
    problem = rest[: test_match.start()] if test_match else rest
    key = _normalize_key(problem)
    return key or None


def _extract_humaneval_prompt(prompt_text: str) -> Optional[str]:
    # 0-shot humaneval prompts are the full function header; few-shot repeats blocks.
    if "def " not in prompt_text:
        return _normalize_key(prompt_text) or None
    idx = prompt_text.rfind("def ")
    return _normalize_key(prompt_text[idx:]) or None


def _extract_gsm8k_question(prompt_text: str) -> Optional[str]:
    matches = list(_QUESTION_RE.finditer(prompt_text))
    if not matches:
        return None
    start = matches[-1].end()
    rest = prompt_text[start:]
    answer_match = _GSM8K_ANSWER_RE.search(rest)
    question = rest[: answer_match.start()] if answer_match else rest
    key = _normalize_key(question)
    return key or None


def _extract_arc_question(prompt_text: str) -> Optional[str]:
    matches = list(_QUESTION_RE.finditer(prompt_text))
    if not matches:
        return None
    start = matches[-1].end()
    rest = prompt_text[start:]
    choice_match = _ARC_CHOICE_RE.search(rest)
    question = rest[: choice_match.start()] if choice_match else rest.split("\nYour response")[0]
    key = _normalize_key(question)
    return key or None


def _extract_last_question(prompt_text: str) -> Optional[str]:
    matches = list(_QUESTION_RE.finditer(prompt_text))
    if not matches:
        return _normalize_key(prompt_text) or None
    start = matches[-1].end()
    rest = prompt_text[start:]
    for sep in ("\nAnswer:", "\nSolution:", "\n\n"):
        pos = rest.find(sep)
        if pos >= 0:
            rest = rest[:pos]
            break
    key = _normalize_key(rest)
    return key or None


def _prompt_token_key(token_ids: list[int], pad_id: Optional[int]) -> tuple[int, ...]:
    if pad_id is None:
        return tuple(int(t) for t in token_ids)
    return tuple(int(t) for t in token_ids if int(t) != int(pad_id))


def trace_prompt_token_key(sample: dict, tokenizer) -> Optional[tuple[int, ...]]:
    prompt = sample.get("prompt")
    if prompt is None:
        return None
    ids = prompt[0] if isinstance(prompt[0], list) else prompt
    return _prompt_token_key(list(ids), tokenizer.pad_token_id)


def build_doc_index(task, task_name: str) -> dict[str, Any]:
    """Map normalized doc keys to lm-eval instances (first wins on collision)."""
    index: dict[str, Any] = {}
    for inst in task.instances:
        doc = getattr(inst, "doc", None) or {}
        key = doc_lookup_key(task_name, doc)
        if key and key not in index:
            index[key] = inst
    return index


def match_trace_to_instance(
    sample: dict,
    *,
    task_name: str,
    tokenizer,
    doc_index: dict[str, Any],
    token_index: Optional[dict[tuple[int, ...], Any]] = None,
) -> tuple[Any | None, str]:
    """
    Match a trace sample to an lm-eval instance.

    Returns (instance, method) where method is 'doc_key', 'token_key', or ''.
    """
    prompt_text = decode_prompt_text(sample, tokenizer)
    if prompt_text:
        doc_key = extract_doc_key_from_prompt(task_name, prompt_text)
        if doc_key and doc_key in doc_index:
            return doc_index[doc_key], "doc_key"

    if token_index is not None:
        token_key = trace_prompt_token_key(sample, tokenizer)
        if token_key is not None and token_key in token_index:
            return token_index[token_key], "token_key"

    return None, ""
