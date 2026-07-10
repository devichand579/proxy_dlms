"""Dataset -> pool rows ``{id, source, text, meta}``."""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional

from datasets import Dataset

from .gsm8k import _gsm8k_format_lm_eval_fewshot
from .minerva_math import _minerva_math_format_lm_eval_fixed_fewshot
from .mcqa import _format_mcqa_chat_fewshot


def _index_id(_row: Dict[str, Any], i: int) -> str:
    """Id suffix: row index as string."""
    return str(i)


def _meta_without_text(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: row[k] for k in row if k != "text"}


def _sample_rows_k(
    base: List[Dict[str, Any]],
    k: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Random ``k`` rows without replacement."""
    if k <= 0 or not base:
        return []
    return rng.sample(base, k)


def _dataset_to_pool(
    ds: Dataset,
    source: str,
    to_text: Callable[[Dict[str, Any]], str],
    id_fn: Callable[[Dict[str, Any], int], str],
) -> List[Dict[str, Any]]:
    pool: List[Dict[str, Any]] = []
    for i, row in enumerate(ds):
        row = dict(row)
        try:
            text = to_text(row)
        except Exception:
            continue
        if not text:
            continue
        eid = f"{source}:{id_fn(row, i)}"
        pool.append(
            {
                "id": eid,
                "source": source,
                "text": text,
                "meta": _meta_without_text(row),
            }
        )
    return pool


def _rows_to_pool(
    rows: List[Dict[str, Any]],
    source: str,
    id_fn: Callable[[Dict[str, Any], int], str],
    *,
    meta_extra: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Like ``_dataset_to_pool`` but rows already have ``text``."""
    pool: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        row = dict(row)
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        eid = f"{source}:{id_fn(row, i)}"
        meta = _meta_without_text(row)
        if meta_extra:
            meta.update(meta_extra)
        pool.append({"id": eid, "source": source, "text": text, "meta": meta})
    return pool


def _gsm8k_pool_with_fewshot(
    ds_train: Dataset,
    test_pool: List[Dict[str, Any]],
    rng_fewshot: random.Random,
    num_fewshot: int,
) -> List[Dict[str, Any]]:
    pool: List[Dict[str, Any]] = []
    for i, row in enumerate(ds_train):
        row = dict(row)
        try:
            text = _gsm8k_format_lm_eval_fewshot(row, test_pool, rng_fewshot, num_fewshot)
        except Exception:
            continue
        if not text:
            continue
        eid = f"gsm8k:{_index_id(row, i)}"
        meta = _meta_without_text(row)
        meta["gsm8k_num_fewshot"] = num_fewshot
        meta["gsm8k_fewshot_pool"] = "test"
        meta["gsm8k_eval_pool"] = "train"
        pool.append({"id": eid, "source": "gsm8k", "text": text, "meta": meta})
    return pool


def _minerva_math_pool_with_fixed_fewshot(
    ds: Dataset,
    fewshot: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    pool: List[Dict[str, Any]] = []
    for i, row in enumerate(ds):
        row = dict(row)
        try:
            text = _minerva_math_format_lm_eval_fixed_fewshot(row, fewshot)
        except Exception:
            continue
        if not text:
            continue
        eid = f"minerva_math:{_index_id(row, i)}"
        meta = _meta_without_text(row)
        meta["minerva_math_fewshot"] = "fixed_4"
        meta["minerva_math_dataset"] = "qwedsacf/competition_math"
        pool.append({"id": eid, "source": "minerva_math", "text": text, "meta": meta})
    return pool


def _arc_csqa_pool_train_with_fewshot_from_test(
    ds_train: Dataset,
    test_pool: List[Dict[str, Any]],
    rng: random.Random,
    source: str,
    id_fn: Callable[[Dict[str, Any], int], str],
    num_fewshot: int = 5,
) -> List[Dict[str, Any]]:
    pool: List[Dict[str, Any]] = []
    for i, row in enumerate(ds_train):
        row = dict(row)
        try:
            text = _format_mcqa_chat_fewshot(row, test_pool, rng, num_fewshot)
        except Exception:
            continue
        if not text:
            continue
        eid = f"{source}:{id_fn(row, i)}"
        meta = _meta_without_text(row)
        meta["mcqa_fewshot"] = num_fewshot
        # ARC few-shot rows come from the public test split; CSQA test has no labels.
        meta["mcqa_fewshot_pool"] = (
            "validation" if source == "commonsense_qa" else "test"
        )
        meta["mcqa_eval_pool"] = "train"
        pool.append({"id": eid, "source": source, "text": text, "meta": meta})
    return pool


