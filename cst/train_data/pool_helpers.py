"""Dataset -> pool rows ``{id, source, text, meta}``."""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, Iterable, List, Optional

from datasets import Dataset

from cst.eval_data.lmeval_task import lmeval_fewshot_text, load_lmeval_task

from .mcqa import _format_mcqa_chat_fewshot
from .minerva_math import _minerva_math_format_lm_eval_fixed_fewshot


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


def _iter_docs(docs: Iterable[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in docs:
        out.append(dict(row))
    return out


def pool_from_lmeval_task(
    source: str,
    task_name: str,
    *,
    seed: int,
    num_fewshot: Optional[int] = None,
    query_docs: Optional[Iterable[Any]] = None,
    id_fn: Callable[[Dict[str, Any], int], str] = _index_id,
) -> List[Dict[str, Any]]:
    """Build a text pool with lm-eval ``fewshot_context`` (same sampler as eval)."""
    task, ns = load_lmeval_task(task_name, num_fewshot=num_fewshot, seed=int(seed))
    if query_docs is not None:
        docs = _iter_docs(query_docs)
    elif hasattr(task, "has_training_docs") and task.has_training_docs():
        docs = _iter_docs(task.training_docs())
    else:
        raise RuntimeError(
            f"lm-eval task {task_name!r} has no training split; pass query_docs for source {source!r}"
        )
    pool: List[Dict[str, Any]] = []
    for i, doc in enumerate(docs):
        try:
            text = lmeval_fewshot_text(task, doc, ns)
        except Exception:
            continue
        eid = f"{source}:{id_fn(doc, i)}"
        meta = _meta_without_text(doc)
        meta["lmeval_task"] = str(task_name)
        meta["num_fewshot"] = int(ns)
        meta["fewshot_sampler"] = "lmeval"
        pool.append({"id": eid, "source": source, "text": text, "meta": meta})
    return pool


def _arc_csqa_pool_train_with_fewshot_from_test(
    ds_train: Dataset,
    test_pool: List[Dict[str, Any]],
    rng: random.Random,
    source: str,
    id_fn: Callable[[Dict[str, Any], int], str],
    num_fewshot: int = 5,
) -> List[Dict[str, Any]]:
    """CSQA (no matching generate-until lm-eval task): train queries + labeled few-shot pool."""
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
        meta["mcqa_fewshot_pool"] = "validation" if source == "commonsense_qa" else "test"
        meta["mcqa_eval_pool"] = "train"
        pool.append({"id": eid, "source": source, "text": text, "meta": meta})
    return pool


def _minerva_math_pool_with_fixed_fewshot(
    ds: Dataset,
    fewshot: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Fallback when lm-eval ``minerva_math500`` cannot load (missing ``lm-eval[math]``)."""
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
        meta["lmeval_task"] = "minerva_math500"
        meta["num_fewshot"] = len(fewshot)
        meta["fewshot_sampler"] = "first_n_fallback"
        pool.append({"id": eid, "source": "minerva_math", "text": text, "meta": meta})
    return pool
