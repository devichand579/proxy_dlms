"""BAAI/TACO train → prompt rows for the ``taco`` pool."""

from __future__ import annotations

from typing import Any, Dict, List

from datasets import Dataset, load_dataset

# Skip huge columns in meta (prompt is in ``text``).
_TACO_HEAVY_KEYS = frozenset({"question", "solutions", "input_output"})

_TACO_TRAIN_PARQUET_URLS = [
    f"https://huggingface.co/datasets/BAAI/TACO/resolve/main/ALL/train-{i:05d}-of-00009.parquet"
    for i in range(9)
]


def _taco_prompt_text(question: str, starter_code: str) -> str:
    q = question.strip()
    st = starter_code.strip()
    if st:
        return f"{q}\n\n{st}"
    return q


def _load_taco_train_from_parquet() -> Dataset:
    raw = load_dataset("parquet", data_files={"train": _TACO_TRAIN_PARQUET_URLS})
    return raw["train"]


def _load_taco_train_from_hub() -> Dataset:
    """Load TACO train from Hub (try no config, then ``ALL``)."""
    attempts = (
        lambda: load_dataset("BAAI/TACO", split="train"),
        lambda: load_dataset("BAAI/TACO", name="ALL", split="train"),
        lambda: load_dataset("BAAI/TACO", "ALL", split="train"),
    )
    last: Exception | None = None
    for fn in attempts:
        try:
            return fn()
        except Exception as e:
            last = e
            continue
    assert last is not None
    raise last


def _load_taco_train() -> Dataset:
    """TACO train: Parquet shards first, else Hub."""
    last_err: Exception | None = None
    for loader in (_load_taco_train_from_parquet, _load_taco_train_from_hub):
        try:
            ds = loader()
            if len(ds) > 0:
                return ds
        except Exception as e:
            last_err = e
    if last_err is not None:
        raise last_err
    raise RuntimeError("BAAI/TACO train: all loaders returned 0 rows")


def _load_taco_train_rows() -> List[Dict[str, Any]]:
    """Rows with ``text`` and remaining columns for meta (minus heavy keys)."""
    try:
        ds = _load_taco_train()
    except Exception as e:
        print(f"[cst.train_data] BAAI/TACO load failed: {e}")
        return []

    if len(ds) == 0:
        print("[cst.train_data] BAAI/TACO train split loaded with 0 rows (check config / cache).")
        return []

    out: List[Dict[str, Any]] = []
    n = len(ds)
    for i in range(n):
        row = dict(ds[i])
        q = str(row.get("question", "") or "").strip()
        st = str(row.get("starter_code", "") or "").strip()
        text = _taco_prompt_text(q, st)
        if not text:
            continue
        slim: Dict[str, Any] = {"text": text}
        for k, v in row.items():
            if k in _TACO_HEAVY_KEYS:
                continue
            slim[k] = v
        out.append(slim)
    if not out and n > 0:
        print(
            "[cst.train_data] BAAI/TACO: train has rows but none produced text "
            "(check ``question`` / ``starter_code`` columns)."
        )
    return out
