"""Build per-source pools and mixed ``sample_uniform`` batches."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from datasets import Dataset, load_dataset

from .taco import _load_taco_train_rows
from .gsm8k import (
    _gsm8k_text,
    _load_gsm8k_test_rows_for_fewshot,
    _load_gsm8k_train,
)
from .minerva_math import (
    _load_minerva_math_train,
    _minerva_math_text,
    list_minerva_math_fewshot_samples,
)
from .mcqa import (
    _format_mcqa_challenge_chat_row,
    _load_arc_challenge_test_rows_for_fewshot,
    _load_arc_challenge_train,
    _load_arc_easy_test_rows_for_fewshot,
    _load_arc_easy_train,
    _load_commonsense_qa_test_rows_for_fewshot,
    _load_commonsense_qa_train,
    _mcqa_hf_id,
)
from .pool_helpers import (
    _arc_csqa_pool_train_with_fewshot_from_test,
    _dataset_to_pool,
    _gsm8k_pool_with_fewshot,
    _index_id,
    _minerva_math_pool_with_fixed_fewshot,
    _rows_to_pool,
    _sample_rows_k,
)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    loader: Callable[[], Dataset]
    to_text: Callable[[Dict[str, Any]], str]
    id_key: Callable[[Dict[str, Any], int], str]


def _empty_dataset() -> Dataset:
    """Unused placeholder for ``taco`` in ``SourceSpec``."""
    return load_dataset("openai/gsm8k", "main", split="train[:0]")


_MCQA_LOADERS: Dict[str, Tuple[Callable[[], Any], Callable[[], List[Dict[str, Any]]]]] = {
    "arc_easy": (_load_arc_easy_train, _load_arc_easy_test_rows_for_fewshot),
    "arc_challenge": (_load_arc_challenge_train, _load_arc_challenge_test_rows_for_fewshot),
    "commonsense_qa": (_load_commonsense_qa_train, _load_commonsense_qa_test_rows_for_fewshot),
}


def _allocation_proportional(n_total: int, pool_sizes: List[int]) -> List[int]:
    """Largest-remainder split of ``n_total`` across ``pool_sizes``."""
    k = len(pool_sizes)
    if k <= 0 or n_total <= 0:
        return [0] * k
    tot = sum(pool_sizes)
    if tot == 0:
        return [0] * k
    quotas = [n_total * pool_sizes[i] / tot for i in range(k)]
    floors = [int(q) for q in quotas]
    rem = n_total - sum(floors)
    order = sorted(range(k), key=lambda i: quotas[i] - floors[i], reverse=True)
    for j in range(rem):
        floors[order[j]] += 1
    return floors


def _allocate_counts_proportional_no_replacement(
    num_samples: int,
    pool_sizes: List[int],
) -> List[int]:
    """Proportional counts per pool, capped, then redistributed to sum to ``num_samples``."""
    k = len(pool_sizes)
    if k == 0 or num_samples <= 0:
        return [0] * k
    capacity = sum(pool_sizes)
    if num_samples > capacity:
        raise RuntimeError(
            f"Requested {num_samples} samples without replacement but only {capacity} rows exist across pools."
        )
    raw = _allocation_proportional(num_samples, pool_sizes)
    counts = [min(raw[i], pool_sizes[i]) for i in range(k)]
    short = num_samples - sum(counts)
    i = 0
    while short > 0:
        if counts[i] < pool_sizes[i]:
            counts[i] += 1
            short -= 1
        i = (i + 1) % k
    return counts


def default_source_specs() -> List[SourceSpec]:
    return [
        SourceSpec("gsm8k", _load_gsm8k_train, _gsm8k_text, _index_id),
        SourceSpec("minerva_math", _load_minerva_math_train, _minerva_math_text, _index_id),
        SourceSpec("arc_easy", _load_arc_easy_train, _format_mcqa_challenge_chat_row, _mcqa_hf_id),
        SourceSpec("arc_challenge", _load_arc_challenge_train, _format_mcqa_challenge_chat_row, _mcqa_hf_id),
        SourceSpec("commonsense_qa", _load_commonsense_qa_train, _format_mcqa_challenge_chat_row, _mcqa_hf_id),
        SourceSpec("taco", _empty_dataset, _minerva_math_text, _index_id),
    ]


def load_pools(
    specs: Optional[Sequence[SourceSpec]] = None,
    *,
    gsm8k_num_fewshot: int = 4,
    gsm8k_fewshot_seed: int = 0,
    minerva_math_fixed_fewshot: bool = True,
    arc_csqa_num_fewshot: int = 5,
    arc_csqa_fewshot_seed: int = 0,
) -> Dict[str, List[Dict[str, Any]]]:
    """Load full pools: ``{id, source, text, meta}``. TACO: train rows, prompt = question + starter_code; large fields dropped from meta."""
    specs = list(specs) if specs is not None else default_source_specs()
    pools: Dict[str, List[Dict[str, Any]]] = {}
    rng_gsm8k = random.Random(gsm8k_fewshot_seed)
    rng_arc = random.Random(arc_csqa_fewshot_seed)

    for spec in specs:
        name = spec.name
        try:
            if name == "gsm8k":
                ds_train = _load_gsm8k_train()
                if gsm8k_num_fewshot > 0:
                    test_rows = _load_gsm8k_test_rows_for_fewshot()
                    pools[name] = _gsm8k_pool_with_fewshot(
                        ds_train, test_rows, rng_gsm8k, gsm8k_num_fewshot
                    )
                else:
                    pools[name] = _dataset_to_pool(ds_train, "gsm8k", _gsm8k_text, _index_id)
            elif name == "minerva_math" and minerva_math_fixed_fewshot:
                pools[name] = _minerva_math_pool_with_fixed_fewshot(
                    _load_minerva_math_train(), list_minerva_math_fewshot_samples()
                )
            elif name == "minerva_math" and not minerva_math_fixed_fewshot:
                pools[name] = _dataset_to_pool(
                    _load_minerva_math_train(), "minerva_math", _minerva_math_text, _index_id
                )
            elif name in _MCQA_LOADERS:
                load_train, load_test_fs = _MCQA_LOADERS[name]
                if arc_csqa_num_fewshot > 0:
                    pools[name] = _arc_csqa_pool_train_with_fewshot_from_test(
                        load_train(),
                        load_test_fs(),
                        rng_arc,
                        name,
                        _mcqa_hf_id,
                        arc_csqa_num_fewshot,
                    )
                else:
                    pools[name] = _dataset_to_pool(
                        load_train(),
                        name,
                        _format_mcqa_challenge_chat_row,
                        _mcqa_hf_id,
                    )
            elif name == "taco":
                taco_rows = _load_taco_train_rows()
                if not taco_rows:
                    raise RuntimeError("No TACO rows (BAAI/TACO train).")
                pools[name] = _rows_to_pool(
                    taco_rows,
                    "taco",
                    _index_id,
                    meta_extra={"python_format": "taco_direct"},
                )
            else:
                ds = spec.loader()
                pools[name] = _dataset_to_pool(ds, name, spec.to_text, spec.id_key)
        except Exception as e:
            print(f"[cst.train_data] Warning: failed to load source {name!r}: {e}")
            pools[name] = []
    return pools


def sample_uniform(
    num_samples: int,
    *,
    seed: int = 0,
    specs: Optional[Sequence[SourceSpec]] = None,
    gsm8k_num_fewshot: int = 4,
    gsm8k_fewshot_seed: Optional[int] = None,
    minerva_math_fixed_fewshot: bool = True,
    arc_csqa_num_fewshot: int = 5,
    arc_csqa_fewshot_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """``num_samples`` distinct rows: proportional split across pools, random sample per pool, then shuffle (seeded)."""
    if num_samples <= 0:
        return []
    spec_list = list(specs) if specs is not None else default_source_specs()
    rng = random.Random(seed)
    fs_seed = int(gsm8k_fewshot_seed) if gsm8k_fewshot_seed is not None else seed
    arc_seed = int(arc_csqa_fewshot_seed) if arc_csqa_fewshot_seed is not None else seed
    pools = load_pools(
        spec_list,
        gsm8k_num_fewshot=gsm8k_num_fewshot,
        gsm8k_fewshot_seed=fs_seed,
        minerva_math_fixed_fewshot=minerva_math_fixed_fewshot,
        arc_csqa_num_fewshot=arc_csqa_num_fewshot,
        arc_csqa_fewshot_seed=arc_seed,
    )
    names = [s.name for s in spec_list]
    active = [n for n in names if pools.get(n)]
    if not active:
        raise RuntimeError("No examples loaded from any source.")
    sizes = [len(pools[n]) for n in active]
    counts = _allocate_counts_proportional_no_replacement(num_samples, sizes)
    out: List[Dict[str, Any]] = []

    for name, k in zip(active, counts):
        if k == 0:
            continue
        pool = pools.get(name, [])
        if not pool:
            print(f"[cst.train_data] Warning: source {name!r} is empty; skipping its quota.")
            continue
        chosen = _sample_rows_k(pool, k, rng)
        for row in chosen:
            out.append(dict(row))

    rng.shuffle(out)
    return out


def sample_uniform_batched(
    num_samples: int,
    batch_size: int,
    *,
    seed: int = 0,
    specs: Optional[Sequence[SourceSpec]] = None,
    gsm8k_num_fewshot: int = 4,
    gsm8k_fewshot_seed: Optional[int] = None,
    minerva_math_fixed_fewshot: bool = True,
    arc_csqa_num_fewshot: int = 5,
    arc_csqa_fewshot_seed: Optional[int] = None,
) -> List[List[Dict[str, Any]]]:
    """Same draw as :func:`sample_uniform`, but split into consecutive batches of up to ``batch_size`` rows."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    flat = sample_uniform(
        num_samples,
        seed=seed,
        specs=specs,
        gsm8k_num_fewshot=gsm8k_num_fewshot,
        gsm8k_fewshot_seed=gsm8k_fewshot_seed,
        minerva_math_fixed_fewshot=minerva_math_fixed_fewshot,
        arc_csqa_num_fewshot=arc_csqa_num_fewshot,
        arc_csqa_fewshot_seed=arc_csqa_fewshot_seed,
    )
    if not flat:
        return []
    return [flat[i : i + batch_size] for i in range(0, len(flat), batch_size)]
