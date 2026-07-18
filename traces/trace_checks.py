"""Trace validation, repair (mask-diff / phantoms / trim), and NPZ round-trip checks."""

from __future__ import annotations

import copy
import json
import logging
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Optional

import numpy as np

from traces.trace_npz import (
    iter_jsonl_trace_samples,
    iter_npz_traces,
    load_trace_samples,
    read_jsonl_sample_at_index,
)
from traces.utils import as_gen_vec, is_fdm_trace_name, squeeze_len, squeeze_to_1d

logger = logging.getLogger(__name__)

STEP_FIELDS_REQUIRED = ("mask_index", "predictions", "confidences", "entropies")
STEP_FIELDS_OPTIONAL = ("selected_positions", "selected_tokens")
# Survives NPZ via sample metadata sidecar; set after repair drops the trailing step.
TRIMMED_TAIL_KEY = "schedulerlens_trimmed_tail"


def sample_tail_trimmed(sample: dict) -> bool:
    if bool(sample.get(TRIMMED_TAIL_KEY)):
        return True
    meta = sample.get("metadata") or {}
    return bool(meta.get(TRIMMED_TAIL_KEY))


def mark_sample_tail_trimmed(sample: dict) -> None:
    sample[TRIMMED_TAIL_KEY] = True
    meta = dict(sample.get("metadata") or {})
    meta[TRIMMED_TAIL_KEY] = True
    sample["metadata"] = meta


@dataclass
class SampleCheck:
    index: int
    n_steps: int
    gen_length: int
    prompt_len: int
    steps_missing_required: int = 0
    steps_null_optional: int = 0
    steps_with_selected: int = 0
    inferred_unmasks: int = 0
    selection_mask_mismatches: int = 0
    phantom_extra_steps: int = 0
    tail_trimmed: bool = False
    proxy_ready: bool = False
    inferrable: bool = False
    has_final_sequence: bool = False
    issues: list[str] = field(default_factory=list)


@dataclass
class TraceCheck:
    path: str
    format: str
    n_samples: int
    samples_checked: int
    samples_proxy_ready: int
    samples_inferrable: int
    samples_no_transfer: int
    samples_with_any_selected: int
    samples_with_all_selected: int
    samples_missing_required: int
    size_bytes: int
    unique_prompts: int = 0
    duplicate_traces: int = 0
    removable_duplicates: int = 0
    ambiguous_duplicate_prompts: int = 0
    any_missing_unmasking_order: bool = False
    samples_selection_mask_mismatch: int = 0
    samples_phantom_steps: int = 0
    samples_untrimmed_tail: int = 0
    first_metadata: Optional[dict] = None
    pcfg_sample_indices: Optional[list[int]] = None
    issues: list[str] = field(default_factory=list)
    sample_checks: list[SampleCheck] = field(default_factory=list)
    repairs_applied: list[str] = field(default_factory=list)

    @property
    def transfer_status(self) -> str:
        if self.samples_checked == 0:
            return "empty"
        if self.samples_with_all_selected == self.samples_checked:
            return "explicit_all"
        if self.samples_with_any_selected > 0:
            return "explicit_partial"
        if self.samples_inferrable == self.samples_checked:
            return "inferrable"
        if self.samples_proxy_ready > 0:
            return "mixed"
        return "missing_transfer"

    @property
    def hooks_ok(self) -> bool:
        return self.transfer_status in ("explicit_all", "explicit_partial", "inferrable")

    @property
    def any_selection_mask_mismatch(self) -> bool:
        return self.samples_selection_mask_mismatch > 0

    @property
    def any_phantom_steps(self) -> bool:
        return self.samples_phantom_steps > 0

    @property
    def any_untrimmed_tail(self) -> bool:
        return self.samples_untrimmed_tail > 0


# Back-compat aliases for internal call sites / tests.
_squeeze_len = squeeze_len
_gen_vec = as_gen_vec


def mask_diff_selection(steps: list[dict], t: int, *, gen_length: int) -> Optional[np.ndarray]:
    """Boolean length-G mask of positions unmasked between step t and t+1."""
    if t < 0 or t + 1 >= len(steps) or gen_length <= 0:
        return None
    mask0 = as_gen_vec(steps[t].get("mask_index"), gen_length=gen_length, dtype=np.bool_)
    mask1 = as_gen_vec(steps[t + 1].get("mask_index"), gen_length=gen_length, dtype=np.bool_)
    if mask0 is None or mask1 is None:
        return None
    return mask0 & ~mask1


def _unmasked_frozenset(step: dict, *, gen_length: int) -> Optional[frozenset[int]]:
    mask = as_gen_vec(step.get("mask_index"), gen_length=gen_length, dtype=np.bool_)
    if mask is None:
        return None
    return frozenset(int(p) for p in np.where(~mask)[0].tolist())


def permanent_still_sets(steps: list[dict], *, gen_length: int) -> list[Optional[frozenset[int]]]:
    """``still[t]`` = positions unmasked at t and at every later step (permanent from t)."""
    us = [_unmasked_frozenset(st, gen_length=gen_length) for st in steps]
    t_max = len(us)
    if t_max == 0:
        return []
    still: list[Optional[frozenset[int]]] = [None] * t_max
    still[-1] = us[-1] if us[-1] is not None else frozenset()
    for t in range(t_max - 2, -1, -1):
        cur = us[t] if us[t] is not None else frozenset()
        nxt = still[t + 1] if still[t + 1] is not None else frozenset()
        still[t] = cur & nxt
    return still


def count_phantom_extra_steps(
    steps: list[dict],
    *,
    gen_length: int,
    is_fdm: bool = False,
) -> int:
    """Extra traced steps beyond permanent-unmask plateaus (FDM only).

    Non-FDM traces can have genuine no-unmask steps that share a ``still`` value;
    those are not counted as phantoms.
    """
    if not is_fdm or gen_length <= 0 or len(steps) < 2:
        return 0
    still = permanent_still_sets(steps, gen_length=gen_length)
    if not still:
        return 0
    n_plateaus = 1 + sum(1 for t in range(1, len(still)) if still[t] != still[t - 1])
    return max(0, len(steps) - n_plateaus)


def count_inferred_unmasks(steps: list[dict]) -> int:
    total = 0
    gen_length = 0
    for st in steps:
        ln = _squeeze_len(st.get("mask_index"))
        if ln:
            gen_length = ln
            break
    for t in range(len(steps) - 1):
        sel = mask_diff_selection(steps, t, gen_length=gen_length) if gen_length else None
        if sel is None:
            continue
        total += int(np.sum(sel))
    return total


def count_selection_mask_mismatches(steps: list[dict], *, gen_length: int) -> list[str]:
    """Steps where selected_positions count disagrees with the mask-diff unmask count.

    Only steps with a next mask are checked. The final step is dropped during repair
    (no mask_diff), so it is not flagged here.
    """
    issues: list[str] = []
    if gen_length <= 0:
        return issues
    for t in range(len(steps) - 1):
        diff = mask_diff_selection(steps, t, gen_length=gen_length)
        if diff is None:
            continue
        n_diff = int(np.sum(diff))
        sel = as_gen_vec(
            steps[t].get("selected_positions"), gen_length=gen_length, dtype=np.bool_
        )
        if sel is None:
            if n_diff > 0:
                step_id = steps[t].get("step", t)
                issues.append(
                    f"step {step_id}: missing selected_positions but mask_diff {n_diff} "
                    "(repairable)"
                )
            continue
        n_sel = int(np.sum(sel))
        if n_sel != n_diff:
            step_id = steps[t].get("step", t)
            issues.append(
                f"step {step_id}: selected_count {n_sel} != mask_diff {n_diff} (repairable)"
            )
    return issues


def prompt_key(sample: dict) -> Optional[tuple[int, ...]]:
    """Stable key for matching duplicate traces (same eval instance)."""
    prompt = sample.get("prompt")
    if prompt is None:
        return None
    ids = prompt[0] if isinstance(prompt[0], list) else prompt
    return tuple(int(t) for t in ids)


def _step_field_equal(a, b, *, field: str) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    # JSONL may store G or (1, G); NPZ unpack returns (1, G) ndarrays.
    aa = squeeze_to_1d(a)
    bb = squeeze_to_1d(b)
    if field in ("confidences", "entropies"):
        return bool(np.allclose(aa, bb, rtol=1e-5, atol=1e-5, equal_nan=True))
    return bool(np.array_equal(aa, bb))


def duplicate_prompt_stats_from_counter(counts: Counter) -> dict[str, int]:
    unique = len(counts)
    same_prompt_extra = sum(c - 1 for c in counts.values() if c > 1)
    return {
        "unique_prompts": unique,
        "duplicate_traces": max(0, sum(counts.values()) - unique),
        "same_prompt_extra": same_prompt_extra,
        "removable_duplicates": same_prompt_extra,
        "ambiguous_duplicate_prompts": 0,
        "max_copies": max(counts.values()) if counts else 0,
    }


def duplicate_prompt_stats(samples: list[dict]) -> dict[str, int]:
    """Count duplicate traces: extra lines sharing the same prompt token ids."""
    counts = Counter(key for key in (prompt_key(s) for s in samples) if key is not None)
    stats = duplicate_prompt_stats_from_counter(counts)
    stats["duplicate_traces"] = max(0, len(samples) - stats["unique_prompts"])
    return stats


def _deduped_pcfg_sample_indices(samples: Iterable[dict]) -> Optional[list[int]]:
    """First-occurrence ``metadata.sample_idx`` per prompt; None if any kept sample lacks it."""
    seen: set[tuple[int, ...]] = set()
    indices: list[int] = []
    for sample in samples:
        key = prompt_key(sample)
        if key is not None and key in seen:
            continue
        if key is not None:
            seen.add(key)
        meta = sample.get("metadata") or {}
        if "sample_idx" not in meta:
            return None
        try:
            indices.append(int(meta["sample_idx"]))
        except (TypeError, ValueError):
            return None
    return indices


def _build_trace_check(
    *,
    path: str,
    fmt: str,
    n_samples: int,
    size_bytes: int,
    sample_checks: list[SampleCheck],
    dup: dict[str, int],
    any_missing_unmasking_order: bool,
    first_metadata: Optional[dict] = None,
    pcfg_sample_indices: Optional[list[int]] = None,
) -> TraceCheck:
    issues: list[str] = []
    if n_samples == 0:
        issues.append("no samples")
    if dup["removable_duplicates"] > 0:
        issues.append(
            f"duplicate prompts: {dup['removable_duplicates']} removable "
            f"({n_samples} lines, {dup['unique_prompts']} unique prompts)"
        )

    proxy_ready = sum(1 for s in sample_checks if s.proxy_ready)
    inferrable = sum(1 for s in sample_checks if s.inferrable)
    no_transfer = sum(
        1 for s in sample_checks if not s.proxy_ready and not s.inferrable and s.n_steps > 0
    )
    any_sel = sum(1 for s in sample_checks if s.steps_with_selected > 0)
    all_sel = sum(
        1
        for s in sample_checks
        if s.n_steps > 0 and s.steps_with_selected == s.n_steps
    )
    missing_req = sum(1 for s in sample_checks if s.steps_missing_required > 0)
    sel_mismatch = sum(1 for s in sample_checks if s.selection_mask_mismatches > 0)
    phantom = sum(1 for s in sample_checks if s.phantom_extra_steps > 0)
    untrimmed = sum(1 for s in sample_checks if not s.tail_trimmed)
    if sel_mismatch:
        issues.append(
            f"selection/mask-diff count mismatch on {sel_mismatch} checked sample(s) (repairable)"
        )
    if phantom:
        issues.append(
            f"phantom extra steps on {phantom} checked sample(s) (repairable; e.g. FDM hyp forwards)"
        )
    if untrimmed:
        issues.append(
            f"untrimmed trailing step on {untrimmed} checked sample(s) "
            "(repairable; all schedulers)"
        )

    return TraceCheck(
        path=path,
        format=fmt,
        n_samples=n_samples,
        samples_checked=len(sample_checks),
        samples_proxy_ready=proxy_ready,
        samples_inferrable=inferrable,
        samples_no_transfer=no_transfer,
        samples_with_any_selected=any_sel,
        samples_with_all_selected=all_sel,
        samples_missing_required=missing_req,
        size_bytes=size_bytes,
        unique_prompts=dup["unique_prompts"],
        duplicate_traces=dup["duplicate_traces"],
        removable_duplicates=dup["removable_duplicates"],
        ambiguous_duplicate_prompts=dup["ambiguous_duplicate_prompts"],
        any_missing_unmasking_order=any_missing_unmasking_order,
        samples_selection_mask_mismatch=sel_mismatch,
        samples_phantom_steps=phantom,
        samples_untrimmed_tail=untrimmed,
        first_metadata=first_metadata,
        pcfg_sample_indices=pcfg_sample_indices,
        issues=issues,
        sample_checks=sample_checks,
    )


def deduplicate_prompt_traces(samples: list[dict]) -> tuple[list[dict], list[str]]:
    """Drop duplicate traces with the same prompt (keep first occurrence)."""
    repairs: list[str] = []
    if not samples:
        return [], repairs

    seen: set[tuple[int, ...]] = set()
    kept: list[dict] = []
    removed = 0
    for sample in samples:
        key = prompt_key(sample)
        if key is not None and key in seen:
            removed += 1
            continue
        if key is not None:
            seen.add(key)
        kept.append(sample)

    if removed:
        repairs.append(f"removed {removed} duplicate trace(s) by prompt ({len(kept)} kept)")
        return kept, repairs
    return list(samples), repairs


def infer_unmasking_order(steps: list[dict], *, gen_length: int) -> dict[int, int]:
    """Gen index -> first unmask step, from selected_positions or mask_index diffs."""
    order: dict[int, int] = {}
    for t, step in enumerate(steps):
        arr = _gen_vec(step.get("selected_positions"), gen_length=gen_length, dtype=np.bool_)
        if arr is None:
            continue
        for pos in np.nonzero(arr)[0].tolist():
            p = int(pos)
            if p not in order:
                order[p] = t

    if order:
        return order

    for t in range(len(steps) - 1):
        sel = mask_diff_selection(steps, t, gen_length=gen_length)
        if sel is None:
            continue
        for pos in np.nonzero(sel)[0].tolist():
            p = int(pos)
            if p not in order:
                order[p] = t
    return order


def _maybe_repair_unmasking_order(
    sample: dict,
    steps: list[dict],
    gen_length: int,
    repairs: list[str],
) -> None:
    inferred = infer_unmasking_order(steps, gen_length=gen_length)
    if not inferred:
        return

    existing = {int(k): int(v) for k, v in (sample.get("unmasking_order") or {}).items()}
    if not existing:
        sample["unmasking_order"] = inferred
        repairs.append(f"inferred unmasking_order for {len(inferred)} positions")
        return

    missing = {pos: step for pos, step in inferred.items() if pos not in existing}
    if missing:
        sample["unmasking_order"] = {**existing, **missing}
        repairs.append(f"filled {len(missing)} missing unmasking_order entries")


def check_sample(
    trace_dict: dict,
    index: int = 0,
    *,
    is_fdm: bool = False,
) -> SampleCheck:
    issues: list[str] = []
    steps = trace_dict.get("step_traces") or []
    gen_length = int(trace_dict.get("gen_length") or 0)
    prompt_len = int(trace_dict.get("prompt_len") or 0)

    if not steps:
        issues.append("no step_traces")
    if gen_length <= 0:
        issues.append("missing gen_length")

    steps_missing = 0
    steps_null_opt = 0
    steps_with_sel = 0

    for st in steps:
        missing = [f for f in STEP_FIELDS_REQUIRED if st.get(f) is None]
        if missing:
            steps_missing += 1
        if st.get("selected_positions") is None:
            steps_null_opt += 1
        else:
            steps_with_sel += 1
        for fname in STEP_FIELDS_REQUIRED:
            val = st.get(fname)
            if val is None:
                continue
            ln = _squeeze_len(val)
            if ln is not None and gen_length > 0 and ln != gen_length:
                issues.append(
                    f"step {st.get('step', '?')}: {fname} len {ln} != gen_length {gen_length}"
                )
                break

    inferred = count_inferred_unmasks(steps)
    mismatch_issues = count_selection_mask_mismatches(steps, gen_length=gen_length)
    issues.extend(mismatch_issues)
    phantom_extra = count_phantom_extra_steps(
        steps, gen_length=gen_length, is_fdm=is_fdm
    )
    if phantom_extra:
        issues.append(
            f"{phantom_extra} FDM phantom extra steps vs permanent-unmask plateaus "
            f"(repairable; keep first of each plateau)"
        )
    tail_trimmed = sample_tail_trimmed(trace_dict)
    if steps and not tail_trimmed:
        issues.append(
            "trailing step not trimmed (repairable; drop last step for all schedulers)"
        )
    proxy_ready = (
        steps_with_sel > 0
        and steps_missing == 0
        and not mismatch_issues
        and phantom_extra == 0
        and tail_trimmed
    )
    inferrable = (
        steps_with_sel == 0
        and steps_missing == 0
        and inferred > 0
        and gen_length > 0
    )

    if steps_missing:
        issues.append(f"{steps_missing} steps missing required fields")
    if steps_with_sel == 0 and inferred == 0 and steps:
        issues.append("no transfer signal")
    if steps and not trace_dict.get("unmasking_order"):
        if inferred > 0 or steps_with_sel > 0:
            issues.append("missing unmasking_order (repairable)")

    return SampleCheck(
        index=index,
        n_steps=len(steps),
        gen_length=gen_length,
        prompt_len=prompt_len,
        steps_missing_required=steps_missing,
        steps_null_optional=steps_null_opt,
        steps_with_selected=steps_with_sel,
        inferred_unmasks=inferred,
        selection_mask_mismatches=len(mismatch_issues),
        phantom_extra_steps=phantom_extra,
        tail_trimmed=tail_trimmed,
        proxy_ready=proxy_ready,
        inferrable=inferrable,
        has_final_sequence=trace_dict.get("final_sequence") is not None,
        issues=issues,
    )


def check_trace_samples(
    samples: list[dict],
    *,
    path: str = "",
    fmt: str = "memory",
    size_bytes: int = 0,
    max_samples: Optional[int] = None,
    check_last: bool = True,
    collect_pcfg_audit: bool = False,
    is_fdm: Optional[bool] = None,
) -> TraceCheck:
    n = len(samples)
    fdm = is_fdm_trace_name(path) if is_fdm is None else bool(is_fdm)

    indices: list[int] = list(range(n)) if max_samples is None else list(range(min(max_samples, n)))
    if check_last and n > 0 and (n - 1) not in indices:
        indices.append(n - 1)
    indices = sorted(set(indices))

    sample_checks = [
        check_sample(samples[i], index=i, is_fdm=fdm) for i in indices
    ]
    dup = duplicate_prompt_stats(samples)
    first_metadata = (samples[0].get("metadata") or {}) if samples else None
    pcfg_indices = _deduped_pcfg_sample_indices(samples) if collect_pcfg_audit and samples else None

    return _build_trace_check(
        path=path,
        fmt=fmt,
        n_samples=n,
        size_bytes=size_bytes,
        sample_checks=sample_checks,
        dup=dup,
        any_missing_unmasking_order=any(
            sample.get("step_traces") and not sample.get("unmasking_order") for sample in samples
        ),
        first_metadata=first_metadata,
        pcfg_sample_indices=pcfg_indices,
    )


def _stream_trace_check(
    samples: Iterable[dict],
    *,
    max_samples: Optional[int],
    collect_pcfg_audit: bool,
    is_fdm: bool = False,
) -> tuple[
    int,
    Optional[int],
    list[SampleCheck],
    Counter,
    bool,
    Optional[dict],
    Optional[list[int]],
    Optional[dict],
]:
    prompt_counts: Counter = Counter()
    sample_checks: list[SampleCheck] = []
    indices_to_check: set[int] = set()
    if max_samples is not None and max_samples > 0:
        indices_to_check.update(range(max_samples))

    n_samples = 0
    any_missing_unmasking_order = False
    last_index: Optional[int] = None
    last_sample: Optional[dict] = None
    first_metadata: Optional[dict] = None
    pcfg_indices: Optional[list[int]] = [] if collect_pcfg_audit else None
    seen_prompts: set[tuple[int, ...]] = set()

    for sample in samples:
        if first_metadata is None:
            first_metadata = dict(sample.get("metadata") or {})
        key = prompt_key(sample)
        if key is not None:
            prompt_counts[key] += 1
        if sample.get("step_traces") and not sample.get("unmasking_order"):
            any_missing_unmasking_order = True
        if n_samples in indices_to_check:
            sample_checks.append(
                check_sample(sample, index=n_samples, is_fdm=is_fdm)
            )
        if collect_pcfg_audit and pcfg_indices is not None:
            if key is not None and key in seen_prompts:
                pass
            else:
                if key is not None:
                    seen_prompts.add(key)
                meta = sample.get("metadata") or {}
                if "sample_idx" not in meta:
                    pcfg_indices = None
                elif pcfg_indices is not None:
                    try:
                        pcfg_indices.append(int(meta["sample_idx"]))
                    except (TypeError, ValueError):
                        pcfg_indices = None
        last_index = n_samples
        last_sample = sample
        n_samples += 1

    return (
        n_samples,
        last_index,
        sample_checks,
        prompt_counts,
        any_missing_unmasking_order,
        first_metadata,
        pcfg_indices,
        last_sample,
    )


def check_trace_jsonl_streaming(
    path: str | Path,
    *,
    max_samples: Optional[int] = 3,
    check_last: bool = True,
    collect_pcfg_audit: bool = False,
    is_fdm: Optional[bool] = None,
) -> TraceCheck:
    """Stream a JSONL trace file: one sample in memory at a time."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    fdm = is_fdm_trace_name(path) if is_fdm is None else bool(is_fdm)

    (
        n_samples,
        last_index,
        sample_checks,
        prompt_counts,
        any_missing_unmasking_order,
        first_metadata,
        pcfg_indices,
        last_sample,
    ) = _stream_trace_check(
        iter_jsonl_trace_samples(path),
        max_samples=max_samples,
        collect_pcfg_audit=collect_pcfg_audit,
        is_fdm=fdm,
    )

    if check_last and last_index is not None and last_index not in {
        sc.index for sc in sample_checks
    }:
        if last_sample is None:
            last_sample = read_jsonl_sample_at_index(path, last_index)
        sample_checks.append(
            check_sample(last_sample, index=last_index, is_fdm=fdm)
        )

    dup = duplicate_prompt_stats_from_counter(prompt_counts)
    return _build_trace_check(
        path=str(path.resolve()),
        fmt="jsonl",
        n_samples=n_samples,
        size_bytes=path.stat().st_size,
        sample_checks=sample_checks,
        dup=dup,
        any_missing_unmasking_order=any_missing_unmasking_order,
        first_metadata=first_metadata,
        pcfg_sample_indices=pcfg_indices,
    )


def check_trace_npz_streaming(
    path: str | Path,
    *,
    max_samples: Optional[int] = 3,
    check_last: bool = True,
    collect_pcfg_audit: bool = False,
    is_fdm: Optional[bool] = None,
) -> TraceCheck:
    """Stream an NPZ trace file: one sample in memory at a time."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    fdm = is_fdm_trace_name(path) if is_fdm is None else bool(is_fdm)

    (
        n_samples,
        last_index,
        sample_checks,
        prompt_counts,
        any_missing_unmasking_order,
        first_metadata,
        pcfg_indices,
        last_sample,
    ) = _stream_trace_check(
        iter_npz_traces(path),
        max_samples=max_samples,
        collect_pcfg_audit=collect_pcfg_audit,
        is_fdm=fdm,
    )

    if check_last and last_index is not None and last_sample is not None and last_index not in {
        sc.index for sc in sample_checks
    }:
        sample_checks.append(
            check_sample(last_sample, index=last_index, is_fdm=fdm)
        )

    dup = duplicate_prompt_stats_from_counter(prompt_counts)
    return _build_trace_check(
        path=str(path.resolve()),
        fmt="npz",
        n_samples=n_samples,
        size_bytes=path.stat().st_size,
        sample_checks=sample_checks,
        dup=dup,
        any_missing_unmasking_order=any_missing_unmasking_order,
        first_metadata=first_metadata,
        pcfg_sample_indices=pcfg_indices,
    )


def _repair_sample_worker(
    payload: tuple[dict, bool],
) -> tuple[dict, list[str]]:
    """Picklable worker entry for :func:`repair_samples_parallel`."""
    sample, is_fdm = payload
    return repair_sample_from_mask_diff(sample, is_fdm=is_fdm)


def repair_samples_parallel(
    samples: list[dict],
    *,
    is_fdm: bool = False,
    workers: int = 1,
) -> list[tuple[dict, list[str]]]:
    """Repair many samples; use a process pool when ``workers > 1``."""
    if not samples:
        return []
    n_workers = max(1, int(workers))
    if n_workers == 1 or len(samples) == 1:
        return [
            repair_sample_from_mask_diff(s, is_fdm=is_fdm) for s in samples
        ]
    payloads = [(s, is_fdm) for s in samples]
    chunksize = max(1, len(samples) // (n_workers * 4))
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        return list(pool.map(_repair_sample_worker, payloads, chunksize=chunksize))


def iter_repaired_jsonl(
    path: str | Path,
    *,
    is_fdm: Optional[bool] = None,
    workers: int = 1,
    batch_size: int = 32,
) -> Iterator[tuple[Literal["comment", "duplicate", "sample"], Any]]:
    """Stream JSONL with per-sample repair + prompt dedup.

    Yields:
      ``("comment", line)`` — `#` header/comment lines (including ``# task=``)
      ``("duplicate", None)`` — skipped duplicate prompt
      ``("sample", (fixed_dict, repair_msgs))`` — kept repaired sample

    When ``workers > 1``, samples are repaired in batches via a process pool.
    """
    path = Path(path)
    fdm = is_fdm_trace_name(path) if is_fdm is None else bool(is_fdm)
    n_workers = max(1, int(workers))
    batch_n = max(1, int(batch_size))
    seen: set[tuple[int, ...]] = set()
    batch: list[dict] = []

    def _flush_batch() -> Iterator[tuple[Literal["sample"], tuple[dict, list[str]]]]:
        nonlocal batch
        if not batch:
            return
        for fixed, reps in repair_samples_parallel(
            batch, is_fdm=fdm, workers=n_workers
        ):
            yield "sample", (fixed, reps)
        batch = []

    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                yield "comment", line if line.endswith("\n") else f"{line}\n"
                continue

            sample = json.loads(stripped)
            key = prompt_key(sample)
            if key is not None and key in seen:
                yield "duplicate", None
                continue
            if key is not None:
                seen.add(key)

            if n_workers <= 1:
                fixed, reps = repair_sample_from_mask_diff(sample, is_fdm=fdm)
                yield "sample", (fixed, reps)
            else:
                batch.append(sample)
                if len(batch) >= batch_n:
                    yield from _flush_batch()

    if batch:
        yield from _flush_batch()


def stream_repair_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    *,
    is_fdm: Optional[bool] = None,
    workers: int = 1,
) -> tuple[int, list[str], Optional[str]]:
    """Write a repaired JSONL (dedup + mask-diff/phantom/trim). One sample in RAM."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    repairs: list[str] = []
    task_name: Optional[str] = None
    n_out = 0
    removed = 0
    t0 = time.perf_counter()

    with open(output_path, "w", encoding="utf-8") as fout:
        for kind, payload in iter_repaired_jsonl(
            input_path, is_fdm=is_fdm, workers=workers
        ):
            if kind == "comment":
                line = str(payload)
                fout.write(line)
                stripped = line.strip()
                if stripped.startswith("# task="):
                    raw = stripped.split("=", 1)[1].strip()
                    try:
                        task_name = json.loads(raw)
                    except json.JSONDecodeError:
                        task_name = raw
                continue
            if kind == "duplicate":
                removed += 1
                continue
            fixed, reps = payload
            for rep in reps:
                repairs.append(f"sample {n_out}: {rep}")
            fout.write(json.dumps(fixed, ensure_ascii=False) + "\n")
            n_out += 1
            if n_out == 1 or n_out % 50 == 0:
                logger.info(
                    "stream_repair_jsonl: %d kept (%d dupes) %.1fs",
                    n_out,
                    removed,
                    time.perf_counter() - t0,
                )

    if removed:
        repairs.insert(
            0,
            f"removed {removed} duplicate trace(s) by prompt ({n_out} kept)",
        )
    logger.info(
        "stream_repair_jsonl done: %d samples, %d repair msgs, %.1fs -> %s",
        n_out,
        len(repairs),
        time.perf_counter() - t0,
        output_path.name,
    )
    return n_out, repairs, task_name


def stream_repair_npz(
    input_path: str | Path,
    output_path: str | Path,
    *,
    task_name: Optional[str] = None,
    is_fdm: Optional[bool] = None,
    workers: int = 1,
) -> tuple[int, list[str], Optional[str], list[dict]]:
    """Unpack NPZ → repair (phantoms, mask-diff, drop last step) → repack NPZ."""
    from traces.trace_npz import iter_npz_traces, pack_samples_to_npz, read_npz_task_name

    input_path = Path(input_path)
    output_path = Path(output_path)
    repairs: list[str] = []
    resolved_task = task_name if task_name is not None else read_npz_task_name(input_path)
    fdm = is_fdm_trace_name(input_path) if is_fdm is None else bool(is_fdm)
    seen: set[tuple[int, ...]] = set()
    raw_samples: list[dict] = []
    removed = 0

    for sample in iter_npz_traces(input_path):
        key = prompt_key(sample)
        if key is not None and key in seen:
            removed += 1
            continue
        if key is not None:
            seen.add(key)
        raw_samples.append(sample)

    if not raw_samples:
        raise ValueError(f"no samples to repair in {input_path}")

    repaired = repair_samples_parallel(
        raw_samples, is_fdm=fdm, workers=workers
    )
    samples: list[dict] = []
    for i, (fixed, reps) in enumerate(repaired):
        for rep in reps:
            repairs.append(f"sample {i}: {rep}")
        samples.append(fixed)

    pack_samples_to_npz(samples, output_path, task_name=resolved_task)
    if removed:
        repairs.insert(
            0,
            f"removed {removed} duplicate trace(s) by prompt ({len(samples)} kept)",
        )
    return len(samples), repairs, resolved_task, samples


def trace_needs_repair(check: TraceCheck, samples: list[dict] | None = None) -> bool:
    needs_hook_repair = check.transfer_status in ("inferrable", "explicit_partial", "mixed")
    needs_order_repair = check.any_missing_unmasking_order
    if samples:
        needs_order_repair = needs_order_repair or any(
            sample.get("step_traces") and not sample.get("unmasking_order") for sample in samples
        )
    needs_dedup = check.removable_duplicates > 0
    needs_sel_align = check.any_selection_mask_mismatch
    needs_phantom = check.any_phantom_steps
    needs_tail = check.any_untrimmed_tail
    if samples:
        needs_tail = needs_tail or any(
            (sample.get("step_traces") and not sample_tail_trimmed(sample))
            for sample in samples
        )
    return (
        needs_hook_repair
        or needs_order_repair
        or needs_dedup
        or needs_sel_align
        or needs_phantom
        or needs_tail
    )


def conversion_repair_blockers(
    check: TraceCheck,
    samples: list[dict] | None = None,
    *,
    repair_enabled: bool,
) -> list[str]:
    """Return non-empty issues when convert must not proceed."""
    blockers: list[str] = []
    needs_repair = trace_needs_repair(check, samples)

    if needs_repair and not repair_enabled:
        blockers.append("repair required but disabled (--no-repair)")

    if not check.hooks_ok:
        blockers.append(f"hooks not ok: transfer_status={check.transfer_status}")

    if check.removable_duplicates > 0:
        blockers.append(
            f"duplicate prompts remain: {check.removable_duplicates} removable"
        )

    if check.samples_missing_required > 0:
        blockers.append(
            f"{check.samples_missing_required} samples missing required step fields"
        )

    missing_order = check.any_missing_unmasking_order
    if samples:
        missing_order = missing_order or any(
            sample.get("step_traces") and not sample.get("unmasking_order") for sample in samples
        )
    if missing_order and check.transfer_status != "missing_transfer":
        blockers.append("missing unmasking_order after repair")

    if check.any_selection_mask_mismatch:
        blockers.append("selection/mask-diff count mismatch remains after repair")

    if check.any_phantom_steps:
        blockers.append("phantom extra steps remain after repair")

    if check.any_untrimmed_tail:
        blockers.append("trailing step not trimmed after repair")

    return blockers


def check_trace_file(
    path: str | Path,
    *,
    max_samples: Optional[int] = None,
    check_last: bool = True,
) -> TraceCheck:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    fmt = path.suffix.lower().lstrip(".") or "unknown"
    size_bytes = path.stat().st_size
    if fmt == "jsonl":
        return check_trace_jsonl_streaming(
            path,
            max_samples=max_samples,
            check_last=check_last,
        )
    if fmt == "npz":
        return check_trace_npz_streaming(
            path,
            max_samples=max_samples,
            check_last=check_last,
        )

    _, samples = load_trace_samples(path)
    return check_trace_samples(
        samples,
        path=str(path.resolve()),
        fmt=fmt,
        size_bytes=size_bytes,
        max_samples=max_samples,
        check_last=check_last,
    )


def _write_selection_fields(
    step: dict,
    sel: np.ndarray,
    *,
    predictions,
    gen_length: int,
) -> None:
    """Set selected_positions / selected_tokens from a boolean mask + predictions."""
    pred = _gen_vec(predictions, gen_length=gen_length, dtype=np.int32)
    if pred is None:
        pred = np.zeros(gen_length, dtype=np.int32)
    sel_tokens = np.zeros(gen_length, dtype=np.int32)
    sel_tokens[sel] = pred[sel]
    step["selected_positions"] = sel.astype(bool).reshape(1, gen_length).tolist()
    step["selected_tokens"] = sel_tokens.reshape(1, gen_length).tolist()


def _plateau_ranges(still: list[Optional[frozenset[int]]]) -> list[tuple[int, int]]:
    """Inclusive ``(lo, hi)`` index ranges where ``still`` is constant."""
    ranges: list[tuple[int, int]] = []
    t = 0
    while t < len(still):
        j = t
        while j + 1 < len(still) and still[j + 1] == still[t]:
            j += 1
        ranges.append((t, j))
        t = j + 1
    return ranges


def repair_collapse_phantom_steps(
    sample: dict,
    *,
    is_fdm: bool = False,
) -> tuple[dict, list[str]]:
    """Drop FDM phantom / empty-growth plateau extras (FDM filename only).

    Detect FDM from the trace stem (``…_fdm_…``). For each permanent-unmask
    plateau keep the **first** step (outer ``model(x)`` before hyps; Hub checks
    show neighbor/mask continuity ties across the plateau). Rebuild ``selected_*``
    from permanent-set growth. Non-FDM schedulers are left unchanged.
    """
    out = copy.deepcopy(sample)
    steps = out.get("step_traces") or []
    gen_length = int(out.get("gen_length") or 0)
    repairs: list[str] = []
    if not is_fdm or gen_length <= 0 or len(steps) < 2:
        return out, repairs

    extras = count_phantom_extra_steps(steps, gen_length=gen_length, is_fdm=True)
    if extras <= 0:
        return out, repairs

    still = permanent_still_sets(steps, gen_length=gen_length)
    plateaus = _plateau_ranges(still)
    # Always keep the first step of each plateau.
    keep_indices = [lo for lo, _hi in plateaus]

    new_steps: list[dict] = []
    for i, keep_i in enumerate(keep_indices):
        step = copy.deepcopy(steps[keep_i])
        step["step"] = i
        if i + 1 < len(keep_indices):
            lo_i, _hi_i = plateaus[i]
            lo_next, _ = plateaus[i + 1]
            prev_set = still[lo_i] or frozenset()
            next_set = still[lo_next] or frozenset()
            newly = next_set - prev_set
            sel = np.zeros(gen_length, dtype=bool)
            for pos in newly:
                if 0 <= pos < gen_length:
                    sel[pos] = True
            _write_selection_fields(
                step,
                sel,
                predictions=steps[keep_i].get("predictions"),
                gen_length=gen_length,
            )
        else:
            step["selected_positions"] = None
            step["selected_tokens"] = None
        new_steps.append(step)

    out["step_traces"] = new_steps
    inferred = infer_unmasking_order(new_steps, gen_length=gen_length)
    if inferred:
        out["unmasking_order"] = inferred
    repairs.append(
        f"collapsed {len(steps)} -> {len(new_steps)} steps "
        f"(dropped {len(steps) - len(new_steps)} FDM plateau extras; keep first)"
    )
    return out, repairs


def repair_drop_last_step(sample: dict) -> tuple[dict, list[str]]:
    """Drop the final step_trace (no next mask → mask_diff cannot validate it).

    Applies to **all** schedulers (not FDM-only). Run after phantom plateau collapse
    so FDM's last kept plateau is what gets dropped. Idempotent via
    ``schedulerlens_trimmed_tail`` in sample metadata (survives NPZ).

    After the drop, the new last step keeps mask-diff–aligned ``selected_*`` from
    the transition into the removed step; metrics trust that selection.
    """
    out = copy.deepcopy(sample)
    repairs: list[str] = []
    if sample_tail_trimmed(out):
        return out, repairs

    steps = out.get("step_traces") or []
    if len(steps) <= 1:
        mark_sample_tail_trimmed(out)
        repairs.append("marked trimmed_tail (single-step; nothing dropped)")
        return out, repairs

    dropped = steps[-1]
    dropped_sel = _gen_vec(
        dropped.get("selected_positions"),
        gen_length=int(out.get("gen_length") or 0),
        dtype=np.bool_,
    )
    n_sel = int(np.sum(dropped_sel)) if dropped_sel is not None else 0
    out["step_traces"] = steps[:-1]
    for i, st in enumerate(out["step_traces"]):
        st["step"] = i
    mark_sample_tail_trimmed(out)
    repairs.append(
        f"dropped last step (all schedulers; no mask_diff to validate; "
        f"selected_count was {n_sel})"
    )
    gen_length = int(out.get("gen_length") or 0)
    inferred = infer_unmasking_order(out["step_traces"], gen_length=gen_length)
    if inferred:
        out["unmasking_order"] = inferred
    return out, repairs


def repair_sample_from_mask_diff(
    sample: dict,
    *,
    is_fdm: bool = False,
) -> tuple[dict, list[str]]:
    """Repair one sample for SchedulerLens-ready step fields.

    Order:
    1. Collapse FDM plateaus (filename encodes FDM) — keep first of each plateau.
    2. Align ``selected_*`` to mask_diff on every step that has a next mask.
    3. Drop the final step entirely (cannot validate without a next ``mask_index``).
    """
    out, repairs = repair_collapse_phantom_steps(sample, is_fdm=is_fdm)
    steps = out.get("step_traces") or []
    gen_length = int(out.get("gen_length") or 0)
    selection_changed = bool(repairs)

    for t in range(len(steps) - 1):
        if gen_length <= 0:
            break
        sel_diff = mask_diff_selection(steps, t, gen_length=gen_length)
        if sel_diff is None:
            continue
        n_diff = int(np.sum(sel_diff))
        existing = steps[t].get("selected_positions")
        if existing is None:
            if n_diff == 0:
                continue
            _write_selection_fields(
                steps[t],
                sel_diff,
                predictions=steps[t].get("predictions"),
                gen_length=gen_length,
            )
            selection_changed = True
            repairs.append(f"sample step {t}: inferred {n_diff} unmasks")
            continue

        sel_vec = _gen_vec(existing, gen_length=gen_length, dtype=np.bool_)
        n_sel = int(np.sum(sel_vec)) if sel_vec is not None else -1
        if n_sel == n_diff:
            continue

        _write_selection_fields(
            steps[t],
            sel_diff,
            predictions=steps[t].get("predictions"),
            gen_length=gen_length,
        )
        selection_changed = True
        repairs.append(
            f"sample step {t}: replaced selected_count {n_sel} with mask_diff {n_diff}"
        )

    # Drop last step for every scheduler (after plateau collapse + mask-diff align).
    out["step_traces"] = steps
    dropped, drop_reps = repair_drop_last_step(out)
    if drop_reps:
        out = dropped
        repairs.extend(drop_reps)
        selection_changed = True
        steps = out.get("step_traces") or []

    if selection_changed:
        inferred = infer_unmasking_order(steps, gen_length=gen_length)
        if inferred:
            out["unmasking_order"] = inferred
            repairs.append("rebuilt unmasking_order after selection/mask-diff repair")
    else:
        _maybe_repair_unmasking_order(out, steps, gen_length, repairs)
    return out, repairs


def _arrays_close(a, b, *, field: str) -> bool:
    return _step_field_equal(a, b, field=field)


def _token_field_equal(a, b) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    aa = np.asarray(a[0] if isinstance(a, list) else a)
    bb = np.asarray(b[0] if isinstance(b, list) else b)
    return bool(np.array_equal(aa, bb))


def _scalar_equal(a, b) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, float) or isinstance(b, float):
        return bool(np.isclose(float(a), float(b), rtol=1e-9, atol=1e-9, equal_nan=True))
    return a == b


def _compare_roundtrip_samples(orig: dict, recon: dict, *, sample_index: int) -> list[str]:
    issues: list[str] = []
    i = sample_index
    for field in ("prompt_len", "gen_length", "gen_only"):
        ov = orig.get(field)
        rv = recon.get(field)
        # Pack defaults missing gen_only to True.
        if field == "gen_only":
            if ov is None:
                ov = True
            if rv is None:
                rv = True
        if not _scalar_equal(ov, rv):
            issues.append(f"sample {i}: {field} mismatch")
    for field in ("start_time", "end_time"):
        if not _scalar_equal(orig.get(field), recon.get(field)):
            issues.append(f"sample {i}: {field} mismatch")
    if not _token_field_equal(orig.get("prompt"), recon.get("prompt")):
        issues.append(f"sample {i}: prompt mismatch")
    if not _token_field_equal(orig.get("final_sequence"), recon.get("final_sequence")):
        issues.append(f"sample {i}: final_sequence mismatch")
    if (orig.get("metadata") or {}) != (recon.get("metadata") or {}):
        issues.append(f"sample {i}: metadata mismatch")

    orig_order = {int(k): int(v) for k, v in (orig.get("unmasking_order") or {}).items()}
    recon_order = {int(k): int(v) for k, v in (recon.get("unmasking_order") or {}).items()}
    if orig_order != recon_order:
        issues.append(f"sample {i}: unmasking_order mismatch")

    orig_steps = orig.get("step_traces") or []
    recon_steps = recon.get("step_traces") or []
    if len(orig_steps) != len(recon_steps):
        issues.append(f"sample {i}: n_steps {len(orig_steps)} != {len(recon_steps)}")
        return issues
    for t, (ost, rst) in enumerate(zip(orig_steps, recon_steps)):
        for fname in STEP_FIELDS_REQUIRED + STEP_FIELDS_OPTIONAL:
            if not _arrays_close(ost.get(fname), rst.get(fname), field=fname):
                issues.append(f"sample {i} step {t}: {fname} mismatch")
                break
        for fname in ("intervention_applied", "intervention_type"):
            ov = ost.get(fname)
            rv = rst.get(fname)
            # Sidecar defaults missing intervention_applied to False.
            if fname == "intervention_applied":
                if ov is None:
                    ov = False
                if rv is None:
                    rv = False
            if not _scalar_equal(ov, rv):
                issues.append(f"sample {i} step {t}: {fname} mismatch")
                break
        if (ost.get("metadata") or {}) != (rst.get("metadata") or {}):
            issues.append(f"sample {i} step {t}: metadata mismatch")
    return issues


def _roundtrip_indices(n_samples: int, *, check_samples: int) -> list[int]:
    indices = list(range(min(check_samples, n_samples)))
    if n_samples > 0 and (n_samples - 1) not in indices:
        indices.append(n_samples - 1)
    return indices


def validate_npz_roundtrip(
    samples: list[dict] | str | Path,
    npz_path: str | Path,
    *,
    check_samples: int = 3,
) -> list[str]:
    """Verify NPZ unpack matches source samples (spot-check; full field fidelity)."""
    npz_path = Path(npz_path)
    if isinstance(samples, (str, Path)) and Path(samples).suffix.lower() == ".jsonl":
        return _validate_npz_roundtrip_jsonl(Path(samples), npz_path, check_samples=check_samples)

    sample_list = samples
    issues: list[str] = []
    with np.load(npz_path, allow_pickle=False) as data:
        n_npz = int(data["n_samples"])
    if n_npz != len(sample_list):
        issues.append(f"n_samples mismatch: jsonl={len(sample_list)} npz={n_npz}")
        return issues

    recon_by_index: dict[int, dict] = {}
    wanted = set(_roundtrip_indices(len(sample_list), check_samples=check_samples))
    for i, recon in enumerate(iter_npz_traces(npz_path)):
        if i in wanted:
            recon_by_index[i] = recon
        if i >= max(wanted, default=-1):
            break

    for i in sorted(wanted):
        issues.extend(_compare_roundtrip_samples(sample_list[i], recon_by_index[i], sample_index=i))
    return issues


def validate_npz_roundtrip_indexed(
    samples_by_index: dict[int, dict],
    npz_path: str | Path,
    *,
    n_samples: int,
) -> list[str]:
    """Round-trip check using a sparse map of expected repaired samples by index."""
    npz_path = Path(npz_path)
    issues: list[str] = []
    with np.load(npz_path, allow_pickle=False) as data:
        n_npz = int(data["n_samples"])
    if n_npz != n_samples:
        issues.append(f"n_samples mismatch: expected={n_samples} npz={n_npz}")
        return issues
    if not samples_by_index:
        return issues

    wanted = set(samples_by_index)
    recon_by_index: dict[int, dict] = {}
    for i, recon in enumerate(iter_npz_traces(npz_path)):
        if i in wanted:
            recon_by_index[i] = recon
        if i >= max(wanted):
            break

    for i in sorted(wanted):
        if i not in recon_by_index:
            issues.append(f"sample {i}: missing from NPZ during round-trip")
            continue
        issues.extend(
            _compare_roundtrip_samples(samples_by_index[i], recon_by_index[i], sample_index=i)
        )
    return issues


def _validate_npz_roundtrip_jsonl(
    jsonl_path: Path,
    npz_path: Path,
    *,
    check_samples: int = 3,
) -> list[str]:
    from traces.trace_npz import scan_jsonl_trace_shape

    issues: list[str] = []
    shape = scan_jsonl_trace_shape(jsonl_path)
    with np.load(npz_path, allow_pickle=False) as data:
        n_npz = int(data["n_samples"])
    if n_npz != shape.n_samples:
        issues.append(f"n_samples mismatch: jsonl={shape.n_samples} npz={n_npz}")
        return issues

    wanted = _roundtrip_indices(shape.n_samples, check_samples=check_samples)
    recon_by_index: dict[int, dict] = {}
    wanted_set = set(wanted)
    for i, recon in enumerate(iter_npz_traces(npz_path)):
        if i in wanted_set:
            recon_by_index[i] = recon
        if i >= max(wanted_set, default=-1):
            break

    for i in wanted:
        orig = read_jsonl_sample_at_index(jsonl_path, i)
        issues.extend(_compare_roundtrip_samples(orig, recon_by_index[i], sample_index=i))
    return issues


def check_to_row(check: TraceCheck, *, extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    row = {
        "transfer_status": check.transfer_status,
        "hooks_ok": check.hooks_ok,
        "n_samples": check.n_samples,
        "samples_checked": check.samples_checked,
        "samples_proxy_ready": check.samples_proxy_ready,
        "samples_inferrable": check.samples_inferrable,
        "samples_no_transfer": check.samples_no_transfer,
        "samples_with_any_selected": check.samples_with_any_selected,
        "samples_with_all_selected": check.samples_with_all_selected,
        "samples_missing_required": check.samples_missing_required,
        "samples_selection_mask_mismatch": check.samples_selection_mask_mismatch,
        "samples_phantom_steps": check.samples_phantom_steps,
        "samples_untrimmed_tail": check.samples_untrimmed_tail,
        "unique_prompts": check.unique_prompts,
        "duplicate_traces": check.duplicate_traces,
        "removable_duplicates": check.removable_duplicates,
        "ambiguous_duplicate_prompts": check.ambiguous_duplicate_prompts,
        "size_bytes": check.size_bytes,
        "issues": "; ".join(check.issues) if check.issues else "",
    }
    if extra:
        row.update(extra)
    return row
