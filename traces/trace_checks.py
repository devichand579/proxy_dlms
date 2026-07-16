"""Trace content validation, repair, and NPZ round-trip checks."""

from __future__ import annotations

import copy
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from traces.trace_npz import (
    iter_jsonl_trace_samples,
    iter_npz_traces,
    load_trace_samples,
    read_jsonl_sample_at_index,
)

STEP_FIELDS_REQUIRED = ("mask_index", "predictions", "confidences", "entropies")
STEP_FIELDS_OPTIONAL = ("selected_positions", "selected_tokens")


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


def _squeeze_len(data) -> Optional[int]:
    if data is None:
        return None
    arr = np.asarray(data)
    while arr.ndim > 1:
        arr = arr[0]
    return int(arr.shape[0])


def count_inferred_unmasks(steps: list[dict]) -> int:
    total = 0
    for t in range(len(steps) - 1):
        m0 = steps[t].get("mask_index")
        m1 = steps[t + 1].get("mask_index")
        if m0 is None or m1 is None:
            continue
        a0 = np.asarray(m0[0], dtype=bool)
        a1 = np.asarray(m1[0], dtype=bool)
        total += int(np.sum(a0 & ~a1))
    return total


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
    aa = np.asarray(a[0] if isinstance(a, list) else a)
    bb = np.asarray(b[0] if isinstance(b, list) else b)
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
        sel = step.get("selected_positions")
        if sel is None:
            continue
        arr = np.asarray(sel[0], dtype=bool)
        if arr.shape[0] != gen_length:
            continue
        for pos in np.nonzero(arr)[0].tolist():
            p = int(pos)
            if p not in order:
                order[p] = t

    if order:
        return order

    for t in range(len(steps) - 1):
        m0 = steps[t].get("mask_index")
        m1 = steps[t + 1].get("mask_index")
        if m0 is None or m1 is None:
            continue
        mask0 = np.asarray(m0[0], dtype=bool)
        mask1 = np.asarray(m1[0], dtype=bool)
        if mask0.shape[0] != gen_length or mask1.shape[0] != gen_length:
            continue
        for pos in np.nonzero(mask0 & ~mask1)[0].tolist():
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


def check_sample(trace_dict: dict, index: int = 0) -> SampleCheck:
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
    proxy_ready = steps_with_sel > 0 and steps_missing == 0
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
) -> TraceCheck:
    n = len(samples)

    indices: list[int] = list(range(n)) if max_samples is None else list(range(min(max_samples, n)))
    if check_last and n > 0 and (n - 1) not in indices:
        indices.append(n - 1)
    indices = sorted(set(indices))

    sample_checks = [check_sample(samples[i], index=i) for i in indices]
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
            sample_checks.append(check_sample(sample, index=n_samples))
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
) -> TraceCheck:
    """Stream a JSONL trace file: one sample in memory at a time."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

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
    )

    if check_last and last_index is not None and last_index not in {
        sc.index for sc in sample_checks
    }:
        if last_sample is None:
            last_sample = read_jsonl_sample_at_index(path, last_index)
        sample_checks.append(check_sample(last_sample, index=last_index))

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
) -> TraceCheck:
    """Stream an NPZ trace file: one sample in memory at a time."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

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
    )

    if check_last and last_index is not None and last_sample is not None and last_index not in {
        sc.index for sc in sample_checks
    }:
        sample_checks.append(check_sample(last_sample, index=last_index))

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


def stream_repair_jsonl(
    input_path: str | Path,
    output_path: str | Path,
) -> tuple[int, list[str], Optional[str]]:
    """Stream-repair JSONL traces; keeps at most one sample in memory."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    repairs: list[str] = []
    task_name: Optional[str] = None
    seen: set[tuple[int, ...]] = set()
    n_out = 0
    removed = 0

    with open(input_path, "r", encoding="utf-8") as fin, open(
        output_path, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                if stripped.startswith("# task="):
                    payload = stripped.split("=", 1)[1].strip()
                    try:
                        task_name = json.loads(payload)
                    except json.JSONDecodeError:
                        task_name = payload
                fout.write(line if line.endswith("\n") else f"{line}\n")
                continue

            sample = json.loads(stripped)
            key = prompt_key(sample)
            if key is not None and key in seen:
                removed += 1
                continue
            if key is not None:
                seen.add(key)

            fixed, reps = repair_sample_from_mask_diff(sample)
            for rep in reps:
                repairs.append(f"sample {n_out}: {rep}")
            fout.write(json.dumps(fixed, ensure_ascii=False) + "\n")
            n_out += 1

    if removed:
        repairs.insert(
            0,
            f"removed {removed} duplicate trace(s) by prompt ({n_out} kept)",
        )
    return n_out, repairs, task_name


def trace_needs_repair(check: TraceCheck, samples: list[dict] | None = None) -> bool:
    needs_hook_repair = check.transfer_status in ("inferrable", "explicit_partial", "mixed")
    needs_order_repair = check.any_missing_unmasking_order
    if samples:
        needs_order_repair = needs_order_repair or any(
            sample.get("step_traces") and not sample.get("unmasking_order") for sample in samples
        )
    needs_dedup = check.removable_duplicates > 0
    return needs_hook_repair or needs_order_repair or needs_dedup


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


def repair_sample_from_mask_diff(sample: dict) -> tuple[dict, list[str]]:
    """Fill missing selected_positions/tokens from mask_index diffs + predictions."""
    out = copy.deepcopy(sample)
    steps = out.get("step_traces") or []
    gen_length = int(out.get("gen_length") or 0)
    repairs: list[str] = []

    for t in range(len(steps) - 1):
        if steps[t].get("selected_positions") is not None:
            continue
        m0 = steps[t].get("mask_index")
        m1 = steps[t + 1].get("mask_index")
        preds = steps[t].get("predictions")
        if m0 is None or m1 is None or preds is None or gen_length <= 0:
            continue

        mask0 = np.asarray(m0[0], dtype=bool)
        mask1 = np.asarray(m1[0], dtype=bool)
        if mask0.shape[0] != gen_length or mask1.shape[0] != gen_length:
            continue
        sel = mask0 & ~mask1
        if not np.any(sel):
            continue

        pred = np.asarray(preds[0], dtype=np.int32)
        sel_tokens = np.zeros(gen_length, dtype=np.int32)
        sel_tokens[sel] = pred[sel]
        steps[t]["selected_positions"] = sel.reshape(1, gen_length).tolist()
        steps[t]["selected_tokens"] = sel_tokens.reshape(1, gen_length).tolist()
        repairs.append(f"sample step {t}: inferred {int(sel.sum())} unmasks")

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
        if not _scalar_equal(orig.get(field), recon.get(field)):
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
            if not _scalar_equal(ost.get(fname), rst.get(fname)):
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
    """Verify NPZ unpack matches source samples (full JSONL fidelity)."""
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
