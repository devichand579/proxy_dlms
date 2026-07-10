"""Trace content validation, repair, and NPZ round-trip checks."""

from __future__ import annotations

import copy
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from traces.trace_npz import iter_npz_traces, load_trace_samples

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


def duplicate_prompt_stats(samples: list[dict]) -> dict[str, int]:
    """Count duplicate traces: extra lines sharing the same prompt token ids."""
    counts = Counter(key for key in (prompt_key(s) for s in samples) if key is not None)
    unique = len(counts)
    same_prompt_extra = sum(c - 1 for c in counts.values() if c > 1)
    return {
        "unique_prompts": unique,
        "duplicate_traces": max(0, len(samples) - unique),
        "same_prompt_extra": same_prompt_extra,
        "removable_duplicates": same_prompt_extra,
        "ambiguous_duplicate_prompts": 0,
        "max_copies": max(counts.values()) if counts else 0,
    }


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
    _, samples = load_trace_samples(path)
    n = len(samples)

    indices: list[int] = list(range(n)) if max_samples is None else list(range(min(max_samples, n)))
    if check_last and n > 0 and (n - 1) not in indices:
        indices.append(n - 1)
    indices = sorted(set(indices))

    sample_checks = [check_sample(samples[i], index=i) for i in indices]
    issues: list[str] = []
    if n == 0:
        issues.append("no samples")

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
    dup = duplicate_prompt_stats(samples)
    if dup["removable_duplicates"] > 0:
        issues.append(
            f"duplicate prompts: {dup['removable_duplicates']} removable "
            f"({n} lines, {dup['unique_prompts']} unique prompts)"
        )

    return TraceCheck(
        path=str(path.resolve()),
        format=fmt,
        n_samples=n,
        samples_checked=len(sample_checks),
        samples_proxy_ready=proxy_ready,
        samples_inferrable=inferrable,
        samples_no_transfer=no_transfer,
        samples_with_any_selected=any_sel,
        samples_with_all_selected=all_sel,
        samples_missing_required=missing_req,
        size_bytes=path.stat().st_size,
        unique_prompts=dup["unique_prompts"],
        duplicate_traces=dup["duplicate_traces"],
        removable_duplicates=dup["removable_duplicates"],
        ambiguous_duplicate_prompts=dup["ambiguous_duplicate_prompts"],
        issues=issues,
        sample_checks=sample_checks,
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


def repair_samples(samples: list[dict]) -> tuple[list[dict], list[str]]:
    samples, dedup_reps = deduplicate_prompt_traces(samples)
    all_repairs: list[str] = list(dedup_reps)
    repaired: list[dict] = []
    for i, sample in enumerate(samples):
        fixed, reps = repair_sample_from_mask_diff(sample)
        repaired.append(fixed)
        for r in reps:
            all_repairs.append(f"sample {i}: {r}")
    return repaired, all_repairs


def _arrays_close(a, b, *, field: str) -> bool:
    return _step_field_equal(a, b, field=field)


def validate_npz_roundtrip(
    samples: list[dict],
    npz_path: str | Path,
    *,
    check_samples: int = 3,
) -> list[str]:
    """Verify NPZ unpack matches source samples on key step fields."""
    issues: list[str] = []
    unpacked = list(iter_npz_traces(npz_path))
    if len(unpacked) != len(samples):
        issues.append(f"n_samples mismatch: jsonl={len(samples)} npz={len(unpacked)}")
        return issues

    indices = list(range(min(check_samples, len(samples))))
    if len(samples) > 0 and (len(samples) - 1) not in indices:
        indices.append(len(samples) - 1)

    for i in indices:
        orig_steps = samples[i].get("step_traces") or []
        recon_steps = unpacked[i].get("step_traces") or []
        if len(orig_steps) != len(recon_steps):
            issues.append(f"sample {i}: n_steps {len(orig_steps)} != {len(recon_steps)}")
            continue
        for t, (ost, rst) in enumerate(zip(orig_steps, recon_steps)):
            for fname in ("mask_index", "predictions", "confidences", "entropies"):
                if not _arrays_close(ost.get(fname), rst.get(fname), field=fname):
                    issues.append(f"sample {i} step {t}: {fname} mismatch")
                    break
            o_pres = ost.get("selected_positions") is not None
            r_pres = rst.get("selected_positions") is not None
            if o_pres != r_pres:
                issues.append(f"sample {i} step {t}: selected_positions presence mismatch")
            elif o_pres and not _arrays_close(
                ost.get("selected_positions"), rst.get("selected_positions"), field="bool"
            ):
                issues.append(f"sample {i} step {t}: selected_positions mismatch")
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
