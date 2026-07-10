"""Compact NPZ storage for eval traces (JSONL -> stacked arrays)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import numpy as np

TRACE_NPZ_VERSION = 2
STEP_FIELDS = (
    "mask_index",
    "predictions",
    "confidences",
    "entropies",
    "selected_positions",
    "selected_tokens",
)
OPTIONAL_STEP_FIELDS = ("selected_positions", "selected_tokens")
FIELD_DTYPES = {
    "mask_index": np.bool_,
    "predictions": np.int32,
    "confidences": np.float32,
    "entropies": np.float32,
    "selected_positions": np.bool_,
    "selected_tokens": np.int32,
}


def _squeeze_batch(row) -> np.ndarray:
    arr = np.asarray(row)
    while arr.ndim > 1:
        arr = arr[0]
    return arr


def _step_array(step_traces: List[dict], field: str, dtype, *, gen_length: int) -> np.ndarray:
    t_max = len(step_traces)
    out = np.zeros((t_max, gen_length), dtype=dtype)
    for t, step in enumerate(step_traces):
        data = step.get(field)
        if data is None:
            continue
        row = _squeeze_batch(data).astype(dtype, copy=False)
        if row.shape[0] != gen_length:
            raise ValueError(
                f"step {t} field {field}: expected length {gen_length}, got {row.shape[0]}"
            )
        out[t] = row
    return out


def _unmask_step_at_pos(unmasking_order: dict, gen_length: int) -> np.ndarray:
    out = np.full(gen_length, -1, dtype=np.int16)
    for pos, step in (unmasking_order or {}).items():
        p = int(pos)
        if 0 <= p < gen_length:
            out[p] = int(step)
    return out


def _step_presence(step_traces: List[dict], field: str) -> np.ndarray:
    t_max = len(step_traces)
    return np.array(
        [step.get(field) is not None for step in step_traces],
        dtype=np.bool_,
    )


def trace_dict_to_arrays(trace_dict: dict) -> Dict[str, np.ndarray]:
    """Pack one JSON trace dict into dense arrays."""
    step_traces = trace_dict.get("step_traces") or []
    if not step_traces:
        raise ValueError("trace has no step_traces")

    gen_length = int(
        trace_dict.get("gen_length")
        or len(_squeeze_batch(step_traces[0].get("mask_index") or [0]))
    )
    packed = {
        "n_steps": np.int32(len(step_traces)),
        "prompt_len": np.int32(int(trace_dict.get("prompt_len") or 0)),
        "unmask_step_at_pos": _unmask_step_at_pos(trace_dict.get("unmasking_order"), gen_length),
    }
    for field, dtype in FIELD_DTYPES.items():
        packed[field] = _step_array(step_traces, field, dtype, gen_length=gen_length)
    for field in OPTIONAL_STEP_FIELDS:
        packed[f"{field}_present"] = _step_presence(step_traces, field)
    return packed


def arrays_to_trace_dict(
    *,
    gen_length: int,
    n_steps: int,
    prompt_len: int,
    arrays: Dict[str, np.ndarray],
    sample_index: int,
) -> dict:
    """Reconstruct one JSON-compatible trace dict from packed arrays."""
    step_traces: List[dict] = []
    for t in range(int(n_steps)):
        step = {"step": t}
        for field in STEP_FIELDS:
            present_key = f"{field}_present"
            if present_key in arrays and not bool(arrays[present_key][sample_index, t]):
                step[field] = None
                continue
            arr = arrays[field][sample_index, t]
            step[field] = arr.reshape(1, gen_length).tolist()
        step["intervention_applied"] = False
        step["intervention_type"] = None
        step["metadata"] = {}
        step_traces.append(step)

    unmask = arrays["unmask_step_at_pos"][sample_index]
    unmasking_order = {
        int(pos): int(step)
        for pos, step in enumerate(unmask.tolist())
        if int(step) >= 0
    }

    return {
        "gen_only": True,
        "prompt": None,
        "final_sequence": None,
        "prompt_len": int(prompt_len),
        "gen_length": int(gen_length),
        "step_traces": step_traces,
        "unmasking_order": unmasking_order,
        "start_time": None,
        "end_time": None,
    }


def _load_jsonl_samples(path: Path) -> tuple[Optional[str], List[dict]]:
    task_name: Optional[str] = None
    samples: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if line.startswith("# task="):
                    try:
                        task_name = json.loads(line.split("=", 1)[1])
                    except json.JSONDecodeError:
                        task_name = line.split("=", 1)[1].strip()
                continue
            samples.append(json.loads(line))
    return task_name, samples


def _load_json_samples(path: Path) -> tuple[Optional[str], List[dict]]:
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if isinstance(doc, list):
        return None, doc
    if isinstance(doc, dict):
        if "samples" in doc:
            return doc.get("task"), list(doc["samples"])
        return doc.get("task"), [doc]
    raise ValueError(f"Unsupported JSON trace format: {path}")


def load_trace_samples(path: Union[str, Path]) -> tuple[Optional[str], List[dict]]:
    """Load all samples from JSONL or JSON trace file."""
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return _load_jsonl_samples(path)
    if path.suffix.lower() == ".json":
        return _load_json_samples(path)
    raise ValueError(f"Unsupported trace format: {path}")


def convert_jsonl_to_npz(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Convert a JSONL/JSON trace file to a compressed NPZ archive."""
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix(".npz")
    else:
        output_path = Path(output_path)

    task_name, samples = load_trace_samples(input_path)
    if not samples:
        raise ValueError(f"No samples found in {input_path}")

    gen_lengths = {int(s.get("gen_length") or 0) for s in samples}
    gen_lengths.discard(0)
    if len(gen_lengths) != 1:
        raise ValueError(f"Mixed gen_length values in {input_path}: {sorted(gen_lengths)}")
    gen_length = next(iter(gen_lengths))

    n_samples = len(samples)
    n_steps = np.zeros(n_samples, dtype=np.int32)
    prompt_len = np.zeros(n_samples, dtype=np.int32)
    max_steps = 0
    per_sample = []
    for i, sample in enumerate(samples):
        packed = trace_dict_to_arrays(sample)
        per_sample.append(packed)
        n_steps[i] = packed["n_steps"]
        prompt_len[i] = packed["prompt_len"]
        max_steps = max(max_steps, int(packed["n_steps"]))

    stacked: Dict[str, np.ndarray] = {
        "version": np.array(str(TRACE_NPZ_VERSION)),
        "n_samples": np.int32(n_samples),
        "gen_length": np.int32(gen_length),
        "n_steps": n_steps,
        "prompt_len": prompt_len,
        "unmask_step_at_pos": np.stack([p["unmask_step_at_pos"] for p in per_sample], axis=0),
    }
    if task_name is not None:
        stacked["task"] = np.array(str(task_name))

    for field, dtype in FIELD_DTYPES.items():
        arr = np.zeros((n_samples, max_steps, gen_length), dtype=dtype)
        for i, packed in enumerate(per_sample):
            t = int(packed["n_steps"])
            arr[i, :t] = packed[field]
        stacked[field] = arr
        if field in OPTIONAL_STEP_FIELDS:
            present = np.zeros((n_samples, max_steps), dtype=np.bool_)
            for i, packed in enumerate(per_sample):
                t = int(packed["n_steps"])
                present[i, :t] = packed[f"{field}_present"]
            stacked[f"{field}_present"] = present

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **stacked)
    return output_path


def iter_npz_traces(path: Union[str, Path]) -> Iterator[dict]:
    """Yield one trace dict per sample from an NPZ trace archive."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        version = str(data["version"])
        if version != str(TRACE_NPZ_VERSION):
            raise ValueError(f"Unsupported NPZ trace version {version!r} in {path}")

        n_samples = int(data["n_samples"])
        gen_length = int(data["gen_length"])
        n_steps = data["n_steps"]
        prompt_len = data["prompt_len"]
        arrays = {field: data[field] for field in STEP_FIELDS}
        for field in OPTIONAL_STEP_FIELDS:
            key = f"{field}_present"
            if key in data:
                arrays[key] = data[key]
        arrays["unmask_step_at_pos"] = data["unmask_step_at_pos"]

        for i in range(n_samples):
            yield arrays_to_trace_dict(
                gen_length=gen_length,
                n_steps=int(n_steps[i]),
                prompt_len=int(prompt_len[i]),
                arrays=arrays,
                sample_index=i,
            )


def compression_ratio(input_path: Union[str, Path], output_path: Union[str, Path]) -> float:
    """Return input_size / output_size for converted files."""
    src = Path(input_path).stat().st_size
    dst = Path(output_path).stat().st_size
    if dst <= 0:
        return float("inf")
    return src / dst
