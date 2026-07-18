"""Compact NPZ storage for eval traces (JSONL/JSON → stacked arrays)."""

from __future__ import annotations

import copy
import gc
import io
import json
import logging
import os
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Union

import numpy as np

from traces.utils import squeeze_to_1d

logger = logging.getLogger(__name__)

DEFAULT_PACK_MEMORY_FRACTION = 0.25
DEFAULT_PACK_JSON_EXPANSION = 2.5
# Deflate level for NPZ (1=fast, 9=small). Uncompressed via TRACE_NPZ_COMPRESS=0.
DEFAULT_NPZ_COMPRESSLEVEL = 1
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


def _token_ids_from_field(value) -> Optional[list[int]]:
    if value is None:
        return None
    return [int(t) for t in squeeze_to_1d(value).tolist()]


def _batched_token_field(token_ids: Optional[list[int]]) -> Optional[list[list[int]]]:
    if token_ids is None:
        return None
    return [token_ids]


def _step_array(step_traces: List[dict], field: str, dtype, *, gen_length: int) -> np.ndarray:
    t_max = len(step_traces)
    out = np.zeros((t_max, gen_length), dtype=dtype)
    for t, step in enumerate(step_traces):
        data = step.get(field)
        if data is None:
            continue
        row = squeeze_to_1d(data).astype(dtype, copy=False)
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


def _sample_sidecar_dict(sample: dict, step_traces: List[dict]) -> dict:
    step_extras: list[dict] = []
    for step in step_traces:
        step_extras.append(
            {
                "intervention_applied": bool(step.get("intervention_applied", False)),
                "intervention_type": step.get("intervention_type"),
                "metadata": step.get("metadata") or {},
            }
        )
    sidecar: dict = {"step_extras": step_extras}
    metadata = sample.get("metadata")
    if metadata:
        sidecar["metadata"] = metadata
    return sidecar


def _encode_sidecar_blob(sidecar: dict) -> bytes:
    return json.dumps(sidecar, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _decode_sidecar_blob(blob: bytes) -> dict:
    if not blob:
        return {}
    return json.loads(blob.decode("utf-8"))


def _parse_jsonl_task_header(line: str) -> Optional[str]:
    if not line.startswith("# task="):
        return None
    payload = line.split("=", 1)[1].strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return payload


def iter_jsonl_trace_samples(path: Union[str, Path]) -> Iterator[dict]:
    """Yield one trace dict per JSONL data line (comments skipped)."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            yield json.loads(line)


def read_jsonl_task_name(path: Union[str, Path]) -> Optional[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                task = _parse_jsonl_task_header(line)
                if task is not None:
                    return task
                continue
            return None
    return None


def read_jsonl_sample_at_index(path: Union[str, Path], index: int) -> dict:
    if index < 0:
        raise IndexError(index)
    for i, sample in enumerate(iter_jsonl_trace_samples(path)):
        if i == index:
            return sample
    raise IndexError(f"sample index {index} out of range")


def _available_json_batch_budget_bytes(*, reserved_bytes: int = 0) -> int:
    """Bytes available for holding parsed JSONL sample dicts during pack."""
    cap = os.environ.get("TRACE_PACK_MEMORY_BYTES", "").strip()
    if cap:
        return max(0, int(cap) - reserved_bytes)
    try:
        import psutil

        available = psutil.virtual_memory().available
    except Exception:
        available = 512 * 1024 * 1024
    fraction = float(os.environ.get("TRACE_PACK_MEMORY_FRACTION", DEFAULT_PACK_MEMORY_FRACTION))
    return max(0, int(available * fraction) - reserved_bytes)


def _estimate_stacked_array_bytes(shape: JsonlTraceShape) -> int:
    """Approximate RAM for pre-allocated NPZ arrays (excluding sidecar text)."""
    n = shape.n_samples
    g = shape.gen_length
    s = shape.max_steps
    total = n * g * 2  # unmask_step_at_pos int16
    total += n * (shape.max_prompt_len + shape.max_final_len) * 4
    total += n * 64  # scalars / token lengths / times
    for field, dtype in FIELD_DTYPES.items():
        total += n * s * g * np.dtype(dtype).itemsize
        if field in OPTIONAL_STEP_FIELDS:
            total += n * s
    return total


def jsonl_pack_batch_size(
    jsonl_path: Union[str, Path],
    shape: JsonlTraceShape,
    *,
    stacked_bytes: int,
) -> int:
    """Choose how many JSONL samples to parse per batch from free RAM."""
    fixed = os.environ.get("TRACE_PACK_BATCH_SAMPLES", "").strip()
    if fixed:
        return max(1, min(int(fixed), shape.n_samples))

    file_bytes = Path(jsonl_path).stat().st_size
    avg_line_bytes = max(1, file_bytes // max(shape.n_samples, 1))
    expansion = float(os.environ.get("TRACE_PACK_JSON_EXPANSION", DEFAULT_PACK_JSON_EXPANSION))
    per_sample_bytes = max(1, int(avg_line_bytes * expansion))

    budget = _available_json_batch_budget_bytes(reserved_bytes=stacked_bytes)
    batch = max(1, budget // per_sample_bytes)

    max_batch = os.environ.get("TRACE_PACK_MAX_BATCH", "").strip()
    if max_batch:
        batch = min(batch, int(max_batch))
    min_batch = int(os.environ.get("TRACE_PACK_MIN_BATCH", "1"))
    batch = max(min_batch, batch)
    return min(batch, shape.n_samples)


def iter_jsonl_trace_batches(
    path: Union[str, Path],
    batch_size: int,
) -> Iterator[list[dict]]:
    """Load JSONL samples in batches of ``batch_size`` dicts."""
    batch: list[dict] = []
    for sample in iter_jsonl_trace_samples(path):
        batch.append(sample)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


@dataclass(frozen=True)
class JsonlTraceShape:
    task_name: Optional[str]
    n_samples: int
    gen_length: int
    max_steps: int
    max_prompt_len: int
    max_final_len: int


def _sample_gen_length(sample: dict) -> int:
    gl = int(sample.get("gen_length") or 0)
    if gl > 0:
        return gl
    step_traces = sample.get("step_traces") or []
    if not step_traces:
        raise ValueError("trace has no step_traces")
    return int(len(squeeze_to_1d(step_traces[0].get("mask_index") or [0])))


def scan_jsonl_trace_shape(path: Union[str, Path]) -> JsonlTraceShape:
    """Single streaming pass: count samples and discover array shapes."""
    path = Path(path)
    task_name: Optional[str] = None
    n_samples = 0
    gen_lengths: set[int] = set()
    max_steps = 0
    max_prompt_len = 0
    max_final_len = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if task_name is None:
                    task_name = _parse_jsonl_task_header(line)
                continue

            sample = json.loads(line)
            step_traces = sample.get("step_traces") or []
            if not step_traces:
                raise ValueError(f"sample {n_samples}: trace has no step_traces")
            max_steps = max(max_steps, len(step_traces))
            gen_lengths.add(_sample_gen_length(sample))

            prompt_tokens = _token_ids_from_field(sample.get("prompt"))
            if prompt_tokens:
                max_prompt_len = max(max_prompt_len, len(prompt_tokens))
            final_tokens = _token_ids_from_field(sample.get("final_sequence"))
            if final_tokens:
                max_final_len = max(max_final_len, len(final_tokens))

            n_samples += 1
            del sample

    if n_samples == 0:
        raise ValueError(f"No samples found in {path}")
    gen_lengths.discard(0)
    if len(gen_lengths) != 1:
        raise ValueError(f"Mixed gen_length values in {path}: {sorted(gen_lengths)}")

    return JsonlTraceShape(
        task_name=task_name,
        n_samples=n_samples,
        gen_length=next(iter(gen_lengths)),
        max_steps=max_steps,
        max_prompt_len=max_prompt_len,
        max_final_len=max_final_len,
    )


def _shape_from_samples(
    samples: list[dict],
    *,
    task_name: Optional[str] = None,
) -> JsonlTraceShape:
    if not samples:
        raise ValueError("No samples to pack")

    gen_lengths: set[int] = set()
    max_steps = 0
    max_prompt_len = 0
    max_final_len = 0
    for sample in samples:
        step_traces = sample.get("step_traces") or []
        if not step_traces:
            raise ValueError("trace has no step_traces")
        max_steps = max(max_steps, len(step_traces))
        gen_lengths.add(_sample_gen_length(sample))
        prompt_tokens = _token_ids_from_field(sample.get("prompt"))
        if prompt_tokens:
            max_prompt_len = max(max_prompt_len, len(prompt_tokens))
        final_tokens = _token_ids_from_field(sample.get("final_sequence"))
        if final_tokens:
            max_final_len = max(max_final_len, len(final_tokens))

    gen_lengths.discard(0)
    if len(gen_lengths) != 1:
        raise ValueError(f"Mixed gen_length values: {sorted(gen_lengths)}")
    return JsonlTraceShape(
        task_name=task_name,
        n_samples=len(samples),
        gen_length=next(iter(gen_lengths)),
        max_steps=max_steps,
        max_prompt_len=max_prompt_len,
        max_final_len=max_final_len,
    )


def _fill_token_row(
    sample: dict,
    index: int,
    field: str,
    matrix: np.ndarray,
    lengths: np.ndarray,
) -> None:
    tokens = _token_ids_from_field(sample.get(field))
    if tokens is None:
        lengths[index] = -1
        return
    lengths[index] = len(tokens)
    if tokens:
        matrix[index, : len(tokens)] = np.asarray(tokens, dtype=np.int32)


def _preallocate_stacked(
    *,
    n_samples: int,
    gen_length: int,
    max_steps: int,
    max_prompt_len: int,
    max_final_len: int,
) -> Dict[str, np.ndarray]:
    stacked: Dict[str, np.ndarray] = {
        "n_samples": np.int32(n_samples),
        "gen_length": np.int32(gen_length),
        "n_steps": np.zeros(n_samples, dtype=np.int32),
        "prompt_len": np.zeros(n_samples, dtype=np.int32),
        "unmask_step_at_pos": np.zeros((n_samples, gen_length), dtype=np.int16),
        "gen_only": np.zeros(n_samples, dtype=np.bool_),
        "start_time": np.full(n_samples, np.nan, dtype=np.float64),
        "end_time": np.full(n_samples, np.nan, dtype=np.float64),
        "prompt_tokens": np.zeros((n_samples, max_prompt_len), dtype=np.int32),
        "prompt_token_len": np.full(n_samples, -1, dtype=np.int32),
        "final_sequence_tokens": np.zeros((n_samples, max_final_len), dtype=np.int32),
        "final_sequence_len": np.full(n_samples, -1, dtype=np.int32),
    }
    for field, dtype in FIELD_DTYPES.items():
        stacked[field] = np.zeros((n_samples, max_steps, gen_length), dtype=dtype)
        if field in OPTIONAL_STEP_FIELDS:
            stacked[f"{field}_present"] = np.zeros((n_samples, max_steps), dtype=np.bool_)
    return stacked


def _pack_sample_into_stacked(
    sample: dict,
    index: int,
    *,
    gen_length: int,
    stacked: Dict[str, np.ndarray],
) -> bytes:
    """Fill pre-allocated stacked arrays for one sample; return encoded sidecar bytes."""
    step_traces = sample.get("step_traces") or []
    if not step_traces:
        raise ValueError("trace has no step_traces")

    n_steps = len(step_traces)
    stacked["n_steps"][index] = np.int32(n_steps)
    stacked["prompt_len"][index] = np.int32(int(sample.get("prompt_len") or 0))
    stacked["unmask_step_at_pos"][index] = _unmask_step_at_pos(
        sample.get("unmasking_order"),
        gen_length,
    )

    for field, dtype in FIELD_DTYPES.items():
        row = _step_array(step_traces, field, dtype, gen_length=gen_length)
        stacked[field][index, :n_steps] = row
        if field in OPTIONAL_STEP_FIELDS:
            stacked[f"{field}_present"][index, :n_steps] = _step_presence(step_traces, field)

    return _encode_sidecar_blob(_sample_sidecar_dict(sample, step_traces))


def _append_sample_to_stacked(
    idx: int,
    sample: dict,
    *,
    gen_length: int,
    stacked: Dict[str, np.ndarray],
    sidecar_offsets: np.ndarray,
    sidecar_data: bytearray,
) -> None:
    sidecar_offsets[idx] = len(sidecar_data)
    sidecar_data.extend(
        _pack_sample_into_stacked(
            sample,
            idx,
            gen_length=gen_length,
            stacked=stacked,
        )
    )
    _fill_token_row(sample, idx, "prompt", stacked["prompt_tokens"], stacked["prompt_token_len"])
    _fill_token_row(
        sample,
        idx,
        "final_sequence",
        stacked["final_sequence_tokens"],
        stacked["final_sequence_len"],
    )
    stacked["gen_only"][idx] = bool(sample.get("gen_only", True))
    if sample.get("start_time") is not None:
        stacked["start_time"][idx] = float(sample["start_time"])
    if sample.get("end_time") is not None:
        stacked["end_time"][idx] = float(sample["end_time"])


def _pack_samples_into_stacked(
    stacked: Dict[str, np.ndarray],
    shape: JsonlTraceShape,
    samples: Iterable[dict],
    *,
    start_index: int = 0,
    sidecar_offsets: Optional[np.ndarray] = None,
    sidecar_data: Optional[bytearray] = None,
    finalize: bool = True,
) -> tuple[np.ndarray, bytes]:
    if sidecar_offsets is None:
        sidecar_offsets = np.zeros(shape.n_samples + 1, dtype=np.int64)
    if sidecar_data is None:
        sidecar_data = bytearray()

    n_packed = 0
    for sample in samples:
        idx = start_index + n_packed
        _append_sample_to_stacked(
            idx,
            sample,
            gen_length=shape.gen_length,
            stacked=stacked,
            sidecar_offsets=sidecar_offsets,
            sidecar_data=sidecar_data,
        )
        n_packed += 1

    if finalize:
        sidecar_offsets[shape.n_samples] = len(sidecar_data)
    return sidecar_offsets, bytes(sidecar_data)


def _npz_compress_settings() -> tuple[bool, int]:
    """Return (use_deflate, compresslevel)."""
    raw = os.environ.get("TRACE_NPZ_COMPRESS", "1").strip().lower()
    use = raw not in ("0", "false", "no", "off")
    level = int(os.environ.get("TRACE_NPZ_COMPRESSLEVEL", str(DEFAULT_NPZ_COMPRESSLEVEL)))
    return use, max(0, min(9, level))


def _write_npz_archive(
    output_path: Union[str, Path],
    stacked: Dict[str, np.ndarray],
    sidecar_offsets: np.ndarray,
    sidecar_bytes: bytes,
    *,
    task_name: Optional[str] = None,
) -> Path:
    output_path = Path(output_path)
    stacked["sidecar_bytes"] = (
        np.frombuffer(sidecar_bytes, dtype=np.uint8) if sidecar_bytes else np.zeros(0, dtype=np.uint8)
    )
    stacked["sidecar_offsets"] = sidecar_offsets
    if task_name is not None:
        stacked["task"] = np.array(str(task_name))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    use_deflate, level = _npz_compress_settings()
    t0 = time.perf_counter()
    compression = zipfile.ZIP_DEFLATED if use_deflate else zipfile.ZIP_STORED
    # ZIP_STORED rejects compresslevel on some Python builds — only pass when deflating.
    zf_kwargs: dict = {"compression": compression}
    if use_deflate:
        zf_kwargs["compresslevel"] = level
    with zipfile.ZipFile(output_path, mode="w", **zf_kwargs) as zf:
        for key, value in stacked.items():
            buf = io.BytesIO()
            np.save(buf, np.asanyarray(value), allow_pickle=False)
            zf.writestr(f"{key}.npy", buf.getvalue())
    logger.info(
        "wrote NPZ %s (%.1f MiB, compress=%s level=%s) in %.1fs",
        output_path.name,
        output_path.stat().st_size / (1024 * 1024),
        use_deflate,
        level if use_deflate else "-",
        time.perf_counter() - t0,
    )
    return output_path


def _pack_trace_shape_to_npz(
    shape: JsonlTraceShape,
    samples: Iterable[dict],
    output_path: Union[str, Path],
    *,
    task_name: Optional[str] = None,
) -> Path:
    task = task_name if task_name is not None else shape.task_name
    stacked = _preallocate_stacked(
        n_samples=shape.n_samples,
        gen_length=shape.gen_length,
        max_steps=shape.max_steps,
        max_prompt_len=shape.max_prompt_len,
        max_final_len=shape.max_final_len,
    )
    sidecar_offsets, sidecar_bytes = _pack_samples_into_stacked(stacked, shape, samples)
    return _write_npz_archive(output_path, stacked, sidecar_offsets, sidecar_bytes, task_name=task)


def pack_jsonl_to_npz(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    task_name: Optional[str] = None,
) -> Path:
    """Convert JSONL to NPZ using RAM-aware batching (two passes over the file)."""
    input_path = Path(input_path)
    shape = scan_jsonl_trace_shape(input_path)
    task = task_name if task_name is not None else shape.task_name

    stacked = _preallocate_stacked(
        n_samples=shape.n_samples,
        gen_length=shape.gen_length,
        max_steps=shape.max_steps,
        max_prompt_len=shape.max_prompt_len,
        max_final_len=shape.max_final_len,
    )
    stacked_bytes = _estimate_stacked_array_bytes(shape)
    batch_size = jsonl_pack_batch_size(input_path, shape, stacked_bytes=stacked_bytes)
    budget = _available_json_batch_budget_bytes(reserved_bytes=stacked_bytes)
    avg_line = max(1, input_path.stat().st_size // max(shape.n_samples, 1))
    logger.info(
        "JSONL pack: %d samples, batch_size=%d (budget=%.1f GiB, ~%.1f KiB/line, arrays=%.1f GiB)",
        shape.n_samples,
        batch_size,
        budget / (1024**3),
        avg_line / 1024,
        stacked_bytes / (1024**3),
    )

    sidecar_offsets = np.zeros(shape.n_samples + 1, dtype=np.int64)
    sidecar_data = bytearray()
    sample_index = 0
    batch_i = 0
    t_pack = time.perf_counter()
    for batch in iter_jsonl_trace_batches(input_path, batch_size):
        _pack_samples_into_stacked(
            stacked,
            shape,
            batch,
            start_index=sample_index,
            sidecar_offsets=sidecar_offsets,
            sidecar_data=sidecar_data,
            finalize=False,
        )
        sample_index += len(batch)
        batch_i += 1
        if batch_i == 1 or sample_index == shape.n_samples or batch_i % 4 == 0:
            logger.info(
                "JSONL pack progress: %d/%d samples (%.1fs)",
                sample_index,
                shape.n_samples,
                time.perf_counter() - t_pack,
            )
        del batch
        # Full GC every batch is expensive; run periodically.
        if batch_i % 4 == 0:
            gc.collect()

    sidecar_offsets[shape.n_samples] = len(sidecar_data)
    return _write_npz_archive(
        output_path,
        stacked,
        sidecar_offsets,
        bytes(sidecar_data),
        task_name=task,
    )


def pack_samples_to_npz(
    samples: list[dict],
    output_path: Union[str, Path],
    *,
    task_name: Optional[str] = None,
) -> Path:
    """Pack in-memory trace samples into an NPZ archive."""
    shape = _shape_from_samples(samples, task_name=task_name)
    return _pack_trace_shape_to_npz(shape, samples, output_path, task_name=task_name)


def arrays_to_trace_dict(
    *,
    gen_length: int,
    n_steps: int,
    prompt_len: int,
    arrays: Dict[str, np.ndarray],
    sample_index: int,
    prompt_tokens: Optional[np.ndarray] = None,
    prompt_token_len: Optional[np.ndarray] = None,
    final_sequence_tokens: Optional[np.ndarray] = None,
    final_sequence_len: Optional[np.ndarray] = None,
    gen_only: Optional[np.ndarray] = None,
    start_time: Optional[np.ndarray] = None,
    end_time: Optional[np.ndarray] = None,
    sidecar_blob: Optional[bytes] = None,
) -> dict:
    """Reconstruct one JSON-compatible trace dict from packed arrays."""
    step_traces: List[dict] = []
    sidecar = _decode_sidecar_blob(sidecar_blob or b"")
    step_extras = sidecar.get("step_extras") or []

    for t in range(int(n_steps)):
        extras = step_extras[t] if t < len(step_extras) else {}
        step = {"step": t}
        for field in STEP_FIELDS:
            present_key = f"{field}_present"
            if present_key in arrays and not bool(arrays[present_key][sample_index, t]):
                step[field] = None
                continue
            # Keep a (1, G) ndarray view — avoid .tolist() (huge Python object graphs).
            arr = arrays[field][sample_index, t]
            step[field] = np.asarray(arr).reshape(1, gen_length)
        step["intervention_applied"] = bool(extras.get("intervention_applied", False))
        step["intervention_type"] = extras.get("intervention_type")
        step["metadata"] = extras.get("metadata") or {}
        step_traces.append(step)

    unmask = arrays["unmask_step_at_pos"][sample_index]
    unmasking_order = {
        int(pos): int(step)
        for pos, step in enumerate(unmask.tolist())
        if int(step) >= 0
    }

    trace: dict = {
        "gen_only": bool(gen_only[sample_index]) if gen_only is not None else True,
        "prompt": None,
        "final_sequence": None,
        "prompt_len": int(prompt_len),
        "gen_length": int(gen_length),
        "step_traces": step_traces,
        "unmasking_order": unmasking_order,
        "start_time": None,
        "end_time": None,
    }

    if start_time is not None:
        val = float(start_time[sample_index])
        trace["start_time"] = None if np.isnan(val) else val
    if end_time is not None:
        val = float(end_time[sample_index])
        trace["end_time"] = None if np.isnan(val) else val

    if prompt_tokens is not None and prompt_token_len is not None:
        plen = int(prompt_token_len[sample_index])
        if plen >= 0:
            trace["prompt"] = _batched_token_field(
                [int(t) for t in prompt_tokens[sample_index, :plen].tolist()]
            )

    if final_sequence_tokens is not None and final_sequence_len is not None:
        flen = int(final_sequence_len[sample_index])
        if flen >= 0:
            trace["final_sequence"] = _batched_token_field(
                [int(t) for t in final_sequence_tokens[sample_index, :flen].tolist()]
            )

    metadata = sidecar.get("metadata")
    if metadata:
        trace["metadata"] = metadata
    return trace


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


def read_npz_task_name(path: Union[str, Path]) -> Optional[str]:
    """Return the optional ``task`` string stored in an NPZ archive."""
    with np.load(path, allow_pickle=False) as data:
        if "task" in data:
            return str(data["task"])
    return None


def _load_npz_samples(path: Path) -> tuple[Optional[str], List[dict]]:
    return read_npz_task_name(path), list(iter_npz_traces(path))


def load_trace_samples(path: Union[str, Path]) -> tuple[Optional[str], List[dict]]:
    """Load all samples from JSONL, JSON, or NPZ trace file."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl_task_name(path), list(iter_jsonl_trace_samples(path))
    if suffix == ".json":
        return _load_json_samples(path)
    if suffix == ".npz":
        return _load_npz_samples(path)
    raise ValueError(f"Unsupported trace format: {path}")


def convert_jsonl_to_npz(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    *,
    samples: Optional[List[dict]] = None,
    task_name: Optional[str] = None,
) -> Path:
    """Convert a JSONL/JSON trace file to an NPZ archive (no repair)."""
    input_path = Path(input_path)
    output_path = Path(output_path or input_path.with_suffix(".npz"))

    if samples is not None:
        return pack_samples_to_npz(samples, output_path, task_name=task_name)
    if input_path.suffix.lower() == ".jsonl":
        return pack_jsonl_to_npz(input_path, output_path, task_name=task_name)

    loaded_task, loaded_samples = load_trace_samples(input_path)
    return pack_samples_to_npz(
        loaded_samples,
        output_path,
        task_name=task_name if task_name is not None else loaded_task,
    )


@dataclass
class RepairedPackResult:
    """Result of repair+pack without writing an intermediate repaired JSONL."""

    path: Path
    n_samples: int
    task_name: Optional[str]
    repairs: list[str] = field(default_factory=list)
    # Sample index -> repaired dict for round-trip spot checks (not full file).
    roundtrip_samples: Dict[int, dict] = field(default_factory=dict)


def pack_repaired_jsonl_to_npz(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    task_name: Optional[str] = None,
    roundtrip_check_samples: int = 3,
    is_fdm: Optional[bool] = None,
    workers: int = 1,
) -> RepairedPackResult:
    """Repair while packing JSONL → NPZ (no intermediate repaired JSONL).

    Two streaming passes via ``iter_repaired_jsonl``:
    1) measure post-repair shape / dedup count
    2) pack into preallocated arrays; stash a few samples for round-trip
    """
    from traces.trace_checks import _roundtrip_indices, iter_repaired_jsonl

    input_path = Path(input_path)
    output_path = Path(output_path)
    header_task = read_jsonl_task_name(input_path)
    task = task_name if task_name is not None else header_task

    t0 = time.perf_counter()
    n_samples = 0
    removed = 0
    gen_lengths: set[int] = set()
    max_steps = 0
    max_prompt_len = 0
    max_final_len = 0
    repair_msgs: list[str] = []

    for kind, payload in iter_repaired_jsonl(
        input_path, is_fdm=is_fdm, workers=workers
    ):
        if kind == "comment":
            continue
        if kind == "duplicate":
            removed += 1
            continue
        fixed, reps = payload
        step_traces = fixed.get("step_traces") or []
        if not step_traces:
            raise ValueError(f"sample {n_samples}: repaired trace has no step_traces")
        max_steps = max(max_steps, len(step_traces))
        gen_lengths.add(_sample_gen_length(fixed))
        prompt_tokens = _token_ids_from_field(fixed.get("prompt"))
        if prompt_tokens:
            max_prompt_len = max(max_prompt_len, len(prompt_tokens))
        final_tokens = _token_ids_from_field(fixed.get("final_sequence"))
        if final_tokens:
            max_final_len = max(max_final_len, len(final_tokens))
        for rep in reps:
            if len(repair_msgs) < 40:
                repair_msgs.append(f"sample {n_samples}: {rep}")
        n_samples += 1
        if n_samples == 1 or n_samples % 50 == 0:
            logger.info(
                "repair+pack scan: %d kept (%d dupes dropped) %.1fs",
                n_samples,
                removed,
                time.perf_counter() - t0,
            )

    if n_samples == 0:
        raise ValueError(f"No samples to pack after repair in {input_path}")
    gen_lengths.discard(0)
    if len(gen_lengths) != 1:
        raise ValueError(f"Mixed gen_length after repair in {input_path}: {sorted(gen_lengths)}")

    shape = JsonlTraceShape(
        task_name=task,
        n_samples=n_samples,
        gen_length=next(iter(gen_lengths)),
        max_steps=max_steps,
        max_prompt_len=max_prompt_len,
        max_final_len=max_final_len,
    )
    logger.info(
        "repair+pack scan done: n=%d max_steps=%d gen=%d (%.1fs)%s",
        shape.n_samples,
        shape.max_steps,
        shape.gen_length,
        time.perf_counter() - t0,
        f", dropped {removed} dupes" if removed else "",
    )
    if removed:
        repair_msgs.insert(
            0,
            f"removed {removed} duplicate trace(s) by prompt ({n_samples} kept)",
        )

    wanted = set(_roundtrip_indices(n_samples, check_samples=roundtrip_check_samples))
    stacked = _preallocate_stacked(
        n_samples=shape.n_samples,
        gen_length=shape.gen_length,
        max_steps=shape.max_steps,
        max_prompt_len=shape.max_prompt_len,
        max_final_len=shape.max_final_len,
    )
    sidecar_offsets = np.zeros(shape.n_samples + 1, dtype=np.int64)
    sidecar_data = bytearray()
    roundtrip_samples: Dict[int, dict] = {}
    idx = 0
    t1 = time.perf_counter()
    for kind, payload in iter_repaired_jsonl(
        input_path, is_fdm=is_fdm, workers=workers
    ):
        if kind != "sample":
            continue
        fixed, _reps = payload
        if idx in wanted:
            # Deep copy so packing helpers cannot mutate the round-trip reference.
            roundtrip_samples[idx] = copy.deepcopy(fixed)
        _append_sample_to_stacked(
            idx,
            fixed,
            gen_length=shape.gen_length,
            stacked=stacked,
            sidecar_offsets=sidecar_offsets,
            sidecar_data=sidecar_data,
        )
        idx += 1
        if idx == 1 or idx == n_samples or idx % 50 == 0:
            logger.info(
                "repair+pack write: %d/%d samples (%.1fs)",
                idx,
                n_samples,
                time.perf_counter() - t1,
            )
        if idx % 32 == 0:
            gc.collect()

    sidecar_offsets[n_samples] = len(sidecar_data)
    _write_npz_archive(
        output_path,
        stacked,
        sidecar_offsets,
        bytes(sidecar_data),
        task_name=task,
    )
    logger.info(
        "repair+pack total %.1fs -> %s",
        time.perf_counter() - t0,
        output_path.name,
    )
    return RepairedPackResult(
        path=output_path,
        n_samples=n_samples,
        task_name=task,
        repairs=repair_msgs,
        roundtrip_samples=roundtrip_samples,
    )


def _sidecar_blob_for_sample(
    sidecar_bytes: np.ndarray,
    sidecar_offsets: np.ndarray,
    sample_index: int,
) -> bytes:
    if sidecar_bytes.size == 0:
        return b""
    start = int(sidecar_offsets[sample_index])
    end = int(sidecar_offsets[sample_index + 1])
    if start >= end:
        return b""
    return bytes(sidecar_bytes[start:end])


def iter_npz_traces(path: Union[str, Path]) -> Iterator[dict]:
    """Yield one trace dict per sample from an NPZ trace archive."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
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

        prompt_tokens = data["prompt_tokens"] if "prompt_tokens" in data else None
        prompt_token_len = data["prompt_token_len"] if "prompt_token_len" in data else None
        final_sequence_tokens = (
            data["final_sequence_tokens"] if "final_sequence_tokens" in data else None
        )
        final_sequence_len = data["final_sequence_len"] if "final_sequence_len" in data else None
        gen_only = data["gen_only"] if "gen_only" in data else None
        start_time = data["start_time"] if "start_time" in data else None
        end_time = data["end_time"] if "end_time" in data else None
        sidecar_bytes = data["sidecar_bytes"] if "sidecar_bytes" in data else None
        sidecar_offsets = data["sidecar_offsets"] if "sidecar_offsets" in data else None

        for i in range(n_samples):
            sidecar_blob = None
            if sidecar_bytes is not None and sidecar_offsets is not None:
                sidecar_blob = _sidecar_blob_for_sample(sidecar_bytes, sidecar_offsets, i)
            yield arrays_to_trace_dict(
                gen_length=gen_length,
                n_steps=int(n_steps[i]),
                prompt_len=int(prompt_len[i]),
                arrays=arrays,
                sample_index=i,
                prompt_tokens=prompt_tokens,
                prompt_token_len=prompt_token_len,
                final_sequence_tokens=final_sequence_tokens,
                final_sequence_len=final_sequence_len,
                gen_only=gen_only,
                start_time=start_time,
                end_time=end_time,
                sidecar_blob=sidecar_blob,
            )


def compression_ratio(input_path: Union[str, Path], output_path: Union[str, Path]) -> float:
    """Return input_size / output_size for converted files."""
    src = Path(input_path).stat().st_size
    dst = Path(output_path).stat().st_size
    if dst <= 0:
        return float("inf")
    return src / dst
