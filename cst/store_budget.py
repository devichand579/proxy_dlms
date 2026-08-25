"""CPU / GPU / disk budgets for CST hidden-store auto-batching."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

logger = logging.getLogger("cst.store_budget")

STORAGE_ELEM_BYTES = 2  # bfloat16
_DISK_WRITE_FACTOR = 1.15  # final .pt plus a little slack (no second copy)


def disk_free_bytes(path: str | Path) -> int:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return int(shutil.disk_usage(p).free)


def cpu_available_bytes() -> Optional[int]:
    """Unused host RAM from MemAvailable, or None if unknown."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def gpu_free_bytes(device: str | torch.device | None = "cuda") -> Optional[int]:
    if not torch.cuda.is_available():
        return None
    try:
        dev = torch.device(device if device is not None else "cuda")
        if dev.type != "cuda":
            return None
        idx = int(dev.index) if dev.index is not None else torch.cuda.current_device()
        free, _total = torch.cuda.mem_get_info(idx)
        return int(free)
    except Exception:
        return None


def model_weight_bytes(model_dir: str | Path) -> int:
    """On-disk parameter files (bf16/fp16 weights ≈ VRAM once loaded)."""
    root = Path(model_dir)
    n = 0
    for pat in ("*.safetensors", "*.bin", "*.pt"):
        for p in root.glob(pat):
            if p.name.startswith("training_args"):
                continue
            try:
                n += int(p.stat().st_size)
            except OSError:
                pass
    return n


def read_model_config_sizes(model_dir: str | Path) -> Tuple[int, int, int]:
    """``(d_model, vocab_size, n_layers)`` from ``config.json``."""
    cfg_path = Path(model_dir) / "config.json"
    data: Dict[str, Any] = {}
    if cfg_path.is_file():
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
    d = int(data.get("d_model") or data.get("hidden_size") or 4096)
    v = int(data.get("vocab_size") or data.get("embedding_size") or 126464)
    n = int(data.get("n_layers") or data.get("num_hidden_layers") or 32)
    return d, v, n


def estimate_chunk_bytes(
    batch_size: int,
    packed_seq_len: int,
    d_model: int,
    n_records: int,
    *,
    elem_bytes: int = STORAGE_ELEM_BYTES,
) -> int:
    """Uncompressed size of one ``[T, B, L, D]`` bf16 chunk."""
    return (
        max(int(n_records), 1)
        * max(int(batch_size), 1)
        * max(int(packed_seq_len), 1)
        * max(int(d_model), 1)
        * int(elem_bytes)
    )


def estimate_cpu_live_bytes(
    batch_size: int,
    packed_seq_len: int,
    d_model: int,
    n_records: int,
    *,
    stream_to_disk: bool,
) -> int:
    """Host tensors held during collect (stream keeps a few steps, not T)."""
    one = estimate_chunk_bytes(batch_size, packed_seq_len, d_model, 1)
    if stream_to_disk:
        return 3 * one
    return estimate_chunk_bytes(batch_size, packed_seq_len, d_model, n_records)


def estimate_gpu_live_bytes(
    batch_size: int,
    full_seq_len: int,
    d_model: int,
    vocab_size: int,
    n_layers: int,
) -> int:
    """Activation / logits working set on top of already-loaded weights."""
    b = max(int(batch_size), 1)
    seq = max(int(full_seq_len), 1)
    d = max(int(d_model), 1)
    hidden = b * seq * d * STORAGE_ELEM_BYTES
    logits = b * seq * max(int(vocab_size), 1) * STORAGE_ELEM_BYTES
    layers = hidden * max(int(n_layers), 1)
    return hidden + logits + layers


def _max_b_for_budget(per_sample: int, budget: int) -> int:
    if per_sample <= 0 or budget <= 0:
        return 0
    return max(int(budget) // int(per_sample), 0)


def max_inference_batch_size(
    *,
    requested: int,
    packed_seq_len: int,
    full_seq_len: int,
    d_model: int,
    vocab_size: int,
    n_layers: int,
    n_records: int,
    scratch: str | Path,
    stream_to_disk: bool = True,
    disk_headroom_gb: float = 20.0,
    cpu_headroom_gb: float = 16.0,
    vram_headroom_gb: float = 8.0,
    max_chunk_gb: Optional[float] = None,
    model_bytes: int = 0,
    device: str = "cuda",
) -> Tuple[int, Dict[str, int]]:
    """Largest B in ``[1, requested]`` that fits disk, RAM, VRAM, and optional chunk cap."""
    req = max(int(requested), 1)
    packed = max(int(packed_seq_len), 1)
    full = max(int(full_seq_len), 1)
    d = max(int(d_model), 1)
    t = max(int(n_records), 1)
    limits: Dict[str, int] = {"requested": req}

    per_disk = int(_DISK_WRITE_FACTOR * estimate_chunk_bytes(1, packed, d, t))
    disk_budget = disk_free_bytes(scratch) - int(float(disk_headroom_gb) * 1e9)
    limits["disk"] = _max_b_for_budget(per_disk, disk_budget)

    cpu_free = cpu_available_bytes()
    if cpu_free is None:
        limits["cpu"] = req
    else:
        cpu_budget = int(cpu_free) - int(float(cpu_headroom_gb) * 1e9)
        per_cpu = estimate_cpu_live_bytes(
            1, packed, d, t, stream_to_disk=stream_to_disk
        )
        limits["cpu"] = _max_b_for_budget(per_cpu, cpu_budget)

    gpu_free = gpu_free_bytes(device)
    if gpu_free is None:
        limits["vram"] = req
    else:
        gpu_budget = int(gpu_free) - int(model_bytes) - int(float(vram_headroom_gb) * 1e9)
        per_gpu = estimate_gpu_live_bytes(1, full, d, vocab_size, n_layers)
        limits["vram"] = _max_b_for_budget(per_gpu, gpu_budget)

    if max_chunk_gb is None:
        limits["chunk"] = req
    else:
        cap = int(float(max_chunk_gb) * 1e9)
        per_chunk = estimate_chunk_bytes(1, packed, d, t)
        limits["chunk"] = _max_b_for_budget(per_chunk, cap)

    chosen = min(req, limits["disk"], limits["cpu"], limits["vram"], limits["chunk"])
    chosen = max(int(chosen), 1)
    if chosen < req:
        logger.info(
            "store auto-batch: B=%d (yaml max %d); limits disk=%d cpu=%d vram=%d "
            "max_chunk=%d | packed_L=%d full_L=%d T=%d d=%d",
            chosen,
            req,
            limits["disk"],
            limits["cpu"],
            limits["vram"],
            limits["chunk"],
            packed,
            full,
            t,
            d,
        )
    else:
        logger.info(
            "store auto-batch: keeping B=%d (disk=%d cpu=%d vram=%d max_chunk=%d)",
            chosen,
            limits["disk"],
            limits["cpu"],
            limits["vram"],
            limits["chunk"],
        )
    return chosen, limits
