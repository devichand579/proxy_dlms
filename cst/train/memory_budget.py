"""Pick CST train pin / microbatch / anchor-block from free RAM and VRAM.

Same idea as store auto-batch: YAML values are optional *caps*. Disk is not a
primary constraint (two Hub files on scratch is only a prefetch nicety).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from cst.store_budget import cpu_available_bytes, disk_free_bytes, gpu_free_bytes

logger = logging.getLogger("cst.train.memory")

_MMAP_SLACK_GB = 8.0
_HOST_OVERHEAD_GB = 6.0
_MIN_GPU_RESERVE_GB = 12.0
_ACT_FUDGE = 1.6
_FIXED_ACT_GB = 1.5


def cst_numel(d: int, k: int, w: int) -> int:
    return int(d) * int(k) * (1 + int(w)) + int(k)


def bytes_per_cst(d: int, k: int, w: int, *, has_moments: bool = True) -> int:
    """fp32 params + (optional) two bf16 Adam moment buffers."""
    n = cst_numel(d, k, w)
    params = n * 4
    moments = n * 2 * 2 if has_moments else 0
    return int(params + moments)


def _gb(n: int | float) -> float:
    return float(n) / 1e9


def _is_blank(v: Any) -> bool:
    return v is None or str(v).strip().lower() in ("", "null", "none")


def _as_bool(v: Any, default: bool) -> bool:
    if _is_blank(v):
        return default
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "on")
    return bool(v)


def _as_cap_float(v: Any) -> Optional[float]:
    if _is_blank(v):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x < 0:
        return None
    return x


def _as_cap_int(v: Any) -> Optional[int]:
    if _is_blank(v):
        return None
    try:
        x = int(v)
    except (TypeError, ValueError):
        return None
    if x < 1:
        return None
    return x


def activation_bytes(
    *,
    anchor_block: int,
    microbatch: int,
    seq_len: int,
    d: int,
    k: int,
    w: int,
    prefetch_block: bool,
    bytes_per_step: int,
) -> int:
    """Conservative live activation + streamed-CST reservation for one block.

    Worst case (late in the sweep) the live window is entirely unpinned, so we
    reserve ``live_csts * bytes_per_step`` on top of whatever is already pinned.
    """
    g = max(int(anchor_block), 1)
    mb = max(int(microbatch), 1)
    seq = max(int(seq_len), 1)
    live = g + int(w) - 1
    if prefetch_block:
        live = live + g
    live = max(int(live), 1)
    win = g * mb * int(w) * seq * int(d) * 2
    enc = g * int(w) * mb * seq * int(k) * 4
    tgt = (g + int(w)) * mb * seq * int(d) * 4
    act = int(_ACT_FUDGE * (win + enc + tgt)) + int(_FIXED_ACT_GB * 1e9)
    streamed = live * int(bytes_per_step)
    return int(act + streamed)


def _mb_ladder(n: int) -> List[int]:
    n = max(int(n), 1)
    out: List[int] = []
    for x in (n, 96, 64, 48, 32, 24, 16, 12, 8, 4, 2, 1):
        if 1 <= x <= n and x not in out:
            out.append(x)
    return out


def choose_chunk_compute(
    *,
    batch: int,
    seq_len: int,
    d: int,
    k: int,
    w: int,
    bytes_per_step: int,
    device: Any,
    cap_anchor_block: Optional[int] = None,
    cap_microbatch: Optional[int] = None,
    prefetch_block: bool = True,
    vram_headroom_gb: float = 8.0,
) -> Tuple[int, int]:
    """Largest microbatch, then largest G, that fit in *currently free* VRAM."""
    b = max(int(batch), 1)
    max_mb = min(b, cap_microbatch) if cap_microbatch is not None else b
    max_g = cap_anchor_block if cap_anchor_block is not None else 32
    max_g = max(int(max_g), 1)

    gpu_free = gpu_free_bytes(device)
    if gpu_free is None:
        return max_g, max_mb
    budget = int(gpu_free) - int(float(vram_headroom_gb) * 1e9)
    if budget <= 0:
        logger.warning(
            "train auto-memory: free VRAM %.1f GB is at/under headroom; using G=1 mb=1",
            _gb(gpu_free),
        )
        return 1, 1

    for mb in _mb_ladder(max_mb):
        for g in range(max_g, 0, -1):
            need = activation_bytes(
                anchor_block=g,
                microbatch=mb,
                seq_len=seq_len,
                d=d,
                k=k,
                w=w,
                prefetch_block=prefetch_block,
                bytes_per_step=bytes_per_step,
            )
            if need <= budget:
                logger.info(
                    "train auto-memory: chunk B=%d L=%d -> anchor_block=%d microbatch=%d "
                    "(need %.1f GB, free %.1f GB)",
                    b,
                    seq_len,
                    g,
                    mb,
                    _gb(need),
                    _gb(gpu_free),
                )
                return g, mb
    logger.warning(
        "train auto-memory: no (G, mb) fit in %.1f GB free; using G=1 mb=1",
        _gb(gpu_free),
    )
    return 1, 1


@dataclass
class TrainMemoryPlan:
    gpu_pinned_gb: float
    pinned_bytes: int
    use_residency: bool
    prefetch_chunks: bool
    prefetch_block: bool
    checkpoint_every: int
    cap_anchor_block: Optional[int]
    cap_microbatch: Optional[int]
    vram_headroom_gb: float
    cpu_headroom_gb: float
    extra: Dict[str, float] = field(default_factory=dict)


def choose_train_memory_plan(
    *,
    max_steps: int,
    d: int,
    k: int,
    w: int,
    device: Any,
    n_chunks: int = 1,
    auto_memory: bool = True,
    cap_pinned_gb: Optional[float] = None,
    cap_anchor_block: Optional[int] = None,
    cap_microbatch: Optional[int] = None,
    checkpoint_every: Optional[int] = None,
    prefetch_chunks: bool = True,
    cpu_headroom_gb: float = 16.0,
    vram_headroom_gb: float = 8.0,
    has_moments: bool = True,
    scratch: Optional[Any] = None,
    next_chunk_bytes: Optional[int] = None,
) -> TrainMemoryPlan:
    """Choose how much bank to pin. Microbatch / G are chosen per chunk later."""
    bps = bytes_per_cst(d, k, w, has_moments=has_moments)
    bank = bps * max(int(max_steps), 1)
    t = max(int(max_steps), 1)

    cpu_free = cpu_available_bytes()
    gpu_free = gpu_free_bytes(device)
    extra = {
        "bytes_per_cst_gb": _gb(bps),
        "bank_gb": _gb(bank),
        "cpu_free_gb": _gb(cpu_free or 0),
        "gpu_free_gb": _gb(gpu_free or 0),
    }

    do_prefetch = bool(prefetch_chunks)
    if do_prefetch and scratch is not None and next_chunk_bytes:
        try:
            free = disk_free_bytes(scratch)
            need = int(next_chunk_bytes) + int(20 * 1e9)
            if free < need:
                logger.info(
                    "train prefetch: skip Hub overlap (disk free %.1f GB < %.1f GB)",
                    _gb(free),
                    _gb(need),
                )
                do_prefetch = False
        except OSError:
            pass

    nbat = max(int(n_chunks), 1)
    if checkpoint_every is not None and int(checkpoint_every) >= 1:
        ck_every = int(checkpoint_every)
    else:
        # ~8 full shard dumps per epoch; always at least once (last file).
        ck_every = max(1, (nbat + 7) // 8)

    if not auto_memory:
        pin_gb = float(cap_pinned_gb or 0.0)
        pin_bytes = int(pin_gb * 1e9)
        pin_bytes = min(pin_bytes, bank)
        return TrainMemoryPlan(
            gpu_pinned_gb=_gb(pin_bytes),
            pinned_bytes=pin_bytes,
            use_residency=pin_bytes > 0,
            prefetch_chunks=do_prefetch,
            prefetch_block=True,
            checkpoint_every=ck_every,
            cap_anchor_block=cap_anchor_block or 16,
            cap_microbatch=cap_microbatch,
            vram_headroom_gb=float(vram_headroom_gb),
            cpu_headroom_gb=float(cpu_headroom_gb),
            extra=extra,
        )

    host_budget = 0
    if cpu_free is not None:
        host_budget = int(cpu_free) - int(float(cpu_headroom_gb) * 1e9)
        host_budget -= int((_MMAP_SLACK_GB + _HOST_OVERHEAD_GB) * 1e9)
        host_budget = max(host_budget, 0)
        max_unpinned = host_budget // bps if bps > 0 else t
        min_pin_steps = max(0, t - int(max_unpinned))
    else:
        max_unpinned = t
        min_pin_steps = 0
    extra["host_budget_gb"] = _gb(host_budget)

    gpu_budget = 0
    if gpu_free is not None:
        gpu_budget = int(gpu_free) - int(float(vram_headroom_gb) * 1e9)
        gpu_budget = max(gpu_budget, 0)
    extra["gpu_budget_gb"] = _gb(gpu_budget)

    if gpu_free is None:
        pin_steps = 0
    else:
        reserve = min(
            gpu_budget * 0.30,
            max(int(_MIN_GPU_RESERVE_GB * 1e9), gpu_budget // 4 if gpu_budget else 0),
        )
        if gpu_budget <= int(_MIN_GPU_RESERVE_GB * 1e9):
            reserve = max(gpu_budget // 2, 0)
        pin_from_gpu_steps = (gpu_budget - int(reserve)) // bps if bps > 0 else 0
        pin_from_gpu_steps = max(0, min(t, int(pin_from_gpu_steps)))

        pin_steps = max(min_pin_steps, pin_from_gpu_steps)
        if cap_pinned_gb is not None:
            pin_steps = min(
                pin_steps,
                int(float(cap_pinned_gb) * 1e9) // bps if bps else pin_steps,
            )
        pin_steps = min(max(int(pin_steps), 0), t)

        unpinned = t - pin_steps
        if (
            bps > 0
            and unpinned * bps > host_budget
            and host_budget >= 0
            and cpu_free is not None
        ):
            pin_steps = min(t, max(pin_steps, t - host_budget // bps))

    pin_bytes = pin_steps * bps
    pin_gb = _gb(pin_bytes)
    extra["pin_steps"] = float(pin_steps)
    extra["unpinned_steps"] = float(t - pin_steps)
    # Auto always uses residency (pin 0 = stream the whole bank). Manual pin 0
    # still means "place the whole bank on GPU" at the call site.

    logger.info(
        "train auto-memory: pin %.1f GB (%d/%d CSTs) | bank %.1f GB | "
        "host free %.1f GB (budget %.1f) | gpu free %.1f GB (budget %.1f) | "
        "checkpoint_every=%d prefetch_chunks=%s",
        pin_gb,
        pin_steps,
        t,
        _gb(bank),
        extra["cpu_free_gb"],
        extra["host_budget_gb"],
        extra["gpu_free_gb"],
        extra["gpu_budget_gb"],
        ck_every,
        do_prefetch,
    )
    if gpu_free is not None and cpu_free is not None and (t - pin_steps) * bps > host_budget:
        logger.warning(
            "train auto-memory: parked bank may exceed host RAM (unpinned %.1f GB, budget %.1f GB)",
            _gb((t - pin_steps) * bps),
            extra["host_budget_gb"],
        )

    return TrainMemoryPlan(
        gpu_pinned_gb=pin_gb,
        pinned_bytes=int(pin_bytes),
        use_residency=gpu_free is not None,
        prefetch_chunks=do_prefetch,
        prefetch_block=True,
        checkpoint_every=ck_every,
        cap_anchor_block=cap_anchor_block,
        cap_microbatch=cap_microbatch,
        vram_headroom_gb=float(vram_headroom_gb),
        cpu_headroom_gb=float(cpu_headroom_gb),
        extra=extra,
    )


def plan_from_train_cfg(
    tr: Any,
    *,
    max_steps: int,
    d: int,
    k: int,
    w: int,
    device: Any,
    n_chunks: int,
    next_chunk_bytes: Optional[int] = None,
    scratch: Optional[Any] = None,
) -> TrainMemoryPlan:
    auto = _as_bool(tr.get("auto_memory", True), True)
    return choose_train_memory_plan(
        max_steps=max_steps,
        d=d,
        k=k,
        w=w,
        device=device,
        n_chunks=n_chunks,
        auto_memory=auto,
        cap_pinned_gb=_as_cap_float(tr.get("gpu_pinned_gb")),
        cap_anchor_block=_as_cap_int(tr.get("anchor_block")),
        cap_microbatch=_as_cap_int(tr.get("microbatch_rows")),
        checkpoint_every=_as_cap_int(tr.get("checkpoint_every")),
        prefetch_chunks=_as_bool(tr.get("prefetch_chunks", True), True),
        cpu_headroom_gb=float(tr.get("cpu_headroom_gb", 16.0) or 16.0),
        vram_headroom_gb=float(tr.get("vram_headroom_gb", 8.0) or 8.0),
        has_moments=True,
        scratch=scratch,
        next_chunk_bytes=next_chunk_bytes,
    )
