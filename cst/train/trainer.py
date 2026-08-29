"""CST training: chunk loop, save/load checkpoints, optional file cleanup."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

from cst.model.cst_bank import CSTBank
from cst.model.residency import CSTResidency, steps_for_anchors
from cst.naming import train_checkpoint_names
from cst.storage_dtype import map_float_tensors
from cst.train.hidden_chunk import iter_anchors, load_chunk_H, stack_h_window
from cst.train.loss import windowed_cst_loss_block
from cst.train.memory_budget import TrainMemoryPlan, bytes_per_cst, choose_chunk_compute

logger = logging.getLogger("cst.train")

# Training checkpoints: one ``latest`` (resume) + one ``best`` (lowest ``loss_mean``).
# Short names are directories of per-step shards (legacy single ``.pt`` still loads).
# With ``name_token`` see :func:`train_checkpoint_names`.
CKPT_LATEST = "cst_latest"
CKPT_BEST = "cst_best"
TRAIN_STATE_JSON = "cst_train_state.json"

# Extra VRAM headroom (GB) added on top of train.vram_headroom_gb for the
# lazy-bank per-file dynamic sizing path. See train_on_chunk_files.
#
# Empirically measured live (2026-08-28): activation_bytes()'s "need" estimate
# undershoots the real peak (torch.cuda.max_memory_allocated() for the whole
# file) by a consistent ~78GB gap, fairly stable across different sequence
# lengths (need=121.6GB vs peak=199.6GB at L=560; need=130.9GB vs a crash at
# ~200GB+ at L=606) — i.e. roughly a fixed omission (plausibly activation-
# checkpoint recomputation overhead in decoder_sparsity_penalty, which the
# formula's fudge factor doesn't model), not something that scales with G or
# L the way the rest of the formula assumes. 16GB wasn't enough margin and
# led to a real OOM on the 4th file of a real run. Sized to absorb that gap
# with room to spare; choose_chunk_compute's own search then naturally picks
# a smaller G for large-B/L files (and a bigger one for small-B/L files)
# against the corrected budget — this is what actually makes the per-file
# sizing self-adjusting, not a hardcoded G.
_LAZY_EXTRA_HEADROOM_GB = 90.0

# Free cached-but-unused CUDA/HIP allocator blocks every N lazy-bank blocks.
# expandable_segments is unsupported on this ROCm build (confirmed via the
# startup warning), so the allocator can only satisfy a differently-shaped
# window tensor by finding an exactly-sized free block; after enough
# alloc/free churn from varied per-block tensor shapes, free space fragments
# and the allocator can stall for many minutes searching for one (observed
# live: a plain non_blocking .to(device) call hung with zero page faults,
# zero disk I/O, and zero GPU utilization for 10+ minutes at ~50% VRAM used).
_LAZY_EMPTY_CACHE_EVERY = 4

# How often (in blocks) to log lazy-bank progress. Sizing is now computed
# once per file (see the per-file preamble above), so there's no other
# recurring log line during a file's block loop otherwise.
_LAZY_PROGRESS_LOG_EVERY = 5

# Keys in training checkpoint ``meta`` (full-run resume at batch boundaries + W&B)
META_DATA_NEXT_EPOCH = "data_next_epoch"
# 0-based index of the epoch that will be trained next, or == ``num_epochs`` if already finished
META_DATA_NEXT_BATCH_IDX = "data_next_batch_idx"
META_DATA_NEXT_SEGMENT = "data_next_segment"
META_WANDB_GLOBAL_STEP = "wandb_global_step"
META_WANDB_RUN_ID = "wandb_run_id"
# Saved so we can warn if the user changes ``num_epochs`` between save and load
META_TRAIN_NUM_EPOCHS = "num_epochs_checkpointed"
# Row-batch / dataset draw fingerprint (keeps ``sample_uniform_batched`` aligned on resume)
META_TRAIN_SEED = "train_seed"
META_TRAIN_SAMPLES_PER_EPOCH = "train_samples_per_epoch"
META_TRAIN_INFERENCE_BATCH_SIZE = "train_inference_batch_size"
META_TRAIN_RESAMPLE_EACH_EPOCH = "train_resample_each_epoch"
META_TRAIN_GSM8K_NUM_FEWSHOT = "train_gsm8k_num_fewshot"
META_TRAIN_ARC_CSQA_NUM_FEWSHOT = "train_arc_csqa_num_fewshot"


def merge_resume_fields_into_meta(
    meta: Dict[str, Any],
    *,
    data_next_epoch: int,
    data_next_batch_idx: int,
    data_next_segment: int,
    wandb_global_step: int,
    wandb_run_id: Optional[str] = None,
    num_epochs: Optional[int] = None,
) -> None:
    """In-place: record full-run data position, W&B step/run id, and optional epoch count."""
    meta[META_DATA_NEXT_EPOCH] = int(data_next_epoch)
    meta[META_DATA_NEXT_BATCH_IDX] = int(data_next_batch_idx)
    meta[META_DATA_NEXT_SEGMENT] = int(data_next_segment)
    meta[META_WANDB_GLOBAL_STEP] = int(wandb_global_step)
    if wandb_run_id is not None and str(wandb_run_id).strip():
        meta[META_WANDB_RUN_ID] = str(wandb_run_id).strip()
    elif META_WANDB_RUN_ID in meta:
        del meta[META_WANDB_RUN_ID]
    if num_epochs is not None:
        meta[META_TRAIN_NUM_EPOCHS] = int(num_epochs)


def parse_data_resume_from_meta(
    meta: Dict[str, Any], *, require_all: bool = True
) -> Optional[tuple[int, int, int]]:
    """
    Return ``(data_next_epoch, data_next_batch_idx, data_next_segment)`` for the next
    training batch, or None if not present/invalid.

    ``data_next_segment`` is kept for old checkpoints (always 0 for new saves).
    If ``require_all`` is true, all three int keys must be set.
    """
    if not meta:
        return None
    ke = meta.get(META_DATA_NEXT_EPOCH)
    kb = meta.get(META_DATA_NEXT_BATCH_IDX)
    ks = meta.get(META_DATA_NEXT_SEGMENT)
    if require_all and (ke is None or kb is None or ks is None):
        return None
    try:
        return int(ke), int(kb), int(ks)
    except (TypeError, ValueError):
        return None


def merge_sampling_state_into_meta(
    meta: Dict[str, Any],
    *,
    train_seed: int,
    train_samples_per_epoch: int,
    train_inference_batch_size: int,
    train_resample_each_epoch: bool,
    train_gsm8k_num_fewshot: int,
    train_arc_csqa_num_fewshot: int,
) -> None:
    """In-place: store draw settings so a resumed run can override YAML and match row batches."""
    meta[META_TRAIN_SEED] = int(train_seed)
    meta[META_TRAIN_SAMPLES_PER_EPOCH] = int(train_samples_per_epoch)
    meta[META_TRAIN_INFERENCE_BATCH_SIZE] = int(train_inference_batch_size)
    meta[META_TRAIN_RESAMPLE_EACH_EPOCH] = bool(train_resample_each_epoch)
    meta[META_TRAIN_GSM8K_NUM_FEWSHOT] = int(train_gsm8k_num_fewshot)
    meta[META_TRAIN_ARC_CSQA_NUM_FEWSHOT] = int(train_arc_csqa_num_fewshot)


def parse_sampling_state_from_meta(meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Return a dict of sampling fields if the checkpoint was saved with a full fingerprint;
    else None (legacy checkpoints, or incomplete meta).
    """
    if not meta:
        return None
    keys = (
        META_TRAIN_SEED,
        META_TRAIN_SAMPLES_PER_EPOCH,
        META_TRAIN_INFERENCE_BATCH_SIZE,
        META_TRAIN_RESAMPLE_EACH_EPOCH,
        META_TRAIN_GSM8K_NUM_FEWSHOT,
        META_TRAIN_ARC_CSQA_NUM_FEWSHOT,
    )
    for k in keys:
        if k not in meta:
            return None
    try:
        return {
            "train_seed": int(meta[META_TRAIN_SEED]),
            "train_samples_per_epoch": int(meta[META_TRAIN_SAMPLES_PER_EPOCH]),
            "train_inference_batch_size": int(meta[META_TRAIN_INFERENCE_BATCH_SIZE]),
            "train_resample_each_epoch": bool(meta[META_TRAIN_RESAMPLE_EACH_EPOCH]),
            "train_gsm8k_num_fewshot": int(meta[META_TRAIN_GSM8K_NUM_FEWSHOT]),
            "train_arc_csqa_num_fewshot": int(meta[META_TRAIN_ARC_CSQA_NUM_FEWSHOT]),
        }
    except (TypeError, ValueError):
        return None


def _read_train_state(ckpt_dir: Path, state_basename: str) -> Dict[str, Any]:
    """Read state JSON; if the requested basename is missing, try ``cst_train_state.json``."""
    want = Path(state_basename).name
    cands: List[Path] = [ckpt_dir / want]
    if want != TRAIN_STATE_JSON:
        cands.append(ckpt_dir / TRAIN_STATE_JSON)
    for p in cands:
        if p.is_file():
            try:
                with p.open() as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("could not read %s: %s", p, e)
                return {}
    return {}


def _write_train_state(ckpt_dir: Path, state_basename: str, data: Dict[str, Any]) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    p = ckpt_dir / Path(state_basename).name
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=0)


def remove_legacy_phase_checkpoints(ckpt_dir: str | Path) -> int:
    """Remove obsolete ``cst_phase_*.pt`` files; return count removed."""
    n = 0
    for f in Path(ckpt_dir).glob("cst_phase_*.pt"):
        try:
            f.unlink()
            n += 1
            logger.info("removed obsolete checkpoint %s", f.name)
        except OSError as e:
            logger.warning("could not remove %s: %s", f, e)
    return n


def save_train_checkpoints(
    ckpt_dir: str | Path,
    *,
    cst_bank: CSTBank,
    optimizer: torch.optim.Optimizer,
    ramp: LambdaRamp,
    phase: int,
    train_loss: float,
    train_batches: float,
    name_token: Optional[str] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
    save_optimizer_state: bool = False,
) -> Dict[str, Any]:
    """Write latest/best CST checkpoints and training state; drop obsolete ``cst_phase_*.pt``."""
    ckpt_dir = Path(ckpt_dir)
    latest_n, best_n, state_n = train_checkpoint_names(name_token)
    n_anchors = int(float(train_batches))
    if n_anchors < 1:
        loss_mean = float("inf")
    else:
        loss_mean = float(train_loss) / float(n_anchors)

    meta: Dict[str, Any] = {
        "phase": int(phase),
        "train_loss_sum": float(train_loss),
        "n_anchors": n_anchors,
        "loss_mean": float(loss_mean) if math.isfinite(loss_mean) else None,
    }
    if name_token:
        meta["name_token"] = name_token
    if extra_meta:
        for k, v in extra_meta.items():
            if v is not None:
                meta[k] = v

    save_cst_checkpoint(
        ckpt_dir / latest_n,
        cst_bank=cst_bank,
        optimizer=optimizer,
        ramp=ramp,
        meta=meta,
        save_optimizer_state=save_optimizer_state,
    )
    logger.info(
        "wrote %s loss_sum=%.6f anchors=%d loss_mean=%s",
        latest_n,
        float(train_loss),
        n_anchors,
        f"{loss_mean:.6f}" if math.isfinite(loss_mean) else "inf",
    )

    state = _read_train_state(ckpt_dir, state_n)
    raw_best = state.get("best_loss_mean")
    try:
        best_prev = float(raw_best) if raw_best is not None else float("inf")
    except (TypeError, ValueError):
        best_prev = float("inf")
    if not math.isfinite(best_prev):
        best_prev = float("inf")

    # A lazy bank only ever holds currently-live (not-yet-evicted) steps in
    # memory; already-flushed steps only exist in `latest_n` (written
    # incrementally by flush_and_evict_step). A `best_n` snapshot taken here
    # would silently miss every already-evicted step, so skip "best" tracking
    # entirely for lazy runs — `latest_n` is the only checkpoint that's ever
    # complete-by-run-end, and it's what resume already uses.
    is_lazy = bool(getattr(cst_bank, "lazy", False))
    updated_best = bool(
        not is_lazy and math.isfinite(loss_mean) and (loss_mean < best_prev - 1e-12)
    )
    if updated_best:
        best_meta = dict(meta)
        best_meta["is_best"] = True
        best_meta["loss_mean"] = float(loss_mean)
        save_cst_checkpoint(
            ckpt_dir / best_n,
            cst_bank=cst_bank,
            optimizer=optimizer,
            ramp=ramp,
            meta=best_meta,
            save_optimizer_state=save_optimizer_state,
        )
        _write_train_state(
            ckpt_dir,
            state_n,
            {
                "best_loss_mean": float(loss_mean),
                "best_phase": int(phase),
            },
        )
        logger.info("wrote %s new best loss_mean=%.6f", best_n, loss_mean)
    else:
        bpath = ckpt_dir / best_n
        if bpath.exists() and math.isfinite(best_prev):
            logger.info("kept %s (best loss_mean=%.6f)", best_n, best_prev)

    n_removed = remove_legacy_phase_checkpoints(ckpt_dir)
    return {
        "latest": str(ckpt_dir / latest_n),
        "best": str(ckpt_dir / best_n) if (ckpt_dir / best_n).exists() else None,
        "loss_mean": loss_mean,
        "updated_best": updated_best,
        "legacy_phase_removed": n_removed,
    }


class LambdaRamp:
    """Sparsity weight ramp: ``step()`` returns a value in ``[0,1]`` over ``total_steps`` calls."""

    def __init__(self, total_steps: int) -> None:
        self.total_steps = max(1, int(total_steps))
        self.current = 0

    def step(self) -> float:
        """Advance ramp; return value in ``[0,1]``."""
        self.current += 1
        return min(1.0, self.current / self.total_steps)

    def state_dict(self) -> Dict[str, Any]:
        return {"current": self.current, "total_steps": self.total_steps}

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        self.current = int(d.get("current", 0))
        self.total_steps = int(d.get("total_steps", self.total_steps))


class _DeviceRows:
    """Absolute-indexed view of a host ``[T,B,L,D]`` tensor, staging one window on device.

    ``windowed_cst_loss_full`` indexes ``H_full`` by absolute step, but a block
    of anchors only ever reads rows ``block[0]-w+1 .. block[-1]``. Staging just
    that window keeps the targets off the device budget: the full tensor is
    ~17 GB at 513 frames and sequence length 1024, against ~1 GB per window.
    """

    def __init__(self, H: torch.Tensor, device: torch.device) -> None:
        self._H = H
        self._device = device
        self._lo = 0
        self._buf: Optional[torch.Tensor] = None

    def prepare(
        self,
        lo: int,
        hi: int,
        *,
        b0: int = 0,
        b1: Optional[int] = None,
    ) -> None:
        lo = max(0, int(lo))
        hi = min(int(hi), self._H.shape[0] - 1)
        self._lo = lo
        b1_i = int(self._H.shape[1]) if b1 is None else int(b1)
        cpu_slice = self._H[lo : hi + 1, int(b0) : b1_i]
        if self._device.type == "cuda":
            # `cpu_slice` is a view into a memory-mapped (pageable) host
            # tensor. Transferring straight from pageable memory routed
            # through this ROCm build's SVM/HMM unified-memory path, which
            # was observed live to stall the whole process indefinitely
            # (amdgpu's svm_range_restore_work kernel workqueue hogging CPU,
            # confirmed via dmesg, with zero GPU/page-fault/IO activity from
            # the process itself). Staging through pinned memory first uses
            # the ordinary DMA path instead.
            cpu_slice = cpu_slice.pin_memory()
        self._buf = cpu_slice.to(self._device, non_blocking=True)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self._buf is None:
            raise RuntimeError("_DeviceRows.prepare() must be called before indexing")
        j = int(idx) - self._lo
        if j < 0 or j >= self._buf.shape[0]:
            raise IndexError(f"row {idx} outside staged window [{self._lo}, {self._lo + self._buf.shape[0] - 1}]")
        return self._buf[j]

    def dim(self) -> int:
        return self._H.dim()

    @property
    def shape(self) -> torch.Size:
        return self._H.shape


def train_on_chunk_files(
    paths: List[str | Path],
    cst_bank: CSTBank,
    optimizer: torch.optim.Optimizer,
    ramp: LambdaRamp,
    *,
    w: int,
    lambda_sparsity: float,
    c_tanh: float,
    lambda_pre_dead: float = 0.0,
    device: torch.device,
    anchor_block: int = 24,
    microbatch_rows: Optional[int] = None,
    residency: Optional[CSTResidency] = None,
    memory_plan: Optional[TrainMemoryPlan] = None,
    prefetch_block: bool = True,
    lazy_bank: bool = False,
    on_step_closed: Optional[Callable[[List[int]], None]] = None,
    on_step_created: Optional[Callable[[int, Any], None]] = None,
) -> Dict[str, Any]:
    """Train on each hidden file in blocks of consecutive anchors.

    One backward per ``(anchor_block, microbatch)`` so a Hub chunk with large B
    does not retain G full-batch graphs. Anchor ``t`` touches steps
    ``t-w+1 .. t``; step ``s`` is updated once its window closes, then residency
    can release it. Numerically this is the same sum as one update per chunk.

    When ``memory_plan`` is set, G and microbatch are chosen from free VRAM for
    each file's ``[T,B,L,D]``. YAML caps on the plan still apply.

    ``lazy_bank=True`` (requires ``cst_bank.lazy`` and ``residency=None``) uses
    a different, dynamic per-block sizing path: instead of a single G chosen
    once per file, ``choose_chunk_compute`` is re-run before *every* block from
    live free VRAM (steps evicted by earlier blocks free real memory, so later
    blocks can grow). Each step is created on first touch and, when
    ``on_step_closed`` is given, handed to it (as the list of just-closed step
    indices) instead of being released back to host — the callback is expected
    to flush the step's final weights and delete it forever (see
    :func:`flush_and_evict_step`). This branch leaves the existing
    (``lazy_bank=False``) fixed-block/residency path completely unchanged.

    Returns loss stats and ``per_chunk``.
    """
    cst_bank.train()
    total_loss = 0.0
    n_items = 0
    per_chunk: List[Dict[str, Any]] = []
    min_frames = 2
    copy_stream = None
    if (
        prefetch_block
        and residency is not None
        and torch.cuda.is_available()
        and str(device).startswith("cuda")
    ):
        copy_stream = torch.cuda.Stream(device=device)
    bps = (
        residency.bytes_per_step
        if residency is not None
        else bytes_per_cst(cst_bank.d, cst_bank.k, cst_bank.w, has_moments=True)
    )

    for p in paths:
        H, _meta, _ = load_chunk_H(p)
        n_frames = H.shape[0]
        if n_frames < min_frames:
            logger.warning(
                "skip %s: need initial + at least one forward (>= %d rows), got %d",
                p,
                min_frames,
                n_frames,
            )
            continue
        B = int(H.shape[1])
        L = int(H.shape[2])
        g_use = max(int(anchor_block), 1)
        mb_use = min(B, max(int(microbatch_rows), 1)) if microbatch_rows else B
        if lazy_bank and torch.cuda.is_available() and str(device).startswith("cuda"):
            # torch.cuda.mem_get_info() reports driver-level free memory,
            # which doesn't reflect PyTorch's own cached-but-unused blocks
            # from the steps evicted at the end of the *previous* file until
            # they're explicitly released — observed live: a file right
            # after a big eviction saw "free VRAM" read as ~46GB while actual
            # usage was ~5GB, tripping the G=1 fallback for no real reason.
            torch.cuda.empty_cache()
        if memory_plan is not None:
            g_use, mb_use = choose_chunk_compute(
                batch=B,
                seq_len=L,
                d=int(cst_bank.d),
                k=int(cst_bank.k),
                w=int(w),
                bytes_per_step=int(bps),
                device=device,
                cap_anchor_block=memory_plan.cap_anchor_block,
                cap_microbatch=memory_plan.cap_microbatch,
                prefetch_block=bool(prefetch_block and residency is not None and not lazy_bank),
                vram_headroom_gb=(
                    float(memory_plan.vram_headroom_gb) + _LAZY_EXTRA_HEADROOM_GB
                    if lazy_bank
                    else float(memory_plan.vram_headroom_gb)
                ),
            )
            if lazy_bank:
                # One anchor below the estimate as a further safety margin
                # against the fudge-factor formula's edge cases.
                g_use = max(int(g_use) - 1, 1)
        mb_use = max(min(int(mb_use), B), 1)
        g_use = max(int(g_use), 1)

        # Always stage windows; never ``H.to(device)`` for the full chunk.
        rows = _DeviceRows(H, device)
        lam = ramp.step()
        optimizer.zero_grad(set_to_none=True)

        anchors_all = iter_anchors(n_frames, w)
        chunk_loss_val = 0.0
        n_anchors_in_file = 0
        n_blocks = 0
        n_micro = 0
        stepped: set[int] = set()
        # Anchor-weighted nmse (normalized reconstruction error) across the
        # whole file — a comparable convergence signal across files of
        # different B/L, unlike the raw loss sum.
        nmse_weighted_sum = 0.0
        n_nmse_terms = 0

        if lazy_bank:
            # Fixed block size for the whole file (computed once above, from
            # live free VRAM at file start — still fully dynamic, just not
            # re-picked every block). Keeping window-tensor shapes constant
            # across blocks avoids the alloc/free churn that was driving the
            # HIP allocator into pathological fragmentation/stalls when G was
            # re-maximized every single block (observed live: multi-minute
            # hangs with zero GPU/IO activity, consistent with the allocator
            # or HIP runtime spinning on a fragmented free list). Evictions
            # still free real memory — the benefit shows up as bigger G on
            # the *next* file's fresh sizing, not intra-file churn.
            idx = 0
            while idx < len(anchors_all):
                block = anchors_all[idx : idx + g_use]
                idx += len(block)
                need = steps_for_anchors(block, w)
                n_blocks += 1
                for t in need:
                    if not cst_bank.has(t):
                        mod = cst_bank.get_or_create(t, device)
                        if on_step_created is not None:
                            on_step_created(t, mod)
                        optimizer.param_groups[0]["params"].extend(mod.parameters())

                lo = block[0] - w + 1
                hi = block[-1]
                for b0 in range(0, B, mb_use):
                    b1 = min(B, b0 + mb_use)
                    h_wins = [
                        stack_h_window(H, a, w, b0=b0, b1=b1, device=device) for a in block
                    ]
                    rows.prepare(lo, hi, b0=b0, b1=b1)
                    loss, m = windowed_cst_loss_block(
                        cst_bank,
                        block,
                        h_wins,
                        rows,
                        lambda_sparsity=lambda_sparsity,
                        c_tanh=c_tanh,
                        lambda_ramp=lam,
                        lambda_pre_dead=lambda_pre_dead,
                    )
                    if loss.requires_grad and loss.numel() > 0:
                        loss.backward()
                    total_loss += float(m.get("loss", 0.0))
                    chunk_loss_val += float(m.get("loss", 0.0))
                    nmse_weighted_sum += float(m.get("nmse", 0.0)) * len(block)
                    n_nmse_terms += len(block)
                    n_micro += 1
                    del loss, h_wins

                n_items += len(block)
                n_anchors_in_file += len(block)

                last_anchor = block[-1]
                closed = [
                    s
                    for s in need
                    if s <= last_anchor - w + 1 and s not in stepped
                ]
                if closed:
                    _step_subset(optimizer, cst_bank, closed)
                    stepped.update(closed)
                    if on_step_closed is not None:
                        on_step_closed(closed)

                if n_blocks % _LAZY_PROGRESS_LOG_EVERY == 0:
                    logger.info(
                        "lazy_bank progress: block=%d anchors=%d/%d steps_closed=%d G=%d "
                        "block_loss=%.4f nmse=%.4f",
                        n_blocks,
                        idx,
                        len(anchors_all),
                        len(stepped),
                        g_use,
                        float(m.get("loss", 0.0)),
                        float(m.get("nmse", 0.0)),
                    )

                if (
                    n_blocks % _LAZY_EMPTY_CACHE_EVERY == 0
                    and torch.cuda.is_available()
                    and str(device).startswith("cuda")
                ):
                    torch.cuda.empty_cache()

            trailing = [s for s in steps_for_anchors(anchors_all, w) if s not in stepped]
            if trailing:
                _step_subset(optimizer, cst_bank, trailing)
                stepped.update(trailing)
                if on_step_closed is not None:
                    on_step_closed(trailing)
        else:
            blocks: List[List[int]] = [
                anchors_all[i : i + g_use] for i in range(0, len(anchors_all), g_use)
            ]

            for bi, block in enumerate(blocks):
                nxt = blocks[bi + 1] if bi + 1 < len(blocks) else None
                need = steps_for_anchors(block, w)
                need_next = steps_for_anchors(nxt, w) if nxt else []
                n_blocks += 1
                if residency is not None:
                    residency.ensure(need)
                    if prefetch_block and need_next:
                        if copy_stream is not None:
                            with torch.cuda.stream(copy_stream):
                                residency.ensure(need_next)
                        else:
                            residency.ensure(need_next)

                lo = block[0] - w + 1
                hi = block[-1]
                for b0 in range(0, B, mb_use):
                    b1 = min(B, b0 + mb_use)
                    h_wins = [
                        stack_h_window(H, a, w, b0=b0, b1=b1, device=device) for a in block
                    ]
                    rows.prepare(lo, hi, b0=b0, b1=b1)
                    loss, m = windowed_cst_loss_block(
                        cst_bank,
                        block,
                        h_wins,
                        rows,
                        lambda_sparsity=lambda_sparsity,
                        c_tanh=c_tanh,
                        lambda_ramp=lam,
                        lambda_pre_dead=lambda_pre_dead,
                    )
                    if loss.requires_grad and loss.numel() > 0:
                        loss.backward()
                    total_loss += float(m.get("loss", 0.0))
                    chunk_loss_val += float(m.get("loss", 0.0))
                    nmse_weighted_sum += float(m.get("nmse", 0.0)) * len(block)
                    n_nmse_terms += len(block)
                    n_micro += 1
                    del loss, h_wins

                n_items += len(block)
                n_anchors_in_file += len(block)

                last_anchor = block[-1]
                keep_next = set(need_next)
                closed = [
                    s
                    for s in need
                    if s <= last_anchor - w + 1 and s not in stepped
                ]
                if closed:
                    _step_subset(optimizer, cst_bank, closed)
                    stepped.update(closed)
                    if residency is not None:
                        for s in closed:
                            if s not in keep_next:
                                residency.release(s)
                if copy_stream is not None and need_next:
                    copy_stream.synchronize()

            trailing = [s for s in steps_for_anchors(anchors_all, w) if s not in stepped]
            if trailing:
                _step_subset(optimizer, cst_bank, trailing)
                if residency is not None:
                    for s in trailing:
                        residency.release(s)

        optimizer.zero_grad(set_to_none=True)
        if n_anchors_in_file:
            p_str = str(p)
            chunk_nmse = nmse_weighted_sum / n_nmse_terms if n_nmse_terms else 0.0
            per_chunk.append(
                {
                    "path": p_str,
                    "loss": chunk_loss_val,
                    "n_anchors": n_anchors_in_file,
                    "nmse": chunk_nmse,
                }
            )
            logger.info(
                "train chunk %s loss=%.6f nmse=%.6f n_anchors=%d blocks=%d microbatches=%d "
                "G=%d mb=%d B=%d L=%d (per-step optimizer updates)",
                Path(p).name,
                chunk_loss_val,
                chunk_nmse,
                n_anchors_in_file,
                n_blocks,
                n_micro,
                g_use,
                mb_use,
                B,
                L,
            )
        del H, rows
    return {
        "train_loss": total_loss,
        "train_batches": float(n_items),
        "per_chunk": per_chunk,
        "n_chunks": float(len(per_chunk)),
    }


def _step_subset(optimizer: torch.optim.Optimizer, cst_bank: CSTBank, steps: List[int]) -> None:
    """Run one optimizer update restricted to the parameters of ``steps``."""
    wanted = {id(p) for s in steps for p in cst_bank.get(s).parameters()}
    saved = [g["params"] for g in optimizer.param_groups]
    try:
        for g in optimizer.param_groups:
            g["params"] = [p for p in g["params"] if id(p) in wanted]
        optimizer.step()
    finally:
        for g, orig in zip(optimizer.param_groups, saved):
            g["params"] = orig
    for s in steps:
        for p in cst_bank.get(s).parameters():
            p.grad = None


CKPT_META_JSON = "meta.json"
CKPT_SHARD_FMT = "cst_{:05d}.pt"
CKPT_FORMAT_VERSION = 2


def _is_sharded_ckpt(path: Path) -> bool:
    return path.is_dir() and (path / CKPT_META_JSON).is_file()


def _cst_opt_state(optimizer: torch.optim.Optimizer, mod: torch.nn.Module) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, prm in mod.named_parameters():
        st = optimizer.state.get(prm)
        if not st:
            continue
        entry: Dict[str, Any] = {"step": int(st.get("step", 0))}
        for key in ("exp_avg", "exp_avg_sq"):
            buf = st.get(key)
            if isinstance(buf, torch.Tensor):
                entry[key] = buf.detach().to("cpu", copy=True)
        out[name] = entry
    return out


def _load_cst_opt_state(
    optimizer: torch.optim.Optimizer,
    mod: torch.nn.Module,
    blob: Dict[str, Any],
) -> None:
    for name, prm in mod.named_parameters():
        entry = blob.get(name)
        if not entry:
            continue
        st = optimizer.state[prm]
        st["step"] = int(entry.get("step", 0))
        for key in ("exp_avg", "exp_avg_sq"):
            buf = entry.get(key)
            if isinstance(buf, torch.Tensor):
                st[key] = buf.to(prm.device)


def load_shard_if_exists(
    ckpt_dir: str | Path,
    t: int,
    mod: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> bool:
    """Lazy-bank only: if step ``t`` was already trained (and flushed) by an
    earlier file in this run, load its weights (and optimizer moments, if
    present) into the freshly-created ``mod`` instead of leaving it at random
    init.

    Without this, ``get_or_create`` has no way to know a step was already
    trained — each file would silently discard every prior file's learning
    for that step and start over from scratch, defeating the entire point of
    training across multiple chunk files. Returns whether a shard was found.
    """
    shard_path = Path(ckpt_dir) / CKPT_SHARD_FMT.format(t)
    if not shard_path.is_file():
        return False
    shard = torch.load(shard_path, map_location="cpu", weights_only=False)
    mod.load_state_dict(map_float_tensors(shard["cst"], to_storage=False))
    if optimizer is not None and "optimizer" in shard:
        _load_cst_opt_state(optimizer, mod, shard["optimizer"])
    return True


def _save_one_shard(
    path: Path,
    t: int,
    mod: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    save_optimizer_state: bool,
) -> None:
    """Write one step's shard file. Shared by the full-bank dump and the
    lazy-bank incremental flush-on-evict path."""
    shard: Dict[str, Any] = {
        "cst": map_float_tensors(
            {k: v.detach().to("cpu", copy=True) for k, v in mod.state_dict().items()},
            to_storage=True,
        )
    }
    if save_optimizer_state and optimizer is not None:
        shard["optimizer"] = _cst_opt_state(optimizer, mod)
    torch.save(shard, path / CKPT_SHARD_FMT.format(t))


def flush_and_evict_step(
    ckpt_dir: str | Path,
    t: int,
    cst_bank: CSTBank,
    optimizer: torch.optim.Optimizer,
    *,
    save_optimizer_state: bool = False,
) -> None:
    """Lazy-bank only: write step ``t``'s final shard, then delete it for good.

    Called the moment ``t``'s sliding window has closed (no future anchor this
    epoch will touch it again). Writes into the same directory
    :func:`save_train_checkpoints` writes into, so periodic ``meta.json``
    refreshes remain the completion marker; shards are addressed by step
    index, not enumerated from ``meta.json``, so staleness between refreshes
    is harmless.
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    mod = cst_bank.get(t)
    _save_one_shard(ckpt_dir, t, mod, optimizer, save_optimizer_state)
    params = list(mod.parameters())
    cst_bank.evict(t)
    drop_state = getattr(optimizer, "drop_state", None)
    if drop_state is not None:
        drop_state(params)
    else:
        for p in params:
            optimizer.state.pop(p, None)
    # Also drop the params from param_groups itself — otherwise the optimizer
    # keeps them referenced forever and the tensors (and their storage) are
    # never actually freed, defeating the point of eviction.
    wanted = {id(p) for p in params}
    for g in optimizer.param_groups:
        g["params"] = [p for p in g["params"] if id(p) not in wanted]


def save_cst_checkpoint(
    path: str | Path,
    *,
    cst_bank: CSTBank,
    optimizer: torch.optim.Optimizer,
    ramp: LambdaRamp,
    meta: Optional[Dict[str, Any]] = None,
    save_optimizer_state: bool = False,
) -> None:
    """Write the bank to ``path`` as a directory of per-step shards.

    A single ``.pt`` is not viable at this scale: 512 CSTs at k=4096 is ~86 GB of
    bf16 parameters (~258 GB with fp32 optimizer state), and ``torch.save``
    would need all of it materialised in host RAM at once. Shards are written
    one step at a time, so peak overhead is a single CST.

    ``meta.json`` is written last and is the completion marker -- a run killed
    mid-save leaves a directory that :func:`_is_sharded_ckpt` rejects.

    ``save_optimizer_state`` defaults off: resuming then restores weights but
    restarts the Adam moments.

    For a lazy bank (``cst_bank.lazy``), steps already flushed-and-evicted by
    :func:`flush_and_evict_step` are skipped here — their shard on disk is
    already final and must not be touched; only currently-live steps are
    (re)written.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    stale = path / CKPT_META_JSON
    if stale.is_file():
        stale.unlink()

    for t in range(1, cst_bank.max_steps + 1):
        if not cst_bank.has(t):
            continue
        mod = cst_bank.get(t)
        _save_one_shard(path, t, mod, optimizer, save_optimizer_state)

    doc = {
        "format_version": CKPT_FORMAT_VERSION,
        "max_steps": int(cst_bank.max_steps),
        "d": int(cst_bank.d),
        "k": int(cst_bank.k),
        "w": int(cst_bank.w),
        "storage_float_dtype": "bfloat16",
        "has_optimizer_state": bool(save_optimizer_state),
        "ramp": ramp.state_dict(),
        "meta": meta or {},
    }
    (path / CKPT_META_JSON).write_text(json.dumps(doc, indent=2))


def _load_sharded(
    path: Path,
    cst_bank: CSTBank,
    optimizer: Optional[torch.optim.Optimizer],
    ramp: Optional[LambdaRamp],
    *,
    meta_only: bool = False,
) -> Dict[str, Any]:
    doc = json.loads((path / CKPT_META_JSON).read_text())
    if not meta_only:
        saved_steps = int(doc.get("max_steps", cst_bank.max_steps))
        if saved_steps != cst_bank.max_steps:
            logger.warning(
                "checkpoint has %d steps, bank has %d; loading the overlap only",
                saved_steps,
                cst_bank.max_steps,
            )
        want_opt = optimizer is not None and bool(doc.get("has_optimizer_state"))
        for t in range(1, min(saved_steps, cst_bank.max_steps) + 1):
            shard_path = path / CKPT_SHARD_FMT.format(t)
            if not shard_path.is_file():
                logger.warning("missing checkpoint shard %s; leaving step %d at init", shard_path, t)
                continue
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            mod = cst_bank.get(t)
            mod.load_state_dict(map_float_tensors(shard["cst"], to_storage=False))
            if want_opt and "optimizer" in shard:
                _load_cst_opt_state(optimizer, mod, shard["optimizer"])
    if ramp is not None and doc.get("ramp"):
        ramp.load_state_dict(doc["ramp"])
    return doc.get("meta") or {}


def _load_legacy_pt(
    path: Path,
    cst_bank: CSTBank,
    optimizer: Optional[torch.optim.Optimizer],
    ramp: Optional[LambdaRamp],
    *,
    meta_only: bool = False,
) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not meta_only:
        cst_bank.load_state_dict_from_save(map_float_tensors(payload["cst"], to_storage=False))
        if optimizer is not None and "optimizer" in payload:
            try:
                optimizer.load_state_dict(map_float_tensors(payload["optimizer"], to_storage=False))
            except (ValueError, KeyError) as e:
                logger.warning("legacy optimizer state not loadable (%s); starting moments fresh", e)
    if ramp is not None and "ramp" in payload:
        ramp.load_state_dict(map_float_tensors(payload["ramp"], to_storage=False))
    return payload.get("meta") or {}


def load_cst_checkpoint(
    path: str | Path,
    cst_bank: CSTBank,
    optimizer: torch.optim.Optimizer,
    ramp: LambdaRamp,
    *,
    meta_only: bool = False,
) -> Dict[str, Any]:
    """Load from ``path`` (sharded directory or legacy single ``.pt``); return ``meta``.

    ``meta_only=True`` (lazy-bank resume) skips restoring CST/optimizer
    weights entirely — only the resume cursor (in ``meta``) and ``ramp`` state
    are read, so relaunching after a crash doesn't reintroduce the eager
    full-bank reconstruction cost. Steps flushed during a prior, successfully
    completed chunk file are already correct on disk and never revisited; a
    step that was live at crash time is simply recreated fresh when its file
    is reprocessed (existing file-granularity resume).
    """
    path = Path(path)
    if _is_sharded_ckpt(path):
        return _load_sharded(path, cst_bank, optimizer, ramp, meta_only=meta_only)
    return _load_legacy_pt(path, cst_bank, optimizer, ramp, meta_only=meta_only)


def load_cst_bank_weights(path: str | Path, cst_bank: CSTBank) -> Dict[str, Any]:
    """Load only CST weights from ``path``; return ``meta``."""
    path = Path(path)
    if _is_sharded_ckpt(path):
        return _load_sharded(path, cst_bank, None, None)
    return _load_legacy_pt(path, cst_bank, None, None)


def delete_files(paths: List[str | Path]) -> None:
    """Unlink each path; log a warning on error."""
    for p in paths:
        try:
            Path(p).unlink(missing_ok=True)
        except OSError as e:
            logger.warning("could not delete %s: %s", p, e)
