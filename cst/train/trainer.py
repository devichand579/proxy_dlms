"""CST training: chunk loop, save/load checkpoints, optional file cleanup."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

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

    updated_best = bool(math.isfinite(loss_mean) and (loss_mean < best_prev - 1e-12))
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
        self._buf = self._H[lo : hi + 1, int(b0) : b1_i].to(
            self._device, non_blocking=True
        )

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
) -> Dict[str, Any]:
    """Train on each hidden file in blocks of consecutive anchors.

    One backward per ``(anchor_block, microbatch)`` so a Hub chunk with large B
    does not retain G full-batch graphs. Anchor ``t`` touches steps
    ``t-w+1 .. t``; step ``s`` is updated once its window closes, then residency
    can release it. Numerically this is the same sum as one update per chunk.

    When ``memory_plan`` is set, G and microbatch are chosen from free VRAM for
    each file's ``[T,B,L,D]``. YAML caps on the plan still apply.

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
                prefetch_block=bool(prefetch_block and residency is not None),
                vram_headroom_gb=float(memory_plan.vram_headroom_gb),
            )
        mb_use = max(min(int(mb_use), B), 1)
        g_use = max(int(g_use), 1)

        # Always stage windows; never ``H.to(device)`` for the full chunk.
        rows = _DeviceRows(H, device)
        lam = ramp.step()
        optimizer.zero_grad(set_to_none=True)

        anchors_all = iter_anchors(n_frames, w)
        blocks: List[List[int]] = [
            anchors_all[i : i + g_use] for i in range(0, len(anchors_all), g_use)
        ]
        chunk_loss_val = 0.0
        n_anchors_in_file = 0
        n_blocks = 0
        n_micro = 0
        stepped: set[int] = set()

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
            per_chunk.append(
                {
                    "path": p_str,
                    "loss": chunk_loss_val,
                    "n_anchors": n_anchors_in_file,
                }
            )
            logger.info(
                "train chunk %s loss=%.6f n_anchors=%d blocks=%d microbatches=%d "
                "G=%d mb=%d B=%d L=%d (per-step optimizer updates)",
                Path(p).name,
                chunk_loss_val,
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
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    stale = path / CKPT_META_JSON
    if stale.is_file():
        stale.unlink()

    for t in range(1, cst_bank.max_steps + 1):
        mod = cst_bank.get(t)
        shard: Dict[str, Any] = {
            "cst": map_float_tensors(
                {k: v.detach().to("cpu", copy=True) for k, v in mod.state_dict().items()},
                to_storage=True,
            )
        }
        if save_optimizer_state:
            shard["optimizer"] = _cst_opt_state(optimizer, mod)
        torch.save(shard, path / CKPT_SHARD_FMT.format(t))

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
) -> Dict[str, Any]:
    doc = json.loads((path / CKPT_META_JSON).read_text())
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
) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
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
) -> Dict[str, Any]:
    """Load from ``path`` (sharded directory or legacy single ``.pt``); return ``meta``."""
    path = Path(path)
    if _is_sharded_ckpt(path):
        return _load_sharded(path, cst_bank, optimizer, ramp)
    return _load_legacy_pt(path, cst_bank, optimizer, ramp)


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
