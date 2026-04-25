"""CST training: chunk loop, save/load checkpoints, optional file cleanup."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from cst.model.cst_bank import CSTBank
from cst.naming import train_checkpoint_names
from cst.storage_dtype import map_float_tensors
from cst.train.hidden_chunk import iter_anchor_samples, load_chunk_H
from cst.train.loss import windowed_cst_loss_full

logger = logging.getLogger("cst.train")

# Training checkpoints: one ``latest`` (resume) + one ``best`` (lowest ``loss_mean``); no per-phase .pt files.
# Short basenames below; with ``name_token`` see :func:`train_checkpoint_names` for ``cst_{...}_train_*.pt``.
CKPT_LATEST = "cst_latest.pt"
CKPT_BEST = "cst_best.pt"
TRAIN_STATE_JSON = "cst_train_state.json"

# Keys in training checkpoint ``meta`` (full-run resume at collect segment boundaries + W&B)
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
    collection segment, or None if not present/invalid.

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
        if bpath.is_file() and math.isfinite(best_prev):
            logger.info("kept %s (best loss_mean=%.6f)", best_n, best_prev)

    n_removed = remove_legacy_phase_checkpoints(ckpt_dir)
    return {
        "latest": str(ckpt_dir / latest_n),
        "best": str(ckpt_dir / best_n) if (ckpt_dir / best_n).is_file() else None,
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
) -> Dict[str, Any]:
    """Train on each hidden file: one optimizer step per file, summed loss over anchors. Returns loss stats and ``per_chunk``."""
    cst_bank.train()
    total_loss = 0.0
    n_items = 0
    per_chunk: List[Dict[str, Any]] = []
    min_frames = 2
    for p in paths:
        H, _meta, _ = load_chunk_H(p)
        H = H.to(device)
        n_frames = H.shape[0]
        if n_frames < min_frames:
            logger.warning(
                "skip %s: need initial + at least one forward (>= %d rows), got %d",
                p,
                min_frames,
                n_frames,
            )
            continue
        lam = ramp.step()
        optimizer.zero_grad(set_to_none=True)
        anchor_losses: List[torch.Tensor] = []
        n_anchors_in_file = 0
        for anchor, h_win in iter_anchor_samples(H, w, device=device):
            loss, m = windowed_cst_loss_full(
                cst_bank,
                anchor,
                h_win,
                H,
                lambda_sparsity=lambda_sparsity,
                c_tanh=c_tanh,
                lambda_ramp=lam,
                lambda_pre_dead=lambda_pre_dead,
            )
            total_loss += float(m.get("loss", 0.0))
            n_items += 1
            n_anchors_in_file += 1
            if loss.requires_grad and loss.numel() > 0:
                anchor_losses.append(loss)
        if anchor_losses:
            chunk_loss = torch.stack(anchor_losses).sum()
            chunk_loss_val = float(chunk_loss.detach().item())
            p_str = str(p)
            per_chunk.append(
                {
                    "path": p_str,
                    "loss": chunk_loss_val,
                    "n_anchors": n_anchors_in_file,
                }
            )
            logger.info(
                "train chunk %s loss=%.6f n_anchors=%d (optimizer step)",
                Path(p).name,
                chunk_loss_val,
                n_anchors_in_file,
            )
            if chunk_loss.requires_grad:
                chunk_loss.backward()
                optimizer.step()
    return {
        "train_loss": total_loss,
        "train_batches": float(n_items),
        "per_chunk": per_chunk,
        "n_chunks": float(len(per_chunk)),
    }


def save_cst_checkpoint(
    path: str | Path,
    *,
    cst_bank: CSTBank,
    optimizer: torch.optim.Optimizer,
    ramp: LambdaRamp,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Write CST state, optimizer, ramp, and ``meta`` to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = {
        "cst": cst_bank.state_dict_for_save(),
        "optimizer": optimizer.state_dict(),
        "ramp": ramp.state_dict(),
        "meta": meta or {},
    }
    to_save = {
        "cst": map_float_tensors(raw["cst"], to_storage=True),
        "optimizer": map_float_tensors(raw["optimizer"], to_storage=True),
        "ramp": map_float_tensors(raw["ramp"], to_storage=True),
        "meta": raw["meta"],
        "storage_float_dtype": "bfloat16",
    }
    torch.save(to_save, path)


def load_cst_checkpoint(
    path: str | Path,
    cst_bank: CSTBank,
    optimizer: torch.optim.Optimizer,
    ramp: LambdaRamp,
) -> Dict[str, Any]:
    """Load from ``path``; return checkpoint ``meta``."""
    payload = torch.load(path, map_location="cpu")
    cst_sd = map_float_tensors(payload["cst"], to_storage=False)
    cst_bank.load_state_dict_from_save(cst_sd)
    if "optimizer" in payload:
        optimizer.load_state_dict(map_float_tensors(payload["optimizer"], to_storage=False))
    if "ramp" in payload:
        ramp.load_state_dict(map_float_tensors(payload["ramp"], to_storage=False))
    return payload.get("meta") or {}


def load_cst_bank_weights(path: str | Path, cst_bank: CSTBank) -> Dict[str, Any]:
    """Load only CST weights from ``path``; return ``meta``."""
    payload = torch.load(path, map_location="cpu")
    cst_sd = map_float_tensors(payload["cst"], to_storage=False)
    cst_bank.load_state_dict_from_save(cst_sd)
    return payload.get("meta") or {}


def delete_files(paths: List[str | Path]) -> None:
    """Unlink each path; log a warning on error."""
    for p in paths:
        try:
            Path(p).unlink(missing_ok=True)
        except OSError as e:
            logger.warning("could not delete %s: %s", p, e)
