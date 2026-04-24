"""CST training entry: collection + `run_train` (used by `launch_cst mode=train`)."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from omegaconf import DictConfig, OmegaConf

from cst.naming import cst_train_name_token, cst_train_staging_prefix, model_label_from_path, resolve_train_resume_latest

logger = logging.getLogger("cst.run")


def _train_cst_on_chunks_and_checkpoint(
    batch_paths: List[str],
    *,
    ckpt_phase: int,
    cst_cfg: Any,
    cst_bank: Any,
    optimizer: Any,
    ramp: Any,
    tr: Any,
    device: torch.device,
    wb: Any,
    global_step: int,
    ckpt_dir: Path,
    log_wandb: bool,
    name_token: str,
    # Optional extra W&B / logging fields (epoch-based training).
    epoch: Optional[int] = None,
    segment: Optional[int] = None,
) -> int:
    """Fit CST on ``batch_paths``, checkpoint, delete chunks; return updated ``global_step``."""
    from cst.train.trainer import delete_files, save_train_checkpoints, train_on_chunk_files

    logger.info("=== train on %d hidden files ===", len(batch_paths))
    cst_bank.to_device_all(device)
    stats = train_on_chunk_files(
        batch_paths,
        cst_bank,
        optimizer,
        ramp,
        w=int(cst_cfg.w),
        lambda_sparsity=float(tr.get("lambda_sparsity", 1e-3)),
        c_tanh=float(tr.get("c_tanh", 1.0)),
        lambda_pre_dead=float(tr.get("lambda_pre_dead", 0.0)),
        device=device,
    )
    if log_wandb and wb is not None:
        chunks = stats.get("per_chunk", [])
        n_ch = len(chunks)
        for i, ch in enumerate(chunks):
            global_step += 1
            row = {
                "train/chunk_loss": ch["loss"],
                "train/chunk_anchors": ch["n_anchors"],
                "train/chunk_index": float(i + 1),
                "train/chunks_in_phase": float(n_ch),
                "phase": ckpt_phase,
            }
            if epoch is not None:
                row["train/epoch"] = int(epoch)
            if segment is not None:
                row["train/segment"] = int(segment)
            wb.log(row, step=global_step)
        if not chunks:
            global_step += 1
            row2 = {
                "train/chunk_loss": 0.0,
                "train/chunks_in_phase": 0.0,
                "train/no_files_trained": 1.0,
                "phase": ckpt_phase,
            }
            if epoch is not None:
                row2["train/epoch"] = int(epoch)
            if segment is not None:
                row2["train/segment"] = int(segment)
            wb.log(row2, step=global_step)
        wb.log(
            {
                "train/loss_sum_phase": float(stats.get("train_loss", 0.0)),
                "train/anchor_count_phase": float(stats.get("train_batches", 0.0)),
                "phase": ckpt_phase,
            },
            step=global_step,
        )

    ck_info = save_train_checkpoints(
        ckpt_dir,
        cst_bank=cst_bank,
        optimizer=optimizer,
        ramp=ramp,
        phase=ckpt_phase,
        train_loss=float(stats.get("train_loss", 0.0)),
        train_batches=float(stats.get("train_batches", 0.0)),
        name_token=name_token,
    )
    if log_wandb and wb is not None:
        wbd: dict[str, Any] = {
            "train/ckpt_new_best": 1.0 if ck_info.get("updated_best") else 0.0,
            "train/legacy_phases_removed": float(ck_info.get("legacy_phase_removed", 0)),
        }
        lm = ck_info.get("loss_mean", float("inf"))
        if isinstance(lm, (int, float)) and math.isfinite(float(lm)):
            wbd["train/ckpt_loss_mean"] = float(lm)
        wb.log(wbd, step=global_step)
    delete_files(batch_paths)
    return global_step


def run_train(cfg: DictConfig) -> int:
    try:
        import wandb
    except ImportError:
        wandb = None

    tr = cfg.train
    device = torch.device(str(tr.get("device", "cuda")))
    staging = Path(str(tr.get("staging_dir", "outputs/cst_staging")))
    staging.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(str(tr.get("checkpoint_dir", "outputs/cst_ckpt")))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    from cst.model.cst_bank import CSTBank
    from cst.train.collect_batch import collect_hidden_until_size_gb_from_row_batches, load_model_for_collect
    from cst.train.trainer import LambdaRamp, load_cst_checkpoint
    from cst.train_data.compose import sample_uniform_batched

    tg = tr.get("target_staging_gb")
    if tg is None or str(tg).strip().lower() in ("", "null", "none"):
        target_gb = 1.0
    else:
        target_gb = float(tg)
    if target_gb <= 0:
        raise ValueError("train.target_staging_gb must be positive (decimal GB on disk).")
    inference_bs = int(tr.inference_batch_size)
    seed = int(tr.get("seed", 0))
    use_wandb = bool(tr.get("wandb", cfg.get("wandb", True)))

    sp_cfg = tr.get("samples_per_epoch")
    if sp_cfg is None or str(sp_cfg).strip().lower() in ("", "null", "none"):
        raise ValueError("train.samples_per_epoch must be set to a positive integer")
    try:
        n_sp = int(sp_cfg)
    except (TypeError, ValueError) as e:
        raise ValueError("train.samples_per_epoch must be a positive integer") from e
    if n_sp < 1:
        raise ValueError("train.samples_per_epoch must be >= 1")
    num_epochs = int(tr.get("num_epochs", 1))
    if num_epochs < 1:
        raise ValueError("train.num_epochs must be >= 1")

    cst_cfg = cfg.cst
    _model_lbl = model_label_from_path(str(cfg.model_path))
    _sched_m = str(tr.get("scheduler_mode", "pd"))
    _k, _w = int(cst_cfg.k), int(cst_cfg.w)
    cst_name_token = cst_train_name_token(_model_lbl, _sched_m, _k, _w)
    cst_bank = CSTBank(
        max_steps=int(cst_cfg.max_steps),
        d=int(cst_cfg.d_model),
        k=int(cst_cfg.k),
        w=int(cst_cfg.w),
        max_on_gpu=int(cst_cfg.get("max_on_gpu", cst_cfg.max_steps)),
        initial_threshold=float(cst_cfg.get("jumprelu_initial_threshold", 0.03)),
        bandwidth=float(cst_cfg.get("jumprelu_bandwidth", 1.0)),
    )
    lr = float(tr.get("learning_rate", 3e-4))
    optimizer = torch.optim.Adam(cst_bank.parameters(), lr=lr)
    ramp_total = int(tr.get("ramp_total_steps", 10_000))
    ramp = LambdaRamp(ramp_total)

    rp_cfg = tr.get("resume_checkpoint")
    rp_str = str(rp_cfg).strip() if rp_cfg is not None else ""
    if rp_str and rp_str.lower() not in ("null", "none", "") and Path(rp_str).is_file():
        load_cst_checkpoint(rp_str, cst_bank, optimizer, ramp)
        logger.info("resumed CST from %s", rp_str)
    elif bool(tr.get("resume_from_latest", False)):
        resume_p = resolve_train_resume_latest(ckpt_dir, cst_name_token)
        if resume_p is not None and resume_p.is_file():
            load_cst_checkpoint(str(resume_p), cst_bank, optimizer, ramp)
            logger.info("resumed CST from %s (train.resume_from_latest)", resume_p)
        else:
            logger.warning(
                "train.resume_from_latest true but no checkpoint in %s (tried cst_%s_latest.pt, cst_latest.pt)",
                ckpt_dir,
                cst_name_token,
            )

    n_cst_params = sum(p.numel() for p in cst_bank.parameters())
    n_cst_trainable = sum(p.numel() for p in cst_bank.parameters() if p.requires_grad)
    logger.info(
        "CST parameters: total=%s trainable=%s",
        f"{n_cst_params:,}",
        f"{n_cst_trainable:,}",
    )

    global_step = 0
    wb = None
    if use_wandb and wandb is not None:
        wb = wandb.init(
            project=str(cfg.get("wandb_project", "strace-cst")),
            name=str(tr.get("wandb_name", "cst-train")),
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
        )
        wb.log(
            {
                "cst/total_parameters": n_cst_params,
                "cst/trainable_parameters": n_cst_trainable,
            },
            step=global_step,
        )

    max_collect_batches = int(tr.get("max_collect_batches", 50_000))
    gsm_fs = int(tr.get("gsm8k_num_fewshot", 4))
    arc_fs = int(tr.get("arc_csqa_num_fewshot", 5))
    # If false: one draw of samples_per_epoch at start (seed only); all epochs reuse the same row_batches.
    # If true (legacy): resample each epoch with seed + epoch * 1_000_000.
    resample_each_epoch = bool(tr.get("resample_each_epoch", True))
    row_batches: Optional[List[List[Dict[str, Any]]]] = None

    for epoch in range(num_epochs):
        if resample_each_epoch or row_batches is None:
            draw_seed = seed + epoch * 1_000_000 if resample_each_epoch else seed
            row_batches = sample_uniform_batched(
                n_sp,
                inference_bs,
                seed=draw_seed,
                gsm8k_num_fewshot=gsm_fs,
                arc_csqa_num_fewshot=arc_fs,
            )
        nbat = len(row_batches)
        logger.info(
            "=== epoch %d / %d: %d row batches (samples=%d, batch=%d, resample_each_epoch=%s) ===",
            epoch + 1,
            num_epochs,
            nbat,
            n_sp,
            inference_bs,
            resample_each_epoch,
        )
        if nbat == 0:
            logger.warning("epoch %d: no batches, skipping", epoch)
            continue
        bi = 0
        seg = 0
        while bi < nbat:
            rem = nbat - bi
            this_max = max(1, min(max_collect_batches, rem))
            logger.info(
                "=== collect epoch %d / %d segment %d (batch %d / %d) ===",
                epoch + 1,
                num_epochs,
                seg + 1,
                bi,
                nbat,
            )
            model, tokenizer = load_model_for_collect(
                str(cfg.model_path),
                device=str(device),
                compile_model=bool(tr.get("compile_model", False)),
            )
            batch_paths: List[str] = []
            try:
                batch_paths, next_bi, coll_sum = collect_hidden_until_size_gb_from_row_batches(
                    model,
                    tokenizer,
                    str(cfg.model_path),
                    staging,
                    row_batches=row_batches,
                    start_idx=bi,
                    target_gb=target_gb,
                    device=str(device),
                    mode_key=str(tr.get("scheduler_mode", "pd")),
                    gen_length=int(tr.get("gen_length", 256)),
                    steps=int(tr.get("steps", 256)),
                    block_length=tr.get("block_length"),
                    top_k=int(tr.get("top_k", 1)),
                    max_prompt_length=int(tr.get("max_prompt_length", 512)),
                    trace_name_prefix=cst_train_staging_prefix(
                        _model_lbl, _sched_m, _k, _w, epoch, seg
                    ),
                    max_batches=this_max,
                    staging_name_token=cst_name_token,
                )
                if not batch_paths:
                    logger.warning(
                        "collect returned no files (start_idx=%d, n_batches=%d); stopping epoch pass",
                        bi,
                        nbat,
                    )
                    break
                logger.info(
                    "collect segment: wrote %.3f GB in %d batches, next batch idx %d (target ≥ %.3f GB; last segment may be smaller)",
                    coll_sum.get("written_gb", 0.0),
                    coll_sum.get("n_batches", 0),
                    coll_sum.get("next_idx", next_bi),
                    target_gb,
                )
                if wb:
                    wb.log(
                        {
                            "collect/epoch": epoch,
                            "collect/segment": seg,
                            "collect/written_gb": coll_sum.get("written_gb", 0),
                            "collect/n_batches": coll_sum.get("n_batches", 0),
                            "collect/total_gen_time": coll_sum.get("total_gen_time", 0),
                            "collect/next_batch_idx": float(coll_sum.get("next_idx", next_bi)),
                        },
                        step=global_step,
                    )
                global_step += int(coll_sum.get("n_batches", 0))
            finally:
                del model
                del tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            bi = int(coll_sum.get("next_idx", next_bi))
            ckpt_phase = epoch * 10_000 + seg
            global_step = _train_cst_on_chunks_and_checkpoint(
                batch_paths,
                ckpt_phase=ckpt_phase,
                cst_cfg=cst_cfg,
                cst_bank=cst_bank,
                optimizer=optimizer,
                ramp=ramp,
                tr=tr,
                device=device,
                wb=wb,
                global_step=global_step,
                ckpt_dir=ckpt_dir,
                log_wandb=use_wandb and wb is not None,
                name_token=cst_name_token,
                epoch=epoch,
                segment=seg,
            )
            seg += 1

    if wb:
        wandb.finish()
    return 0
