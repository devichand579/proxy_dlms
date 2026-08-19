#!/usr/bin/env python3
"""
CST entry point: store hiddens, train CST, or eval CST.

Config: config/cst.yaml. Examples:
  python hydra_runners/run_cst.py mode=store store.scope=train
  python hydra_runners/run_cst.py mode=train
  python hydra_runners/run_cst.py mode=eval eval.checkpoint=...
Or: ./scripts/cst.sh --mode store|train|eval
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from repo_env_defaults import apply_repo_env_defaults

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from cst.hf_hiddens.config import hf_token, resolve_eval_repo_id, resolve_org, resolve_train_repo_id
from cst.hf_hiddens.jobs import dispatch_from_config
from cst.hf_hiddens.store import fetch_manifest
from cst.model.cst_bank import CSTBank
from cst.model.residency import CSTResidency, budget_report
from cst.train.adam_bf16 import AdamBF16
from cst.naming import (
    cst_eval_task_name_token,
    cst_hidden_segment_prefix,
    cst_train_name_token,
    model_label_from_path,
    resolve_train_best_in_dir,
    resolve_train_resume_latest,
)
from cst.train.collect_batch import collect_hidden_until_size_gb_from_row_batches, load_model_for_collect
from cst.train.trainer import (
    LambdaRamp,
    META_TRAIN_NUM_EPOCHS,
    META_WANDB_GLOBAL_STEP,
    META_WANDB_RUN_ID,
    delete_files,
    load_cst_checkpoint,
    merge_resume_fields_into_meta,
    merge_sampling_state_into_meta,
    parse_data_resume_from_meta,
    parse_sampling_state_from_meta,
    save_train_checkpoints,
    train_on_chunk_files,
)
from cst.train_data.compose import sample_uniform_batched
from cst.eval_data.compose import eval_num_fewshot, load_eval_row_batches

_CONFIG_DIR = _REPO_ROOT / "config"
logger = logging.getLogger("run_cst")

_EVAL_DEFAULTS = {
    "hidden_dir": "outputs/eval_hidden",
    "checkpoint": "outputs/cst_ckpt/cst_best.pt",
    "model_path": None,
    "ntf_conf_low": 0.2,
    "ntf_conf_high": 0.8,
    "ntf_first_non_eos": False,
    "device": "cuda",
    "w": None,
    "wandb_name": "cst-eval",
    "hidden_source": "local",
}


def _coerce_use_eos_harness(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "on")
    return True if v is None else bool(v)


def _merge_eval_cfg(cfg: DictConfig) -> DictConfig:
    base = OmegaConf.create(_EVAL_DEFAULTS)
    o = cfg.get("eval")
    if o is not None:
        return OmegaConf.merge(base, o)
    return base


def _sampling_state_from_config(tr: Any) -> Dict[str, Any]:
    return {
        "train_seed": int(tr.get("seed", 0)),
        "train_samples_per_epoch": int(tr.get("samples_per_epoch", 2048)),
        "train_inference_batch_size": int(tr.inference_batch_size),
        "train_resample_each_epoch": bool(tr.get("resample_each_epoch", False)),
        "train_gsm8k_num_fewshot": int(tr.get("gsm8k_num_fewshot", 4)),
        "train_arc_csqa_num_fewshot": int(tr.get("arc_csqa_num_fewshot", 5)),
    }


def _target_staging_gb(store_cfg: Any, tr: Any) -> float:
    tg = store_cfg.get("target_staging_gb")
    if tg is None or str(tg).strip().lower() in ("", "null", "none"):
        tg = tr.get("target_staging_gb")
    target_gb = 1.0 if tg is None or str(tg).strip().lower() in ("", "null", "none") else float(tg)
    if target_gb <= 0:
        raise ValueError("store.target_staging_gb / train.target_staging_gb must be positive.")
    return target_gb


def _store_hiddens_segments(
    cfg: DictConfig,
    *,
    scope: str,
    name_token: str,
    repo_id: str,
    row_batches: List[List[Dict[str, Any]]],
    sampling_state: Dict[str, Any],
    sched_m: str,
    wandb_name: str,
) -> int:
    """Collect hiddens by segment and upload each slice to the Hub."""
    try:
        import wandb
    except ImportError:
        wandb = None

    tr = cfg.train
    store_cfg = cfg.get("store") or OmegaConf.create({})
    device = torch.device(str(tr.get("device", "cuda")))
    staging = Path(str(tr.get("staging_dir", "outputs/cst_staging")))
    staging.mkdir(parents=True, exist_ok=True)
    target_gb = _target_staging_gb(store_cfg, tr)
    use_wandb = bool(tr.get("wandb", cfg.get("wandb", True)))
    capture_context = "train" if scope == "train" else "eval"

    nbat = len(row_batches)
    logger.info(
        "store %s hiddens -> %s (%s): %d row batches, %.3f GB per segment",
        scope,
        repo_id,
        name_token,
        nbat,
        target_gb,
    )
    if nbat == 0:
        raise RuntimeError(f"store.scope={scope}: no row batches to collect")

    wb = None
    global_step = 0
    if use_wandb and wandb is not None:
        wb = wandb.init(
            project=str(cfg.get("wandb_project", "strace-cst")),
            name=str(wandb_name),
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
        )

    bi, seg = 0, 0
    while bi < nbat:
        seg_prefix = cst_hidden_segment_prefix(name_token, seg)
        logger.info("=== collect segment %d (batch %d / %d) ===", seg + 1, bi, nbat)

        model, tokenizer = load_model_for_collect(
            str(cfg.model_path),
            device=str(device),
            compile_model=bool(tr.get("compile_model", False)),
        )
        ueh = _coerce_use_eos_harness(tr.get("use_eos_harness", True))
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
                mode_key=sched_m,
                gen_length=int(tr.get("gen_length", 256)),
                steps=int(tr.get("steps", 256)),
                block_length=tr.get("block_length"),
                top_k=int(tr.get("top_k", 1)),
                max_prompt_length=int(tr.get("max_prompt_length", 512)),
                staging_name_token=name_token,
                segment=seg,
                capture_context=capture_context,
                use_eos_harness=ueh,
            )
        finally:
            del model
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not batch_paths:
            logger.warning("collect returned no files (start_idx=%d); stopping", bi)
            break

        logger.info(
            "collect segment: wrote %.3f GB in %d batches, next idx %d",
            coll_sum.get("written_gb", 0.0),
            coll_sum.get("n_batches", 0),
            coll_sum.get("next_idx", next_bi),
        )

        dispatch_from_config(
            tr,
            {
                "type": "upload_hidden_segment",
                "repo_id": repo_id,
                "name_token": name_token,
                "local_paths": list(batch_paths),
                "segment": seg,
                "prefix": seg_prefix,
                "next_batch_idx": int(coll_sum.get("next_idx", next_bi)),
                "sampling_fingerprint": dict(sampling_state),
                "scope": scope,
                "delete_local_after": bool(tr.get("delete_local_after_hf_push", False)),
            },
        )

        if wb:
            wb.log(
                {
                    "collect/segment": seg,
                    "collect/written_gb": coll_sum.get("written_gb", 0),
                    "collect/n_batches": coll_sum.get("n_batches", 0),
                    "collect/next_batch_idx": float(coll_sum.get("next_idx", next_bi)),
                },
                step=global_step,
            )
        global_step += int(coll_sum.get("n_batches", 0))

        bi = int(coll_sum.get("next_idx", next_bi))
        if bool(tr.get("delete_local_chunks_after_train", True)):
            delete_files(batch_paths)
        seg += 1

    logger.info("collect finished: %s on %s", name_token, repo_id)
    if wb:
        wandb.finish()
    return 0


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
    epoch: Optional[int] = None,
    segment: Optional[int] = None,
    data_next_epoch: int = 0,
    data_next_batch_idx: int = 0,
    data_next_segment: int = 0,
    num_epochs: int = 1,
    wandb_run_id_for_meta: Optional[str] = None,
    sampling_state: Optional[Dict[str, Any]] = None,
    residency: Any = None,
) -> int:
    logger.info("=== train on %d hidden files ===", len(batch_paths))
    if residency is None:
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
        anchor_block=int(tr.get("anchor_block", 24)),
        residency=residency,
    )
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        logger.info(
            "peak GPU memory this phase: %.1f GB (tune train.anchor_block from this)",
            torch.cuda.max_memory_allocated(device) / 1e9,
        )
        torch.cuda.reset_peak_memory_stats(device)
    if log_wandb and wb is not None:
        chunks = stats.get("per_chunk", [])
        n_ch = len(chunks)
        for i, ch in enumerate(chunks):
            global_step += 1
            row: Dict[str, Any] = {
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
            row2: Dict[str, Any] = {
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

    extra_meta: Dict[str, Any] = {}
    run_id: Optional[str] = None
    if wb is not None and getattr(wb, "id", None):
        run_id = str(wb.id)
    elif wandb_run_id_for_meta:
        run_id = str(wandb_run_id_for_meta)
    merge_resume_fields_into_meta(
        extra_meta,
        data_next_epoch=int(data_next_epoch),
        data_next_batch_idx=int(data_next_batch_idx),
        data_next_segment=int(data_next_segment),
        wandb_global_step=int(global_step),
        wandb_run_id=run_id,
        num_epochs=int(num_epochs),
    )
    if sampling_state is not None:
        merge_sampling_state_into_meta(
            extra_meta,
            train_seed=int(sampling_state["train_seed"]),
            train_samples_per_epoch=int(sampling_state["train_samples_per_epoch"]),
            train_inference_batch_size=int(sampling_state["train_inference_batch_size"]),
            train_resample_each_epoch=bool(sampling_state["train_resample_each_epoch"]),
            train_gsm8k_num_fewshot=int(sampling_state["train_gsm8k_num_fewshot"]),
            train_arc_csqa_num_fewshot=int(sampling_state["train_arc_csqa_num_fewshot"]),
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
        extra_meta=extra_meta,
        save_optimizer_state=bool(tr.get("save_optimizer_state", False)),
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
    if bool(tr.get("delete_local_chunks_after_train", True)):
        delete_files(batch_paths)
    return global_step


# ---------------------------------------------------------------------------
# mode=store store.scope=train
# ---------------------------------------------------------------------------


def store_train_hiddens(cfg: DictConfig) -> int:
    """Store train hidden chunks on the Hub."""
    tr = cfg.train
    store_cfg = cfg.get("store") or OmegaConf.create({})

    sp_cfg = store_cfg.get("samples")
    if sp_cfg is None or str(sp_cfg).strip().lower() in ("", "null", "none"):
        sp_cfg = tr.get("samples_per_epoch")
    if sp_cfg is None or str(sp_cfg).strip().lower() in ("", "null", "none"):
        raise ValueError("store.samples / train.samples_per_epoch must be set")
    n_sp = int(sp_cfg)
    if n_sp < 1:
        raise ValueError("store.samples must be >= 1")

    inference_bs = int(tr.inference_batch_size)
    seed = int(tr.get("seed", 0))
    gsm_fs = int(tr.get("gsm8k_num_fewshot", 4))
    arc_fs = int(tr.get("arc_csqa_num_fewshot", 5))

    model_lbl = model_label_from_path(str(cfg.model_path))
    sched_m = str(tr.get("scheduler_mode", "pd"))
    k, w = int(cfg.cst.k), int(cfg.cst.w)
    lam = float(tr.get("lambda_sparsity", 1e-3))
    name_token = cst_train_name_token(model_lbl, sched_m, k, w, lam)
    hf_train_repo = resolve_train_repo_id(tr)

    sampling_state = {
        "train_seed": int(seed),
        "train_samples_per_epoch": int(n_sp),
        "train_inference_batch_size": int(inference_bs),
        "train_resample_each_epoch": False,
        "train_gsm8k_num_fewshot": int(gsm_fs),
        "train_arc_csqa_num_fewshot": int(arc_fs),
    }

    row_batches = sample_uniform_batched(
        n_sp, inference_bs, seed=seed, gsm8k_num_fewshot=gsm_fs, arc_csqa_num_fewshot=arc_fs
    )

    return _store_hiddens_segments(
        cfg,
        scope="train",
        name_token=name_token,
        repo_id=hf_train_repo,
        row_batches=row_batches,
        sampling_state=sampling_state,
        sched_m=sched_m,
        wandb_name=str(tr.get("wandb_name", "cst-store-train")),
    )


# ---------------------------------------------------------------------------
# mode=store store.scope=eval
# ---------------------------------------------------------------------------


def store_eval_hiddens(cfg: DictConfig) -> int:
    """Store eval hidden chunks on the Hub."""
    tr = cfg.train
    store = cfg.get("store") or OmegaConf.create({})

    task = store.get("task_name")
    if task is None or str(task).strip().lower() in ("", "null", "none"):
        raise ValueError("store.scope=eval requires store.task_name")

    sched_m = str(store.get("scheduler_tag") or tr.get("scheduler_mode", "pd"))
    k, w = int(cfg.cst.k), int(cfg.cst.w)
    lam = float(tr.get("lambda_sparsity", 1e-3))
    name_token = store.get("name_token")
    if name_token is None or str(name_token).strip().lower() in ("", "null", "none"):
        name_token = cst_eval_task_name_token(
            model_label_from_path(str(cfg.model_path)), sched_m, str(task), k, w, lam
        )
    else:
        name_token = str(name_token).strip()

    repo_id = resolve_eval_repo_id(tr, store=store)
    inference_bs = int(tr.inference_batch_size)
    seed = int(tr.get("seed", 0))

    nfs = store.get("num_fewshot")
    num_fewshot = int(nfs) if nfs not in (None, "", "null", "none") else eval_num_fewshot(str(task))

    sp_cfg = store.get("samples")
    limit = None
    if sp_cfg is not None and str(sp_cfg).strip().lower() not in ("", "null", "none"):
        limit = int(sp_cfg)
        if limit < 1:
            raise ValueError("store.samples must be >= 1 when set")

    row_batches = load_eval_row_batches(
        str(task),
        inference_bs,
        limit=limit,
        num_fewshot=num_fewshot,
        seed=seed,
    )

    sampling_state = {
        "eval_task": str(task),
        "eval_num_fewshot": int(num_fewshot),
        "eval_limit": limit,
        "eval_inference_batch_size": int(inference_bs),
        "eval_seed": int(seed),
        "eval_scheduler": sched_m,
    }

    return _store_hiddens_segments(
        cfg,
        scope="eval",
        name_token=name_token,
        repo_id=repo_id,
        row_batches=row_batches,
        sampling_state=sampling_state,
        sched_m=sched_m,
        wandb_name=str(store.get("wandb_name") or tr.get("wandb_name", "cst-store-eval")),
    )


def run_store(cfg: DictConfig) -> int:
    store = cfg.get("store") or OmegaConf.create({})
    scope = str(store.get("scope", "train")).strip().lower()
    if scope == "train":
        return store_train_hiddens(cfg)
    if scope == "eval":
        return store_eval_hiddens(cfg)
    raise ValueError(f"store.scope must be 'train' or 'eval', got {scope!r}")


# ---------------------------------------------------------------------------
# mode=train
# ---------------------------------------------------------------------------


def run_train(cfg: DictConfig) -> int:
    """Train CST from Hub chunks, downloading and deleting locals each step."""
    try:
        import wandb
    except ImportError:
        wandb = None

    tr = cfg.train
    hs = str(tr.get("hidden_source", "hf_hub")).strip().lower()
    if hs not in ("hf", "hf_hub", "hub"):
        raise ValueError(
            "mode=train requires Hub hiddens (train.hidden_source=hf_hub). "
            "Store first: python hydra_runners/run_cst.py mode=store store.scope=train"
        )

    device = torch.device(str(tr.get("device", "cuda")))
    staging = Path(str(tr.get("staging_dir", "outputs/cst_staging")))
    staging.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(str(tr.get("checkpoint_dir", "outputs/cst_ckpt")))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tg = tr.get("target_staging_gb")
    target_gb = 1.0 if tg is None or str(tg).strip().lower() in ("", "null", "none") else float(tg)
    if target_gb <= 0:
        raise ValueError("train.target_staging_gb must be positive.")

    use_wandb = bool(tr.get("wandb", cfg.get("wandb", True)))
    num_epochs = int(tr.get("num_epochs", 1))
    if num_epochs < 1:
        raise ValueError("train.num_epochs must be >= 1")

    cst_cfg = cfg.cst
    model_lbl = model_label_from_path(str(cfg.model_path))
    sched_m = str(tr.get("scheduler_mode", "pd"))
    k, w = int(cst_cfg.k), int(cst_cfg.w)
    lam = float(tr.get("lambda_sparsity", 1e-3))
    name_token = cst_train_name_token(model_lbl, sched_m, k, w, lam)
    hf_train_repo = resolve_train_repo_id(tr)
    cache_dir = Path(str(tr.get("hf_local_cache_dir", "outputs/cst_hf_cache")))

    cst_bank = CSTBank(
        max_steps=int(cst_cfg.max_steps),
        d=int(cst_cfg.d_model),
        k=int(cst_cfg.k),
        w=int(cst_cfg.w),
        max_on_gpu=int(cst_cfg.get("max_on_gpu", cst_cfg.max_steps)),
        initial_threshold=float(cst_cfg.get("jumprelu_initial_threshold", 0.03)),
        bandwidth=float(cst_cfg.get("jumprelu_bandwidth", 1.0)),
    )
    # bf16 moments halve optimizer state (172 GB -> 86 GB for a 512-step k=4096
    # bank), which is what makes the full bank fit across GPU + host RAM.
    optimizer = AdamBF16(cst_bank.parameters(), lr=float(tr.get("learning_rate", 3e-4)))
    ramp = LambdaRamp(int(tr.get("ramp_total_steps", 10_000)))

    ckpt_meta: Dict[str, Any] = {}
    loaded_ckpt_path: Optional[str] = None
    rp_str = str(tr.get("resume_checkpoint") or "").strip()
    if rp_str and rp_str.lower() not in ("null", "none", "") and Path(rp_str).is_file():
        ckpt_meta = load_cst_checkpoint(rp_str, cst_bank, optimizer, ramp)
        loaded_ckpt_path = rp_str
        logger.info("resumed CST from %s", rp_str)
    elif bool(tr.get("resume_from_latest", False)):
        resume_p = resolve_train_resume_latest(ckpt_dir, name_token)
        if resume_p is not None and resume_p.is_file():
            ckpt_meta = load_cst_checkpoint(str(resume_p), cst_bank, optimizer, ramp)
            loaded_ckpt_path = str(resume_p)
            logger.info("resumed CST from %s (train.resume_from_latest)", resume_p)

    resume_data_from_checkpoint = bool(tr.get("resume_data_from_checkpoint", True))
    resume_trip: Optional[tuple[int, int, int]] = None
    global_step = 0
    wandb_run_id_resumed: Optional[str] = None
    if loaded_ckpt_path:
        try:
            global_step = int(ckpt_meta.get(META_WANDB_GLOBAL_STEP, 0) or 0)
        except (TypeError, ValueError):
            global_step = 0
        wr = ckpt_meta.get(META_WANDB_RUN_ID)
        if wr is not None and str(wr).strip():
            wandb_run_id_resumed = str(wr).strip()
        if resume_data_from_checkpoint:
            resume_trip = parse_data_resume_from_meta(ckpt_meta, require_all=True)
            n_ep_saved = ckpt_meta.get(META_TRAIN_NUM_EPOCHS)
            if n_ep_saved is not None and int(n_ep_saved) != int(num_epochs):
                logger.warning(
                    "Checkpoint num_epochs=%s != config num_epochs=%s; using saved cursor.",
                    n_ep_saved,
                    num_epochs,
                )

    sampling_state = _sampling_state_from_config(tr)
    if loaded_ckpt_path and resume_data_from_checkpoint and bool(tr.get("resume_override_sampling", True)):
        sfp = parse_sampling_state_from_meta(ckpt_meta)
        if sfp is not None:
            sampling_state = sfp

    manifest = fetch_manifest(
        hf_train_repo,
        name_token,
        cache_dir=cache_dir,
        revision=str(tr.get("hf_hiddens_revision", "main")),
        token=hf_token(tr),
        create_if_missing=False,
    )
    manifest_fp = manifest.sampling_fingerprint_from_manifest()
    if manifest_fp is not None:
        sampling_state = manifest_fp

    n_cst_params = sum(p.numel() for p in cst_bank.parameters())
    logger.info("CST parameters: total=%s", f"{n_cst_params:,}")
    logger.info("train from Hub: %s (%s)", hf_train_repo, name_token)

    # Residency: pin as much of the bank on GPU as ``gpu_pinned_gb`` allows and
    # stream the rest from host RAM on the anchor sweep. gpu_pinned_gb=0 keeps
    # the old behaviour (whole bank resident via to_device_all), which is fine
    # for small banks but is ~344 GB of state at max_steps=512, k=4096.
    pinned_gb = float(tr.get("gpu_pinned_gb", 0.0) or 0.0)
    bud = budget_report(
        max_steps=int(cst_cfg.max_steps),
        d=int(cst_cfg.d_model),
        k=int(cst_cfg.k),
        w=int(cst_cfg.w),
        anchor_block=int(tr.get("anchor_block", 24)),
        pinned_gb=pinned_gb,
        batch=int(tr.get("inference_batch_size", 2)),
        seq_len=int(tr.get("gen_length", 256)) + int(tr.get("max_prompt_length", 512)),
        has_moments=True,
    )
    logger.info(
        "memory budget: bank %.0f GB (%.3f GB/CST) | pinned %.0f GB | %d CSTs live "
        "| predicted GPU %.0f GB, host %.0f GB",
        bud["bank_total_gb"], bud["per_cst_gb"], bud["pinned_gb"],
        int(bud["live_csts"]), bud["gpu_gb"], bud["host_gb"],
    )
    logger.info(
        "  tune with train.anchor_block (GPU) and train.gpu_pinned_gb (splits bank GPU/host); "
        "seq_len assumed worst-case gen_length + max_prompt_length"
    )
    residency: Optional[CSTResidency] = None
    if pinned_gb > 0.0:
        residency = CSTResidency(
            cst_bank,
            device,
            optimizer=optimizer,
            pinned_bytes=int(pinned_gb * 1e9),
        )
    else:
        cst_bank.set_device(device)

    if resume_trip is not None:
        r_e, r_b, _r_s = resume_trip
        if r_e >= int(num_epochs):
            logger.info("Training already complete (pass %d >= num_epochs %d).", r_e, num_epochs)
            return 0
        logger.info("Resume pass=%d batch=%d wandb_step=%d", r_e, r_b, global_step)
        if r_b < 0:
            resume_trip = None
            global_step = 0
            wandb_run_id_resumed = None
            r_e, r_b = 0, 0
    else:
        r_e, r_b = 0, 0

    wb = None
    if use_wandb and wandb is not None:
        init_kwargs: Dict[str, Any] = {
            "project": str(cfg.get("wandb_project", "strace-cst")),
            "name": str(tr.get("wandb_name", "cst-train")),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "reinit": True,
        }
        if wandb_run_id_resumed:
            init_kwargs["id"] = wandb_run_id_resumed
            init_kwargs["resume"] = "allow"
        wb = wandb.init(**init_kwargs)
        if wb is not None and str(getattr(wb, "id", "")).strip():
            wandb_run_id_resumed = str(wb.id)
        if global_step == 0:
            wb.log({"cst/total_parameters": n_cst_params}, step=global_step)

    nbat = manifest.total_batch_count()
    if nbat == 0:
        raise FileNotFoundError(
            f"no hiddens in manifest for {name_token} @ {hf_train_repo}. "
            "Run: python hydra_runners/run_cst.py mode=store store.scope=train"
        )

    for cst_pass in range(num_epochs):
        logger.info("=== CST pass %d / %d (%d batches) ===", cst_pass + 1, num_epochs, nbat)
        if resume_trip is not None and cst_pass < r_e:
            continue
        if resume_trip is not None and cst_pass == r_e and r_b >= nbat:
            r_e += 1
            resume_trip = (r_e, 0, 0)
            if r_e >= num_epochs:
                return 0
            continue
        bi = r_b if resume_trip is not None and cst_pass == r_e else 0

        step_in_pass = 0
        while bi < nbat:
            step_in_pass += 1
            dl_dir = staging / "hf_dl" / name_token / f"pass{cst_pass:03d}_step{step_in_pass:04d}"
            logger.info(
                "=== download pass %d step %d (batch %d / %d, target %.3f GB) ===",
                cst_pass + 1,
                step_in_pass,
                bi,
                nbat,
                target_gb,
            )
            try:
                result = dispatch_from_config(
                    tr,
                    {
                        "type": "download_hiddens_until_gb",
                        "repo_id": str(hf_train_repo),
                        "name_token": name_token,
                        "start_batch_idx": bi,
                        "target_gb": target_gb,
                        "local_dir": str(dl_dir),
                    },
                )
                batch_paths = [str(p) for p in result.get("local_paths") or []]
                next_bi = int(result.get("next_batch_idx", bi))
                coll_sum = dict(result.get("summary") or {})
                if not batch_paths:
                    raise RuntimeError(f"HF download returned no paths: {coll_sum}")
            except FileNotFoundError as e:
                logger.error("Hub data missing — run mode=store first: %s", e)
                break

            if wb:
                wb.log(
                    {
                        "download/pass": cst_pass,
                        "download/step": step_in_pass,
                        "download/written_gb": coll_sum.get("written_gb", 0),
                        "download/n_batches": coll_sum.get("n_batches", 0),
                    },
                    step=global_step,
                )
            global_step += int(coll_sum.get("n_batches", 0))

            bi = int(coll_sum.get("next_idx", next_bi))
            d_ne = int(cst_pass) + 1 if bi >= nbat else int(cst_pass)
            d_nb = 0 if bi >= nbat else int(bi)

            global_step = _train_cst_on_chunks_and_checkpoint(
                batch_paths,
                ckpt_phase=int(cst_pass) * 100_000 + step_in_pass,
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
                name_token=name_token,
                epoch=cst_pass,
                segment=step_in_pass,
                data_next_epoch=d_ne,
                data_next_batch_idx=d_nb,
                data_next_segment=0,
                num_epochs=int(num_epochs),
                wandb_run_id_for_meta=wandb_run_id_resumed if (not use_wandb or wb is None) else None,
                sampling_state=sampling_state,
                residency=residency,
            )

    if wb:
        wandb.finish()
    return 0


# ---------------------------------------------------------------------------
# mode=eval
# ---------------------------------------------------------------------------


def run_eval(cfg: DictConfig) -> int:
    """Score a CST checkpoint on saved eval hiddens."""
    try:
        import wandb
    except ImportError:
        wandb = None

    from cst.eval.run_eval_cst import run_eval_on_dir
    from cst.hf_hiddens.segment import download_all_manifest_chunks

    ev = _merge_eval_cfg(cfg)
    hidden_dir = str(ev.get("hidden_dir", _EVAL_DEFAULTS["hidden_dir"]))
    hidden_source = str(ev.get("hidden_source", "local")).strip().lower()

    if hidden_source in ("hf", "hf_hub", "hub"):
        tr = cfg.train
        store = cfg.get("store") or OmegaConf.create({})
        task = ev.get("task_name") or store.get("task_name")
        if not task or str(task).strip().lower() in ("", "null", "none"):
            raise ValueError("eval.hidden_source=hf_hub requires eval.task_name (or store.task_name)")
        sched = ev.get("scheduler_tag") or tr.get("scheduler_mode", "pd")
        lam = float(ev.get("lambda_sparsity", tr.get("lambda_sparsity", 1e-3)))
        name_token = ev.get("name_token")
        if not name_token or str(name_token).strip().lower() in ("", "null", "none"):
            name_token = cst_eval_task_name_token(
                model_label_from_path(str(cfg.model_path)),
                str(sched),
                str(task),
                int(cfg.cst.k),
                int(cfg.cst.w),
                lam,
            )
        repo_id = resolve_eval_repo_id(tr, eval_cfg=ev)
        cache = Path(str(tr.get("hf_local_cache_dir", "outputs/cst_hf_cache"))) / "eval_dl" / name_token
        hidden_dir = str(
            download_all_manifest_chunks(
                repo_id,
                str(name_token),
                local_dir=cache,
                cache_dir=str(tr.get("hf_local_cache_dir", "outputs/cst_hf_cache")),
                revision=str(tr.get("hf_hiddens_revision", "main")),
                token=hf_token(tr),
            )
        )

    ck = str(ev.get("checkpoint", _EVAL_DEFAULTS["checkpoint"]))
    ckp = Path(ck)
    if not ckp.is_file() and ckp.parent.is_dir():
        alt = resolve_train_best_in_dir(ckp.parent)
        if alt is not None and alt.is_file():
            ck = str(alt)

    ntf_1st = ev.get("ntf_first_non_eos", _EVAL_DEFAULTS["ntf_first_non_eos"])
    ntf_1st_b = (
        ntf_1st.strip().lower() in ("true", "1", "yes", "on")
        if isinstance(ntf_1st, str)
        else bool(ntf_1st)
    )

    out = run_eval_on_dir(
        hidden_dir=hidden_dir,
        ckpt_path=ck,
        cfg_bank=OmegaConf.to_container(cfg.cst, resolve=True),
        device=str(ev.get("device", "cuda")),
        w=int(ev.get("w")) if ev.get("w") is not None else None,
        model_path=str(ev.get("model_path")) if ev.get("model_path") not in (None, "", "null") else None,
        ntf_conf_low=float(ev.get("ntf_conf_low", 0.2)),
        ntf_conf_high=float(ev.get("ntf_conf_high", 0.8)),
        ntf_first_non_eos=ntf_1st_b,
        gpu_pinned_gb=float(ev.get("gpu_pinned_gb", 0.0) or 0.0),
    )
    print(OmegaConf.to_yaml(out))

    use_wb = ev.get("wandb")
    if use_wb is None:
        use_wb = cfg.get("wandb", True)
    if bool(use_wb) and wandb is not None:
        wb = wandb.init(
            project=str(cfg.get("wandb_project", "strace-cst")),
            name=str(ev.get("wandb_name", "cst-eval")),
            reinit=True,
        )
        if "summary" in out:
            wb.log({f"eval/{k}": v for k, v in out["summary"].items()})
        wb.finish()
    return 0


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------


def dispatch(cfg: DictConfig) -> int:
    mode = str(cfg.get("mode", "train")).strip().lower()
    if mode in ("store", "collect"):
        return run_store(cfg)
    if mode == "train":
        return run_train(cfg)
    if mode == "eval":
        return run_eval(cfg)
    raise ValueError(f"unknown mode={mode!r} (use store, train, or eval)")


@hydra.main(version_base="1.2", config_path=str(_CONFIG_DIR), config_name="cst")
def main(cfg: DictConfig) -> None:
    apply_repo_env_defaults()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(int(dispatch(cfg)))


if __name__ == "__main__":
    main()
