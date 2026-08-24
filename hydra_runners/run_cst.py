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

from cst.hf_hiddens.config import hf_token, resolve_eval_repo_id, resolve_train_repo_id
from cst.hf_hiddens.store import (
    download_all_chunks,
    download_hidden_chunk,
    fetch_manifest,
    upload_hidden_chunk,
)
from cst.model.cst_bank import CSTBank
from cst.naming import (
    cst_ckpt_name_token,
    cst_hidden_name_token,
    model_label_from_path,
    resolve_train_best_in_dir,
    resolve_train_resume_latest,
)
from cst.train.collect_batch import collect_hidden_row_batch, load_model_for_collect
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


def _is_blank(v: Any) -> bool:
    return v is None or str(v).strip().lower() in ("", "null", "none")


def _section_model_path(cfg: DictConfig, section: Any) -> str:
    mp = section.get("model_path") if section is not None else None
    if _is_blank(mp):
        mp = cfg.get("model_path")
    if _is_blank(mp):
        raise ValueError("model_path is required")
    return str(mp)


def _hidden_name_token(
    section: Any,
    model_path: str,
    *,
    task_name: Optional[str] = None,
) -> str:
    override = section.get("name_token") if section is not None else None
    if not _is_blank(override):
        return str(override).strip()
    sched = "pd"
    if section is not None and not _is_blank(section.get("scheduler_mode")):
        sched = str(section.get("scheduler_mode"))
    gen_length = 256 if section is None else int(section.get("gen_length", 256))
    steps = 256 if section is None else int(section.get("steps", 256))
    return cst_hidden_name_token(
        model_label_from_path(model_path),
        sched,
        gen_length,
        steps,
        task_name=task_name,
    )


def _sampling_state_from_store(store_cfg: Any, *, n_samples: int, inference_bs: int) -> Dict[str, Any]:
    return {
        "train_seed": int(store_cfg.get("seed", 0)),
        "train_samples_per_epoch": int(n_samples),
        "train_inference_batch_size": int(inference_bs),
        "train_resample_each_epoch": False,
        "train_gsm8k_num_fewshot": int(store_cfg.get("gsm8k_num_fewshot", 4)),
        "train_arc_csqa_num_fewshot": int(store_cfg.get("arc_csqa_num_fewshot", 5)),
    }


def _store_hiddens(
    cfg: DictConfig,
    *,
    scope: str,
    name_token: str,
    repo_id: str,
    row_batches: List[List[Dict[str, Any]]],
    sampling_state: Dict[str, Any],
    sched_m: str,
) -> int:
    """Collect each inference batch and upload that .pt to the Hub."""
    store_cfg = cfg.get("store") or OmegaConf.create({})
    device = torch.device(str(store_cfg.get("device", "cuda")))
    scratch = Path(str(store_cfg.get("scratch_dir", "outputs/cst_staging")))
    scratch.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(str(store_cfg.get("hf_local_cache_dir", "outputs/cst_hf_cache")))
    capture_context = "train" if scope == "train" else "eval"
    delete_local = bool(store_cfg.get("delete_local_after_hf_push", True))
    model_path = _section_model_path(cfg, store_cfg)

    nbat = len(row_batches)
    logger.info(
        "store %s hiddens -> %s (%s): %d batches (upload each batch)",
        scope,
        repo_id,
        name_token,
        nbat,
    )
    if nbat == 0:
        raise RuntimeError(f"store.scope={scope}: no row batches to collect")

    manifest = fetch_manifest(
        repo_id,
        name_token,
        cache_dir=cache_dir,
        revision=str(store_cfg.get("hf_hiddens_revision", "main")),
        token=hf_token(store_cfg),
        create_if_missing=True,
    )
    model, tokenizer = load_model_for_collect(
        model_path,
        device=str(device),
        compile_model=bool(store_cfg.get("compile_model", False)),
    )
    ueh = _coerce_use_eos_harness(store_cfg.get("use_eos_harness", True))
    try:
        for bi, rows in enumerate(row_batches):
            logger.info("=== collect batch %d / %d ===", bi + 1, nbat)
            out_path, meta = collect_hidden_row_batch(
                model,
                tokenizer,
                model_path,
                scratch,
                rows=rows,
                batch_idx=bi,
                name_token=name_token,
                device=str(device),
                mode_key=sched_m,
                gen_length=int(store_cfg.get("gen_length", 256)),
                steps=int(store_cfg.get("steps", 256)),
                block_length=store_cfg.get("block_length"),
                top_k=int(store_cfg.get("top_k", 1)),
                max_prompt_length=int(store_cfg.get("max_prompt_length", 512)),
                capture_context=capture_context,
                use_eos_harness=ueh,
            )
            nbytes = Path(out_path).stat().st_size
            rec = upload_hidden_chunk(
                repo_id,
                name_token,
                out_path,
                manifest=manifest,
                sampling_fingerprint=dict(sampling_state),
                cache_dir=cache_dir,
                revision=str(store_cfg.get("hf_hiddens_revision", "main")),
                token=hf_token(store_cfg),
                private_repo=bool(store_cfg.get("hf_hiddens_private", False)),
                delete_local_after=delete_local,
                scope=scope,
            )
            logger.info(
                "uploaded %s (%.3f GB, gen_time=%.2fs)",
                rec.filename,
                nbytes / 1e9,
                float(meta.get("gen_time", 0.0)),
            )
    finally:
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("collect finished: %s on %s", name_token, repo_id)
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
    batch_idx: Optional[int] = None,
    data_next_epoch: int = 0,
    data_next_batch_idx: int = 0,
    num_epochs: int = 1,
    wandb_run_id_for_meta: Optional[str] = None,
    sampling_state: Optional[Dict[str, Any]] = None,
) -> int:
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
            row: Dict[str, Any] = {
                "train/chunk_loss": ch["loss"],
                "train/chunk_anchors": ch["n_anchors"],
                "train/chunk_index": float(i + 1),
                "train/chunks_in_phase": float(n_ch),
                "phase": ckpt_phase,
            }
            if epoch is not None:
                row["train/epoch"] = int(epoch)
            if batch_idx is not None:
                row["train/batch_idx"] = int(batch_idx)
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
            if batch_idx is not None:
                row2["train/batch_idx"] = int(batch_idx)
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
        data_next_segment=0,
        wandb_global_step=int(global_step),
        wandb_run_id=run_id,
        num_epochs=int(num_epochs),
    )
    if sampling_state is not None and all(
        k in sampling_state
        for k in (
            "train_seed",
            "train_samples_per_epoch",
            "train_inference_batch_size",
            "train_resample_each_epoch",
            "train_gsm8k_num_fewshot",
            "train_arc_csqa_num_fewshot",
        )
    ):
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
    store_cfg = cfg.get("store") or OmegaConf.create({})

    sp_cfg = store_cfg.get("samples")
    if _is_blank(sp_cfg):
        raise ValueError("store.samples must be set")
    n_sp = int(sp_cfg)
    if n_sp < 1:
        raise ValueError("store.samples must be >= 1")

    inference_bs = int(store_cfg.inference_batch_size)
    seed = int(store_cfg.get("seed", 0))
    gsm_fs = int(store_cfg.get("gsm8k_num_fewshot", 4))
    arc_fs = int(store_cfg.get("arc_csqa_num_fewshot", 5))

    model_path = _section_model_path(cfg, store_cfg)
    sched_m = str(store_cfg.get("scheduler_mode", "pd"))
    name_token = _hidden_name_token(store_cfg, model_path)
    hf_train_repo = resolve_train_repo_id(store_cfg)

    sampling_state = _sampling_state_from_store(store_cfg, n_samples=n_sp, inference_bs=inference_bs)

    row_batches = sample_uniform_batched(
        n_sp, inference_bs, seed=seed, gsm8k_num_fewshot=gsm_fs, arc_csqa_num_fewshot=arc_fs
    )

    return _store_hiddens(
        cfg,
        scope="train",
        name_token=name_token,
        repo_id=hf_train_repo,
        row_batches=row_batches,
        sampling_state=sampling_state,
        sched_m=sched_m,
    )


# ---------------------------------------------------------------------------
# mode=store store.scope=eval
# ---------------------------------------------------------------------------


def store_eval_hiddens(cfg: DictConfig) -> int:
    """Store eval hidden chunks on the Hub."""
    store = cfg.get("store") or OmegaConf.create({})

    task = store.get("task_name")
    if _is_blank(task):
        raise ValueError("store.scope=eval requires store.task_name")

    model_path = _section_model_path(cfg, store)
    sched_m = str(store.get("scheduler_mode", "pd"))
    name_token = _hidden_name_token(store, model_path, task_name=str(task))

    repo_id = resolve_eval_repo_id(store, store=store)
    inference_bs = int(store.inference_batch_size)
    seed = int(store.get("seed", 0))

    nfs = store.get("num_fewshot")
    num_fewshot = int(nfs) if not _is_blank(nfs) else eval_num_fewshot(str(task))

    sp_cfg = store.get("samples")
    limit = None
    if not _is_blank(sp_cfg):
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
        "eval_gen_length": int(store.get("gen_length", 256)),
        "eval_steps": int(store.get("steps", 256)),
    }

    return _store_hiddens(
        cfg,
        scope="eval",
        name_token=name_token,
        repo_id=repo_id,
        row_batches=row_batches,
        sampling_state=sampling_state,
        sched_m=sched_m,
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
    scratch = Path(str(tr.get("scratch_dir", tr.get("staging_dir", "outputs/cst_staging"))))
    scratch.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(str(tr.get("checkpoint_dir", "outputs/cst_ckpt")))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = bool(tr.get("wandb", cfg.get("wandb", True)))
    num_epochs = int(tr.get("num_epochs", 1))
    if num_epochs < 1:
        raise ValueError("train.num_epochs must be >= 1")

    cst_cfg = cfg.cst
    model_path = _section_model_path(cfg, tr)
    k, w = int(cst_cfg.k), int(cst_cfg.w)
    lam = float(tr.get("lambda_sparsity", 1e-3))
    hidden_token = _hidden_name_token(tr, model_path)
    ckpt_token = cst_ckpt_name_token(hidden_token, k, w, lam)
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
    optimizer = torch.optim.Adam(cst_bank.parameters(), lr=float(tr.get("learning_rate", 3e-4)))
    ramp = LambdaRamp(int(tr.get("ramp_total_steps", 10_000)))

    ckpt_meta: Dict[str, Any] = {}
    loaded_ckpt_path: Optional[str] = None
    rp_str = str(tr.get("resume_checkpoint") or "").strip()
    if rp_str and rp_str.lower() not in ("null", "none", "") and Path(rp_str).is_file():
        ckpt_meta = load_cst_checkpoint(rp_str, cst_bank, optimizer, ramp)
        loaded_ckpt_path = rp_str
        logger.info("resumed CST from %s", rp_str)
    elif bool(tr.get("resume_from_latest", False)):
        resume_p = resolve_train_resume_latest(ckpt_dir, ckpt_token)
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

    sampling_state: Optional[Dict[str, Any]] = None
    if loaded_ckpt_path and resume_data_from_checkpoint and bool(tr.get("resume_override_sampling", True)):
        sfp = parse_sampling_state_from_meta(ckpt_meta)
        if sfp is not None:
            sampling_state = sfp

    manifest = fetch_manifest(
        hf_train_repo,
        hidden_token,
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
    logger.info("train from Hub: %s (%s)", hf_train_repo, hidden_token)

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

    all_chunks = manifest.all_chunks_sorted()
    nbat = manifest.total_batch_count()
    if nbat == 0:
        raise FileNotFoundError(
            f"no hiddens in manifest for {hidden_token} @ {hf_train_repo}. "
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
        pending = [c for c in all_chunks if int(c.batch_idx) >= int(bi)]
        dl_dir = scratch / "hf_dl" / hidden_token
        for ch in pending:
            logger.info(
                "=== download pass %d batch %d / %d (%s) ===",
                cst_pass + 1,
                int(ch.batch_idx),
                nbat,
                ch.filename,
            )
            try:
                dest = download_hidden_chunk(
                    str(hf_train_repo),
                    hidden_token,
                    ch.filename,
                    dl_dir,
                    expected_bytes=ch.bytes,
                    revision=str(tr.get("hf_hiddens_revision", "main")),
                    token=hf_token(tr),
                )
            except FileNotFoundError as e:
                logger.error("Hub data missing — run mode=store first: %s", e)
                break

            if wb:
                wb.log(
                    {
                        "download/pass": cst_pass,
                        "download/batch_idx": int(ch.batch_idx),
                        "download/bytes": float(ch.bytes),
                    },
                    step=global_step,
                )
            global_step += 1

            next_bi = int(ch.batch_idx) + 1
            d_ne = int(cst_pass) + 1 if next_bi >= nbat else int(cst_pass)
            d_nb = 0 if next_bi >= nbat else int(next_bi)

            global_step = _train_cst_on_chunks_and_checkpoint(
                [str(dest)],
                ckpt_phase=int(cst_pass) * 100_000 + int(ch.batch_idx) + 1,
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
                name_token=ckpt_token,
                epoch=cst_pass,
                batch_idx=int(ch.batch_idx),
                data_next_epoch=d_ne,
                data_next_batch_idx=d_nb,
                num_epochs=int(num_epochs),
                wandb_run_id_for_meta=wandb_run_id_resumed if (not use_wandb or wb is None) else None,
                sampling_state=sampling_state,
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

    ev = _merge_eval_cfg(cfg)
    hidden_dir = str(ev.get("hidden_dir", _EVAL_DEFAULTS["hidden_dir"]))
    hidden_source = str(ev.get("hidden_source", "local")).strip().lower()

    if hidden_source in ("hf", "hf_hub", "hub"):
        store = cfg.get("store") or OmegaConf.create({})
        task = ev.get("task_name") or store.get("task_name")
        if _is_blank(task):
            raise ValueError("eval.hidden_source=hf_hub requires eval.task_name (or store.task_name)")
        model_path = _section_model_path(cfg, ev)
        name_token = _hidden_name_token(ev, model_path, task_name=str(task))
        repo_id = resolve_eval_repo_id(ev, store=store, eval_cfg=ev)
        cache_dir = str(ev.get("hf_local_cache_dir", "outputs/cst_hf_cache"))
        cache = Path(cache_dir) / "eval_dl" / name_token
        hidden_dir = str(
            download_all_chunks(
                repo_id,
                str(name_token),
                local_dir=cache,
                cache_dir=cache_dir,
                revision=str(ev.get("hf_hiddens_revision", "main")),
                token=hf_token(ev),
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
