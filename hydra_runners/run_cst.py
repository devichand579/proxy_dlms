#!/usr/bin/env python3
"""
CST entry point: store hiddens, train CST, or eval CST.

Config: config/cst.yaml. Examples:
  python hydra_runners/run_cst.py mode=store store.scope=train
  python hydra_runners/run_cst.py mode=store store.scope=eval store.task_name=all
  python hydra_runners/run_cst.py mode=train
  python hydra_runners/run_cst.py mode=eval eval.checkpoint=...
Or: ./scripts/cst.sh --mode store|train|eval
"""

from __future__ import annotations

import logging
import math
import sys
from collections import defaultdict
from datetime import datetime
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
from cst.hf_hiddens.manifest import (
    fingerprint_conflicts,
    fingerprint_progress,
    store_fingerprint_required_keys,
)
from cst.hf_hiddens.store import (
    HubStorageFullError,
    download_all_chunks,
    download_hidden_chunk,
    fetch_manifest,
    list_hub_chunks,
    push_manifest,
    upload_hidden_chunk,
)
from cst.pre_logits import DiskFullError
from cst.store_budget import (
    max_inference_batch_size,
    model_weight_bytes,
    read_model_config_sizes,
)
from cst.model.cst_bank import CSTBank
from cst.model.residency import CSTResidency, budget_report
from cst.train.adam_bf16 import AdamBF16
from cst.naming import (
    cst_ckpt_name_token,
    cst_hidden_name_token,
    model_label_from_path,
    resolve_train_best_in_dir,
    resolve_train_resume_latest,
)
from cst.train.collect_batch import (
    collect_hidden_row_batch,
    load_model_for_collect,
)
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
from cst.eval_data.compose import eval_num_fewshot, expand_eval_task_names, load_eval_rows
from cst.train_data.compose import sample_uniform
from load_model import is_dream_model_dir, load_tokenizer, prepend_bos_for_dream
from sampling_defaults import effective_gen_length_steps

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


def _cst_log_name_part(raw: Any) -> str:
    if _is_blank(raw):
        return ""
    s = str(raw).strip().replace("/", "_").replace(",", "_").replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")[:80]


def _cst_log_stem(cfg: DictConfig) -> str:
    mode = str(cfg.get("mode", "train")).strip().lower()
    if mode in ("store", "collect"):
        store = cfg.get("store") or OmegaConf.create({})
        scope = str(store.get("scope", "train")).strip().lower() or "train"
        parts = ["cst", "store", scope]
        task = _cst_log_name_part(store.get("task_name"))
        if task:
            parts.append(task)
        return "_".join(parts)
    if mode == "train":
        return "cst_train"
    if mode == "eval":
        parts = ["cst", "eval"]
        ev = cfg.get("eval") or OmegaConf.create({})
        task = _cst_log_name_part(ev.get("task_name"))
        if task:
            parts.append(task)
        return "_".join(parts)
    return f"cst_{mode or 'run'}"


def _setup_cst_file_logging(cfg: DictConfig) -> Path:
    """Write store/train/eval logger output to ``logs/cst_{mode}_*.log`` (same dir as eval.py)."""
    log_dir = _REPO_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{_cst_log_stem(cfg)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(fh)
    has_console = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root.handlers
    )
    if not has_console:
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        root.addHandler(sh)
    return log_file


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
    ignore_name_token_override: bool = False,
    gen_length: Optional[int] = None,
    steps: Optional[int] = None,
) -> str:
    override = section.get("name_token") if section is not None else None
    if not ignore_name_token_override and not _is_blank(override):
        return str(override).strip()
    sched = "pd"
    if section is not None and not _is_blank(section.get("scheduler_mode")):
        sched = str(section.get("scheduler_mode"))
    if gen_length is None:
        gen_length = 256 if section is None else int(section.get("gen_length", 256))
    if steps is None:
        steps = 256 if section is None else int(section.get("steps", 256))
    return cst_hidden_name_token(
        model_label_from_path(model_path),
        sched,
        int(gen_length),
        int(steps),
        task_name=task_name,
    )


def _optional_int(v: Any) -> Optional[int]:
    if _is_blank(v):
        return None
    return int(v)


def _store_yaml_gen_length_steps(store_cfg: Any) -> tuple[int, int]:
    return (
        int(store_cfg.get("gen_length", 512)),
        int(store_cfg.get("steps", 512)),
    )


def _row_task_name(row: Dict[str, Any]) -> str:
    src = str(row.get("source") or "").strip()
    if src:
        return src
    meta = row.get("meta") or {}
    return str(meta.get("task") or meta.get("lmeval_task") or "").strip()


def _batch_gen_length_steps(
    rows: List[Dict[str, Any]],
    default_gen: int,
    default_steps: int,
) -> tuple[int, int]:
    keys = {
        effective_gen_length_steps(_row_task_name(r), default_gen, default_steps)
        for r in rows
    }
    if len(keys) == 1:
        return keys.pop()
    return int(default_gen), int(default_steps)


# Same prompt-length buckets as eval.py: (0, 32], (32, 64], …, 1024+.
_PROMPT_LENGTH_BUCKETS = [32, 64, 128, 256, 384, 512, 640, 768, 896, 1024, float("inf")]


def _prompt_length_bucket(length: int) -> float:
    for bound in _PROMPT_LENGTH_BUCKETS:
        if length <= bound:
            return bound
    return _PROMPT_LENGTH_BUCKETS[-1]


def _prompt_token_lengths(
    texts: List[str],
    tokenizer,
    model_path: str,
) -> List[int]:
    """Token lengths generate() will see (Dream BOS included, no pad)."""
    if is_dream_model_dir(model_path):
        texts = [prepend_bos_for_dream(tokenizer, t) for t in texts]
    enc = tokenizer(texts, add_special_tokens=False, padding=False)
    return [len(ids) for ids in enc["input_ids"]]


def _bucket_row_batches(
    flat: List[Dict[str, Any]],
    batch_size: int,
    default_gen: int,
    default_steps: int,
    *,
    tokenizer,
    model_path: str,
) -> List[List[Dict[str, Any]]]:
    """Batch by gen_length, then eval.py prompt-length buckets (sorted within bucket)."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    gen_groups: Dict[tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in flat:
        key = effective_gen_length_steps(
            _row_task_name(row), default_gen, default_steps
        )
        gen_groups[key].append(row)
    out: List[List[Dict[str, Any]]] = []
    for gen_key in sorted(gen_groups.keys(), reverse=True):
        rows = gen_groups[gen_key]
        lengths = _prompt_token_lengths(
            [str(r.get("text", "")) for r in rows], tokenizer, model_path
        )
        by_bucket: Dict[float, List[int]] = defaultdict(list)
        for i, L in enumerate(lengths):
            by_bucket[_prompt_length_bucket(int(L))].append(i)
        for bkey in sorted(
            by_bucket.keys(), key=lambda x: x if x != float("inf") else 1e9
        ):
            indices = by_bucket[bkey]
            indices.sort(key=lambda idx: lengths[idx])
            for start in range(0, len(indices), batch_size):
                chunk_idx = indices[start : start + batch_size]
                out.append([rows[i] for i in chunk_idx])
    return out


def _log_store_batching(row_batches: List[List[Dict[str, Any]]], batch_size: int) -> None:
    if not row_batches:
        return
    sizes = [len(b) for b in row_batches]
    n = sum(sizes)
    mid = sorted(sizes)[len(sizes) // 2]
    logger.info(
        "store batching: %d rows -> %d batches (max B=%d, prompt-length buckets like eval.py; "
        "batch sizes min=%d median=%d max=%d)",
        n,
        len(row_batches),
        int(batch_size),
        min(sizes),
        mid,
        max(sizes),
    )


def _peek_hub_inference_batch_size(
    store_cfg: Any,
    repo_id: str,
    name_token: str,
    *,
    scope: str,
) -> Optional[int]:
    """Fingerprint B already on Hub, or None if this is a fresh folder."""
    if not bool(store_cfg.get("resume", True)):
        return None
    try:
        manifest = fetch_manifest(
            repo_id,
            name_token,
            cache_dir=Path(str(store_cfg.get("hf_local_cache_dir", "outputs/cst_hf_cache"))),
            revision=str(store_cfg.get("hf_hiddens_revision", "main")),
            token=hf_token(store_cfg),
            create_if_missing=True,
        )
    except Exception as e:
        logger.info("store auto-batch: could not read Hub fingerprint (%s)", e)
        return None
    fp = manifest.sampling_fingerprint
    if not fp:
        return None
    key = (
        "train_inference_batch_size"
        if str(scope).strip().lower() == "train"
        else "eval_inference_batch_size"
    )
    raw = fp.get(key)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _worst_store_seq_stats(
    rows: List[Dict[str, Any]],
    tokenizer,
    model_path: str,
    default_gen: int,
    default_steps: int,
) -> tuple[int, int, int]:
    """``(packed_L, full_L, T)`` worst case over this draw (few-shot drop can only shrink L)."""
    if not rows:
        g, s = int(default_gen), int(default_steps)
        return g, g, s + 1
    lengths = _prompt_token_lengths(
        [str(r.get("text", "")) for r in rows], tokenizer, model_path
    )
    packed = 1
    full = 1
    t_max = 1
    for i, row in enumerate(rows):
        g, s = effective_gen_length_steps(
            _row_task_name(row), default_gen, default_steps
        )
        p = int(lengths[i]) if i < len(lengths) else int(g)
        packed = max(packed, p + int(g))
        full = max(full, p + int(g))
        t_max = max(t_max, int(s) + 1)
    return packed, full, t_max


def _choose_store_batch_size(
    store_cfg: Any,
    *,
    yaml_bs: int,
    rows: List[Dict[str, Any]],
    tokenizer,
    model_path: str,
    default_gen: int,
    default_steps: int,
    repo_id: str,
    name_token: str,
    scope: str,
) -> int:
    """YAML cap, or lower from free disk/RAM/VRAM. Resume keeps the fingerprint B."""
    requested = max(int(yaml_bs), 1)
    raw_auto = store_cfg.get("auto_batch_size", True)
    if _is_blank(raw_auto):
        auto = True
    elif isinstance(raw_auto, str):
        auto = raw_auto.strip().lower() in ("true", "1", "yes")
    else:
        auto = bool(raw_auto)
    if not auto:
        return requested
    peeked = _peek_hub_inference_batch_size(
        store_cfg, repo_id, name_token, scope=scope
    )
    if peeked is not None:
        logger.info(
            "store resume: using fingerprint batch size %d (auto-batch not recomputed)",
            peeked,
        )
        return max(int(peeked), 1)
    packed, full, n_rec = _worst_store_seq_stats(
        rows, tokenizer, model_path, default_gen, default_steps
    )
    d_model, vocab, n_layers = read_model_config_sizes(model_path)
    mcg = store_cfg.get("max_chunk_gb", None)
    max_chunk_gb: Optional[float]
    if _is_blank(mcg):
        max_chunk_gb = None
    else:
        max_chunk_gb = float(mcg)
    dh = store_cfg.get("disk_headroom_gb", 20)
    ch = store_cfg.get("cpu_headroom_gb", 16)
    vh = store_cfg.get("vram_headroom_gb", 8)
    scratch = Path(str(store_cfg.get("scratch_dir", "outputs/cst_staging")))
    chosen, _limits = max_inference_batch_size(
        requested=requested,
        packed_seq_len=packed,
        full_seq_len=full,
        d_model=d_model,
        vocab_size=vocab,
        n_layers=n_layers,
        n_records=n_rec,
        scratch=scratch,
        stream_to_disk=bool(store_cfg.get("stream_to_disk", True)),
        disk_headroom_gb=20.0 if _is_blank(dh) else float(dh),
        cpu_headroom_gb=16.0 if _is_blank(ch) else float(ch),
        vram_headroom_gb=8.0 if _is_blank(vh) else float(vh),
        max_chunk_gb=max_chunk_gb,
        model_bytes=model_weight_bytes(model_path),
        device=str(store_cfg.get("device", "cuda")),
    )
    if chosen != requested:
        logger.info(
            "store auto-batch: lowered inference_batch_size %d -> %d "
            "(fingerprint will record %d; resume keeps that B)",
            requested,
            chosen,
            chosen,
        )
    return chosen


def _store_generation_fingerprint(
    store_cfg: Any,
    sched_m: str,
    *,
    gen_length: Optional[int] = None,
    steps: Optional[int] = None,
) -> Dict[str, Any]:
    yg, ys = _store_yaml_gen_length_steps(store_cfg)
    return {
        "scheduler": str(sched_m),
        "gen_length": int(yg if gen_length is None else gen_length),
        "steps": int(ys if steps is None else steps),
        "block_length": _optional_int(store_cfg.get("block_length")),
        "top_k": int(store_cfg.get("top_k", 1)),
        "use_eos_harness": _coerce_use_eos_harness(store_cfg.get("use_eos_harness", True)),
    }


def _sampling_state_from_store(
    store_cfg: Any,
    *,
    n_samples: int,
    inference_bs: int,
    sched_m: str,
) -> Dict[str, Any]:
    gen = _store_generation_fingerprint(store_cfg, sched_m)
    return {
        "train_seed": int(store_cfg.get("seed", 0)),
        "train_samples_per_epoch": int(n_samples),
        "train_inference_batch_size": int(inference_bs),
        "train_resample_each_epoch": False,
        "train_gsm8k_num_fewshot": eval_num_fewshot("gsm8k"),
        "train_arc_csqa_num_fewshot": eval_num_fewshot("arc_challenge_chat"),
        "train_scheduler": gen["scheduler"],
        "train_gen_length": gen["gen_length"],
        "train_steps": gen["steps"],
        "train_block_length": gen["block_length"],
        "train_top_k": gen["top_k"],
        "train_use_eos_harness": gen["use_eos_harness"],
        "train_bucket_by_gen_length": True,
        "train_bucket_by_prompt_length": True,
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
    """Collect each inference batch and upload that .pt to the Hub.

    With ``store.resume=true`` (default), batches already on Hub are skipped
    when the sampling fingerprint matches this run (same seed, sample count,
    batch size, and generation settings). Skip uses the manifest *and* chunk
    files under ``{name_token}/chunks/``, so a crash after the .pt commit but
    before the manifest write does not re-upload that batch.
    """
    store_cfg = cfg.get("store") or OmegaConf.create({})
    device = torch.device(str(store_cfg.get("device", "cuda")))
    scratch = Path(str(store_cfg.get("scratch_dir", "outputs/cst_staging")))
    scratch.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(str(store_cfg.get("hf_local_cache_dir", "outputs/cst_hf_cache")))
    capture_context = "train" if scope == "train" else "eval"
    delete_local = bool(store_cfg.get("delete_local_after_hf_push", True))
    resume = bool(store_cfg.get("resume", True))
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

    hub_revision = str(store_cfg.get("hf_hiddens_revision", "main"))
    hub_tok = hf_token(store_cfg)
    manifest = fetch_manifest(
        repo_id,
        name_token,
        cache_dir=cache_dir,
        revision=hub_revision,
        token=hub_tok,
        create_if_missing=True,
    )
    stored_idx = {int(c.batch_idx) for c in manifest.all_chunks_sorted()}
    hub_chunks = (
        list_hub_chunks(repo_id, name_token, revision=hub_revision, token=hub_tok)
        if resume
        else []
    )
    hub_idx = {int(c.batch_idx) for c in hub_chunks}
    present_idx = stored_idx | hub_idx
    skip: set[int] = set()
    total_question_tokens = 0
    total_gen_tokens = 0
    total_cst_train_tokens = 0
    total_samples = 0
    if present_idx and resume:
        if stored_idx or manifest.sampling_fingerprint:
            conflicts = fingerprint_conflicts(
                manifest.sampling_fingerprint,
                sampling_state,
                required_keys=store_fingerprint_required_keys(scope),
            )
            if conflicts:
                raise RuntimeError(
                    f"store resume refused for {repo_id}/{name_token}: Hub data is not "
                    f"the same draw. Re-run with the original store.seed / store.samples / "
                    f"store.inference_batch_size (and generation settings), use a new "
                    f"store.name_token, or pass store.resume=false to overwrite. "
                    + "; ".join(conflicts)
                )
        else:
            logger.warning(
                "store resume: %d Hub chunk file(s) not in manifest (likely crash after "
                ".pt upload, before manifest); skipping re-upload",
                len(hub_idx),
            )
        skip = {i for i in present_idx if 0 <= i < nbat}
        hub_by_idx = {int(c.batch_idx): c for c in hub_chunks}
        orphans = [hub_by_idx[i] for i in sorted(skip) if i not in stored_idx and i in hub_by_idx]
        if orphans:
            for rec in orphans:
                manifest.upsert_chunk(rec)
            if manifest.sampling_fingerprint is None:
                manifest.sampling_fingerprint = dict(sampling_state)
            try:
                push_manifest(
                    manifest,
                    repo_id,
                    revision=hub_revision,
                    token=hub_tok,
                    cache_dir=cache_dir,
                )
                logger.info(
                    "store resume: recorded %d Hub chunk(s) into manifest without re-upload: %s",
                    len(orphans),
                    ", ".join(r.filename for r in orphans),
                )
            except Exception as e:
                logger.warning(
                    "store resume: could not write recovered chunks into manifest (%s); "
                    "will still skip re-upload this run",
                    e,
                )
            stored_idx = {int(c.batch_idx) for c in manifest.all_chunks_sorted()}
        prog = fingerprint_progress(manifest.sampling_fingerprint)
        total_question_tokens = prog["cst_question_tokens"]
        total_gen_tokens = prog["cst_gen_tokens"]
        total_cst_train_tokens = prog["cst_train_tokens"]
        total_samples = prog["cst_train_samples"]
        sampling_state["cst_question_tokens"] = int(total_question_tokens)
        sampling_state["cst_gen_tokens"] = int(total_gen_tokens)
        sampling_state["cst_train_tokens"] = int(total_cst_train_tokens)
        sampling_state["cst_train_samples"] = int(total_samples)
        extra = sorted(i for i in present_idx if i >= nbat)
        if extra:
            logger.warning(
                "store resume: Hub has %d chunk(s) with batch_idx >= %d; leaving them in the manifest",
                len(extra),
                nbat,
            )
        logger.info(
            "store resume: skipping %d / %d batches already on Hub; collecting %d remaining",
            len(skip),
            nbat,
            nbat - len(skip),
        )
    elif stored_idx and not resume:
        logger.warning(
            "store.resume=false: Hub already has %d chunks for %s; re-collecting all %d batches",
            len(stored_idx),
            name_token,
            nbat,
        )

    remaining = [i for i in range(nbat) if i not in skip]
    if not remaining:
        logger.info(
            "store resume: all %d batches already on Hub (%s / %s); nothing to collect | "
            "samples=%d cst_train_tokens=%d (question=%d + gen=%d)",
            nbat,
            name_token,
            repo_id,
            total_samples,
            total_cst_train_tokens,
            total_question_tokens,
            total_gen_tokens,
        )
        return 0

    model, tokenizer = load_model_for_collect(
        model_path,
        device=str(device),
        compile_model=bool(store_cfg.get("compile_model", False)),
    )
    stream_to_disk = bool(store_cfg.get("stream_to_disk", True))
    dh = store_cfg.get("disk_headroom_gb", 40)
    disk_headroom = 40.0 if _is_blank(dh) else float(dh)

    ueh = _coerce_use_eos_harness(store_cfg.get("use_eos_harness", True))
    try:
        for bi, rows in enumerate(row_batches):
            if bi in skip:
                continue
            logger.info("=== collect batch %d / %d ===", bi + 1, nbat)
            b_gen, b_steps = _batch_gen_length_steps(
                rows,
                int(store_cfg.get("gen_length", 512)),
                int(store_cfg.get("steps", 512)),
            )
            if (b_gen, b_steps) != (
                int(store_cfg.get("gen_length", 512)),
                int(store_cfg.get("steps", 512)),
            ):
                logger.info(
                    "batch %d: gen_length=%d steps=%d (ARC/CSQA override)",
                    bi + 1,
                    b_gen,
                    b_steps,
                )
            try:
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
                    gen_length=int(b_gen),
                    steps=int(b_steps),
                    block_length=store_cfg.get("block_length"),
                    top_k=int(store_cfg.get("top_k", 1)),
                    capture_context=capture_context,
                    use_eos_harness=ueh,
                    stream_to_disk=stream_to_disk,
                    disk_headroom_gb=disk_headroom,
                    scratch_dir=scratch,
                    disk_label=f"batch {bi + 1}/{nbat}",
                )
            except DiskFullError:
                raise
            b_q = int(meta.get("n_question_tokens", 0))
            b_g = int(meta.get("n_gen_tokens", 0))
            b_t = int(meta.get("cst_train_tokens", b_q + b_g))
            b_n = int(meta.get("batch_size", len(rows)))
            nbytes = Path(out_path).stat().st_size
            try:
                rec = upload_hidden_chunk(
                    repo_id,
                    name_token,
                    out_path,
                    manifest=manifest,
                    sampling_fingerprint=dict(
                        sampling_state,
                        cst_question_tokens=int(total_question_tokens + b_q),
                        cst_gen_tokens=int(total_gen_tokens + b_g),
                        cst_train_tokens=int(total_cst_train_tokens + b_t),
                        cst_train_samples=int(total_samples + b_n),
                    ),
                    cache_dir=cache_dir,
                    revision=str(store_cfg.get("hf_hiddens_revision", "main")),
                    token=hf_token(store_cfg),
                    private_repo=bool(store_cfg.get("hf_hiddens_private", False)),
                    delete_local_after=delete_local,
                    scope=scope,
                    skip_if_exists=resume,
                )
            except (HubStorageFullError, DiskFullError) as e:
                try:
                    Path(out_path).unlink(missing_ok=True)
                except OSError:
                    pass
                n_on_hub = len(manifest.all_chunks_sorted())
                kind = "Hub storage" if isinstance(e, HubStorageFullError) else "local disk"
                logger.warning(
                    "%s full while uploading batch %d / %d for %s; "
                    "finishing store early. %d batch(es) already on Hub. "
                    "Free space and re-run the same command to resume. (%s)",
                    kind,
                    bi + 1,
                    nbat,
                    name_token,
                    n_on_hub,
                    e,
                )
                raise
            total_question_tokens += b_q
            total_gen_tokens += b_g
            total_cst_train_tokens += b_t
            total_samples += b_n
            sampling_state["cst_question_tokens"] = int(total_question_tokens)
            sampling_state["cst_gen_tokens"] = int(total_gen_tokens)
            sampling_state["cst_train_tokens"] = int(total_cst_train_tokens)
            sampling_state["cst_train_samples"] = int(total_samples)
            logger.info(
                "uploaded %s (%.3f GB, gen_time=%.2fs, cst_tokens=%d question=%d gen=%d, total_cst_tokens=%d)",
                rec.filename,
                nbytes / 1e9,
                float(meta.get("gen_time", 0.0)),
                b_t,
                b_q,
                b_g,
                total_cst_train_tokens,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(
        "collect finished: %s on %s | samples=%d cst_train_tokens=%d (question=%d + gen=%d)",
        name_token,
        repo_id,
        total_samples,
        total_cst_train_tokens,
        total_question_tokens,
        total_gen_tokens,
    )
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
    store_cfg = cfg.get("store") or OmegaConf.create({})

    sp_cfg = store_cfg.get("samples")
    if _is_blank(sp_cfg):
        raise ValueError("store.samples must be set")
    n_sp = int(sp_cfg)
    if n_sp < 1:
        raise ValueError("store.samples must be >= 1")

    inference_bs = int(store_cfg.inference_batch_size)
    seed = int(store_cfg.get("seed", 0))

    model_path = _section_model_path(cfg, store_cfg)
    sched_m = str(store_cfg.get("scheduler_mode", "pd"))
    yaml_g, yaml_s = _store_yaml_gen_length_steps(store_cfg)
    name_token = _hidden_name_token(
        store_cfg, model_path, gen_length=yaml_g, steps=yaml_s
    )
    hf_train_repo = resolve_train_repo_id(store_cfg)

    flat = sample_uniform(n_sp, seed=seed)
    tokenizer = load_tokenizer(model_path)
    inference_bs = _choose_store_batch_size(
        store_cfg,
        yaml_bs=inference_bs,
        rows=flat,
        tokenizer=tokenizer,
        model_path=model_path,
        default_gen=yaml_g,
        default_steps=yaml_s,
        repo_id=hf_train_repo,
        name_token=name_token,
        scope="train",
    )
    sampling_state = _sampling_state_from_store(
        store_cfg, n_samples=n_sp, inference_bs=inference_bs, sched_m=sched_m
    )
    row_batches = _bucket_row_batches(
        flat,
        inference_bs,
        yaml_g,
        yaml_s,
        tokenizer=tokenizer,
        model_path=model_path,
    )
    _log_store_batching(row_batches, inference_bs)

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
    """Store eval hidden chunks on the Hub.

    ``store.task_name=all`` (or a comma-separated list) stores each task into
    its own folder under the same ``hiddens-eval`` repo. Resume is per folder.
    """
    store = cfg.get("store") or OmegaConf.create({})

    task = store.get("task_name")
    if _is_blank(task):
        raise ValueError("store.scope=eval requires store.task_name (or store.task_name=all)")

    tasks = expand_eval_task_names(task)
    multi = len(tasks) > 1
    if multi and not _is_blank(store.get("name_token")):
        logger.warning(
            "store.name_token=%s ignored for multi-task eval store; each task uses "
            "the auto folder {model}_{scheduler}_{gen_length}_{steps}_task_{task}",
            store.get("name_token"),
        )
    logger.info("eval store tasks (%d): %s", len(tasks), ", ".join(tasks))
    rc = 0
    for i, t in enumerate(tasks, start=1):
        logger.info("=== eval store task %d / %d: %s ===", i, len(tasks), t)
        rc = _store_eval_one_task(
            cfg, task_name=t, ignore_name_token_override=multi
        )
        if rc:
            return rc
    return 0


def _store_eval_one_task(
    cfg: DictConfig,
    *,
    task_name: str,
    ignore_name_token_override: bool = False,
) -> int:
    """Store one lm-eval task's hidden chunks on the Hub."""
    store = cfg.get("store") or OmegaConf.create({})

    model_path = _section_model_path(cfg, store)
    sched_m = str(store.get("scheduler_mode", "pd"))
    yaml_g, yaml_s = _store_yaml_gen_length_steps(store)
    gen_length, steps = effective_gen_length_steps(str(task_name), yaml_g, yaml_s)
    name_token = _hidden_name_token(
        store,
        model_path,
        task_name=str(task_name),
        ignore_name_token_override=ignore_name_token_override,
        gen_length=gen_length,
        steps=steps,
    )

    repo_id = resolve_eval_repo_id(store, store=store)
    inference_bs = int(store.inference_batch_size)
    seed = int(store.get("seed", 0))

    nfs = store.get("num_fewshot")
    num_fewshot = int(nfs) if not _is_blank(nfs) else eval_num_fewshot(str(task_name))

    sp_cfg = store.get("samples")
    limit = None
    if not _is_blank(sp_cfg):
        limit = int(sp_cfg)
        if limit < 1:
            raise ValueError("store.samples must be >= 1 when set")

    rows = load_eval_rows(
        str(task_name),
        limit=limit,
        num_fewshot=num_fewshot,
        seed=seed,
    )
    tokenizer = load_tokenizer(model_path)
    inference_bs = _choose_store_batch_size(
        store,
        yaml_bs=inference_bs,
        rows=rows,
        tokenizer=tokenizer,
        model_path=model_path,
        default_gen=gen_length,
        default_steps=steps,
        repo_id=repo_id,
        name_token=name_token,
        scope="eval",
    )
    row_batches = _bucket_row_batches(
        rows,
        inference_bs,
        yaml_g,
        yaml_s,
        tokenizer=tokenizer,
        model_path=model_path,
    )
    _log_store_batching(row_batches, inference_bs)

    gen = _store_generation_fingerprint(
        store, sched_m, gen_length=gen_length, steps=steps
    )
    if (gen_length, steps) != (yaml_g, yaml_s):
        logger.info(
            "eval task %s: gen_length=%d steps=%d (overrode store %d/%d)",
            task_name,
            gen_length,
            steps,
            yaml_g,
            yaml_s,
        )
    sampling_state = {
        "eval_task": str(task_name),
        "eval_num_fewshot": int(num_fewshot),
        "eval_limit": limit,
        "eval_inference_batch_size": int(inference_bs),
        "eval_seed": int(seed),
        "eval_scheduler": gen["scheduler"],
        "eval_gen_length": gen["gen_length"],
        "eval_steps": gen["steps"],
        "eval_block_length": gen["block_length"],
        "eval_top_k": gen["top_k"],
        "eval_use_eos_harness": gen["use_eos_harness"],
        "eval_bucket_by_prompt_length": True,
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
    try:
        if scope == "train":
            return store_train_hiddens(cfg)
        if scope == "eval":
            return store_eval_hiddens(cfg)
    except HubStorageFullError as e:
        logger.warning(
            "store finished early (Hub storage full). Batches already on Hub are kept; "
            "re-run the same command after freeing space to resume. (%s)",
            e,
        )
        return 0
    except DiskFullError as e:
        logger.warning(
            "store finished early (local disk full). Batches already on Hub are kept; "
            "free disk (staging + Hugging Face cache) and re-run the same command to resume. (%s)",
            e,
        )
        return 0
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
    # bf16 moments halve optimizer state (172 GB -> 86 GB for a 512-step k=4096
    # bank), which is what makes the full bank fit across GPU + host RAM.
    optimizer = AdamBF16(cst_bank.parameters(), lr=float(tr.get("learning_rate", 3e-4)))
    ramp = LambdaRamp(int(tr.get("ramp_total_steps", 10_000)))

    ckpt_meta: Dict[str, Any] = {}
    loaded_ckpt_path: Optional[str] = None
    rp_str = str(tr.get("resume_checkpoint") or "").strip()
    rp = Path(rp_str) if rp_str and rp_str.lower() not in ("null", "none", "") else None
    if rp is not None and (rp.is_file() or rp.is_dir()):
        ckpt_meta = load_cst_checkpoint(str(rp), cst_bank, optimizer, ramp)
        loaded_ckpt_path = str(rp)
        logger.info("resumed CST from %s", rp)
    elif bool(tr.get("resume_from_latest", False)):
        resume_p = resolve_train_resume_latest(ckpt_dir, ckpt_token)
        if resume_p is not None and (resume_p.is_file() or resume_p.is_dir()):
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
        n_tok = manifest_fp.get("cst_train_tokens")
        n_q = manifest_fp.get("cst_question_tokens")
        n_g = manifest_fp.get("cst_gen_tokens")
        n_s = manifest_fp.get("cst_train_samples")
        if n_tok is not None:
            logger.info(
                "CST train tokens (question + gen_length, no few-shot/pad): %s "
                "(samples=%s question=%s gen=%s)",
                f"{int(n_tok):,}",
                f"{int(n_s):,}" if n_s is not None else "?",
                f"{int(n_q):,}" if n_q is not None else "?",
                f"{int(n_g):,}" if n_g is not None else "?",
            )

    n_cst_params = sum(p.numel() for p in cst_bank.parameters())
    logger.info("CST parameters: total=%s", f"{n_cst_params:,}")
    logger.info("train from Hub: %s (%s)", hf_train_repo, hidden_token)

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
            if sampling_state and sampling_state.get("cst_train_tokens") is not None:
                wbd = {
                    "data/cst_train_tokens": float(sampling_state["cst_train_tokens"]),
                }
                if sampling_state.get("cst_question_tokens") is not None:
                    wbd["data/cst_question_tokens"] = float(sampling_state["cst_question_tokens"])
                if sampling_state.get("cst_gen_tokens") is not None:
                    wbd["data/cst_gen_tokens"] = float(sampling_state["cst_gen_tokens"])
                if sampling_state.get("cst_train_samples") is not None:
                    wbd["data/cst_train_samples"] = float(sampling_state["cst_train_samples"])
                wb.log(wbd, step=global_step)

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

    ev = _merge_eval_cfg(cfg)
    hidden_dir = str(ev.get("hidden_dir", _EVAL_DEFAULTS["hidden_dir"]))
    hidden_source = str(ev.get("hidden_source", "local")).strip().lower()

    if hidden_source in ("hf", "hf_hub", "hub"):
        store = cfg.get("store") or OmegaConf.create({})
        task = ev.get("task_name") or store.get("task_name")
        if _is_blank(task):
            raise ValueError("eval.hidden_source=hf_hub requires eval.task_name (or store.task_name)")
        model_path = _section_model_path(cfg, ev)
        yaml_g, yaml_s = _store_yaml_gen_length_steps(ev)
        if store is not None and not _is_blank(store.get("gen_length")):
            yaml_g, yaml_s = _store_yaml_gen_length_steps(store)
        gen_length, steps = effective_gen_length_steps(str(task), yaml_g, yaml_s)
        name_token = _hidden_name_token(
            ev,
            model_path,
            task_name=str(task),
            gen_length=gen_length,
            steps=steps,
        )
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
    if not ckp.is_file() and not ckp.is_dir() and ckp.parent.is_dir():
        alt = resolve_train_best_in_dir(ckp.parent)
        if alt is not None:
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
    yaml_out = OmegaConf.to_yaml(out)
    print(yaml_out)
    logger.info("eval results:\n%s", yaml_out.rstrip())

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
    log_file = _setup_cst_file_logging(cfg)
    logger.info("logging to %s", log_file)
    sys.exit(int(dispatch(cfg)))


if __name__ == "__main__":
    main()
