"""LM generation → save pre-logit tensors to ``.pt`` files (single batch or until a size cap)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from cst import save_hidden_records
from load_model import infer_diffusion_token_ids, load_model_and_tokenizer
from schedulers.generate import generate


def _mode_map(short: str) -> tuple[str, Optional[str]]:
    m = {
        "pd": ("pure_diffusion", "confidence"),
        "bd": ("block_diffusion", "confidence"),
        "eb": ("eb_sampler", "confidence"),
        "fdm": ("fdm", None),
        "fhs": ("fhs", None),
        "top_k_c": ("top_k", "confidence"),
        "top_k_m": ("top_k", "top_margin"),
        "dus": ("dus", "confidence"),
    }
    short = short.strip().lower()
    if short in m:
        return m[short]
    return short, "confidence"


def resolve_block_length(mode: str, gen_length: int, block_length: Optional[int]) -> Optional[int]:
    if block_length is not None:
        return int(block_length)
    if mode == "fhs":
        return None
    if mode == "block_diffusion":
        return 32
    if mode == "fdm":
        return 64
    if mode == "dus":
        return 16
    return gen_length


@torch.no_grad()
def collect_hidden_batch_loaded(
    model: torch.nn.Module,
    tokenizer,
    model_path: str,
    texts: List[str],
    out_path: str,
    *,
    device: str = "cuda",
    mode_key: str = "pd",
    gen_length: int = 256,
    steps: int = 256,
    block_length: Optional[int] = None,
    top_k: int = 1,
    max_prompt_length: int = 512,
    trace_name: Optional[str] = None,
    staging_name_token: Optional[str] = None,
) -> Dict[str, Any]:
    """One generate pass; writes ``save_hidden_records`` payload to ``out_path``."""
    mode, strategy = _mode_map(mode_key)
    if strategy is None:
        strategy = "confidence"
    bl = resolve_block_length(mode, gen_length, block_length)
    mask_id, eos_id, eot_id = infer_diffusion_token_ids(model_path)

    enc = tokenizer(
        texts,
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    _, steps_taken, blocks, gen_time = generate(
        model,
        input_ids,
        attention_mask,
        steps=steps,
        gen_length=gen_length,
        block_length=bl,
        mode=mode,
        strategy=strategy,
        top_k=top_k,
        trace_name=trace_name,
        upload_to_wandb=False,
        mask_id=mask_id,
        eos_id=eos_id,
        eot_id=eot_id,
    )

    if not hasattr(model, "records"):
        raise RuntimeError("Model must use PreLogitsCaptureWrapper (capture_pre_logits=True).")
    rec = model.records
    um = getattr(model, "unmasked_records", None)
    dm = getattr(model, "diffusion_masked_records", None)
    meta: Dict[str, Any] = {
        "steps_taken": int(steps_taken),
        "blocks": int(blocks),
        "gen_time": float(gen_time),
        "batch_size": int(input_ids.shape[0]),
        "prompt_len": int(input_ids.shape[1]),
        "gen_length": int(gen_length),
        "mask_id": int(mask_id),
        "mode": mode,
        "strategy": strategy,
    }
    if staging_name_token:
        meta["cst_name_token"] = str(staging_name_token)
        meta["context"] = "train"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if (
        um is not None
        and dm is not None
        and len(um) == len(rec) == len(dm)
    ):
        save_hidden_records(
            rec,
            out_path,
            extra=meta,
            token_unmasked=um,
            token_diffusion_masked=dm,
        )
    else:
        save_hidden_records(rec, out_path, extra=meta)
    if hasattr(model, "clear_records"):
        model.clear_records()
    return meta


@torch.no_grad()
def load_model_for_collect(
    model_path: str,
    *,
    device: str = "cuda",
    compile_model: bool = False,
) -> Tuple[torch.nn.Module, Any]:
    """``load_model_and_tokenizer`` with ``capture_pre_logits=True``."""
    return load_model_and_tokenizer(
        model_path,
        device=device,
        compile_model=compile_model,
        capture_pre_logits=True,
    )


def _bytes_from_gb(gb: float) -> int:
    """``gb`` × 1e9 bytes."""
    return max(0, int(float(gb) * 1e9))


@torch.no_grad()
def collect_hidden_until_size_gb_from_row_batches(
    model: torch.nn.Module,
    tokenizer,
    model_path: str,
    staging_dir: str | Path,
    *,
    row_batches: Sequence[Sequence[Dict[str, Any]]],
    start_idx: int,
    target_gb: float,
    device: str = "cuda",
    mode_key: str = "pd",
    gen_length: int = 256,
    steps: int = 256,
    block_length: Optional[int] = None,
    top_k: int = 1,
    max_prompt_length: int = 512,
    trace_name_prefix: str = "cst_collect",
    max_batches: int = 50_000,
    staging_name_token: Optional[str] = None,
) -> Tuple[List[str], int, Dict[str, Any]]:
    """Write batches from ``row_batches[start_idx:]`` to disk as ``.pt`` files until reaching ``target_gb`` or out of batches.
    Returns (paths, next_idx, summary).
    """
    target_b = _bytes_from_gb(target_gb)
    staging_dir = Path(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    paths: List[str] = []
    total_bytes = 0
    total_gen_time = 0.0
    b = int(start_idx)
    n = len(row_batches)
    n_seg = 0
    while total_bytes < target_b and b < n and n_seg < max_batches:
        rows = row_batches[b]
        texts = [str(r.get("text", "")) for r in rows]
        if not any(texts):
            raise ValueError(f"row_batches[{b}] has no non-empty text")
        out_path = staging_dir / f"{trace_name_prefix}_{b:06d}.pt"
        meta = collect_hidden_batch_loaded(
            model,
            tokenizer,
            model_path,
            texts,
            str(out_path),
            device=device,
            mode_key=mode_key,
            gen_length=gen_length,
            steps=steps,
            block_length=block_length,
            top_k=top_k,
            max_prompt_length=max_prompt_length,
            trace_name=f"{trace_name_prefix}_{b}",
            staging_name_token=staging_name_token,
        )
        paths.append(str(out_path))
        total_bytes += out_path.stat().st_size
        total_gen_time += float(meta.get("gen_time", 0.0))
        b += 1
        n_seg += 1

    summary: Dict[str, Any] = {
        "target_gb": float(target_gb),
        "written_bytes": total_bytes,
        "written_gb": total_bytes / 1e9,
        "n_batches": n_seg,
        "next_idx": b,
        "total_gen_time": total_gen_time,
    }
    return paths, b, summary
