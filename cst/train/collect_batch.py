"""Run generation and write pre-logit hidden .pt chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from cst import save_hidden_records
from cst.naming import cst_hidden_chunk_basename, cst_hidden_segment_prefix
from load_model import (
    infer_diffusion_token_ids,
    is_dream_model_dir,
    load_model_and_tokenizer,
    prepend_bos_for_dream,
)
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
    use_eos_harness: bool = True,
    capture_context: str = "train",
) -> Dict[str, Any]:
    """Run one batch and write a hidden chunk file."""
    mode, strategy = _mode_map(mode_key)
    if strategy is None:
        strategy = "confidence"
    bl = resolve_block_length(mode, gen_length, block_length)
    mask_id, eos_id, eot_id = infer_diffusion_token_ids(model_path)

    if is_dream_model_dir(model_path):
        texts = [prepend_bos_for_dream(tokenizer, t) for t in texts]

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
        use_eos_harness=use_eos_harness,
        model_path=model_path,
    )

    if not hasattr(model, "records"):
        raise RuntimeError("Model must use PreLogitsCaptureWrapper (capture_pre_logits=True).")
    rec = model.records
    orig_prompt_len = int(input_ids.shape[1])

    # === NEW TENSOR SLICING LOGIC ===
    B = len(texts)
    few_shot_lens = []

    # 1. Calculate few-shot lengths dynamically
    for text in texts:
        parts = text.split("\n\n")
        if len(parts) > 1:
            prefix_str = "\n\n".join(parts[:-1]) + "\n\n"
            fs_len = len(tokenizer.encode(prefix_str, add_special_tokens=False))
            few_shot_lens.append(fs_len)
        else:
            few_shot_lens.append(0)

    orig_seq_len = rec[0].shape[1] if len(rec) > 0 else orig_prompt_len
    gen_len = orig_seq_len - orig_prompt_len

    # 2. Find actual unpadded lengths using the existing attention_mask
    unpadded_lens = attention_mask.sum(dim=1).tolist()

    q_lens = []
    for b in range(B):
        # Safeguard: ensure fs_len never swallows the actual question
        fs = min(few_shot_lens[b], max(0, int(unpadded_lens[b]) - 1))
        few_shot_lens[b] = fs
        q_lens.append(int(unpadded_lens[b]) - fs)

    # 3. Calculate the new, highly-compressed prompt length
    new_prompt_len = max(q_lens) if q_lens else 0
    new_seq_len = new_prompt_len + gen_len

    # 4. Shift and pack the dense tensors
    if new_prompt_len < orig_prompt_len:
        for i in range(len(rec)):
            t = rec[i]
            if t.dim() == 3:
                # Initialize a tightly-packed tensor with zeros (which act as safe padding)
                new_t = torch.zeros((B, new_seq_len, t.shape[2]), dtype=t.dtype, device=t.device)
                for b in range(B):
                    fs = few_shot_lens[b]
                    q = q_lens[b]
                    # Copy the isolated target question
                    new_t[b, :q, :] = t[b, fs : fs + q, :]
                    # Copy the generation block, perfectly aligned to new_prompt_len
                    new_t[b, new_prompt_len : new_prompt_len + gen_len, :] = t[b, orig_prompt_len : orig_prompt_len + gen_len, :]
                rec[i] = new_t

        prompt_len = new_prompt_len
    else:
        prompt_len = orig_prompt_len
    # ================================

    save_hidden_records(rec, out_path, prompt_len=prompt_len)
    meta: Dict[str, Any] = {
        "steps_taken": int(steps_taken),
        "blocks": int(blocks),
        "gen_time": float(gen_time),
        "batch_size": int(input_ids.shape[0]),
        "prompt_len": prompt_len,
        "gen_length": int(gen_length),
        "mask_id": int(mask_id),
        "mode": mode,
        "strategy": strategy,
    }
    if staging_name_token:
        meta["cst_name_token"] = str(staging_name_token)
        meta["context"] = str(capture_context)
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
    """Load model with pre-logit capture enabled."""
    return load_model_and_tokenizer(
        model_path,
        device=device,
        compile_model=compile_model,
        capture_pre_logits=True,
    )


def _bytes_from_gb(gb: float) -> int:
    """Convert GB target to bytes."""
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
    staging_name_token: Optional[str] = None,
    segment: int = 0,
    capture_context: str = "train",
    use_eos_harness: bool = True,
) -> Tuple[List[str], int, Dict[str, Any]]:
    """Collect chunks until target_gb; return paths, next index, summary."""
    target_b = _bytes_from_gb(target_gb)
    staging_dir = Path(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    paths: List[str] = []
    total_bytes = 0
    total_gen_time = 0.0
    b = int(start_idx)
    n = len(row_batches)
    n_seg = 0
    while total_bytes < target_b and b < n:
        rows = row_batches[b]
        texts = [str(r.get("text", "")) for r in rows]
        if not any(texts):
            raise ValueError(f"row_batches[{b}] has no non-empty text")
        if not staging_name_token:
            raise ValueError("collect_hidden_until_size_gb_from_row_batches requires staging_name_token")
        out_name = cst_hidden_chunk_basename(staging_name_token, segment, b)
        trace_name = f"{cst_hidden_segment_prefix(staging_name_token, segment)}_{b:06d}"
        out_path = staging_dir / out_name
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
            trace_name=trace_name,
            staging_name_token=staging_name_token,
            use_eos_harness=use_eos_harness,
            capture_context=capture_context,
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
