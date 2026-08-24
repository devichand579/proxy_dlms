"""Run generation and write pre-logit hidden .pt chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from cst import save_hidden_records
from cst.naming import cst_hidden_chunk_basename
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

    # Same as eval.py: left-pad so generate() sees pads on the left of the prompt.
    if getattr(tokenizer, "padding_side", None) != "left":
        tokenizer.padding_side = "left"

    enc = tokenizer(
        texts,
        add_special_tokens=False,
        padding=True,
        truncation=False,
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

    # 2. Find actual unpadded lengths using the existing attention_mask.
    # Left pad: [PAD × left][few-shot][question][GEN]; content starts at left_pad.
    unpadded_lens = attention_mask.sum(dim=1).tolist()
    left_pads = [orig_prompt_len - int(u) for u in unpadded_lens]

    q_lens = []
    for b in range(B):
        # Safeguard: ensure fs_len never swallows the actual question
        fs = min(few_shot_lens[b], max(0, int(unpadded_lens[b]) - 1))
        few_shot_lens[b] = fs
        q_lens.append(int(unpadded_lens[b]) - fs)

    # 3. Packed layout matches generate: left-pad questions, gen on the right.
    #    [zeros][question][GEN]; prompt_len = max question length in the batch.
    new_prompt_len = max(q_lens) if q_lens else 0
    new_seq_len = new_prompt_len + gen_len
    prompt_len = new_prompt_len if new_prompt_len > 0 else orig_prompt_len

    if rec and new_prompt_len > 0:
        for i in range(len(rec)):
            t = rec[i]
            if t.dim() != 3:
                continue
            new_t = torch.zeros((B, new_seq_len, t.shape[2]), dtype=t.dtype, device=t.device)
            for b in range(B):
                fs = few_shot_lens[b]
                q = q_lens[b]
                src = left_pads[b] + fs
                dst = new_prompt_len - q
                new_t[b, dst : dst + q, :] = t[b, src : src + q, :]
                new_t[b, new_prompt_len : new_prompt_len + gen_len, :] = t[
                    b, orig_prompt_len : orig_prompt_len + gen_len, :
                ]
            rec[i] = new_t

    n_question_tokens = int(sum(q_lens))
    n_gen_tokens = int(B) * int(gen_len)
    cst_train_tokens = n_question_tokens + n_gen_tokens
    save_hidden_records(
        rec,
        out_path,
        prompt_len=prompt_len,
        extra_meta={
            "n_samples": int(B),
            "n_question_tokens": n_question_tokens,
            "n_gen_tokens": n_gen_tokens,
            "cst_train_tokens": cst_train_tokens,
        },
    )
    meta: Dict[str, Any] = {
        "steps_taken": int(steps_taken),
        "blocks": int(blocks),
        "gen_time": float(gen_time),
        "batch_size": int(input_ids.shape[0]),
        "prompt_len": prompt_len,
        "gen_length": int(gen_length),
        "n_question_tokens": n_question_tokens,
        "n_gen_tokens": n_gen_tokens,
        "cst_train_tokens": cst_train_tokens,
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
    model, tokenizer = load_model_and_tokenizer(
        model_path,
        device=device,
        compile_model=compile_model,
        capture_pre_logits=True,
    )
    if getattr(tokenizer, "padding_side", None) != "left":
        tokenizer.padding_side = "left"
    return model, tokenizer


@torch.no_grad()
def collect_hidden_row_batch(
    model: torch.nn.Module,
    tokenizer,
    model_path: str,
    out_dir: str | Path,
    *,
    rows: List[Dict[str, Any]],
    batch_idx: int,
    name_token: str,
    device: str = "cuda",
    mode_key: str = "pd",
    gen_length: int = 256,
    steps: int = 256,
    block_length: Optional[int] = None,
    top_k: int = 1,
    capture_context: str = "train",
    use_eos_harness: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """Collect one inference batch and write ``{name_token}_{batch:06d}.pt``."""
    texts = [str(r.get("text", "")) for r in rows]
    if not any(texts):
        raise ValueError(f"row_batches[{batch_idx}] has no non-empty text")
    if not name_token:
        raise ValueError("collect_hidden_row_batch requires name_token")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = cst_hidden_chunk_basename(name_token, batch_idx)
    out_path = out_dir / out_name
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
        trace_name=out_path.stem,
        staging_name_token=name_token,
        use_eos_harness=use_eos_harness,
        capture_context=capture_context,
    )
    return str(out_path), meta
