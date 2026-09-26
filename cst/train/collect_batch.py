"""Run generation and write pre-logit hidden .pt chunks."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from cst.naming import cst_hidden_chunk_basename
from cst.pre_logits import DiskFullError, HiddenStepSink, save_hidden_records
from load_model import (
    infer_diffusion_token_ids,
    is_dream_model_dir,
    load_model_and_tokenizer,
    prepend_bos_for_dream,
)
from schedulers.generate import generate

logger = logging.getLogger("cst.collect")


def _mode_map(short: str) -> tuple[str, Optional[str]]:
    m = {
        "pd": ("pure_diffusion", "confidence"),
        "pd_r": ("pure_diffusion", "random"),
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


def few_shot_prefix_len(text: str, tokenizer) -> int:
    """Token length of everything before the last ``\\n\\n``-separated block (0 if none)."""
    parts = text.split("\n\n")
    if len(parts) <= 1:
        return 0
    prefix_str = "\n\n".join(parts[:-1]) + "\n\n"
    return len(tokenizer.encode(prefix_str, add_special_tokens=False))


@dataclass
class QuestionGenPackPlan:
    """Drop left-pad + few-shot; keep left-padded questions + generation region."""

    few_shot_lens: List[int]
    q_lens: List[int]
    left_pads: List[int]
    orig_prompt_len: int
    new_prompt_len: int
    gen_len: int
    new_seq_len: int
    n_question_tokens: int
    n_gen_tokens: int

    def pack(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() != 3:
            return t
        b = int(t.shape[0])
        d = int(t.shape[2])
        new_t = torch.zeros(
            (b, self.new_seq_len, d),
            dtype=t.dtype,
            device=t.device,
        )
        orig_p = self.orig_prompt_len
        gen = self.gen_len
        new_p = self.new_prompt_len
        src_len = int(t.shape[1])
        g_take = max(0, min(gen, src_len - orig_p, self.new_seq_len - new_p))
        if g_take > 0:
            new_t[:, new_p : new_p + g_take, :] = t[:, orig_p : orig_p + g_take, :]
        for i in range(b):
            fs = self.few_shot_lens[i]
            q = self.q_lens[i]
            src = self.left_pads[i] + fs
            dst = new_p - q
            q_take = max(0, min(q, src_len - src, self.new_seq_len - dst))
            if q_take > 0:
                new_t[i, dst : dst + q_take, :] = t[i, src : src + q_take, :]
        return new_t


def build_question_gen_pack_plan(
    texts: List[str],
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gen_length: int,
) -> QuestionGenPackPlan:
    """Pack plan from the padded prompt batch (before generation)."""
    b = len(texts)
    orig_prompt_len = int(input_ids.shape[1])
    gen_len = int(gen_length)
    few_shot_lens = [few_shot_prefix_len(text, tokenizer) for text in texts]
    unpadded_lens = attention_mask.sum(dim=1).tolist()
    left_pads = [orig_prompt_len - int(u) for u in unpadded_lens]
    q_lens: List[int] = []
    for i in range(b):
        fs = min(few_shot_lens[i], max(0, int(unpadded_lens[i]) - 1))
        few_shot_lens[i] = fs
        q_lens.append(int(unpadded_lens[i]) - fs)
    new_prompt_len = max(q_lens) if q_lens else 0
    if new_prompt_len <= 0:
        new_prompt_len = orig_prompt_len
        new_seq_len = orig_prompt_len + gen_len
    else:
        new_seq_len = new_prompt_len + gen_len
    return QuestionGenPackPlan(
        few_shot_lens=few_shot_lens,
        q_lens=q_lens,
        left_pads=left_pads,
        orig_prompt_len=orig_prompt_len,
        new_prompt_len=int(new_prompt_len),
        gen_len=gen_len,
        new_seq_len=int(new_seq_len),
        n_question_tokens=int(sum(q_lens)),
        n_gen_tokens=int(b) * gen_len,
    )


def _hidden_d_model(model: torch.nn.Module) -> int:
    cfg = getattr(model, "config", None)
    src = cfg if cfg is not None else model
    for key in ("d_model", "hidden_size"):
        val = getattr(src, key, None)
        if val is not None:
            return int(val)
    return 4096


def require_store_disk(
    scratch: Path,
    chunk_bytes: int,
    *,
    headroom_gb: float,
    label: str,
) -> None:
    """Refuse a collect if the chunk plus headroom would not fit."""
    need = int(1.15 * max(int(chunk_bytes), 1) + float(headroom_gb) * 1e9)
    free = int(shutil.disk_usage(scratch).free)
    if free < need:
        raise DiskFullError(
            f"need {need / 1e9:.1f} GB free at {scratch} for {label} "
            f"(chunk ~{chunk_bytes / 1e9:.1f} GB plus {headroom_gb:.0f} GB headroom); "
            f"have {free / 1e9:.1f} GB. Delete leftover staging files or lower "
            "store.inference_batch_size, then re-run the same command to resume."
        )


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
    stream_to_disk: bool = True,
    disk_headroom_gb: Optional[float] = None,
    scratch_dir: Optional[str | Path] = None,
    disk_label: Optional[str] = None,
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

    plan = build_question_gen_pack_plan(
        texts, tokenizer, input_ids, attention_mask, gen_length
    )
    prompt_len = plan.new_prompt_len
    n_question_tokens = plan.n_question_tokens
    n_gen_tokens = plan.n_gen_tokens
    cst_train_tokens = n_question_tokens + n_gen_tokens
    extra_meta = {
        "n_samples": int(len(texts)),
        "n_question_tokens": n_question_tokens,
        "n_gen_tokens": n_gen_tokens,
        "cst_train_tokens": cst_train_tokens,
    }
    if disk_headroom_gb is not None:
        est_bytes = (
            (int(steps) + 1)
            * int(input_ids.shape[0])
            * int(plan.new_seq_len)
            * _hidden_d_model(model)
            * 2
        )
        logger.info(
            "packed_L=%d T<=%d B=%d est_chunk=%.2f GB (bf16, before EOS-early-stop)",
            int(plan.new_seq_len),
            int(steps) + 1,
            int(input_ids.shape[0]),
            est_bytes / 1e9,
        )
        require_store_disk(
            Path(scratch_dir) if scratch_dir is not None else Path(out_path).parent,
            est_bytes,
            headroom_gb=float(disk_headroom_gb),
            label=disk_label or Path(out_path).name,
        )
    out_p = Path(out_path)
    sink: Optional[HiddenStepSink] = None
    if stream_to_disk and hasattr(model, "configure_capture"):
        sink = HiddenStepSink(out_p)
        model.configure_capture(pack_fn=plan.pack, sink=sink)
    elif hasattr(model, "configure_capture"):
        model.configure_capture(pack_fn=plan.pack, sink=None)
    elif hasattr(model, "clear_records"):
        model.clear_records()

    success = False
    try:
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

        if sink is not None:
            sink.finalize({"prompt_len": int(prompt_len), **extra_meta})
        else:
            if hasattr(model, "finish_capture"):
                rec = model.finish_capture()
            elif hasattr(model, "records"):
                rec = model.records
            else:
                raise RuntimeError(
                    "Model must use PreLogitsCaptureWrapper (capture_pre_logits=True)."
                )
            save_hidden_records(
                rec,
                out_path,
                prompt_len=prompt_len,
                extra_meta=extra_meta,
            )
        success = True
    finally:
        if sink is not None and not success:
            sink.unlink()
        elif not success:
            Path(out_path).unlink(missing_ok=True)
        if hasattr(model, "configure_capture"):
            model.configure_capture(pack_fn=None, sink=None)
        elif hasattr(model, "clear_records"):
            model.clear_records()

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
    stream_to_disk: bool = True,
    disk_headroom_gb: Optional[float] = None,
    scratch_dir: Optional[str | Path] = None,
    disk_label: Optional[str] = None,
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
        stream_to_disk=stream_to_disk,
        disk_headroom_gb=disk_headroom_gb,
        scratch_dir=scratch_dir,
        disk_label=disk_label,
    )
    return str(out_path), meta
