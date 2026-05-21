"""Capture hidden states for one prompt."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from cst.train.collect_batch import collect_hidden_batch_loaded
from load_model import load_model_and_tokenizer


@torch.no_grad()
def capture_prompt_to_pt(
    model_path: str,
    prompt: str,
    out_pt: str,
    *,
    device: str = "cuda",
    mode_key: str = "pd",
    gen_length: int = 256,
    steps: int = 256,
    block_length: Optional[int] = None,
    top_k: int = 1,
    max_prompt_length: int = 512,
    compile_model: bool = False,
    use_eos_harness: bool = True,
) -> Dict[str, Any]:
    """Run the model on a prompt and save a hidden-state .pt file."""
    model, tok = load_model_and_tokenizer(
        model_path,
        device=device,
        compile_model=compile_model,
        capture_pre_logits=True,
    )
    return collect_hidden_batch_loaded(
        model,
        tok,
        model_path,
        [str(prompt)],
        out_pt,
        device=device,
        mode_key=mode_key,
        gen_length=gen_length,
        steps=steps,
        block_length=block_length,
        top_k=top_k,
        max_prompt_length=max_prompt_length,
        trace_name="tag_single",
        use_eos_harness=use_eos_harness,
    )


def default_tag_pt_path(
    out_dir: str | Path,
    *,
    name: str = "tag_prompt",
) -> str:
    """Default path for a captured hidden .pt under out_dir."""
    p = Path(out_dir) / f"{name}.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)
