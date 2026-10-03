import time
from pathlib import Path
from typing import Optional, Union

import torch

from load_model import (
    LLADA_DEFAULT_EOS_ID,
    LLADA_DEFAULT_EOT_ID,
    LLADA_DEFAULT_MASK_ID,
    infer_b44_eos_unmask_safety,
    infer_diffusion_token_ids,
)
from schedulers.llada_schedulers import generate_llada
from schedulers.top_k import generate_topk
from schedulers.fhs import generate_fhs
from schedulers.eb_sampler import generate_EB
from schedulers.dus import generate_dus
from schedulers.fdm import generate_fdm
from schedulers.proxy_swap import generate_proxy


@torch.no_grad()
def generate(
    model,
    input_ids,
    attention_mask,
    steps=256,
    gen_length=256,
    block_length=256,
    mode="pure_diffusion",
    strategy="confidence",
    top_k=1,
    trace_name=None,
    upload_to_wandb=True,
    intervention_mode=None,
    intervention_steps=None,
    intervention_k=1,
    mask_id=LLADA_DEFAULT_MASK_ID,
    eos_id=LLADA_DEFAULT_EOS_ID,
    eot_id: Optional[int] = LLADA_DEFAULT_EOT_ID,
    model_path: Optional[Union[str, Path]] = None,
    use_eos_harness: bool = True,
    invalid_token_ids: Optional[torch.Tensor] = None,
):
    """Batched mask-diffusion `generate` for all rows in `input_ids`.

    With `model_path`, ids come from `infer_diffusion_token_ids` and B.4 may apply
    (`infer_b44_eos_unmask_safety`, e.g. `llada1_5`). `use_eos_harness=False` disables
    early EOS return in schedulers (reference-style full tail decode in eval).
    """
    b44 = False
    if model_path is not None:
        mask_id, eos_id, eot_id = infer_diffusion_token_ids(model_path)
        b44 = infer_b44_eos_unmask_safety(model_path)
    start_time = time.time()
    num_samples = input_ids.shape[0]
    assert num_samples >= 1, "Need at least one sample"
    assert steps <= gen_length, "Steps must be less than or equal to gen_length"

    common_trace_kw = dict(
        trace_name=trace_name,
        upload_to_wandb=upload_to_wandb,
        intervention_mode=intervention_mode,
        intervention_steps=intervention_steps,
        intervention_k=intervention_k,
        mask_id=mask_id,
        eos_id=eos_id,
        eot_id=eot_id,
        use_eos_harness=use_eos_harness,
        logits_eos_inf=b44,
        confidence_eos_eot_inf=b44,
    )

    if mode == "pc_swap":
        gen, steps_taken, blocks = generate_proxy(
            model,
            input_ids,
            attention_mask,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            mode=mode,
            top_k=top_k,
            mask_id=mask_id,
            eos_id=eos_id,
            eot_id=eot_id,
            logits_eos_inf=b44,
            confidence_eos_eot_inf=b44,
            use_eos_harness=use_eos_harness,
            model_path=model_path,
            trace_name=trace_name,
            upload_to_wandb=upload_to_wandb,
            intervention_mode=intervention_mode,
            intervention_steps=intervention_steps,
            intervention_k=intervention_k,
        )
    elif mode == "pure_diffusion":
        assert block_length <= gen_length, "Block length must be less than or equal to gen_length"
        assert strategy in ("confidence", "random"), "strategy must be confidence or random"
        gen, steps_taken, blocks = generate_llada(
            model, input_ids, attention_mask,
            steps=steps, gen_length=gen_length, block_length=block_length,
            temperature=0.0, remasking=strategy, **common_trace_kw,
        )
    elif mode == "block_diffusion":
        assert block_length < gen_length, "Block length must be less than gen_length"
        assert strategy in ("confidence", "random"), "strategy must be confidence or random"
        gen, steps_taken, blocks = generate_llada(
            model, input_ids, attention_mask,
            steps=steps, gen_length=gen_length, block_length=block_length,
            temperature=0.0, remasking=strategy, **common_trace_kw,
        )
    elif mode == "top_k":
        assert block_length <= gen_length, "Block length must be less than or equal to gen_length"
        assert strategy in ("confidence", "top_margin", "entropy", "random"), (
            "strategy must be confidence, top_margin, entropy, or random"
        )
        gen, steps_taken, blocks = generate_topk(
            model, input_ids, attention_mask,
            steps=steps, gen_length=gen_length, block_length=block_length,
            temperature=0.0, top_k=top_k, metric=strategy, **common_trace_kw,
        )
    elif mode == "fhs":
        assert block_length is None, "Block length must be None"
        gen, steps_taken = generate_fhs(
            model, input_ids, attention_mask,
            steps=steps, gen_length=gen_length, temperature=1.0,
            pass_time_to_model=True, invalid_token_ids=invalid_token_ids,
            **common_trace_kw,
        )
        blocks = 0
    elif mode == "eb_sampler":
        assert block_length <= gen_length, "Block length must be less than or equal to gen_length"
        assert strategy in ("confidence", "margin", "entropy"), (
            "strategy must be confidence, margin, or entropy"
        )
        gen, steps_taken, blocks = generate_EB(
            model, input_ids, attention_mask,
            steps=steps, gen_length=gen_length, block_length=block_length,
            gamma=0.1, E=strategy, temperature=0.0, **common_trace_kw,
        )
    elif mode == "dus":
        assert block_length <= gen_length, "Block length must be less than or equal to gen_length"
        assert strategy in ("confidence", "entropy"), "strategy must be confidence or entropy"
        gen, steps_taken, blocks = generate_dus(
            model, input_ids, attention_mask,
            gen_length=gen_length, block_length=block_length, temperature=0.0,
            remasking=strategy, scheduler="dilated", base=2, base_skip=1,
            confidence_threshold=0.1, **common_trace_kw,
        )
    elif mode == "fdm":
        assert block_length <= gen_length, "Block length must be less than or equal to gen_length"
        # mask_id / eos_id / eot_id come from common_trace_kw only (avoid duplicate mask_id kwarg).
        gen, steps_taken, blocks = generate_fdm(
            model, input_ids, attention_mask,
            steps=steps, gen_length=gen_length, block_length=block_length,
            K1=2, N=4, eta1=0.8, eta2=0.7, gamma=0.5, temperature=0.0,
            global_scale=1.0, **common_trace_kw,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    total_time = time.time() - start_time
    return gen, steps_taken, blocks, total_time
