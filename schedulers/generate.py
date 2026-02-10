import torch
import time

from schedulers.llada_schedulers import generate_llada
from schedulers.top_k import generate_topk
from schedulers.fhs import generate_fhs
from schedulers.eb_sampler import generate_EB
from schedulers.dus import generate_dus
from schedulers.fdm import generate_fdm


@torch.no_grad()
def generate(model, input_ids, attention_mask, steps=256, gen_length=256, block_length=256, mode="pure_diffusion", strategy="confidence", trace_name=None, upload_to_wandb=True, intervention_mode=None, intervention_steps=None, intervention_k=1):
    """
    Run batched diffusion generation. All samples in input_ids are processed in one batch.

    Modes: pure_diffusion, block_diffusion, top_k_confidence, top_k_margin, FHS, EB_sampler, Dilated Scheduling, FDM
    Strategies: confidence, random, top_margin, entropy
    """
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
    )

    if mode == "pure_diffusion":
        assert strategy in ("confidence", "random"), "strategy must be confidence or random"
        assert block_length == gen_length, "Block length must be equal to gen_length"
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
        assert block_length == gen_length, "Block length must be equal to gen_length"
        assert strategy in ("confidence", "top_margin", "entropy", "random"), (
            "strategy must be confidence, top_margin, entropy, or random"
        )
        gen, steps_taken, blocks = generate_topk(
            model, input_ids, attention_mask,
            steps=steps, gen_length=gen_length, block_length=block_length,
            temperature=0.0, metric=strategy, **common_trace_kw,
        )
    elif mode == "fhs":
        assert block_length is None, "Block length must be None"
        gen, steps_taken = generate_fhs(
            model, input_ids, attention_mask,
            steps=steps, gen_length=gen_length, temperature=1.0,
            pass_time_to_model=True, **common_trace_kw,
        )
        blocks = 0
    elif mode == "eb_sampler":
        assert block_length == gen_length, "Block length must be equal to gen_length"
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
        assert block_length is None, "Block length must be None"
        gen, steps_taken = generate_fdm(
            model, input_ids, attention_mask,
            steps=steps, gen_length=gen_length,
            K1=2, N=4, eta1=0.8, eta2=0.7, gamma=0.5, temperature=0.0,
            mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False,
            global_scale=1.0, **common_trace_kw,
        )
        blocks = 0
    else:
        raise ValueError(f"Unknown mode: {mode}")

    total_time = time.time() - start_time
    return gen, steps_taken, blocks, total_time
