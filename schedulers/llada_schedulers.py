'''
    LLaDA paper based schedulers
'''


import torch
import numpy as np
import os
from typing import Optional

from load_model import LLADA_DEFAULT_EOS_ID, LLADA_DEFAULT_EOT_ID, LLADA_DEFAULT_MASK_ID
from schedulers.utils import (
    sample_tokens,
    get_num_transfer_tokens,
    get_mask_by_confidence,
    check_eos_stop_condition,
    check_all_generated_unmasked,
    mask_eos_eot_logits,
)

if os.environ.get('TRACING_ENABLED', 'false').lower() == 'true':
    from traces.trace_wrapper import trace_scheduler
else:
    def trace_scheduler(*args, **kwargs):
        return lambda x: x


@torch.no_grad()
@trace_scheduler()  # tracing decorator
def generate_llada(
    model,
    prompt,
    attention_mask=None,
    steps=256,
    gen_length=256,
    block_length=256,
    temperature=0.,
    remasking='confidence',
    mask_id=LLADA_DEFAULT_MASK_ID,
    eos_id=LLADA_DEFAULT_EOS_ID,
    eot_id: Optional[int] = LLADA_DEFAULT_EOT_ID,
    logits_eos_inf=False,
    confidence_eos_eot_inf=False,
    trace_name=None, 
    upload_to_wandb=True,
    intervention_mode=None, 
    intervention_steps=None, 
    intervention_k=1,
    use_eos_harness: bool = True,
):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        attention_mask: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        remasking: Remasking strategy. 'confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        logits_eos_inf: Whether to set the logits of EOS token to -inf. See Appendix B.4 of LLaDA for details
        confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf. See Appendix B.4 of LLaDA for details
        trace_name: Optional trace name. If provided, trace will be saved to disk.
        upload_to_wandb: If True and trace_name is provided, upload trace to wandb as artifact.
        intervention_mode: Optional intervention mode. "drop" to prevent tokens from being unmasked, "swap" to swap tokens.
        intervention_steps: Optional list/set of step indices (0-indexed) at which to apply intervention.
        intervention_k: Number of tokens to drop/swap per intervention step. Default 1.
        use_eos_harness: If False, never return early on EOS (full step budget / reference-style).
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    prompt_len = prompt.shape[1]
    steps_taken = 0  

    for num_block in range(num_blocks):
        block_start = prompt_len + num_block * block_length
        block_end = prompt_len + (num_block + 1) * block_length
        
        # Early stopping - check if block is already complete
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        if not block_mask_index.any():
            continue  # Skip completed blocks
        
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        
        for i in range(steps):
            # Cache mask_index - compute once per step
            mask_index = (x == mask_id)
            
            
            steps_taken += 1  # Increment step counter
            logits = model(x, attention_mask=attention_mask).logits

            mask_eos_eot_logits(logits, logits_eos_inf, confidence_eos_eot_inf, eos_id, eot_id)

            x0 = sample_tokens(logits, temperature)

            if remasking == 'confidence':
                x0_p = get_mask_by_confidence(logits, x0)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, block_end:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            del logits, x0_p
            torch.cuda.empty_cache()

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                k = num_transfer_tokens[j, i].item()
                if k > 0:
                    _, select_index = torch.topk(confidence[j], k=k, dim=-1)
                    transfer_index[j, select_index] = True
            
            x[transfer_index] = x0[transfer_index]
            
            # Early stopping: Check if EOS is decoded and all tokens before it are unmasked
            if use_eos_harness and check_eos_stop_condition(x, eos_id, mask_id, prompt_len):
                return x, steps_taken, num_block + 1  # Stop generation early
            if check_all_generated_unmasked(x, mask_id, prompt_len):
                return x, steps_taken, num_block + 1

    return x, steps_taken, num_blocks