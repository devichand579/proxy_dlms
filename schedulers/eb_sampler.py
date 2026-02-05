'''
    EB (Entropy-Based) sampler
    generate_EB(model, prompt, attention_mask=None, steps=256, gen_length=256, block_length=256, 
                gamma=0.1, E='confidence', temperature=0., mask_id=126336, 
                logits_eos_inf=False, confidence_eos_eot_inf=False, trace_name=None, intervention_mode=None, intervention_steps=None, intervention_k=1)
'''

import os
import torch
import torch.nn.functional as F

from schedulers.utils import add_gumbel_noise, get_mask_by_confidence, get_mask_by_entropy, check_eos_stop_condition

if os.environ.get('TRACING_ENABLED', 'False') == 'True':
    from traces.trace_wrapper import trace_scheduler
else:
    def trace_scheduler(*args, **kwargs):
        return lambda x: x

@ torch.no_grad()
@trace_scheduler(mask_id=126336)  # tracing decorator
def generate_EB(
    model,
    prompt,
    attention_mask=None,
    steps=256,
    gen_length=256,
    block_length=256,
    gamma=0.1,
    E='confidence',
    temperature=0.,
    mask_id=126336,
    logits_eos_inf=False,
    confidence_eos_eot_inf=False,
    trace_name=None,  
    upload_to_wandb=True,  
    intervention_mode=None, 
    intervention_steps=None, 
    intervention_k=1  
):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        attention_mask: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length.
        gamma: Entropy threshold for EB selection. Lower values select fewer tokens.
        E: Ranking metric. 'confidence', 'margin', or 'entropy'.
        temperature: Categorical distribution sampling temperature.
        mask_id: The token id of [MASK] is 126336.
        logits_eos_inf: Whether to set the logits of EOS token to -inf. See Appendix B.4 of LLaDA for details
        confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf. See Appendix B.4 of LLaDA for details
        trace_name: Optional trace name. If provided, trace will be saved to disk.
        upload_to_wandb: If True and trace_name is provided, upload trace to wandb as artifact.
        intervention_mode: Optional intervention mode. "drop" to prevent tokens from being unmasked, "swap" to swap tokens.
        intervention_steps: Optional list/set of step indices (0-indexed) at which to apply intervention.
        intervention_k: Number of tokens to drop/swap per intervention step. Default 1.
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
    eos_id, eot_id = 126081, 126348
    steps_taken = 0  

    for num_block in range(num_blocks):
        block_start = prompt_len + num_block * block_length
        block_end = prompt_len + (num_block + 1) * block_length
        
        # Early stopping - check if block is already complete
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        if not block_mask_index.any():
            continue  # Skip completed blocks
        
        for i in range(steps):
            # Cache mask_index - compute once per step
            mask_index = (x == mask_id)
            block_mask = mask_index[:, block_start:block_end]
            
            if not block_mask.any():
                break
            
            steps_taken += 1  
            logits = model(x, attention_mask=attention_mask).logits

            # Set EOS/EOT to -inf before sampling to prevent them from being predicted
            if logits_eos_inf:
                logits[:, :, eos_id] = -torch.inf

            if confidence_eos_eot_inf:
                logits[:, :, eos_id] = -torch.inf
                logits[:, :, eot_id] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

            p = F.softmax(logits.to(torch.float64), dim=-1)

            # Compute EB ranking score
            if E == 'confidence':
                e = get_mask_by_confidence(logits, x0)
            elif E == 'margin':
                top2 = torch.topk(p, 2, dim=-1).values
                e = top2[..., 0] - top2[..., 1]
            elif E == 'entropy':
                entropy = get_mask_by_entropy(logits)
                e = -entropy  # low entropy = high confidence
            else:
                raise NotImplementedError(E)

            # Compute entropy for EB constraint
            h = -(p * torch.log(p + 1e-9)).sum(-1)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)

            # EB selection per batch
            for j in range(prompt.shape[0]):
                block_positions = [
                    pos for pos in range(block_start, block_end)
                    if mask_index[j, pos]
                ]

                if not block_positions:
                    continue

                # Sort by ranking score (highest first)
                block_positions = sorted(
                    block_positions,
                    key=lambda pos: e[j, pos].item(),
                    reverse=True
                )

                # Select tokens while cumulative entropy constraint holds
                H = []
                for pos in block_positions:
                    H.append(h[j, pos].item())

                    # Constraint: sum(H) - max(H) <= gamma
                    if sum(H) - max(H) <= gamma:
                        transfer_index[j, pos] = True
                    else:
                        break

            # Fallback: if entropy too strict, select most confident token
            if not transfer_index.any():
                for j in range(prompt.shape[0]):
                    block_positions = [
                        pos for pos in range(block_start, block_end)
                        if mask_index[j, pos]
                    ]
                    if not block_positions:
                        continue

                    # Choose most confident token
                    best_pos = max(block_positions, key=lambda pos: e[j, pos].item())
                    transfer_index[j, best_pos] = True

            x[transfer_index] = x0[transfer_index]
            
            # Early stopping: Check if EOS is decoded and all tokens before it are unmasked
            if check_eos_stop_condition(x, eos_id, mask_id, prompt_len):
                return x, steps_taken, num_block + 1  # Stop generation early

            del logits, logits_with_noise, p, e, h
            torch.cuda.empty_cache()

    return x, steps_taken, num_blocks