'''
    DUS (Dilated Unmasking Scheduler) schedulers
    1. Binary Search Schedule
    generate_dus(model, prompt, attention_mask=None, gen_length=256, block_length=32,
                      temperature=0., remasking='confidence', mask_id=126336, scheduler='binary',
                      logits_eos_inf=False, confidence_eos_eot_inf=False, trace_name=None, intervention_mode=None, intervention_steps=None, intervention_k=1)
    
    2. Dilated Schedule
    generate_dus(model, prompt, attention_mask=None, gen_length=256, block_length=32,
                      temperature=0., remasking='confidence', mask_id=126336, scheduler='dilated',
                      base=2, base_skip=1, logits_eos_inf=False, confidence_eos_eot_inf=False, trace_name=None, intervention_mode=None, intervention_steps=None, intervention_k=1)
    
    Note: Number of steps is determined automatically by the schedule (binary/dilated),
    not by a steps parameter. The schedule creates hierarchical levels of unmasking.
'''


import torch
from typing import Optional, List
from schedulers.utils import add_gumbel_noise, get_mask_by_confidence, get_mask_by_entropy, check_eos_stop_condition
import os

if os.environ.get('TRACING_ENABLED', 'false').lower() == 'true':
    from traces.trace_wrapper import trace_scheduler
else:
    def trace_scheduler(*args, **kwargs):
        return lambda x: x


def dilated_unmask_levels(start: int, end: int, base: int = 2, skip_exp: int = 1) -> List[List[int]]:
    """
    Generate coarse-to-fine (dilated) unmasking schedule over [start..end].
    
    Creates multiple levels of positions to unmask, starting with sparse
    positions at regular intervals and progressively filling in gaps.
    
    Args:
        start: Starting position (inclusive)
        end: Ending position (inclusive)
        base: Dilation base factor (must be >= 1)
        skip_exp: Exponent for initial stride calculation (must be >= 1)
        
    Returns:
        List of lists, where each inner list contains positions to unmask
        at that level
        
    Raises:
        ValueError: If base or skip_exp is less than 1
    """
    if base < 1 or skip_exp < 1:
        raise ValueError("base and skip_exp must be >= 1")
    if base == 1:
        return [list(range(start, end + 1))]

    length = end - start + 1
    stride = length // (base ** skip_exp)
    levels = []
    revealed = set()

    # Dilated rounds
    while stride >= 1:
        this_round = [
            i
            for i in range(start, end + 1)
            if (i - start) % stride == 0 and i not in revealed
        ]
        if this_round:
            levels.append(this_round)
            revealed.update(this_round)
        stride //= base

    # Final round – reveal any leftovers
    remainder = [i for i in range(start, end + 1) if i not in revealed]
    if remainder:
        levels.append(remainder)

    return levels


def binary_search_levels(start: int, end: int, level: int = 0, out: Optional[List[List[int]]] = None) -> List[List[int]]:
    """
    Generate binary search unmasking schedule over [start..end].
    
    Creates a hierarchical schedule where positions are revealed in
    binary search order, with midpoints revealed first at each level.
    
    Args:
        start: Starting position (inclusive)
        end: Ending position (inclusive)
        level: Current recursion level (used internally)
        out: Output list to accumulate results (used internally)
        
    Returns:
        List of lists, where each inner list contains positions to unmask
        at that level
    """
    if out is None:
        out = []
    if start > end:
        return out
    if len(out) <= level:
        out.append([])
    mid = (start + end) // 2
    out[level].append(mid)
    binary_search_levels(start, mid - 1, level + 1, out)
    binary_search_levels(mid + 1, end, level + 1, out)
    return out


def merge_last_level(levels: List[List[int]]) -> List[List[int]]:
    """
    Merge the last level with the previous one if it's significantly smaller.
    
    This optimization prevents having a tiny final level by merging it
    with the previous level when the last level has fewer elements.
    
    Args:
        levels: List of unmasking levels
        
    Returns:
        Modified levels list with potentially merged final level
    """
    if len(levels) >= 2 and len(levels[-1]) < len(levels[-2]):
        # Merge last level into previous
        levels[-2].extend(levels[-1])
        levels.pop()
    return levels


@ torch.no_grad()
@trace_scheduler()  # tracing decorator
def generate_dus(
    model,
    prompt,
    attention_mask=None,
    gen_length=256,
    block_length=32,
    temperature=0.,
    remasking='confidence',
    mask_id=126336,
    scheduler='binary',
    base=2,
    base_skip=1,
    confidence_threshold=0.1,
    logits_eos_inf=False,
    confidence_eos_eot_inf=False,
    trace_name=None,
    upload_to_wandb=True,
    intervention_mode=None,
    intervention_steps=None,
    intervention_k=1,
):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        attention_mask: A tensor of shape (1, L).
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length.
        temperature: Categorical distribution sampling temperature.
        remasking: Remasking strategy. 'confidence', 'entropy', or 'none'.
        mask_id: The token id of [MASK] is 126336.
        scheduler: Scheduling method. 'binary' for binary search, 'dilated' for coarse-to-fine.
        base: Base value for dilated unmasking levels.
        base_skip: Exponent for initial division in dilated scheduling.
        confidence_threshold: Threshold below which tokens are remasked.
        logits_eos_inf: Whether to set the logits of EOS token to -inf. See Appendix B.4 of LLaDA for details
        confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf. See Appendix B.4 of LLaDA for details
        trace_name: Optional trace name. If provided, trace will be saved to disk.
        upload_to_wandb: If True and trace_name is provided, upload trace to wandb as artifact.
        intervention_mode: Optional intervention mode. "drop" to prevent tokens from being unmasked, "swap" to swap tokens.
        intervention_steps: Optional list/set of step indices (0-indexed) at which to apply intervention.
        intervention_k: Number of tokens to drop/swap per intervention step. Default 1.
    
    Note: Number of steps is determined automatically by the schedule algorithm,
    not by a parameter. Binary search and dilated schedules create hierarchical
    levels based on block_length.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

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
        
        # Generate schedule for this block
        if scheduler == 'dilated':
            schedule = dilated_unmask_levels(block_start, block_end - 1, base=base, skip_exp=base_skip)
        elif scheduler == 'binary':
            schedule = binary_search_levels(block_start, block_end - 1)
        else:
            raise NotImplementedError(scheduler)
        
        schedule = merge_last_level(schedule)
        
        for step_idx, step_indices in enumerate(schedule):
            # Cache mask_index - compute once per step
            mask_index = (x == mask_id)
            if not mask_index.any():
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

            # Update positions scheduled for this step
            for pos in step_indices:
                for j in range(prompt.shape[0]):
                    if x[j, pos] == mask_id:  # Only update if still masked
                        x[j, pos] = x0[j, pos]

            # Apply remasking if specified
            if remasking != 'none' and step_idx < len(schedule) - 1:  # Don't remask in final step
                if remasking == 'confidence':
                    confidence = get_mask_by_confidence(logits, x0)
                    # Remask low-confidence predictions in current block
                    for pos in step_indices:
                        for j in range(prompt.shape[0]):
                            if pos < confidence.shape[1] and confidence[j, pos] < confidence_threshold:
                                x[j, pos] = mask_id
                    del confidence
                elif remasking == 'entropy':
                    entropy = get_mask_by_entropy(logits)
                    # Remask high-entropy predictions in current block
                    for pos in step_indices:
                        for j in range(prompt.shape[0]):
                            if pos < entropy.shape[1] and entropy[j, pos] > confidence_threshold:
                                x[j, pos] = mask_id
                    del entropy
                else:
                    raise NotImplementedError(remasking)

            # Early stopping: Check if EOS is decoded and all tokens before it are unmasked
            if check_eos_stop_condition(x, eos_id, mask_id, prompt_len):
                return x, steps_taken, num_block + 1  # Stop generation early

            del logits, logits_with_noise
            torch.cuda.empty_cache()

    return x, steps_taken, num_blocks