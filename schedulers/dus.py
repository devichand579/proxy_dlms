import gc
import re
from typing import Optional, Tuple, Union, List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from utils import add_gumbel_noise, get_num_transfer_tokens, get_mask_by_confidence, get_mask_by_entropy, check_stop_generation




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


@torch.no_grad()
def generate_scheduled(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.,
    cfg_scale: float = 0.,
    remasking: str = 'low_confidence',
    mask_id: int = 126336,
    stop_on_eos: bool = True,
    stop_tokens: Optional[List[str]] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    confidence_threshold: float = 0.3,
    scheduler: str = 'binary',
    base: int = 2,
    base_skip: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate text using structured unmasking scheduler.
    
    This function implements scheduled generation where tokens are revealed
    according to a predefined schedule (binary search or dilated) rather than
    confidence-based selection. Supports remasking based on confidence thresholds.
    
    Args:
        model: The masked language model for prediction
        prompt: Input prompt tensor of shape (1, prompt_len)
        steps: Number of sampling steps (for compatibility, not used in scheduling)
        gen_length: Total length of text to generate
        block_length: Length of each generation block
        temperature: Sampling temperature (0 = deterministic)
        cfg_scale: Classifier-free guidance scale (0 = no guidance)
        remasking: Strategy for token remasking ('low_confidence', 'high_entropy', 'none')
        mask_id: Token ID for the [MASK] token
        stop_on_eos: Whether to stop generation early on EOS tokens
        stop_tokens: List of strings that trigger early stopping
        tokenizer: Tokenizer for processing stop tokens
        confidence_threshold: Threshold below which tokens are remasked
        scheduler: Scheduling method ('binary' for binary search, 'dilated' for coarse-to-fine)
        base: Base value for dilated unmasking levels
        base_skip: Exponent for initial division in dilated scheduling
        
    Returns:
        Tuple of (generated_sequence, unmasking_stages) where:
            - generated_sequence: Complete sequence including prompt
            - unmasking_stages: Tensor tracking when each token was unmasked
    """
    if stop_tokens is None:
        stop_tokens = []
        
    if tokenizer and stop_tokens:
        stop_tokens = tokenizer(stop_tokens, add_special_tokens=False)["input_ids"]
    else:
        stop_on_eos = False

    device = model.device
    prompt_len = prompt.shape[1]
    
    # Initialize generation tensors
    x = torch.full((1, prompt_len + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = prompt.clone().to(device)
    
    unmasking_stage = torch.zeros_like(x, dtype=torch.long, device='cpu')
    unmasking_stage[:, :prompt_len] = -1

    # Validate block configuration
    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length

    total_steps = 0
    stop_generation = False
    
    for num_block in range(num_blocks):
        if stop_generation:
            break
        block_start = prompt_len + num_block * block_length
        block_end = block_start + block_length - 1
        
        if scheduler == 'dilated':
            schedule = dilated_unmask_levels(block_start, block_end, base=base, skip_exp=base_skip)
        elif scheduler == 'binary':
            schedule = binary_search_levels(block_start, block_end)
        else:
            raise NotImplementedError(f"Unknown schedule: {scheduler}")
        
        schedule = merge_last_level(schedule)
        
        for step_idx, step_indices in enumerate(schedule):
            if stop_generation:
                break

            # Generate predictions for all masked positions
            mask_index = (x == mask_id)
            if not mask_index.any():
                break

            with torch.no_grad():
                if cfg_scale > 0.:
                    prompt_index = (x != mask_id)
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    combined_x = torch.cat([x, un_x], dim=0)
                    logits = model(combined_x).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = logits + cfg_scale * (logits - un_logits)
                else:
                    logits = model(x).logits

            # Clean cache
            gc.collect()
            torch.cuda.empty_cache()

            if temperature == 0:
                x0 = torch.argmax(logits, dim=-1)
            else:
                logits = add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(logits, dim=-1)

            # Update positions scheduled for this step
            for pos in step_indices:
                if x[0, pos] == mask_id:  # Only update if still masked
                    x[0, pos] = x0[0, pos]
                    unmasking_stage[0, pos] = total_steps
            
            total_steps += 1

            # Apply remasking if specified
            if remasking != 'none' and step_idx < len(schedule) - 1:  # Don't remask in final step
                if remasking == 'low_confidence':
                    confidence = get_mask_by_confidence(logits, x0)
                elif remasking == 'high_entropy':
                    confidence = get_mask_by_entropy(logits)
                else:
                    continue  # Skip remasking

                # Remask low-confidence predictions in current block
                for pos in step_indices:
                    if pos < confidence.shape[1] and confidence[0, pos] < confidence_threshold:
                        x[0, pos] = mask_id
                        unmasking_stage[0, pos] = -1

            # Check for early stopping
            if check_stop_generation(x, stop_tokens, stop_on_eos, mask_id, prompt_len):
                stop_generation = True
                break

    return x, unmasking_stage