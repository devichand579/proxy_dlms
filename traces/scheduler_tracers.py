"""
Traced versions of schedulers that inject tracing and intervention support.

These are wrapper functions that call the original schedulers but add
tracing and intervention capabilities.
"""

import torch
import numpy as np
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import sys
from pathlib import Path

from .trace_generator import TracingContext, InterventionHook

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from schedulers.utils import add_gumbel_noise, get_num_transfer_tokens


@torch.no_grad()
def traced_pure_diffusion(
    model,
    prompt,
    tracing_context: Optional[TracingContext] = None,
    attention_mask=None,
    steps=128,
    gen_length=128,
    temperature=0.,
    remasking='low_confidence',
    mask_id=126336,
    logits_eos_inf=False,
    confidence_eos_eot_inf=False
) -> Tuple[torch.Tensor, int]:
    """
    Traced version of pure_diffusion scheduler.
    
    This is a drop-in replacement that adds tracing and intervention support.
    """
    if tracing_context is None:
        tracing_context = TracingContext()
    
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    
    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)
    
    mask_index = (x == mask_id)
    num_transfer_tokens = get_num_transfer_tokens(mask_index, steps)
    
    num_steps_taken = 0
    
    for i in range(steps):
        mask_index = (x == mask_id)
        
        if not mask_index.any():
            break
        
        # Record step start
        tracing_context.record_step_start(i, x, mask_id)
        
        logits = model(x, attention_mask=attention_mask).logits
        num_steps_taken += 1
        
        # Record logits
        tracing_context.record_logits(logits)
        
        if logits_eos_inf:
            logits[:, :, 126081] = -torch.inf
        
        if confidence_eos_eot_inf:
            logits[:, :, 126081] = -torch.inf
            logits[:, :, 126348] = -torch.inf
        
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)
        
        if remasking == 'low_confidence':
            p = F.softmax(logits, dim=-1)
            x0_p = torch.squeeze(
                torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
        elif remasking == 'random':
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
        else:
            raise NotImplementedError(remasking)
        
        if confidence_eos_eot_inf:
            x0_p = torch.where(x0 == 126081, torch.zeros_like(x0_p), x0_p)
            x0_p = torch.where(x0 == 126348, torch.zeros_like(x0_p), x0_p)
        
        # Record predictions and confidences
        tracing_context.record_predictions(x0, x0_p)
        
        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, -np.inf)
        
        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
        for j in range(confidence.shape[0]):
            _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
            transfer_index[j, select_index] = True
        
        # Apply interventions
        transfer_index, x0, intervention_applied = tracing_context.apply_interventions(
            transfer_index, x0, x, mask_id
        )
        
        # Apply transfers
        x[transfer_index] = x0[transfer_index]
        
        # Record transfers
        tracing_context.record_transfer(transfer_index, x0, x)
    
    return x, num_steps_taken
