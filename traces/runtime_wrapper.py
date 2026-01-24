"""
Runtime wrapper that intercepts scheduler operations to inject tracing.

This module provides a way to trace and intervene in schedulers without
modifying their code by intercepting tensor operations at runtime.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
import contextlib

from .trace_generator import TracingContext, InterventionHook, StepTrace


class TensorInterceptor:
    """
    Intercepts tensor operations to record and modify scheduler behavior.
    """
    
    def __init__(self, tracing_context: TracingContext):
        self.tracing_context = tracing_context
        self.step_count = 0
        self._original_tensor_setitem = None
        self._active = False
        
    def __enter__(self):
        """Activate interception."""
        self._active = True
        # Patch torch.Tensor.__setitem__ to intercept updates
        self._original_tensor_setitem = torch.Tensor.__setitem__
        
        def intercepted_setitem(self_tensor, key, value):
            # Check if this is a transfer operation (x[transfer_index] = x0[transfer_index])
            if isinstance(key, torch.Tensor) and key.dtype == torch.bool:
                # This might be a transfer_index update
                if self._active and hasattr(self_tensor, '_is_sequence_tensor'):
                    # Record the transfer
                    positions = torch.nonzero(key[0] if key.dim() > 0 else key, as_tuple=False)
                    if positions.numel() > 0:
                        # This is a transfer operation
                        self.tracing_context.record_transfer(key, value, self_tensor)
            
            # Call original setitem
            return self._original_tensor_setitem(self_tensor, key, value)
        
        torch.Tensor.__setitem__ = intercepted_setitem
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Deactivate interception."""
        self._active = False
        if self._original_tensor_setitem:
            torch.Tensor.__setitem__ = self._original_tensor_setitem


def wrap_scheduler_call(
    scheduler_fn: Callable,
    model: torch.nn.Module,
    prompt: torch.Tensor,
    tracing_context: TracingContext,
    **kwargs
) -> torch.Tensor:
    """
    Wrap a scheduler call to inject tracing.
    
    This function modifies the scheduler's execution by:
    1. Intercepting model forward passes
    2. Recording predictions and confidences
    3. Applying interventions before token selection
    4. Recording final transfers
    """
    # Store original model forward
    original_forward = model.forward
    
    def traced_forward(*args, **forward_kwargs):
        """Intercept model forward to record logits."""
        result = original_forward(*args, **forward_kwargs)
        if hasattr(result, 'logits') and tracing_context.trace.steps:
            tracing_context.trace.steps[-1].logits = result.logits.clone()
        return result
    
    model.forward = traced_forward
    
    try:
        # Call scheduler
        result = scheduler_fn(model, prompt, **kwargs)
        
        # Extract final sequence
        if isinstance(result, tuple):
            final_seq = result[0]
        else:
            final_seq = result
        
        tracing_context.trace.final_sequence = final_seq.clone()
        return final_seq
        
    finally:
        # Restore original forward
        model.forward = original_forward
