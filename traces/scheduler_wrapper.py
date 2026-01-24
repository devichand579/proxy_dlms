"""
Wrapper to inject tracing and interventions into any scheduler.

This module provides decorators and wrappers that make any scheduler
traceable and intervenable without modifying the scheduler code itself.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
import functools
import inspect

from .trace_generator import TracingContext, InterventionHook, GenerationTrace


def traceable_scheduler(scheduler_fn: Callable) -> Callable:
    """
    Decorator to make any scheduler traceable.
    
    The scheduler must follow the common pattern:
    - Takes (model, prompt, ...) as first arguments
    - Updates x[transfer_index] = x0[transfer_index] at some point
    - Returns (x, ...) or just x
    
    Usage:
        @traceable_scheduler
        def my_scheduler(model, prompt, ...):
            ...
    """
    @functools.wraps(scheduler_fn)
    def wrapped(*args, **kwargs):
        # Extract tracing_context if provided
        tracing_context = kwargs.pop('tracing_context', None)
        intervention_hook = kwargs.pop('intervention_hook', None)
        
        if tracing_context is None:
            tracing_context = TracingContext(intervention_hook=intervention_hook)
        
        # Store in kwargs for scheduler to access
        kwargs['tracing_context'] = tracing_context
        
        # Call original scheduler
        result = scheduler_fn(*args, **kwargs)
        
        # Extract trace
        if isinstance(result, tuple):
            return result + (tracing_context.trace,)
        else:
            return result, tracing_context.trace
    
    return wrapped


def with_tracing(scheduler_fn: Callable, intervention_hook: Optional[InterventionHook] = None):
    """
    Wrap a scheduler function to add tracing support.
    
    Args:
        scheduler_fn: The scheduler function to wrap
        intervention_hook: Optional intervention hook
        
    Returns:
        Wrapped function that returns (result, trace)
    """
    tracing_context = TracingContext(intervention_hook=intervention_hook)
    
    @functools.wraps(scheduler_fn)
    def wrapped(*args, **kwargs):
        # Inject tracing_context
        kwargs['tracing_context'] = tracing_context
        
        # Initialize trace
        if len(args) > 1:
            prompt = args[1]  # Assuming prompt is second argument
            tracing_context.trace.prompt = prompt.clone()
        
        # Call scheduler
        result = scheduler_fn(*args, **kwargs)
        
        # Extract final sequence
        if isinstance(result, tuple):
            final_seq = result[0]
        else:
            final_seq = result
        
        tracing_context.trace.final_sequence = final_seq.clone()
        
        return result, tracing_context.trace
    
    return wrapped


def inject_tracing_into_scheduler(scheduler_fn: Callable) -> Callable:
    """
    Inject tracing into a scheduler by wrapping key operations.
    
    This is a more invasive approach that intercepts tensor operations.
    For most use cases, schedulers should be modified to accept tracing_context.
    """
    # This would require more sophisticated interception
    # For now, we'll use the simpler approach of modifying schedulers
    # to accept tracing_context as a parameter
    pass
