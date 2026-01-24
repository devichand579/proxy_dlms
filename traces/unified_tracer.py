"""
Unified tracing and intervention system for masked diffusion schedulers.

This module provides a single interface to trace and intervene in any scheduler
by passing scheduler type, trace type, and model as arguments.

Usage:
    from traces.unified_tracer import trace_generation
    
    result, trace = trace_generation(
        scheduler_type="pure_diffusion",
        model=model,
        prompt=prompt,
        intervention_type="force_early",
        intervention_params={"positions": [10, 20], "tokens": [123, 456]},
        **scheduler_kwargs
    )
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
import importlib
import sys
from pathlib import Path

from .trace_generator import TracingContext, InterventionHook, GenerationTrace, StepTrace


# Map scheduler names to their module paths
SCHEDULER_MAP = {
    "pure_diffusion": ("schedulers.pure_diffusion", "generate_pure_diffusion"),
    "block_diffusion": ("schedulers.block_diffusion", "generate"),
    "top_k": ("schedulers.top_k", "generate_block_topk"),
    "eb_sampler": ("schedulers.eb_sampler", "generate_EB"),
    "fdm": ("schedulers.fdm", "generate_fdm"),
    "fhs": ("schedulers.fhs", "generate_fhs"),
    "dus": ("schedulers.dus", "generate_scheduled"),
}


def load_scheduler(scheduler_type: str) -> Callable:
    """
    Dynamically load a scheduler function by name.
    
    Args:
        scheduler_type: Name of the scheduler (e.g., "pure_diffusion", "top_k")
        
    Returns:
        The scheduler function
    """
    if scheduler_type not in SCHEDULER_MAP:
        raise ValueError(
            f"Unknown scheduler type: {scheduler_type}. "
            f"Available: {list(SCHEDULER_MAP.keys())}"
        )
    
    module_path, function_name = SCHEDULER_MAP[scheduler_type]
    
    try:
        module = importlib.import_module(module_path)
        scheduler_fn = getattr(module, function_name)
        return scheduler_fn
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Failed to load scheduler {scheduler_type} from {module_path}.{function_name}: {e}"
        )


def create_intervention_from_type(
    intervention_type: str,
    intervention_params: Dict[str, Any],
    gen_length: int,
    prompt_len: int
) -> InterventionHook:
    """
    Create an intervention hook from intervention type and parameters.
    
    Args:
        intervention_type: Type of intervention:
            - "force_early": Force tokens to unmask early
            - "force_late": Force tokens to unmask late
            - "force_at_step": Force tokens at specific steps
            - "prevent": Prevent tokens from unmasking
            - "force_token": Force specific token values
            - "none": No intervention
        intervention_params: Parameters for the intervention:
            - For "force_early": {"positions": [int], "tokens": [int]} or {"positions": [int]}
            - For "force_late": {"positions": [int], "steps": [int]}
            - For "force_at_step": {"steps": [int], "positions": [int], "tokens": [int]}
            - For "prevent": {"positions": [int]}
            - For "force_token": {"positions": [int], "tokens": [int]}
        gen_length: Length of generation
        prompt_len: Length of prompt
        
    Returns:
        InterventionHook configured with the interventions
    """
    hook = InterventionHook()
    
    if intervention_type == "none" or intervention_type is None:
        return hook
    
    # Convert relative positions (from generation start) to absolute positions
    def to_absolute_pos(rel_pos: int) -> int:
        """Convert relative position in generation to absolute position in sequence."""
        if rel_pos < 0:
            # Negative means from end
            return prompt_len + gen_length + rel_pos
        return prompt_len + rel_pos
    
    if intervention_type == "force_early":
        positions = intervention_params.get("positions", [])
        tokens = intervention_params.get("tokens", [])
        
        if len(tokens) == 0:
            # If no tokens specified, we'll use model predictions
            tokens = [None] * len(positions)
        elif len(tokens) == 1:
            # Single token for all positions
            tokens = tokens * len(positions)
        
        for pos, token in zip(positions, tokens):
            abs_pos = to_absolute_pos(pos)
            if token is not None:
                hook.force_unmask_early[abs_pos] = token
            else:
                # Will use model prediction
                hook.force_unmask_early[abs_pos] = None
    
    elif intervention_type == "force_late":
        positions = intervention_params.get("positions", [])
        steps = intervention_params.get("steps", [])
        
        if len(steps) == 1:
            steps = steps * len(positions)
        
        for pos, step in zip(positions, steps):
            abs_pos = to_absolute_pos(pos)
            hook.force_unmask_late[abs_pos] = step
    
    elif intervention_type == "force_at_step":
        steps = intervention_params.get("steps", [])
        positions = intervention_params.get("positions", [])
        tokens = intervention_params.get("tokens", [])
        
        if len(tokens) == 0:
            tokens = [None] * len(positions)
        elif len(tokens) == 1:
            tokens = tokens * len(positions)
        
        if len(steps) == 1:
            steps = steps * len(positions)
        
        for step, pos, token in zip(steps, positions, tokens):
            abs_pos = to_absolute_pos(pos)
            if token is not None:
                hook.force_unmask[(step, abs_pos)] = token
            else:
                # Will use model prediction
                hook.force_unmask[(step, abs_pos)] = None
    
    elif intervention_type == "prevent":
        positions = intervention_params.get("positions", [])
        for pos in positions:
            abs_pos = to_absolute_pos(pos)
            hook.prevent_unmask.add(abs_pos)
    
    elif intervention_type == "force_token":
        positions = intervention_params.get("positions", [])
        tokens = intervention_params.get("tokens", [])
        
        if len(tokens) == 1:
            tokens = tokens * len(positions)
        
        for pos, token in zip(positions, tokens):
            abs_pos = to_absolute_pos(pos)
            hook.force_token_value[abs_pos] = token
    
    else:
        raise ValueError(
            f"Unknown intervention type: {intervention_type}. "
            f"Available: force_early, force_late, force_at_step, prevent, force_token, none"
        )
    
    return hook


def create_traced_scheduler(scheduler_fn: Callable, scheduler_type: str) -> Callable:
    """
    Create a traced version of a scheduler by wrapping it.
    
    This works by intercepting the scheduler's key operations:
    - Model forward passes (to record logits)
    - Token selection (to apply interventions)
    - Token transfers (to record decisions)
    """
    # For schedulers that follow the common pattern, we can create a generic wrapper
    # For now, we'll use a simpler approach: check if there's a traced version
    
    # Try to import traced version from scheduler_tracers
    try:
        from .scheduler_tracers import traced_pure_diffusion
        if scheduler_type == "pure_diffusion":
            return traced_pure_diffusion
    except ImportError:
        pass
    
    # Otherwise, create a generic wrapper
    @torch.no_grad()
    def traced_wrapper(model, prompt, tracing_context=None, **kwargs):
        """Generic traced wrapper for schedulers."""
        if tracing_context is None:
            tracing_context = TracingContext()
        
        # Store original model forward
        original_forward = model.forward
        
        def traced_forward(*args, **forward_kwargs):
            result = original_forward(*args, **forward_kwargs)
            if hasattr(result, 'logits') and tracing_context.trace.steps:
                tracing_context.trace.steps[-1].logits = result.logits.clone()
            return result
        
        model.forward = traced_forward
        
        try:
            # Call original scheduler (it will ignore tracing_context)
            result = scheduler_fn(model, prompt, **{k: v for k, v in kwargs.items() if k != 'tracing_context'})
            
            # Extract final sequence
            if isinstance(result, tuple):
                final_seq = result[0]
            else:
                final_seq = result
            
            tracing_context.trace.final_sequence = final_seq.clone()
            return result
            
        finally:
            model.forward = original_forward
    
    return traced_wrapper


def trace_generation(
    scheduler_type: str,
    model: torch.nn.Module,
    prompt: torch.Tensor,
    intervention_type: Optional[str] = None,
    intervention_params: Optional[Dict[str, Any]] = None,
    trace_enabled: bool = True,
    attention_mask: Optional[torch.Tensor] = None,
    **scheduler_kwargs
) -> Tuple[torch.Tensor, Optional[GenerationTrace]]:
    """
    Trace generation with any scheduler and optional interventions.
    
    This is a unified interface that works with any scheduler by dynamically
    loading it and injecting tracing/intervention logic.
    
    Args:
        scheduler_type: Type of scheduler to use (e.g., "pure_diffusion", "top_k", "fdm")
        model: The language model
        prompt: Input prompt tensor [B, L]
        intervention_type: Type of intervention ("force_early", "force_late", etc.) or None
        intervention_params: Parameters for the intervention (see create_intervention_from_type)
        trace_enabled: Whether to record trace (if False, just returns result)
        attention_mask: Optional attention mask
        **scheduler_kwargs: Additional arguments passed to the scheduler
        
    Returns:
        (final_sequence, trace) where trace is None if trace_enabled=False
        
    Example:
        # Force positions 10 and 20 to unmask early with specific tokens
        result, trace = trace_generation(
            scheduler_type="pure_diffusion",
            model=model,
            prompt=prompt,
            intervention_type="force_early",
            intervention_params={
                "positions": [10, 20],
                "tokens": [123, 456]
            },
            steps=128,
            gen_length=128
        )
        
        # Prevent positions 5-10 from unmasking
        result, trace = trace_generation(
            scheduler_type="top_k",
            model=model,
            prompt=prompt,
            intervention_type="prevent",
            intervention_params={"positions": list(range(5, 11))},
            gen_length=128
        )
    """
    # Load the scheduler
    scheduler_fn = load_scheduler(scheduler_type)
    
    # Get generation length from kwargs or use default
    gen_length = scheduler_kwargs.get("gen_length", 128)
    prompt_len = prompt.shape[1]
    
    # Create intervention hook if needed
    intervention_hook = None
    if intervention_type:
        if intervention_params is None:
            intervention_params = {}
        intervention_hook = create_intervention_from_type(
            intervention_type, intervention_params, gen_length, prompt_len
        )
    
    # Create tracing context
    tracing_context = TracingContext(intervention_hook=intervention_hook)
    tracing_context.trace.prompt = prompt.clone()
    
    # Create traced version of scheduler
    traced_scheduler = create_traced_scheduler(scheduler_fn, scheduler_type)
    
    # Prepare kwargs
    if attention_mask is not None:
        scheduler_kwargs['attention_mask'] = attention_mask
    scheduler_kwargs['tracing_context'] = tracing_context
    
    # Call traced scheduler
    result = traced_scheduler(model, prompt, **scheduler_kwargs)
    
    # Extract final sequence
    if isinstance(result, tuple):
        final_seq = result[0]
    else:
        final_seq = result
    
    tracing_context.trace.final_sequence = final_seq.clone()
    
    # Return result and trace
    if trace_enabled:
        return final_seq, tracing_context.trace
    else:
        return final_seq, None


def analyze_bottlenecks(
    trace: GenerationTrace,
    metric: str = "confidence"
) -> Dict[int, float]:
    """
    Analyze which token positions are bottlenecks in generation.
    
    A bottleneck is a position that:
    - Was unmasked late relative to its confidence
    - Had low confidence when unmasked
    - Required many steps before being selected
    
    Args:
        trace: Generation trace to analyze
        metric: Metric to use ("confidence", "steps", "delay")
        
    Returns:
        Dictionary mapping position -> bottleneck score (higher = more bottleneck)
    """
    bottlenecks = {}
    
    for step_trace in trace.steps:
        if step_trace.confidences is None:
            continue
        
        # Find positions that were masked but had high confidence
        mask_index = step_trace.mask_index[0]
        confidences = step_trace.confidences[0]
        
        for pos in range(mask_index.shape[0]):
            if mask_index[pos]:  # Still masked
                conf = confidences[pos].item()
                if conf > 0.5:  # High confidence but not selected
                    if pos not in bottlenecks:
                        bottlenecks[pos] = []
                    bottlenecks[pos].append({
                        "step": step_trace.step,
                        "confidence": conf,
                        "unmasked_at": trace.get_unmasking_step(pos)
                    })
    
    # Compute bottleneck scores
    scores = {}
    for pos, history in bottlenecks.items():
        if history:
            # Average confidence when it should have been unmasked
            avg_conf = sum(h["confidence"] for h in history) / len(history)
            # Delay in steps
            unmasked_at = history[0]["unmasked_at"]
            if unmasked_at is not None:
                first_high_conf_step = history[0]["step"]
                delay = unmasked_at - first_high_conf_step
            else:
                delay = 999  # Never unmasked
            
            # Bottleneck score: high confidence * delay
            scores[pos] = avg_conf * delay
    
    return scores
