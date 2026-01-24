"""
Traces module for causal intervention and tracing in masked diffusion schedulers.

This module provides a unified interface to trace and intervene in any scheduler.

Main entry point:
    from traces import trace_generation
    
    result, trace = trace_generation(
        scheduler_type="pure_diffusion",
        model=model,
        prompt=prompt,
        intervention_type="force_early",
        intervention_params={"positions": [10, 20]},
        **scheduler_kwargs
    )
"""

from .unified_tracer import (
    trace_generation,
    create_intervention_from_type,
    load_scheduler,
    analyze_bottlenecks
)

from .trace_generator import (
    TracingContext,
    InterventionHook,
    GenerationTrace,
    StepTrace,
    create_intervention_hook
)

__all__ = [
    "trace_generation",
    "create_intervention_from_type",
    "load_scheduler",
    "analyze_bottlenecks",
    "TracingContext",
    "InterventionHook",
    "GenerationTrace",
    "StepTrace",
    "create_intervention_hook",
]
