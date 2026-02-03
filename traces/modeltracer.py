"""
Fully automated tracing wrapper that requires NO code changes to schedulers.

This module intercepts tensor operations to automatically detect:
- Model forward calls -> records logits and tracks steps
- argmax operations -> records predictions
- topk operations -> detects transfer_index creation
- Tensor setitem operations -> detects transfers and applies interventions

Usage:
    # Fully automated - no scheduler changes needed
    traced_scheduler = auto_trace_scheduler(generate_llada, mask_id=126336)
    result, trace = traced_scheduler(model, prompt, ...)
    
    # Or use decorator
    @auto_trace_scheduler(mask_id=126336)
    def my_scheduler(...):
        ...
"""

from typing import Optional, Callable, Dict, Any
import functools
import time
import torch
import inspect

from traces.trace_generator import TracingContext, InterventionHook


class ModelTracer:
    """
    Fully automated tracing wrapper that intercepts operations in schedulers.
    
    This wrapper automatically detects:
    1. Model forward calls -> records logits and tracks steps
    2. argmax operations -> records predictions
    3. topk operations -> detects transfer_index creation
    4. Tensor setitem -> detects transfers and applies interventions
    """
    
    def __init__(
        self,
        scheduler_fn: Callable,
        mask_id: int = 126336,
        tracing_context: Optional[TracingContext] = None,
        intervention_hook: Optional[InterventionHook] = None,
        trace_name: Optional[str] = None
    ):
        self.scheduler_fn = scheduler_fn
        self.mask_id = mask_id
        self.tracing_context = tracing_context or TracingContext(
            intervention_hook=intervention_hook,
            trace_name=trace_name
        )
        
        # State tracking for automated detection
        self._step_counter = 0
        self._last_logits = None
        self._last_x0 = None
        self._last_confidence = None
        self._last_x = None
        self._pending_transfer_index = None
        self._pending_x0 = None
        
        # Store original functions for restoration
        self._original_setitem = None
        self._original_argmax = None
        self._original_topk = None
        self._original_forward = None
    
    def _intercept_model_forward(self, model: torch.nn.Module):
        """Create a patched forward method that captures logits and tracks steps."""
        original_forward = model.forward
        
        def traced_forward(*args, **kwargs):
            output = original_forward(*args, **kwargs)
            
            if self.tracing_context is not None:
                # Capture logits if available
                if hasattr(output, 'logits'):
                    self._last_logits = output.logits
                    self.tracing_context.record_logits(output.logits)
                
                # Track step: each model call typically represents a step
                # Try to find the sequence tensor (x) from forward args
                if len(args) > 0:
                    x = args[0] if isinstance(args[0], torch.Tensor) else None
                    if x is not None and x.dim() == 2:
                        self._last_x = x
                        # Record step start
                        self.tracing_context.record_step_start(
                            self._step_counter, x, self.mask_id
                        )
                        self._step_counter += 1
            
            return output
        
        return original_forward, traced_forward
    
    def _intercept_argmax(self, original_argmax):
        """Intercept torch.argmax to capture predictions."""
        def traced_argmax(input, dim=None, keepdim=False, *, out=None):
            result = original_argmax(input, dim=dim, keepdim=keepdim, out=out)
            
            # If this is called on logits (or logits_with_noise), it's likely x0
            if self.tracing_context is not None and self._last_logits is not None:
                # Check if input shape matches last logits (without vocab dim)
                if isinstance(input, torch.Tensor) and input.dim() >= 2:
                    # This might be predictions
                    if dim == -1 or (dim is None and input.dim() == 3):
                        self._last_x0 = result
                        # Try to infer confidence from logits
                        if self._last_logits is not None:
                            # Use softmax or max as confidence proxy
                            with torch.no_grad():
                                probs = torch.softmax(self._last_logits, dim=-1)
                                confidence = torch.max(probs, dim=-1)[0]
                                self._last_confidence = confidence
                                self.tracing_context.record_predictions(result, confidence)
            
            return result
        
        return traced_argmax
    
    def _intercept_topk(self, original_topk):
        """Intercept torch.topk to detect transfer_index creation."""
        def traced_topk(input, k, dim=None, largest=True, sorted=True, *, out=None):
            values, indices = original_topk(
                input, k, dim=dim, largest=largest, sorted=sorted, out=out
            )
            
            # If this is called on confidence, it's likely creating transfer_index
            if self.tracing_context is not None:
                # Check if input looks like confidence (1D or 2D, float values)
                if isinstance(input, torch.Tensor) and input.dim() <= 2:
                    # Store for potential transfer detection
                    # We'll create a boolean mask from indices
                    if input.dim() == 1:
                        # Single batch
                        transfer_mask = torch.zeros_like(input, dtype=torch.bool)
                        transfer_mask[indices] = True
                        self._pending_transfer_index = transfer_mask.unsqueeze(0)
                    elif input.dim() == 2:
                        # Batched - we need to handle this per batch
                        # For now, store indices and we'll handle in setitem
                        self._pending_transfer_index = indices
            
            return values, indices
        
        return traced_topk
    
    def _intercept_tensor_setitem(self, original_setitem):
        """Intercept tensor __setitem__ to detect transfers."""
        tracer_instance = self  # Capture self for closure
        
        def traced_setitem(tensor_self, key, value):
            # Check if this looks like a transfer operation: x[transfer_index] = x0[transfer_index]
            if tracer_instance.tracing_context is not None:
                # Check if key is a boolean tensor (transfer_index pattern)
                if isinstance(key, torch.Tensor) and key.dtype == torch.bool:
                    # Check if value is also a tensor (x0 pattern)
                    if isinstance(value, torch.Tensor):
                        # This looks like a transfer!
                        transfer_index = key.clone()
                        x0 = value.clone()
                        x = tensor_self
                        
                        # Ensure proper shape [1, L] for tracing context
                        if transfer_index.dim() == 1:
                            transfer_index = transfer_index.unsqueeze(0)
                        if x0.dim() == 1:
                            x0 = x0.unsqueeze(0)
                        if x.dim() == 1:
                            x = x.unsqueeze(0)
                        
                        # Apply interventions if needed
                        modified_transfer, modified_x0, intervention_applied = tracer_instance.tracing_context.apply_interventions(
                            transfer_index, x0, x, tracer_instance.mask_id
                        )
                        
                        # Record transfer (use modified values if intervention was applied)
                        tracer_instance.tracing_context.record_transfer(modified_transfer, modified_x0)
                        
                        # If intervention modified the transfer, update the actual operation
                        if intervention_applied:
                            # Update key and value for the actual setitem operation
                            if modified_transfer.shape[0] == 1:
                                key = modified_transfer.squeeze(0)
                            else:
                                key = modified_transfer
                            
                            if modified_x0.shape[0] == 1:
                                value = modified_x0.squeeze(0)
                            else:
                                value = modified_x0
            
            # Call original setitem with potentially modified key/value
            return original_setitem(tensor_self, key, value)
        
        return traced_setitem
    
    def __call__(self, *args, **kwargs):
        """Execute scheduler with fully automated tracing."""
        start_time = time.time()
        
        # Extract model (usually first arg)
        model = args[0] if len(args) > 0 else kwargs.get('model')
        if model is None:
            raise ValueError("Model must be provided as first argument or 'model' kwarg")
        
        # Extract prompt (usually second arg)
        prompt = args[1] if len(args) > 1 else kwargs.get('prompt')
        if prompt is not None and self.tracing_context.trace.prompt is None:
            self.tracing_context.trace.prompt = prompt.clone()
        
        # Extract mask_id from kwargs or use default
        mask_id = kwargs.get('mask_id', self.mask_id)
        self.mask_id = mask_id
        
        # Patch model forward
        original_forward, traced_forward = self._intercept_model_forward(model)
        self._original_forward = original_forward
        model.forward = traced_forward
        
        # Patch torch functions
        self._original_argmax = torch.argmax
        self._original_topk = torch.topk
        torch.argmax = self._intercept_argmax(torch.argmax)
        torch.topk = self._intercept_topk(torch.topk)
        
        # Patch tensor __setitem__
        self._original_setitem = torch.Tensor.__setitem__
        torch.Tensor.__setitem__ = self._intercept_tensor_setitem(torch.Tensor.__setitem__)
        
        try:
            # Call original scheduler - everything captured automatically
            result = self.scheduler_fn(*args, **kwargs)
        finally:
            # Restore all patched methods
            model.forward = self._original_forward
            torch.argmax = self._original_argmax
            torch.topk = self._original_topk
            torch.Tensor.__setitem__ = self._original_setitem
        
        # Extract final sequence
        if isinstance(result, tuple):
            final_seq = result[0] if len(result) > 0 else None
        else:
            final_seq = result
        
        if final_seq is not None and self.tracing_context.trace.final_sequence is None:
            self.tracing_context.trace.final_sequence = final_seq.clone()
        
        # Record timestamps
        end_time = time.time()
        self.tracing_context.record_timestamps(start_time, end_time)
        
        # Return result with trace
        if isinstance(result, tuple):
            # Check if context already in result
            if len(result) > 0 and isinstance(result[-1], TracingContext):
                return result
            return result + (self.tracing_context,)
        else:
            return result, self.tracing_context


def auto_trace_scheduler(
    scheduler_fn: Optional[Callable] = None,
    mask_id: int = 126336,
    intervention_hook: Optional[InterventionHook] = None,
    trace_name: Optional[str] = None
):
    """
    Fully automated tracing wrapper - requires NO code changes to schedulers!
    
    This wrapper automatically intercepts and traces:
    - Model forward calls -> records logits and tracks steps
    - torch.argmax operations -> records predictions and confidence
    - torch.topk operations -> detects transfer_index creation
    - Tensor setitem (x[transfer_index] = x0) -> detects transfers and applies interventions
    
    All tracing happens automatically - no manual calls needed in scheduler code!
    
    Usage as decorator:
        @auto_trace_scheduler(mask_id=126336)
        def my_scheduler(model, prompt, ...):
            # No tracing code needed - everything is automatic!
            ...
    
    Usage as function:
        traced = auto_trace_scheduler(generate_llada, mask_id=126336)
        result, trace = traced(model, prompt, ...)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            # Extract or create tracing context
            tracing_context = kwargs.pop('tracing_context', None)
            ctx = kwargs.pop('ctx', None)
            
            if ctx is not None:
                tracing_context = ctx
            elif tracing_context is None:
                # Create new context
                hook = kwargs.pop('intervention_hook', intervention_hook)
                name = kwargs.pop('trace_name', trace_name)
                tracing_context = TracingContext(
                    intervention_hook=hook,
                    trace_name=name
                )
            
            # Create model tracer
            tracer = ModelTracer(
                func,
                mask_id=mask_id,
                tracing_context=tracing_context,
                trace_name=trace_name
            )
            
            return tracer(*args, **kwargs)
        
        return wrapped
    
    # Support both decorator and function call syntax
    if scheduler_fn is None:
        # Called as @auto_trace_scheduler(...)
        return decorator
    else:
        # Called as auto_trace_scheduler(func, ...)
        return decorator(scheduler_fn)
