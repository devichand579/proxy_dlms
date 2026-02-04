"""
 Tracing wrapper for schedulers.
"""

from typing import Optional, Callable
import functools
import time
import torch
from traces.trace_generator import TracingContext, InterventionHook


class ModelTracer:
    """
    tracing wrapper that intercepts operations in schedulers.
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
        
        # State tracking
        self._step_counter = 0
        self._last_x0 = None  
        self._last_confidence = None
        self._last_x = None
        self._pending_transfer_index = None
        self._in_intervention = False  # Flag to prevent recursion during intervention
        
        # Store original 
        self._original_setitem = None
        self._original_argmax = None
        self._original_topk = None
        self._original_forward = None
    
    def _intercept_model_forward(self, model: torch.nn.Module):
        """Create a patched forward method that tracks steps."""
        original_forward = model.forward
        
        def traced_forward(*args, **kwargs):
            output = original_forward(*args, **kwargs)
            
            if self.tracing_context is not None:
                # Track step: each model call typically represents a step
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
            if self.tracing_context is not None:
                # Check if input looks like logits (3D tensor) or logits_with_noise
                if isinstance(input, torch.Tensor) and input.dim() >= 2:
                    # This might be predictions
                    if dim == -1 or (dim is None and input.dim() == 3):
                        # Store the full x0 tensor for later use in setitem
                        self._last_x0 = result.clone() if isinstance(result, torch.Tensor) else result
                        # Compute confidence from input (logits or logits_with_noise)
                        if input.dim() == 3:  # [batch, length, vocab] - logits
                            with torch.no_grad():
                                probs = torch.softmax(input, dim=-1)
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
                    # Create boolean mask from indices 
                    if input.dim() == 1:
                        transfer_mask = torch.zeros_like(input, dtype=torch.bool)
                        transfer_mask[indices] = True
                        self._pending_transfer_index = transfer_mask.unsqueeze(0)
                    elif input.dim() == 2:
                        transfer_mask = torch.zeros_like(input, dtype=torch.bool)
                        transfer_mask[0, indices] = True
                        self._pending_transfer_index = transfer_mask
            
            return values, indices
        
        return traced_topk
    
    def _intercept_tensor_setitem(self, original_setitem):
        """Intercept tensor __setitem__ to detect transfers. Assumes batch size 1."""
        tracer_instance = self  
        
        def traced_setitem(tensor_self, key, value):
            # Skip if we're already in an intervention to prevent recursion
            if tracer_instance._in_intervention:
                return tracer_instance._original_setitem(tensor_self, key, value)
            
            # Detect block boundary assignment: x0_p[:, block_end:] = -np.inf
            # This pattern sets positions beyond block_end to -inf
            if tracer_instance.tracing_context is not None:
                # Check if this is a slice assignment like [:, block_end:] = -inf
                if isinstance(key, tuple) and len(key) == 2:
                    first_slice, second_slice = key
                    # Pattern: [:, block_end:] where first slice is : and second is block_end:
                    if (first_slice == slice(None) and 
                        isinstance(second_slice, slice) and 
                        second_slice.start is not None and 
                        second_slice.stop is None):
                        # Check if value is -inf (can be scalar, numpy, or tensor of -inf)
                        is_neg_inf = False
                        try:
                            import numpy as np
                            if isinstance(value, (int, float)):
                                is_neg_inf = (value == float('-inf')) or (value == np.NINF) or (value < -1e10)
                            elif isinstance(value, np.ndarray):
                                # Check if all values are -inf
                                is_neg_inf = np.all(np.isinf(value)) and np.all(value < 0) if value.size > 0 else False
                            elif isinstance(value, torch.Tensor):
                                # Check if all values are -inf
                                is_neg_inf = torch.all(value == -torch.inf) if value.numel() > 0 else False
                            elif hasattr(value, '__float__'):
                                val_float = float(value)
                                is_neg_inf = (val_float == float('-inf')) or (val_float < -1e10)
                        except:
                            # Fallback: check if it's a very large negative number
                            try:
                                if isinstance(value, (int, float)):
                                    is_neg_inf = value < -1e10
                            except:
                                pass
                        
                        if is_neg_inf:
                            # Extract block_end from slice
                            block_end = second_slice.start
                            tracer_instance.tracing_context._current_block_end = block_end
                            # Calculate block_start from block_end and block_length
                            if tracer_instance.tracing_context._block_length is not None:
                                block_start = block_end - tracer_instance.tracing_context._block_length
                                tracer_instance.tracing_context._current_block_start = block_start
                
                # Check if key is a boolean tensor (transfer_index pattern)
                if isinstance(key, torch.Tensor) and key.dtype == torch.bool:
                    # Check if value is also a tensor
                    if isinstance(value, torch.Tensor):
                        # This looks like a transfer!
                        transfer_index = key.clone()
                        x = tensor_self.clone()
                        
                        # Ensure [1, L] shape
                        if transfer_index.dim() == 1:
                            transfer_index = transfer_index.unsqueeze(0)
                        # If already 2D, assume it's [1, L]
                        
                        if x.dim() == 1:
                            x = x.unsqueeze(0)
                        # If already 2D, assume it's [1, L]
                    
                        if tracer_instance._last_x0 is not None:
                            try:
                                x0_full = tracer_instance._last_x0.clone()
                                
                                # Ensure [1, L] shape
                                if x0_full.dim() == 1:
                                    x0_trace = x0_full.unsqueeze(0)
                                else:
                                    x0_trace = x0_full  
                                
                                # Verify all shapes match [1, L] before tracing
                                if (transfer_index.shape == x0_trace.shape == x.shape and 
                                    len(transfer_index.shape) == 2 and transfer_index.shape[0] == 1):
                                    # Guard tracing operations against recursive __setitem__ interception
                                    tracer_instance._in_intervention = True
                                    try:
                                        # Apply interventions if needed
                                        modified_transfer, modified_x0, intervention_applied = tracer_instance.tracing_context.apply_interventions(
                                            transfer_index, x0_trace, x, tracer_instance.mask_id
                                        )
                                        
                                        # Record transfer
                                        tracer_instance.tracing_context.record_transfer(modified_transfer, modified_x0)
                                        
                                        # Actually apply the intervention by modifying key and value
                                        if intervention_applied:
                                            # Prepare key and value (flag will catch any __setitem__ calls)
                                            # Use minimal operations to extract values
                                            if key.dim() == 1:
                                                # Extract 1D key
                                                if modified_transfer.dim() == 2:
                                                    new_key = modified_transfer[0]
                                                else:
                                                    new_key = modified_transfer
                                                
                                                # Extract corresponding values
                                                if modified_x0.dim() == 2:
                                                    # modified_x0 is [1, L], new_key is [L] bool
                                                    # Extract values where new_key is True
                                                    new_value = modified_x0[0, new_key]
                                                else:
                                                    # modified_x0 is [L], new_key is [L] bool
                                                    new_value = modified_x0[new_key]
                                            else:
                                                # Key is 2D [1, L]
                                                new_key = modified_transfer
                                                # modified_x0 is [1, L], new_key is [1, L] bool
                                                new_value = modified_x0[new_key]
                                            
                                            # This bypasses any method resolution that might hit our patched version
                                            tracer_instance._original_setitem(tensor_self, new_key, new_value)
                                            
                                            return tensor_self
                                    finally:
                                        # Always clear flag
                                        tracer_instance._in_intervention = False
                            except Exception as e:
                                if not tracer_instance._in_intervention:
                                    import warnings
                                    warnings.warn(f"Tracing failed (non-fatal): {e}")
                                pass
            
            # Call original setitem with original key/value
            return original_setitem(tensor_self, key, value)
        
        return traced_setitem
    
    def __call__(self, *args, **kwargs):
        """Execute scheduler with fully automated tracing."""
        start_time = time.time()
        
        # Extract trace_name from scheduler kwargs if provided (takes precedence)
        trace_name_from_kwargs = kwargs.get('trace_name', None)
        if trace_name_from_kwargs is not None:
            self.tracing_context.trace_name = trace_name_from_kwargs
        
        # Extract upload_to_wandb from scheduler kwargs if provided (takes precedence)
        upload_to_wandb_from_kwargs = kwargs.get('upload_to_wandb', None)
        if upload_to_wandb_from_kwargs is not None:
            self.tracing_context.upload_to_wandb = upload_to_wandb_from_kwargs
        
        # Extract intervention parameters from scheduler kwargs and create hook if provided
        intervention_mode = kwargs.get('intervention_mode', None)
        intervention_steps = kwargs.get('intervention_steps', None)
        intervention_k = kwargs.get('intervention_k', 1)
        
        if intervention_mode is not None:
            from traces.trace_generator import InterventionHook
            if intervention_steps is None:
                intervention_steps = set()
            elif isinstance(intervention_steps, (list, tuple)):
                intervention_steps = set(intervention_steps)
            elif isinstance(intervention_steps, set):
                pass  # Already a set
            else:
                intervention_steps = {intervention_steps}  # Single step
            
            intervention_hook = InterventionHook(
                mode=intervention_mode,
                steps=intervention_steps,
                k=intervention_k
            )
            self.tracing_context.intervention_hook = intervention_hook
        
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
        
        # Extract block_length from scheduler kwargs (used for block boundary calculation)
        block_length = kwargs.get('block_length', None)
        if block_length is not None:
            self.tracing_context._block_length = block_length
        
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
            # Call original scheduler
            result = self.scheduler_fn(*args, **kwargs)
        finally:
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
        
        # Upload to wandb if enabled
        if self.tracing_context.upload_to_wandb and self.tracing_context.trace_name is not None:
            try:
                self.tracing_context.trace.upload_to_wandb(
                    self.tracing_context.trace_name,
                    base_dir=self.tracing_context.base_dir
                )
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to upload trace to wandb: {e}")
        
        # Automatically save trace to disk if trace_name is provided
        # Skip if uploading to wandb (file already saved during upload)
        should_save_to_disk = (
            self.tracing_context.trace_name is not None 
            and self.tracing_context.save_to_disk
            and not self.tracing_context.upload_to_wandb
        )
        if should_save_to_disk:
            try:
                self.tracing_context.save_trace()
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to save trace: {e}")
        
        # Return original result 
        return result


def trace_scheduler(
    scheduler_fn: Optional[Callable] = None,
    mask_id: int = 126336,
    intervention_hook: Optional[InterventionHook] = None,
    trace_name: Optional[str] = None,
    save_predictions: bool = True,
    save_confidences: bool = True,
    upload_to_wandb: bool = False
):
    """
    tracing wrapper - requires NO code changes to schedulers!
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
                # trace_name can come from decorator or from scheduler kwargs
                name = trace_name  
                tracing_context = TracingContext(
                    intervention_hook=hook,
                    trace_name=name,
                    save_predictions=save_predictions,
                    save_confidences=save_confidences,
                    upload_to_wandb=upload_to_wandb
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
        # Called as @trace_scheduler(...)
        return decorator
    else:
        # Called as trace_scheduler(func, ...)
        return decorator(scheduler_fn)
