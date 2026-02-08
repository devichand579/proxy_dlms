"""
 Tracing wrapper for schedulers.
"""

from typing import Optional, Callable, List
import contextvars
import functools
import time
import torch
import torch.nn.functional as F
import numpy as np
from traces.trace_generator import TracingContext, InterventionHook
from traces.utils import is_negative_inf, was_position_masked, was_positions_masked, extract_tensor_value, get_value_item, value_matches_x0, build_transfer_index, build_x0_for_trace, ensure_2d_shape, shapes_match, get_x0_trace, _is_tensor_like

# When set by eval.py, sample traces are appended here and written as one file per task. No per-sample save/upload.
_task_trace_list_ctx: contextvars.ContextVar[Optional[List]] = contextvars.ContextVar("task_trace_list", default=None)


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
        self._last_logits = None  # Store logits for FHS-style schedulers
        self._pending_transfer_index = None
        self._in_intervention = False  # Flag to prevent recursion during intervention
        self._in_setitem = False  # Additional reentrant guard
        
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
                    x = args[0] if _is_tensor_like(args[0]) else None
                    if x is not None and x.dim() == 2:
                        self._last_x = x
                        # Store logits for FHS-style schedulers that don't use argmax
                        if hasattr(output, 'logits') and output.logits is not None:
                            self._last_logits = output.logits.clone()
                            # Also store in tracing context for on-demand prediction computation
                            self.tracing_context._current_logits = output.logits.clone()
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
                if _is_tensor_like(input) and input.dim() >= 2:
                    if dim == -1 or (dim is None and input.dim() == 3):
                        # Store the full x0 tensor for later use in setitem
                        self._last_x0 = result.clone() if _is_tensor_like(result) else result
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
            # Handle dim=None for 1D tensors 
            # For 1D tensors, dim should be -1 (last dimension)
            if dim is None and _is_tensor_like(input) and input.dim() == 1:
                dim = -1
            
            # Call original topk - preserve the named tuple return type
            result = original_topk(
                input, k, dim=dim, largest=largest, sorted=sorted, out=out
            )
            
            # If this is called on confidence, it's likely creating transfer_index
            if self.tracing_context is not None:
                # Check if input looks like confidence (1D or 2D, float values)
                if _is_tensor_like(input) and input.dim() <= 2:
                    # Access indices from the named tuple
                    indices = result.indices
                    # Create boolean mask from indices 
                    if input.dim() == 1:
                        transfer_mask = torch.zeros_like(input, dtype=torch.bool)
                        transfer_mask[indices] = True
                        self._pending_transfer_index = transfer_mask.unsqueeze(0)
                    elif input.dim() == 2:
                        transfer_mask = torch.zeros_like(input, dtype=torch.bool)
                        transfer_mask[0, indices] = True
                        self._pending_transfer_index = transfer_mask
            
            # Return the original named tuple 
            return result
        
        return traced_topk
    
    def _detect_block_boundary(self, key, value):
        """Detect block boundary pattern: x0_p[:, block_end:] = -np.inf"""
        if not (isinstance(key, tuple) and len(key) == 2):
            return False
        
        first_slice, second_slice = key
        if not (first_slice == slice(None) and 
                isinstance(second_slice, slice) and 
                second_slice.start is not None and 
                second_slice.stop is None):
            return False
        
        if not is_negative_inf(value):
            return False
        
        # Extract block boundaries
        block_end = second_slice.start
        self.tracing_context._current_block_end = block_end
        if self.tracing_context._block_length is not None:
            block_start = block_end - self.tracing_context._block_length
            self.tracing_context._current_block_start = block_start
        return True
    
    def _detect_single_position_transfer(self, x, x0_trace, pos_val, value):
        """Detect if single position assignment is a transfer."""
        if pos_val < 0 or x.shape[1] <= pos_val:
            return None
        
        # Check if position was masked
        was_masked = was_position_masked(x, pos_val, self.mask_id)
        is_real_token = get_value_item(value, self.mask_id) != self.mask_id
        
        if not (was_masked and is_real_token):
            return None
        
        # Extract value
        val_item, val_tensor = extract_tensor_value(value, device=x.device, dtype=torch.long)
        if val_item is None or val_tensor is None:
            return None
        
        # Check if matches x0 (for deterministic schedulers)
        matches_x0 = value_matches_x0(x0_trace, pos_val, val_item)
        
        # Record if matches x0 OR was masked (for stochastic schedulers like FHS)
        if matches_x0 or was_masked:
            transfer_positions = torch.tensor([pos_val], device=x.device, dtype=torch.long)
            # Update x0_trace with actual value
            if x0_trace.shape[1] > pos_val:
                x0_trace[0, pos_val] = int(val_item)
            return transfer_positions, val_tensor
        
        return None
    
    def _detect_multiple_positions_transfer(self, x, x0_trace, pos_tensor, value):
        """Detect if multiple positions assignment is a transfer."""
        if (pos_tensor.numel() == 0 or 
            not (pos_tensor >= 0).all() or 
            not (pos_tensor < x.shape[1]).all()):
            return None
        
        # Use duck typing to avoid isinstance recursion
        if not (_is_tensor_like(value) and value.numel() == pos_tensor.numel()):
            return None
        
        # Check which positions were masked
        was_masked_mask = was_positions_masked(x, pos_tensor, self.mask_id)
        
        # Check if values match x0
        matches_x0 = False
        if x0_trace.shape[1] > pos_tensor.max().item():
            x0_vals = x0_trace[0, pos_tensor]
            matches_x0 = (torch.allclose(value.float(), x0_vals.float(), atol=1e-6) or 
                         torch.equal(value, x0_vals))
        
        if matches_x0 or was_masked_mask.any():
            return pos_tensor, value
        
        return None
    
    def _process_transfer(self, tensor_self, original_key, transfer_index, x0_trace, x):
        """Common logic for processing a transfer: apply interventions, record, and apply."""
        self._in_intervention = True
        try:
            # For FHS-style schedulers: if predictions haven't been recorded yet, compute from logits
            if (self.tracing_context._current_predictions is None and 
                self.tracing_context._current_logits is not None):
                # Compute predictions and confidences from logits
                predictions = torch.argmax(self.tracing_context._current_logits, dim=-1)  # [1, L]
                probs = F.softmax(self.tracing_context._current_logits, dim=-1)
                confidences = torch.max(probs, dim=-1)[0]  # [1, L]
                self.tracing_context.record_predictions(predictions, confidences)
            
            # Apply interventions if needed
            modified_transfer, modified_x0, intervention_applied = (
                self.tracing_context.apply_interventions(
                    transfer_index, x0_trace, x, self.mask_id
                )
            )
            
            # Record transfer
            self.tracing_context.record_transfer(modified_transfer, modified_x0)
            
            # Apply intervention if it modified the transfer
            if intervention_applied:
                modified_positions = torch.nonzero(modified_transfer[0], as_tuple=False).squeeze(-1)
                if modified_positions.numel() > 0:
                    modified_values = modified_x0[0, modified_positions]
                    
                    # Handle different key types
                    # Use type() for tuple (safe) and duck typing for tensor (avoids recursion)
                    if isinstance(original_key, tuple):
                        batch_idx, _ = original_key
                        new_key = (batch_idx, modified_positions)
                        self._original_setitem(tensor_self, new_key, modified_values)
                        return "modified"
                    elif _is_tensor_like(original_key):
                        # Boolean tensor case
                        modified_transfer_2d = ensure_2d_shape(modified_transfer)
                        modified_x0_2d = ensure_2d_shape(modified_x0)
                        if original_key.dim() == 1:
                            new_key = modified_transfer_2d[0]
                            new_value = modified_x0_2d[0, new_key]
                        else:
                            new_key = modified_transfer_2d
                            new_value = modified_x0_2d[new_key]
                        self._original_setitem(tensor_self, new_key, new_value)
                        return "modified"
            
            return True  # Handled but not modified
        finally:
            self._in_intervention = False
    
    def _handle_boolean_tensor_transfer(self, tensor_self, key, value):
        """Handle boolean tensor indexing: x[transfer_index] = x0[transfer_index]"""
        # Use duck typing to avoid isinstance recursion
        if not (_is_tensor_like(key) and hasattr(key, 'dtype') and key.dtype == torch.bool and 
                _is_tensor_like(value)):
            return False
        
        if self._last_x0 is None:
            return False
        
        try:
            transfer_index = ensure_2d_shape(key.clone())
            x = ensure_2d_shape(tensor_self.clone())
            x0_trace = ensure_2d_shape(self._last_x0.clone())
            
            if not shapes_match([transfer_index, x0_trace, x]):
                return False
            
            return self._process_transfer(tensor_self, key, transfer_index, x0_trace, x)
        except Exception as e:
            if not self._in_intervention:
                import warnings
                warnings.warn(f"Tracing failed: {e}")
            return False
    
    def _handle_integer_indexing_transfer(self, tensor_self, key, value):
        """Handle integer indexing: x[0, pos] = value or x[0, positions] = values"""
        if not (isinstance(key, tuple) and len(key) == 2):
            return False
        
        batch_idx, pos_idx = key
        if not ((isinstance(batch_idx, int) and batch_idx == 0) or batch_idx == slice(None)):
            return False
        
        x = ensure_2d_shape(tensor_self.clone())
        x0_trace = get_x0_trace(self, x)
        
        # Detect transfer based on position index type
        # Use duck typing to avoid isinstance recursion
        transfer_info = None
        if isinstance(pos_idx, int):
            transfer_info = self._detect_single_position_transfer(x, x0_trace, pos_idx, value)
        elif _is_tensor_like(pos_idx) and hasattr(pos_idx, 'dtype') and pos_idx.dtype in (torch.long, torch.int64, torch.int32, torch.int):
            transfer_info = self._detect_multiple_positions_transfer(x, x0_trace, pos_idx, value)
        
        if transfer_info is None:
            return False
        
        transfer_positions, transfer_values = transfer_info
        
        # Build transfer_index and process
        transfer_index = build_transfer_index(transfer_positions, x.shape[1], x.device)
        x0_for_trace = build_x0_for_trace(x0_trace, transfer_positions, transfer_values)
        
        if not shapes_match([transfer_index, x0_for_trace, x]):
            return False
        
        return self._process_transfer(tensor_self, key, transfer_index, x0_for_trace, x)
    
    def _intercept_tensor_setitem(self, original_setitem):
        """Intercept tensor __setitem__ to detect transfers. Assumes batch size 1."""
        tracer_instance = self  
        
        def traced_setitem(tensor_self, key, value):
            # Reentrant guard to prevent recursion
            if getattr(tracer_instance, '_in_setitem', False):
                return original_setitem(tensor_self, key, value)
            tracer_instance._in_setitem = True
            try:
            # Skip if we're already in an intervention to prevent recursion
                if tracer_instance._in_intervention:
                    return original_setitem(tensor_self, key, value)
                
                if tracer_instance.tracing_context is None:
                    return original_setitem(tensor_self, key, value)
                
                # Try to detect and handle different patterns
                # Detect block boundary first (doesn't return early)
                tracer_instance._detect_block_boundary(key, value)
                
                # Try boolean tensor transfer
                result = tracer_instance._handle_boolean_tensor_transfer(tensor_self, key, value)
                if result:
                    return tensor_self if result == "modified" else original_setitem(tensor_self, key, value)
                
                # Try integer indexing transfer
                result = tracer_instance._handle_integer_indexing_transfer(tensor_self, key, value)
                if result:
                    return tensor_self if result == "modified" else original_setitem(tensor_self, key, value)
                
                # Fallback: call original setitem
                return original_setitem(tensor_self, key, value)
            finally:
                tracer_instance._in_setitem = False
        
        return traced_setitem
    
    def __call__(self, *args, **kwargs):
        """Execute scheduler with fully automated tracing."""
        start_time = time.time()
        
        # Extract mask_id from scheduler kwargs first (before it's used in record_step_start)
        mask_id_from_kwargs = kwargs.get('mask_id', None)
        if mask_id_from_kwargs is not None:
            self.mask_id = mask_id_from_kwargs
        
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

        # If eval.py set the task trace list (context var), append here and skip per-sample save/upload
        task_trace_list = _task_trace_list_ctx.get()
        if task_trace_list is not None:
            try:
                task_trace_list.append(self.tracing_context.trace.to_dict())
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to append trace to task list: {e}")
        else:
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
