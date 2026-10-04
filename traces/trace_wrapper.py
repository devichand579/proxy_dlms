"""Scheduler tracing via forward, argmax, and setitem hooks."""

from typing import Callable, List, Optional
import contextvars
import functools
import time
import warnings
import numpy as np
import torch
from load_model import LLADA_DEFAULT_MASK_ID
from traces.trace_generator import TracingContext, InterventionHook
from traces.utils import is_negative_inf, was_position_masked, was_positions_masked, extract_tensor_value, get_value_item, value_matches_x0, build_transfer_index, build_x0_for_trace, ensure_2d_shape, shapes_match, same_2d_batch_shape, get_x0_trace, _is_tensor_like

# When set by eval.py, sample traces are appended here and written as one file per task. No per-sample save/upload.
_task_trace_list_ctx: contextvars.ContextVar[Optional[List]] = contextvars.ContextVar("task_trace_list", default=None)

# When True, model.forward still runs but does not open a new traced step.
# Used by FDM (and similar) nested hypothesis forwards so they do not inflate n_steps.
_suppress_step_trace_ctx: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "suppress_step_trace", default=False
)
# When True with suppress: refresh predictions/confidences/entropies on the open
# step from this forward (FHS corrector) without opening a new step.
_update_open_step_logits_ctx: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "update_open_step_logits", default=False
)
_active_tracer_ctx: contextvars.ContextVar[Optional["ModelTracer"]] = contextvars.ContextVar(
    "active_tracer", default=None
)


def note_trace_commit(tokens: torch.Tensor) -> None:
    """Selected tokens recorded on the next boolean write come from ``tokens``."""
    tracer = _active_tracer_ctx.get()
    if tracer is None:
        return
    tracer._last_x0 = tokens.detach()


class suppress_step_recording:
    """Skip traced step opens for nested model forwards (e.g. FDM hypotheses).

    Nestable via ContextVar tokens. The forward still runs; step opens and
    ``_last_x`` updates are suppressed.

    Args:
        update_open_step: If True (FHS corrector), overwrite logits-derived
            fields on the current open step from this forward's logits.
    """

    def __init__(self, *, update_open_step: bool = False):
        self.update_open_step = bool(update_open_step)

    def __enter__(self):
        self._token = _suppress_step_trace_ctx.set(True)
        self._upd_token = _update_open_step_logits_ctx.set(self.update_open_step)
        return self

    def __exit__(self, exc_type, exc, tb):
        _update_open_step_logits_ctx.reset(self._upd_token)
        _suppress_step_trace_ctx.reset(self._token)
        return False


class ModelTracer:
    """Wraps a scheduler to record generation traces."""
    
    def __init__(
        self,
        scheduler_fn: Callable,
        mask_id: int = LLADA_DEFAULT_MASK_ID,
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
        # Batch tracing: when prompt has batch_size > 1, one context per sample
        self._batch_contexts = None
        self._batch_size = 1
        
        # Store original 
        self._original_setitem = None
        self._original_argmax = None
        self._original_topk = None
        self._original_forward = None
    
    def _intercept_model_forward(self, model: torch.nn.Module):
        """Patch model.forward to start each traced step."""
        original_forward = model.forward
        
        def traced_forward(*args, **kwargs):
            output = original_forward(*args, **kwargs)

            # Nested hyp / probe forwards (e.g. FDM) must not open steps or clobber
            # _last_x used for transfer detection on the real decode step.
            if _suppress_step_trace_ctx.get():
                if (
                    _update_open_step_logits_ctx.get()
                    and hasattr(output, "logits")
                    and output.logits is not None
                ):
                    self._refresh_open_step_logits(output.logits)
                return output

            if len(args) > 0:
                x = args[0] if _is_tensor_like(args[0]) else None
            else:
                x = None
            if x is not None and x.dim() == 2:
                self._last_x = x
                batch_size = x.shape[0]
                if self._batch_contexts is not None and batch_size == len(self._batch_contexts):
                    if hasattr(output, 'logits') and output.logits is not None:
                        self._last_logits = output.logits.clone()
                    for b in range(batch_size):
                        ctx = self._batch_contexts[b]
                        if output.logits is not None:
                            ctx._current_logits = output.logits[b : b + 1].clone()
                        ctx.record_step_start(self._step_counter, x[b : b + 1], self.mask_id)
                        if output.logits is not None:
                            ctx.record_logits_metrics(output.logits[b : b + 1])
                    self._step_counter += 1
                elif self.tracing_context is not None and batch_size == 1:
                    if hasattr(output, 'logits') and output.logits is not None:
                        self._last_logits = output.logits.clone()
                        self.tracing_context._current_logits = output.logits.clone()
                    self.tracing_context.record_step_start(
                        self._step_counter, x, self.mask_id
                    )
                    if hasattr(output, 'logits') and output.logits is not None:
                        self.tracing_context.record_logits_metrics(output.logits)
                    self._step_counter += 1
            return output
        
        return original_forward, traced_forward

    def _refresh_open_step_logits(self, logits: torch.Tensor) -> None:
        """Overwrite logits metrics on the current open step (no new step)."""
        self._last_logits = logits.clone()
        batch_size = int(logits.shape[0]) if logits.dim() >= 1 else 1
        if self._batch_contexts is not None and batch_size == len(self._batch_contexts):
            for b in range(batch_size):
                ctx = self._batch_contexts[b]
                ctx._current_logits = logits[b : b + 1].clone()
                ctx.record_logits_metrics(logits[b : b + 1])
        elif self.tracing_context is not None:
            self.tracing_context._current_logits = logits.clone()
            self.tracing_context.record_logits_metrics(logits)
    
    def _intercept_argmax(self, original_argmax):
        """Patch argmax to capture x0 for transfer detection."""
        def traced_argmax(input, dim=None, keepdim=False, *, out=None):
            result = original_argmax(input, dim=dim, keepdim=keepdim, out=out)
            
            # If this is called on logits (or logits_with_noise), it's likely x0
            if _is_tensor_like(input) and input.dim() >= 2 and (dim == -1 or (dim is None and input.dim() == 3)):
                self._last_x0 = result.clone() if _is_tensor_like(result) else result
                if input.dim() == 3:
                    with torch.no_grad():
                        probs = torch.softmax(input, dim=-1)
                        confidence = torch.max(probs, dim=-1)[0]
                        self._last_confidence = confidence
            return result
        
        return traced_argmax
    
    def _intercept_topk(self, original_topk):
        """Patch topk to detect boolean transfer_index writes."""
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
        """Detect x0_p[:, block_end:] = -inf block-boundary writes."""
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
        
        block_end = second_slice.start
        if self._batch_contexts is not None:
            for ctx in self._batch_contexts:
                ctx._current_block_end = block_end
                if ctx._block_length is not None:
                    ctx._current_block_start = block_end - ctx._block_length
        elif self.tracing_context is not None:
            self.tracing_context._current_block_end = block_end
            if self.tracing_context._block_length is not None:
                self.tracing_context._current_block_start = (
                    block_end - self.tracing_context._block_length
                )
        return True
    
    def _gen_pos_selected_on_open_step(self, pos_val: int, batch_index: int = 0) -> bool:
        """True if the open step already marked this gen-region position selected."""
        ctx = None
        if self._batch_contexts is not None and 0 <= batch_index < len(self._batch_contexts):
            ctx = self._batch_contexts[batch_index]
        elif self.tracing_context is not None:
            ctx = self.tracing_context
        if ctx is None or not ctx.trace.step_traces:
            return False
        sel = ctx.trace.step_traces[-1].selected_positions
        if sel is None:
            return False
        pl = int(ctx.prompt_len or 0)
        gen_pos = int(pos_val) - pl
        if gen_pos < 0 or gen_pos >= sel.shape[-1]:
            return False
        return bool(sel[0, gen_pos].item())

    def _detect_single_position_transfer(
        self, x, x0_trace, pos_val, value, *, batch_index: int = 0
    ):
        """True if x[0, pos] = value is a single-token unmask or refine."""
        if pos_val < 0 or x.shape[1] <= pos_val:
            return None

        was_masked = was_position_masked(x, pos_val, self.mask_id)
        is_real_token = get_value_item(value, self.mask_id) != self.mask_id
        if not is_real_token:
            return None

        val_item, val_tensor = extract_tensor_value(value, device=x.device, dtype=torch.long)
        if val_item is None or val_tensor is None:
            return None

        matches_x0 = value_matches_x0(x0_trace, pos_val, val_item)
        # Predictor unmask (masked → token) or FHS corrector refine of already-selected pos.
        refine = self._gen_pos_selected_on_open_step(int(pos_val), batch_index=batch_index)
        if not ((was_masked and (matches_x0 or was_masked)) or refine):
            return None

        transfer_positions = torch.tensor([pos_val], device=x.device, dtype=torch.long)
        if x0_trace.shape[1] > pos_val:
            x0_trace[0, pos_val] = int(val_item)
        return transfer_positions, val_tensor
    
    def _detect_multiple_positions_transfer(self, x, x0_trace, pos_tensor, value):
        """True if x[0, positions] = values is a multi-token unmask."""
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
    
    def _process_transfer(self, tensor_self, original_key, transfer_index, x0_trace, x, batch_index=None):
        """Run interventions, record the transfer, and apply token writes."""
        self._in_intervention = True
        try:
            B = transfer_index.shape[0] if transfer_index.dim() >= 1 else 1
            contexts = self._batch_contexts if self._batch_contexts is not None else (
                [self.tracing_context] if self.tracing_context is not None else []
            )
            if not contexts or (self._batch_contexts is not None and B != len(contexts) and batch_index is None):
                return False

            if batch_index is not None and self._batch_contexts is not None:
                ctx = contexts[batch_index]
                modified_transfer, modified_x0, intervention_applied = ctx.apply_interventions(
                    transfer_index, x0_trace, x, self.mask_id
                )
                ctx.record_committed_token_confidences(modified_x0)
                ctx.record_transfer(modified_transfer, modified_x0)
                if intervention_applied and isinstance(original_key, tuple):
                    batch_idx, _ = original_key
                    mod_pos = torch.nonzero(modified_transfer[0], as_tuple=False).squeeze(-1)
                    if mod_pos.numel() > 0:
                        self._original_setitem(
                            tensor_self, (batch_idx, mod_pos), modified_x0[0, mod_pos]
                        )
                        return "modified"
                return True
            if self._batch_contexts is not None and B > 1:
                # Batched: apply interventions and record per sample, then combine
                modified_list = []
                modified_x0_list = []
                any_intervention = False
                for b in range(B):
                    ctx = contexts[b]
                    mod_t, mod_x0, applied = ctx.apply_interventions(
                        transfer_index[b : b + 1],
                        x0_trace[b : b + 1],
                        x[b : b + 1],
                        self.mask_id,
                    )
                    ctx.record_committed_token_confidences(mod_x0)
                    ctx.record_transfer(mod_t, mod_x0)
                    modified_list.append(mod_t)
                    modified_x0_list.append(mod_x0)
                    if applied:
                        any_intervention = True
                modified_transfer = torch.cat(modified_list, dim=0)
                modified_x0 = torch.cat(modified_x0_list, dim=0)
                if any_intervention and _is_tensor_like(original_key) and original_key.dtype == torch.bool:
                    self._original_setitem(
                        tensor_self, modified_transfer, modified_x0[modified_transfer]
                    )
                    return "modified"
                if any_intervention and isinstance(original_key, tuple):
                    for b in range(B):
                        mod_pos = torch.nonzero(modified_transfer[b], as_tuple=False).squeeze(-1)
                        if mod_pos.numel() > 0:
                            self._original_setitem(
                                tensor_self,
                                (b, mod_pos),
                                modified_x0[b, mod_pos],
                            )
                    return "modified"
                return True
            else:
                ctx = contexts[0]
                modified_transfer, modified_x0, intervention_applied = (
                    ctx.apply_interventions(
                        transfer_index, x0_trace, x, self.mask_id
                    )
                )
                ctx.record_committed_token_confidences(modified_x0)
                ctx.record_transfer(modified_transfer, modified_x0)
                if intervention_applied:
                    modified_positions = torch.nonzero(modified_transfer[0], as_tuple=False).squeeze(-1)
                    if modified_positions.numel() > 0:
                        modified_values = modified_x0[0, modified_positions]
                        if isinstance(original_key, tuple):
                            batch_idx, _ = original_key
                            new_key = (batch_idx, modified_positions)
                            self._original_setitem(tensor_self, new_key, modified_values)
                            return "modified"
                        elif _is_tensor_like(original_key):
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
                return True
        finally:
            self._in_intervention = False
    
    def _handle_boolean_tensor_transfer(self, tensor_self, key, value):
        """Handle x[transfer_index] = x0[transfer_index] (batched or single-sample)."""
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
            
            if not same_2d_batch_shape([transfer_index, x0_trace, x]):
                return False

            batch_size = transfer_index.shape[0]
            if batch_size == 1 and not shapes_match([transfer_index, x0_trace, x]):
                return False

            if batch_size > 1:
                if self._batch_contexts is None or len(self._batch_contexts) != batch_size:
                    if not self._in_intervention:
                        warnings.warn(
                            "Batched boolean transfer (x[transfer_index] = x0[transfer_index]) "
                            f"with batch_size={batch_size} but no per-sample tracing contexts; "
                            "selected_positions will not be recorded."
                        )
                    return False
            
            return self._process_transfer(tensor_self, key, transfer_index, x0_trace, x)
        except Exception as e:
            if not self._in_intervention:
                warnings.warn(f"Tracing failed: {e}")
            return False
    
    def _handle_integer_indexing_transfer(self, tensor_self, key, value):
        """Handle x[b, pos] = value and x[b, positions] = values."""
        if not (isinstance(key, tuple) and len(key) == 2):
            return False

        batch_idx, pos_idx = key
        allow_batch_idx = (
            batch_idx == slice(None)
            or (isinstance(batch_idx, int) and (
                batch_idx == 0
                or (self._batch_contexts is not None and 0 <= batch_idx < len(self._batch_contexts))
            ))
        )
        if not allow_batch_idx:
            return False

        x = ensure_2d_shape(tensor_self.clone())
        # For batched integer indexing (x[b, pos]), slice to that row first so get_x0_trace returns [1, L]
        if isinstance(batch_idx, int) and self._batch_contexts is not None and x.shape[0] > 1:
            x = x[batch_idx : batch_idx + 1]
        x0_trace = get_x0_trace(self, x, batch_index=batch_idx if isinstance(batch_idx, int) and self._batch_contexts is not None else None)
        if isinstance(batch_idx, int) and self._batch_contexts is not None and x0_trace.shape[0] > 1:
            x0_trace = x0_trace[batch_idx : batch_idx + 1]

        transfer_info = None
        b_for_sel = batch_idx if isinstance(batch_idx, int) else 0
        if isinstance(pos_idx, int):
            transfer_info = self._detect_single_position_transfer(
                x, x0_trace, pos_idx, value, batch_index=b_for_sel
            )
        elif _is_tensor_like(pos_idx) and hasattr(pos_idx, 'dtype') and pos_idx.dtype in (torch.long, torch.int64, torch.int32, torch.int):
            transfer_info = self._detect_multiple_positions_transfer(x, x0_trace, pos_idx, value)

        if transfer_info is None:
            return False

        transfer_positions, transfer_values = transfer_info
        transfer_index = build_transfer_index(transfer_positions, x.shape[1], x.device)
        x0_for_trace = build_x0_for_trace(x0_trace, transfer_positions, transfer_values)

        if not shapes_match([transfer_index, x0_for_trace, x]):
            return False

        batch_index = batch_idx if isinstance(batch_idx, int) and self._batch_contexts is not None else None
        return self._process_transfer(tensor_self, key, transfer_index, x0_for_trace, x, batch_index=batch_index)
    
    def _intercept_tensor_setitem(self, original_setitem):
        """Patch tensor __setitem__ to detect unmask writes."""
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
        """Run the scheduler with tracing hooks installed."""
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
            if intervention_steps is None:
                intervention_steps = set()
            elif isinstance(intervention_steps, (list, tuple)):
                intervention_steps = set(intervention_steps)
            elif isinstance(intervention_steps, set):
                pass
            else:
                intervention_steps = {intervention_steps}
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

        # Extract prompt (usually second arg) and set up batch contexts when batch_size > 1
        prompt = args[1] if len(args) > 1 else kwargs.get('prompt')
        block_length = kwargs.get('block_length', None)
        gen_length_kw = kwargs.get('gen_length')
        task_trace_list = _task_trace_list_ctx.get()
        if prompt is not None:
            prompt_len = int(prompt.shape[1])
            if gen_length_kw is not None:
                gen_len = int(gen_length_kw)
                self.tracing_context.set_generation_bounds(prompt_len, gen_len)
            batch_size = prompt.shape[0]
            use_batch_contexts = (
                batch_size > 1
                and (
                    self.tracing_context.trace_name is not None
                    or task_trace_list is not None
                )
            )
            if use_batch_contexts:
                base_name = self.tracing_context.trace_name
                self._batch_size = batch_size
                self._batch_contexts = []
                for b in range(batch_size):
                    ctx = TracingContext(
                        intervention_hook=getattr(self.tracing_context, 'intervention_hook', None),
                        trace_name=f"{base_name}_b{b}",
                        upload_to_wandb=self.tracing_context.upload_to_wandb,
                        save_to_disk=self.tracing_context.save_to_disk,
                        base_dir=self.tracing_context.base_dir,
                    )
                    ctx.trace.prompt = prompt[b : b + 1].clone()
                    if gen_length_kw is not None:
                        ctx.set_generation_bounds(prompt_len, int(gen_length_kw))
                    if block_length is not None:
                        ctx._block_length = block_length
                    self._batch_contexts.append(ctx)
            else:
                if self.tracing_context.trace.prompt is None:
                    self.tracing_context.trace.prompt = prompt.clone()
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
        
        token = _active_tracer_ctx.set(self)
        try:
            # Call original scheduler
            result = self.scheduler_fn(*args, **kwargs)
        finally:
            _active_tracer_ctx.reset(token)
            model.forward = self._original_forward
            torch.argmax = self._original_argmax
            torch.topk = self._original_topk
            torch.Tensor.__setitem__ = self._original_setitem
        
        # Extract final sequence
        if isinstance(result, tuple):
            final_seq = result[0] if len(result) > 0 else None
        else:
            final_seq = result
        
        end_time = time.time()
        task_trace_list = _task_trace_list_ctx.get()

        if self._batch_contexts is not None and final_seq is not None:
            for b, ctx in enumerate(self._batch_contexts):
                ctx.trace.final_sequence = final_seq[b : b + 1].clone()
                ctx.record_timestamps(start_time, end_time)
                if task_trace_list is not None:
                    try:
                        task_trace_list.append(ctx.trace.to_dict())
                    except Exception as e:
                        warnings.warn(f"Failed to append batch trace {b} to task list: {e}")
                else:
                    if ctx.upload_to_wandb and ctx.trace_name is not None:
                        try:
                            ctx.trace.upload_to_wandb(ctx.trace_name, base_dir=ctx.base_dir)
                        except Exception as e:
                            warnings.warn(f"Failed to upload batch trace {b} to wandb: {e}")
                    if ctx.trace_name is not None and ctx.save_to_disk and not ctx.upload_to_wandb:
                        try:
                            ctx.save_trace()
                        except Exception as e:
                            warnings.warn(f"Failed to save batch trace {b} to disk: {e}")
        else:
            if final_seq is not None and self.tracing_context.trace.final_sequence is None:
                self.tracing_context.trace.final_sequence = final_seq.clone()
            self.tracing_context.record_timestamps(start_time, end_time)
            if task_trace_list is not None:
                try:
                    task_trace_list.append(self.tracing_context.trace.to_dict())
                except Exception as e:
                    warnings.warn(f"Failed to append trace to task list: {e}")
            else:
                if self.tracing_context.upload_to_wandb and self.tracing_context.trace_name is not None:
                    try:
                        self.tracing_context.trace.upload_to_wandb(
                            self.tracing_context.trace_name,
                            base_dir=self.tracing_context.base_dir
                        )
                    except Exception as e:
                        warnings.warn(f"Failed to upload trace to wandb: {e}")
                should_save_to_disk = (
                    self.tracing_context.trace_name is not None
                    and self.tracing_context.save_to_disk
                    and not self.tracing_context.upload_to_wandb
                )
                if should_save_to_disk:
                    try:
                        self.tracing_context.save_trace()
                    except Exception as e:
                        warnings.warn(f"Failed to save trace: {e}")

        return result


def trace_scheduler(
    scheduler_fn: Optional[Callable] = None,
    mask_id: int = LLADA_DEFAULT_MASK_ID,
    intervention_hook: Optional[InterventionHook] = None,
    trace_name: Optional[str] = None,
    save_predictions: bool = True,
    save_confidences: bool = True,
    upload_to_wandb: bool = False
):
    """Decorator that traces a scheduler without editing its source."""
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
            
            effective_mask_id = kwargs.get("mask_id", mask_id)
            # Create model tracer
            tracer = ModelTracer(
                func,
                mask_id=effective_mask_id,
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
