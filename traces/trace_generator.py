"""
General tracing and causal intervention system for masked diffusion schedulers.

This module provides a plug-and-play tracing wrapper that can:
1. Record all unmasking decisions during generation
2. Perform causal interventions (force tokens early/late)
3. Analyze which tokens are bottlenecks

Works with any scheduler by intercepting token selection through a callback mechanism.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
import copy


@dataclass
class StepTrace:
    """Records the state at a single generation step."""
    step: int
    mask_index: torch.Tensor  # Which positions are still masked [B, L]
    logits: Optional[torch.Tensor] = None  # Model predictions [B, L, V]
    predictions: Optional[torch.Tensor] = None  # Predicted tokens [B, L]
    confidences: Optional[torch.Tensor] = None  # Confidence scores [B, L]
    selected_positions: Optional[torch.Tensor] = None  # Positions that were actually unmasked [B, L] bool
    selected_tokens: Optional[torch.Tensor] = None  # Tokens that were unmasked [B, L]
    intervention_applied: bool = False  # Whether intervention was applied this step
    intervention_type: Optional[str] = None  # Type of intervention if applied
    metadata: Dict = field(default_factory=dict)  # Additional metadata


@dataclass
class GenerationTrace:
    """Complete trace of a generation process."""
    prompt: torch.Tensor
    final_sequence: torch.Tensor
    steps: List[StepTrace] = field(default_factory=list)
    unmasking_order: Dict[int, int] = field(default_factory=dict)  # position -> step when unmasked
    interventions: List[Dict] = field(default_factory=list)  # List of interventions applied
    
    def get_unmasking_step(self, position: int) -> Optional[int]:
        """Get the step at which a position was unmasked."""
        return self.unmasking_order.get(position)
    
    def get_token_at_step(self, step: int, position: int) -> Optional[int]:
        """Get the token at a specific position at a specific step."""
        if step < len(self.steps):
            if self.steps[step].predictions is not None:
                return int(self.steps[step].predictions[0, position].item())
        return None
    
    def get_all_unmasked_positions(self) -> List[int]:
        """Get all positions that were unmasked, in order."""
        return sorted(self.unmasking_order.keys(), key=lambda p: self.unmasking_order[p])


class InterventionHook:
    """
    Hook to intercept and modify token selection decisions.
    
    This allows causal interventions like:
    - Force token at position i to unmask at step s (early intervention)
    - Prevent token at position i from unmasking until step s (late intervention)
    - Force specific token value at position i
    """
    
    def __init__(self):
        self.force_unmask: Dict[Tuple[int, int], int] = {}  # (step, position) -> token_id
        self.force_unmask_early: Dict[int, int] = {}  # position -> token_id (unmask as early as possible)
        self.force_unmask_late: Dict[int, int] = {}  # position -> step (don't unmask until this step)
        self.prevent_unmask: set = set()  # positions to prevent from unmasking
        self.force_token_value: Dict[int, int] = {}  # position -> token_id (force specific token)
        
    def should_unmask(self, step: int, position: int, current_mask: torch.Tensor, 
                      predictions: torch.Tensor, confidences: torch.Tensor) -> Tuple[bool, Optional[int]]:
        """
        Determine if a position should be unmasked at this step.
        
        Returns:
            (should_unmask, forced_token_id)
            - should_unmask: True/False to force, None to let scheduler decide
            - forced_token_id: If not None, use this token instead of prediction
        """
        # Check if we should prevent unmasking
        if position in self.prevent_unmask:
            return False, None
        
        # Check if we should force unmask at this specific step
        if (step, position) in self.force_unmask:
            forced_token = self.force_unmask[(step, position)]
            return True, forced_token
        
        # Check if we should force unmask early (first time this position is considered)
        if position in self.force_unmask_early:
            if current_mask[0, position]:  # Still masked
                forced_token = self.force_unmask_early[position]
                return True, forced_token
        
        # Check if we should force unmask late (only after certain step)
        if position in self.force_unmask_late:
            required_step = self.force_unmask_late[position]
            if step < required_step:
                return False, None
        
        # Check if we should force a specific token value (when unmasked)
        if position in self.force_token_value:
            forced_token = self.force_token_value[position]
            # If already unmasked, we'll apply the forced value
            if not current_mask[0, position]:
                return True, forced_token
        
        # Default: let scheduler decide
        return None, None
    
    def apply_forced_token(self, position: int, predictions: torch.Tensor) -> torch.Tensor:
        """Apply forced token value if specified."""
        if position in self.force_token_value:
            predictions[0, position] = self.force_token_value[position]
        return predictions


class TracingContext:
    """
    Context manager for tracing generation.
    
    This is injected into schedulers to record all decisions.
    """
    
    def __init__(self, intervention_hook: Optional[InterventionHook] = None):
        self.trace = GenerationTrace(prompt=None, final_sequence=None)
        self.intervention_hook = intervention_hook or InterventionHook()
        self.current_step = 0
        self._current_logits = None
        self._current_predictions = None
        self._current_confidences = None
        
    def record_step_start(self, step: int, x: torch.Tensor, mask_id: int):
        """Record the start of a generation step."""
        self.current_step = step
        mask_index = (x == mask_id)
        
        step_trace = StepTrace(
            step=step,
            mask_index=mask_index.clone(),
            logits=None,
            predictions=None,
            confidences=None,
            selected_positions=None,
            selected_tokens=None
        )
        self.trace.steps.append(step_trace)
        
    def record_logits(self, logits: torch.Tensor):
        """Record model logits for current step."""
        if self.trace.steps:
            self.trace.steps[-1].logits = logits.clone()
            self._current_logits = logits
    
    def record_predictions(self, predictions: torch.Tensor, confidences: torch.Tensor):
        """Record predictions and confidences for current step."""
        if self.trace.steps:
            self.trace.steps[-1].predictions = predictions.clone()
            self.trace.steps[-1].confidences = confidences.clone()
            self._current_predictions = predictions
            self._current_confidences = confidences
    
    def apply_interventions(self, transfer_index: torch.Tensor, x0: torch.Tensor, 
                           x: torch.Tensor, mask_id: int) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Apply interventions to token selection.
        
        Returns:
            (modified_transfer_index, modified_x0, intervention_applied)
        """
        if not self.trace.steps:
            return transfer_index, x0, False
        
        step_trace = self.trace.steps[-1]
        mask_index = step_trace.mask_index
        modified_transfer = transfer_index.clone()
        modified_x0 = x0.clone()
        intervention_applied = False
        
        # Get positions that would be unmasked
        positions_to_unmask = torch.nonzero(transfer_index[0], as_tuple=False).squeeze(-1)
        if positions_to_unmask.dim() == 0:
            positions_to_unmask = positions_to_unmask.unsqueeze(0)
        positions_list = positions_to_unmask.tolist()
        
        # Apply interventions
        for pos in positions_list:
            should_unmask, forced_token = self.intervention_hook.should_unmask(
                self.current_step, pos, mask_index, modified_x0, step_trace.confidences
            )
            
            if should_unmask is False:
                # Prevent unmasking
                modified_transfer[0, pos] = False
                intervention_applied = True
                step_trace.intervention_type = "prevent"
            elif should_unmask is True:
                # Force unmasking
                modified_transfer[0, pos] = True
                if forced_token is not None:
                    modified_x0[0, pos] = forced_token
                    intervention_applied = True
                    step_trace.intervention_type = "force_token"
                else:
                    intervention_applied = True
                    step_trace.intervention_type = "force_unmask"
        
        # Check for early unmasking (positions that should be unmasked but weren't selected)
        if step_trace.confidences is not None:
            for pos in range(mask_index.shape[1]):
                if mask_index[0, pos]:  # Still masked
                    should_unmask, forced_token = self.intervention_hook.should_unmask(
                        self.current_step, pos, mask_index, modified_x0, step_trace.confidences
                    )
                    if should_unmask is True:
                        modified_transfer[0, pos] = True
                        if forced_token is not None:
                            modified_x0[0, pos] = forced_token
                        intervention_applied = True
                        step_trace.intervention_type = "force_early"
        
        step_trace.intervention_applied = intervention_applied
        return modified_transfer, modified_x0, intervention_applied
    
    def record_transfer(self, transfer_index: torch.Tensor, x0: torch.Tensor, x: torch.Tensor):
        """Record which tokens were actually transferred."""
        if self.trace.steps:
            step_trace = self.trace.steps[-1]
            step_trace.selected_positions = transfer_index.clone()
            
            # Record selected tokens
            selected_tokens = torch.zeros_like(x0)
            selected_tokens[transfer_index] = x0[transfer_index]
            step_trace.selected_tokens = selected_tokens.clone()
            
            # Update unmasking order
            positions = torch.nonzero(transfer_index[0], as_tuple=False).squeeze(-1)
            if positions.dim() == 0:
                positions = positions.unsqueeze(0)
            for pos in positions.tolist():
                if pos not in self.trace.unmasking_order:
                    self.trace.unmasking_order[pos] = self.current_step


def create_intervention_hook(
    force_early: Optional[Dict[int, int]] = None,
    force_late: Optional[Dict[int, int]] = None,
    force_at_step: Optional[Dict[Tuple[int, int], int]] = None,
    prevent: Optional[List[int]] = None,
    force_tokens: Optional[Dict[int, int]] = None
) -> InterventionHook:
    """
    Create an intervention hook with specified interventions.
    
    Args:
        force_early: {position: token_id} - Force unmask these positions as early as possible
        force_late: {position: step} - Don't unmask these positions until this step
        force_at_step: {(step, position): token_id} - Force unmask at specific step
        prevent: List of positions to prevent from unmasking
        force_tokens: {position: token_id} - Force specific token values when unmasked
        
    Returns:
        InterventionHook configured with the interventions
    """
    hook = InterventionHook()
    
    if force_early:
        hook.force_unmask_early.update(force_early)
    
    if force_late:
        hook.force_unmask_late.update(force_late)
    
    if force_at_step:
        hook.force_unmask.update(force_at_step)
    
    if prevent:
        hook.prevent_unmask.update(prevent)
    
    if force_tokens:
        hook.force_token_value.update(force_tokens)
    
    return hook
