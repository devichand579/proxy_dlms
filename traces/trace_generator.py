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
    metadata: Dict = field(default_factory=dict)  # Additional metadata, can store values to be tracked like mutual information, attention scores, etc.


@dataclass
class GenerationTrace:
    """Complete trace of a generation process."""
    prompt: torch.Tensor
    final_sequence: torch.Tensor
    steps: List[StepTrace] = field(default_factory=list)
    unmasking_order: Dict[int, int] = field(default_factory=dict)  # position -> step when unmasked, use this to calculate Kendall's tau
    interventions: List[Dict] = field(default_factory=list)  # List of interventions applied
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
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
    
    @property
    def wall_time(self) -> Optional[float]:
        if self.start_time is None or self.end_time is None:
            return None
        return self.end_time - self.start_time

class InterventionHook:
    def __init__(
        self,
        mode: Optional[str] = None,   # None | "drop" | "swap"
        steps: Optional[set] = None,  # steps to intervene
        k: int = 1                    # number of tokens to drop/swap
    ):
        self.mode = mode
        self.steps = steps or set()
        self.k = k

    def active(self, step: int) -> bool:
        return self.mode is not None and step in self.steps

class TracingContext:
    """
    Tracing + causal intervention context.
    """

    def __init__(self, intervention_hook: Optional[InterventionHook] = None):
        self.trace = GenerationTrace(prompt=None, final_sequence=None)
        self.intervention_hook = intervention_hook or InterventionHook()
        self.current_step = 0

        self._current_predictions = None
        self._current_confidences = None

    def record_step_start(self, step: int, x: torch.Tensor, mask_id: int):
        self.current_step = step
        mask_index = (x == mask_id)

        step_trace = StepTrace(
            step=step,
            mask_index=mask_index.clone()
        )
        self.trace.steps.append(step_trace)

    def record_logits(self, logits: torch.Tensor):
        if self.trace.steps:
            self.trace.steps[-1].logits = logits.clone()
            
    def record_timestamps(self, start_time: float, end_time: float):
        self.trace.start_time = start_time
        self.trace.end_time = end_time

    def record_predictions(self, predictions: torch.Tensor, confidences: torch.Tensor):
        if self.trace.steps:
            self.trace.steps[-1].predictions = predictions.clone()
            self.trace.steps[-1].confidences = confidences.clone()
            self._current_predictions = predictions
            self._current_confidences = confidences

    def apply_interventions(
        self,
        transfer_index: torch.Tensor,  # scheduler decision [B, L] bool
        x0: torch.Tensor,              # predicted tokens [B, L]
        x: torch.Tensor,               # current sequence
        mask_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:

        if not self.trace.steps:
            return transfer_index, x0, False

        hook = self.intervention_hook
        step_trace = self.trace.steps[-1]
        mask_index = step_trace.mask_index

        modified_transfer = transfer_index.clone()
        modified_x0 = x0.clone()
        intervention_applied = False

        if not hook.active(self.current_step):
            return modified_transfer, modified_x0, False

        S = torch.nonzero(transfer_index[0], as_tuple=False).squeeze(-1)

        R = torch.nonzero(
            mask_index[0] & (~transfer_index[0]),
            as_tuple=False
        ).squeeze(-1)

        if hook.mode == "drop":
            if S.numel() == 0:
                return modified_transfer, modified_x0, False

            k = min(hook.k, S.numel())
            perm = torch.randperm(S.numel(), device=S.device)
            dropped = S[perm[:k]]

            modified_transfer[0, dropped] = False

            intervention_applied = True
            step_trace.intervention_type = "drop"
            step_trace.metadata["dropped_positions"] = dropped.tolist()
            step_trace.metadata["k"] = k
            step_trace.metadata["S_size"] = int(S.numel())

        elif hook.mode == "swap":
            if S.numel() == 0 or R.numel() == 0:
                return modified_transfer, modified_x0, False

            k = min(hook.k, S.numel(), R.numel())
            perm_S = torch.randperm(S.numel(), device=S.device)
            perm_R = torch.randperm(R.numel(), device=R.device)

            removed = S[perm_S[:k]]
            added = R[perm_R[:k]]

            modified_transfer[0, removed] = False

            modified_transfer[0, added] = True
            modified_x0[0, added] = self._current_predictions[0, added]

            intervention_applied = True
            step_trace.intervention_type = "swap"
            step_trace.metadata["swapped_out"] = removed.tolist()
            step_trace.metadata["swapped_in"] = added.tolist()
            step_trace.metadata["k"] = k
            step_trace.metadata["S_size"] = int(S.numel())
            step_trace.metadata["R_size"] = int(R.numel())

        step_trace.intervention_applied = intervention_applied
        return modified_transfer, modified_x0, intervention_applied

    def record_transfer(self, transfer_index: torch.Tensor, x0: torch.Tensor):
        if not self.trace.steps:
            return

        step_trace = self.trace.steps[-1]
        step_trace.selected_positions = transfer_index.clone()

        selected_tokens = torch.zeros_like(x0)
        selected_tokens[transfer_index] = x0[transfer_index]
        step_trace.selected_tokens = selected_tokens.clone()

        positions = torch.nonzero(transfer_index[0], as_tuple=False).squeeze(-1)
        if positions.dim() == 0:
            positions = positions.unsqueeze(0)

        for pos in positions.tolist():
            if pos not in self.trace.unmasking_order:
                self.trace.unmasking_order[pos] = self.current_step

def create_intervention_hook(
    mode: Optional[str] = None,      # None | "drop" | "swap"
    steps: Optional[List[int]] = None,
    k: int = 1
) -> InterventionHook:
    """
    Create a scheduler-relative intervention hook.

    Args:
        mode: "drop" or "swap"
        steps: list of step indices at which to intervene
        k: number of tokens to drop / swap at each step
    """
    return InterventionHook(
        mode=mode,
        steps=set(steps) if steps is not None else set(),
        k=k
    )