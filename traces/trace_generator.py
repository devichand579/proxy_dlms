"""
General tracing and causal intervention system for masked diffusion schedulers.
Works with any scheduler by intercepting token selection through a callback mechanism.
"""

import torch
import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


def _tensor_to_list(tensor: Optional[torch.Tensor]) -> Optional[List]:
    """Convert torch tensor to JSON-serializable list."""
    if tensor is None:
        return None
    # Move to CPU
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Convert unsupported dtypes to supported ones for JSON serialization
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float32)
    elif tensor.dtype == torch.float16:
        tensor = tensor.to(torch.float32)
    elif tensor.dtype == torch.complex64 or tensor.dtype == torch.complex128:
        # Convert complex to real
        tensor = tensor.real.to(torch.float32)
    
    # Convert to numpy, then to list
    return tensor.detach().numpy().tolist()

def _list_to_tensor(data: Optional[List], dtype=None) -> Optional[torch.Tensor]:
    """Convert list back to torch tensor."""
    if data is None:
        return None
    # Try to infer dtype from data if not provided
    if dtype is None:
        # Check if it's boolean-like 
        if isinstance(data, list) and len(data) > 0:
            sample = data[0] if not isinstance(data[0], list) else data[0][0] if isinstance(data[0], list) and len(data[0]) > 0 else None
            if sample is not None and isinstance(sample, bool):
                dtype = torch.bool
    tensor = torch.tensor(data, dtype=dtype)
    return tensor

@dataclass
class StepTrace:
    """Records the state at a single generation step."""
    step: int
    mask_index: torch.Tensor  # Which positions are still masked [1, L]
    predictions: Optional[torch.Tensor] = None  # Predicted tokens [1, L]
    confidences: Optional[torch.Tensor] = None  # Confidence scores [1, L]
    selected_positions: Optional[torch.Tensor] = None  # Positions that were actually unmasked [1, L]
    selected_tokens: Optional[torch.Tensor] = None  # Tokens that were unmasked [1, L]
    intervention_applied: bool = False  # Whether intervention was applied this step
    intervention_type: Optional[str] = None  # Type of intervention if applied
    metadata: Dict = field(default_factory=dict)  # Additional metadata, This can store values to be tracked like mutual information, attention scores, etc.
    
    def to_dict(self) -> Dict:
        """Convert StepTrace to JSON-serializable dictionary."""
        return {
            "step": self.step,
            "mask_index": _tensor_to_list(self.mask_index),
            "predictions": _tensor_to_list(self.predictions) if self.predictions is not None else None,
            "confidences": _tensor_to_list(self.confidences) if self.confidences is not None else None,
            "selected_positions": _tensor_to_list(self.selected_positions) if self.selected_positions is not None else None,
            "selected_tokens": _tensor_to_list(self.selected_tokens) if self.selected_tokens is not None else None,
            "intervention_applied": self.intervention_applied,
            "intervention_type": self.intervention_type,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "StepTrace":
        """Create StepTrace from dictionary."""
        return cls(
            step=data["step"],
            mask_index=_list_to_tensor(data["mask_index"], dtype=torch.bool),
            predictions=_list_to_tensor(data.get("predictions"), dtype=torch.long),
            confidences=_list_to_tensor(data.get("confidences"), dtype=torch.float32),
            selected_positions=_list_to_tensor(data.get("selected_positions"), dtype=torch.bool),
            selected_tokens=_list_to_tensor(data.get("selected_tokens"), dtype=torch.long),
            intervention_applied=data.get("intervention_applied", False),
            intervention_type=data.get("intervention_type"),
            metadata=data.get("metadata", {})
        )


@dataclass
class GenerationTrace:
    """Complete trace of a generation process."""
    prompt: torch.Tensor
    final_sequence: torch.Tensor
    step_traces: List[StepTrace] = field(default_factory=list)
    unmasking_order: Dict[int, int] = field(default_factory=dict)  # position -> step when unmasked, use this to calculate Kendall's tau
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    def get_unmasking_step(self, position: int) -> Optional[int]:
        """Get the step at which a position was unmasked."""
        return self.unmasking_order.get(position)
    
    def get_token_at_step(self, step: int, position: int) -> Optional[int]:
        """Get the token at a specific position at a specific step."""
        if step < len(self.step_traces):
            if self.step_traces[step].predictions is not None:
                return int(self.step_traces[step].predictions[0, position].item())
        return None
    
    def get_all_unmasked_positions(self) -> List[int]:
        """Get all positions that were unmasked, in order."""
        return sorted(self.unmasking_order.keys(), key=lambda p: self.unmasking_order[p])
    
    @property
    def wall_time(self) -> Optional[float]:
        if self.start_time is None or self.end_time is None:
            return None
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict:
        """Convert GenerationTrace to JSON-serializable dictionary."""
        return {
            "prompt": _tensor_to_list(self.prompt) if self.prompt is not None else None,
            "final_sequence": _tensor_to_list(self.final_sequence) if self.final_sequence is not None else None,
            "step_traces": [st.to_dict() for st in self.step_traces],
            "unmasking_order": self.unmasking_order,
            "start_time": self.start_time,
            "end_time": self.end_time
        }
    
    def save_to_json(self, name: str, base_dir: str = "trace_outputs") -> str:
        """
        Save trace to JSON file.
        """
        os.makedirs(base_dir, exist_ok=True)
        filepath = os.path.join(base_dir, f"{name}.json")
        
        trace_dict = self.to_dict()
        with open(filepath, 'w') as f:
            json.dump(trace_dict, f, indent=2)
        
        return filepath
    
    def upload_to_wandb(self, name: Optional[str] = None, base_dir: str = "trace_outputs"):
        """
        Upload trace to wandb as an artifact.
        """
        try:
            import wandb
        except ImportError:
            import warnings
            warnings.warn("wandb not available. Install with: pip install wandb")
            return
        
        if wandb.run is None:
            import warnings
            warnings.warn("wandb run not initialized. Cannot upload trace.")
            return
        
        artifact_name = name or "trace"
        
        # Create wandb artifact
        artifact = wandb.Artifact(
            name=f"trace_{artifact_name}",
            type="trace",
            description=f"Generation trace: {artifact_name}"
        )
        
        # Save trace to JSON file in trace_outputs directory
        os.makedirs(base_dir, exist_ok=True)
        filepath = os.path.join(base_dir, f"{artifact_name}.json")
        
        trace_dict = self.to_dict()
        with open(filepath, 'w') as f:
            json.dump(trace_dict, f, indent=2)
        
        # Add file to artifact and upload
        artifact.add_file(filepath, name=f"{artifact_name}.json")
        wandb.log_artifact(artifact)

    
    @classmethod
    def load_from_json(cls, name: str, base_dir: str = "trace_outputs") -> "GenerationTrace":
        """
        Load trace from JSON file.
        """
        filepath = os.path.join(base_dir, f"{name}.json")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Trace file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            trace_dict = json.load(f)
        
        # Reconstruct the trace
        trace = cls(
            prompt=_list_to_tensor(trace_dict.get("prompt"), dtype=torch.long) if trace_dict.get("prompt") is not None else None,
            final_sequence=_list_to_tensor(trace_dict.get("final_sequence"), dtype=torch.long) if trace_dict.get("final_sequence") is not None else None,
            step_traces=[StepTrace.from_dict(st) for st in trace_dict.get("step_traces", [])],
            unmasking_order=trace_dict.get("unmasking_order", {}),
            start_time=trace_dict.get("start_time"),
            end_time=trace_dict.get("end_time")
        )
        
        return trace

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

    def __init__(
        self, 
        intervention_hook: Optional[InterventionHook] = None,
        save_to_disk: bool = True,
        trace_name: Optional[str] = None,
        base_dir: str = "trace_outputs",
        save_predictions: bool = True,
        save_confidences: bool = True,
        upload_to_wandb: bool = False
    ):
        self.trace = GenerationTrace(prompt=None, final_sequence=None)
        self.intervention_hook = intervention_hook or InterventionHook()
        self.current_step = 0

        self._current_predictions = None
        self._current_confidences = None
        self._current_logits = None  # Store logits for computing predictions on-demand
        
        # Block boundaries 
        self._current_block_start = None
        self._current_block_end = None
        self._block_length = None  
        
        # Disk saving options
        self.save_to_disk = save_to_disk
        self.trace_name = trace_name
        self.base_dir = base_dir
        self._saved_steps = 0  # Track how many steps have been saved
        
        # Performance options
        self.save_predictions = save_predictions
        self.save_confidences = save_confidences
        self.upload_to_wandb = upload_to_wandb

    def record_step_start(self, step: int, x: torch.Tensor, mask_id: int):
        self.current_step = step
        mask_index = (x == mask_id)

        step_trace = StepTrace(
            step=step,
            mask_index=mask_index.clone()
        )
        self.trace.step_traces.append(step_trace)
        
        # Reset block boundaries at start of each step 
        self._current_block_start = None
        self._current_block_end = None

    def record_timestamps(self, start_time: float, end_time: float):
        self.trace.start_time = start_time
        self.trace.end_time = end_time

    def record_predictions(self, predictions: torch.Tensor, confidences: torch.Tensor):
        if self.trace.step_traces:
            if self.save_predictions:
                self.trace.step_traces[-1].predictions = predictions.clone()
            if self.save_confidences:
                self.trace.step_traces[-1].confidences = confidences.clone()
            self._current_predictions = predictions
            self._current_confidences = confidences

    def apply_interventions(
        self,
        transfer_index: torch.Tensor,  # scheduler decision [1, L] bool
        x0: torch.Tensor,              # predicted tokens [1, L]
        x: torch.Tensor,               # current sequence
        mask_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:

        if not self.trace.step_traces:
            return transfer_index, x0, False

        hook = self.intervention_hook
        step_trace = self.trace.step_traces[-1]
        mask_index = step_trace.mask_index

        modified_transfer = transfer_index.clone()
        modified_x0 = x0.clone()
        intervention_applied = False

        if not hook.active(self.current_step):
            return modified_transfer, modified_x0, False

        S = torch.nonzero(transfer_index[0], as_tuple=False).squeeze(-1)

        # For swap: use block boundaries detected from scheduler operations
        # block_end is detected from slice assignment (x0_p[:, block_end:] = -np.inf)
        # block_start is calculated as block_end - block_length
        block_start = self._current_block_start
        block_end = self._current_block_end

        R_all = torch.nonzero(
            mask_index[0] & (~transfer_index[0]),
            as_tuple=False
        ).squeeze(-1)
        
        # For swap: filter R to only include positions within the same block as S
        if hook.mode == "swap" and block_start is not None and block_end is not None:
            R = R_all[(R_all >= block_start) & (R_all < block_end)]
        else:
            R = R_all

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
            
            # Get predictions for added positions
            # For schedulers like FHS that don't use argmax, compute from logits
            if self._current_predictions is not None:
                modified_x0[0, added] = self._current_predictions[0, added]
            elif self._current_logits is not None:
                # Compute predictions from logits (for FHS-style schedulers)
                predictions = torch.argmax(self._current_logits, dim=-1)  # [1, L]
                modified_x0[0, added] = predictions[0, added]
            else:
                # Fallback: use x0 parameter if available
                if x0.shape[1] > added.max().item():
                    modified_x0[0, added] = x0[0, added]
                else:
                    # Can't swap - no predictions available
                    modified_transfer[0, added] = False
                    return modified_transfer, modified_x0, False

            intervention_applied = True
            step_trace.intervention_type = "swap"
            step_trace.metadata["swapped_out"] = removed.tolist()
            step_trace.metadata["swapped_in"] = added.tolist()
            step_trace.metadata["k"] = k
            step_trace.metadata["S_size"] = int(S.numel())
            step_trace.metadata["R_size"] = int(R.numel())
            if block_start is not None and block_end is not None:
                step_trace.metadata["block_start"] = block_start
                step_trace.metadata["block_end"] = block_end

        step_trace.intervention_applied = intervention_applied
        return modified_transfer, modified_x0, intervention_applied

    def record_transfer(self, transfer_index: torch.Tensor, x0: torch.Tensor):
        if not self.trace.step_traces:
            return

        step_trace = self.trace.step_traces[-1]
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
    
    def save_trace(self, name: Optional[str] = None) -> str:
        """
        Save the current trace to JSON file.
        """
        save_name = name or self.trace_name
        if save_name is None:
            raise ValueError("trace_name must be provided in __init__ or as argument to save_trace()")
        
        return self.trace.save_to_json(save_name, self.base_dir)
    
    def save_incremental(self, clear_memory: bool = True):
        """
        Save step traces incrementally to reduce memory usage.
        Saves completed steps to disk and optionally clears them from memory.
        """
        if not self.save_to_disk or self.trace_name is None:
            return
        
        if not self.trace.step_traces:
            return
        
        # Save only new steps since last save
        steps_to_save = self.trace.step_traces[self._saved_steps:]
        if not steps_to_save:
            return
        
        # Create incremental save file
        inc_filepath = os.path.join(self.base_dir, f"{self.trace_name}_steps_{self._saved_steps}_{len(self.trace.step_traces)}.json")
        os.makedirs(self.base_dir, exist_ok=True)
        
        steps_data = [st.to_dict() for st in steps_to_save]
        with open(inc_filepath, 'w') as f:
            json.dump(steps_data, f, indent=2)
        
        # Delete saved step traces from memory if requested
        if clear_memory:
            del self.trace.step_traces[self._saved_steps:]
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self._saved_steps = len(self.trace.step_traces)

def create_intervention_hook(
    mode: Optional[str] = None,      # None | "drop" | "swap"
    steps: Optional[List[int]] = None,
    k: int = 1
) -> InterventionHook:
    """
    Create a scheduler-relative intervention hook.
    """
    return InterventionHook(
        mode=mode,
        steps=set(steps) if steps is not None else set(),
        k=k
    )