"""Tracing and intervention hooks for masked diffusion schedulers."""

import torch
import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


def gen_region_slice(prompt_len: int, gen_length: int) -> slice:
    """Slice index for the generation region."""
    return slice(int(prompt_len), int(prompt_len) + int(gen_length))


def slice_gen_region(
    tensor: torch.Tensor, prompt_len: int, gen_length: int
) -> torch.Tensor:
    """Generation-tail slice of a batch tensor."""
    return tensor[:, gen_region_slice(prompt_len, gen_length)].contiguous()


def logits_to_step_metrics(
    logits: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predictions, max-prob confidences, and entropy from logits.

    Sequence chunks avoid a second full-vocabulary tensor next to ``logits``.
    """
    length = int(logits.shape[1])
    predictions = torch.empty(logits.shape[:2], dtype=torch.long, device=logits.device)
    confidences = torch.empty(logits.shape[:2], dtype=torch.float32, device=logits.device)
    entropies = torch.empty(logits.shape[:2], dtype=torch.float32, device=logits.device)
    with torch.no_grad():
        for start in range(0, length, 32):
            end = min(length, start + 32)
            piece = logits[:, start:end]
            predictions[:, start:end] = piece.argmax(dim=-1)
            probs = torch.softmax(piece.float(), dim=-1)
            confidences[:, start:end] = probs.max(dim=-1).values
            entropies[:, start:end] = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
    return predictions, confidences, entropies


def _tensor_to_list(tensor: Optional[torch.Tensor]) -> Optional[List]:
    """Tensor to JSON-serializable list."""
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
    """JSON list to tensor."""
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
    """One diffusion step in the generation region."""
    step: int
    mask_index: torch.Tensor  # Still masked [1, gen_length]
    predictions: Optional[torch.Tensor] = None  # Argmax token id [1, gen_length]
    # Max softmax probability initially; after a transfer this is the raw-model
    # probability of the sampled proposal, so selected positions match committed tokens.
    confidences: Optional[torch.Tensor] = None  # [1, gen_length]
    entropies: Optional[torch.Tensor] = None  # Shannon entropy [1, gen_length]
    selected_positions: Optional[torch.Tensor] = None  # Unmasked this step [1, gen_length]
    selected_tokens: Optional[torch.Tensor] = None  # Tokens unmasked this step [1, gen_length]
    intervention_applied: bool = False
    intervention_type: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Step as a JSON dict."""
        return {
            "step": self.step,
            "mask_index": _tensor_to_list(self.mask_index),
            "predictions": _tensor_to_list(self.predictions) if self.predictions is not None else None,
            "confidences": _tensor_to_list(self.confidences) if self.confidences is not None else None,
            "entropies": _tensor_to_list(self.entropies) if self.entropies is not None else None,
            "selected_positions": _tensor_to_list(self.selected_positions) if self.selected_positions is not None else None,
            "selected_tokens": _tensor_to_list(self.selected_tokens) if self.selected_tokens is not None else None,
            "intervention_applied": self.intervention_applied,
            "intervention_type": self.intervention_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "StepTrace":
        """StepTrace from a JSON dict."""
        return cls(
            step=data["step"],
            mask_index=_list_to_tensor(data["mask_index"], dtype=torch.bool),
            predictions=_list_to_tensor(data.get("predictions"), dtype=torch.long),
            confidences=_list_to_tensor(data.get("confidences"), dtype=torch.float32),
            entropies=_list_to_tensor(data.get("entropies"), dtype=torch.float32),
            selected_positions=_list_to_tensor(data.get("selected_positions"), dtype=torch.bool),
            selected_tokens=_list_to_tensor(data.get("selected_tokens"), dtype=torch.long),
            intervention_applied=data.get("intervention_applied", False),
            intervention_type=data.get("intervention_type"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class GenerationTrace:
    """Full generation trace."""
    prompt: torch.Tensor
    final_sequence: torch.Tensor
    step_traces: List[StepTrace] = field(default_factory=list)
    unmasking_order: Dict[int, int] = field(default_factory=dict)  # gen index -> step unmasked
    prompt_len: Optional[int] = None
    gen_length: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def get_unmasking_step(self, position: int) -> Optional[int]:
        """First unmask step for a gen-region index."""
        return self.unmasking_order.get(position)

    def get_token_at_step(self, step: int, position: int) -> Optional[int]:
        """Predicted token at a gen index on a given step."""
        if step < len(self.step_traces):
            preds = self.step_traces[step].predictions
            if preds is not None and position < preds.shape[1]:
                return int(preds[0, position].item())
        return None
    
    def get_all_unmasked_positions(self) -> List[int]:
        """Gen indices sorted by first unmask step."""
        return sorted(self.unmasking_order.keys(), key=lambda p: self.unmasking_order[p])
    
    @property
    def wall_time(self) -> Optional[float]:
        if self.start_time is None or self.end_time is None:
            return None
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict:
        """Trace as a JSON dict."""
        return {
            "gen_only": True,
            "prompt": _tensor_to_list(self.prompt) if self.prompt is not None else None,
            "final_sequence": _tensor_to_list(self.final_sequence) if self.final_sequence is not None else None,
            "prompt_len": self.prompt_len,
            "gen_length": self.gen_length,
            "step_traces": [st.to_dict() for st in self.step_traces],
            "unmasking_order": self.unmasking_order,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }
    
    def save_to_json(self, name: str, base_dir: str = "trace_outputs") -> str:
        """Write trace JSON to disk."""
        os.makedirs(base_dir, exist_ok=True)
        filepath = os.path.join(base_dir, f"{name}.json")
        
        trace_dict = self.to_dict()
        with open(filepath, 'w') as f:
            json.dump(trace_dict, f, indent=2)
        
        return filepath
    
    def upload_to_wandb(self, name: Optional[str] = None, base_dir: str = "trace_outputs"):
        """Upload trace as a W&B artifact."""
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
        """Load trace from JSON."""
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
            unmasking_order={int(k): v for k, v in trace_dict.get("unmasking_order", {}).items()},
            prompt_len=trace_dict.get("prompt_len"),
            gen_length=trace_dict.get("gen_length"),
            start_time=trace_dict.get("start_time"),
            end_time=trace_dict.get("end_time"),
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
    """Collects step traces and applies interventions."""

    def __init__(
        self, 
        intervention_hook: Optional[InterventionHook] = None,
        save_to_disk: bool = True,
        trace_name: Optional[str] = None,
        base_dir: str = "trace_outputs",
        save_predictions: bool = True,
        save_confidences: bool = True,
        save_entropies: bool = True,
        upload_to_wandb: bool = False
    ):
        self.trace = GenerationTrace(prompt=None, final_sequence=None)
        self.intervention_hook = intervention_hook or InterventionHook()
        self.current_step = 0
        self.prompt_len: Optional[int] = None
        self.gen_length: Optional[int] = None

        self._current_predictions = None
        self._current_confidences = None
        self._current_logits = None
        
        # Block boundaries 
        self._current_block_start = None
        self._current_block_end = None
        self._block_length = None  
        
        # Disk saving options
        self.save_to_disk = save_to_disk
        self.trace_name = trace_name
        self.base_dir = base_dir
        self._saved_steps = 0  # Track how many steps have been saved
        self._total_flushed = 0  # Absolute step count across all flushes (for unique filenames)

        # Performance options
        self.save_predictions = save_predictions
        self.save_confidences = save_confidences
        self.save_entropies = save_entropies
        self.upload_to_wandb = upload_to_wandb

    def set_generation_bounds(self, prompt_len: int, gen_length: int) -> None:
        """Set prompt_len and gen_length for gen-only fields."""
        self.prompt_len = int(prompt_len)
        self.gen_length = int(gen_length)
        self.trace.prompt_len = self.prompt_len
        self.trace.gen_length = self.gen_length

    def _ensure_generation_bounds(self, x: torch.Tensor) -> None:
        if self.trace.prompt is not None and self.prompt_len is None:
            self.prompt_len = int(self.trace.prompt.shape[1])
        if self.prompt_len is None:
            self.prompt_len = 0
        if self.gen_length is None:
            self.gen_length = int(x.shape[1]) - int(self.prompt_len)
        self.trace.prompt_len = self.prompt_len
        self.trace.gen_length = self.gen_length

    def record_step_start(self, step: int, x: torch.Tensor, mask_id: int):
        self.current_step = step
        self._ensure_generation_bounds(x)
        pl, gl = self.prompt_len, self.gen_length
        mask_gen = (x[:, gen_region_slice(pl, gl)] == mask_id)

        step_trace = StepTrace(
            step=step,
            mask_index=mask_gen.detach().to(device="cpu"),
        )
        self.trace.step_traces.append(step_trace)

        self._current_block_start = None
        self._current_block_end = None

    def record_logits_metrics(self, logits: torch.Tensor) -> None:
        """Store predictions, max-prob confidences, and entropies from logits."""
        if not self.trace.step_traces or self.prompt_len is None or self.gen_length is None:
            return
        predictions, confidences, entropies = logits_to_step_metrics(logits)
        pl, gl = self.prompt_len, self.gen_length
        pred_g = slice_gen_region(predictions, pl, gl).detach().cpu()
        conf_g = slice_gen_region(confidences, pl, gl).detach().cpu()
        ent_g = slice_gen_region(entropies, pl, gl).detach().cpu()
        step_trace = self.trace.step_traces[-1]
        if self.save_predictions:
            step_trace.predictions = pred_g
        if self.save_confidences:
            step_trace.confidences = conf_g
        if self.save_entropies:
            step_trace.entropies = ent_g
        self._current_predictions = predictions.detach().cpu()
        self._current_confidences = conf_g

    def record_committed_token_confidences(
        self,
        tokens: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
    ) -> None:
        """Replace confidence with p(committed token) for stochastic decoding."""
        if not self.trace.step_traces or self.prompt_len is None or self.gen_length is None:
            return
        if probs is not None:
            confidence = probs
        else:
            src = self._current_logits if logits is None else logits
            if src is None:
                return
            token_ids = tokens.to(device=src.device, dtype=torch.long)
            chosen = src.float().gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
            lse = torch.logsumexp(src.float(), dim=-1)
            confidence = (chosen - lse).exp()
        conf_g = slice_gen_region(
            confidence,
            self.prompt_len,
            self.gen_length,
        ).detach().cpu()
        if self.save_confidences:
            self.trace.step_traces[-1].confidences = conf_g
        self._current_confidences = conf_g
        self._current_logits = None

    def record_timestamps(self, start_time: float, end_time: float):
        self.trace.start_time = start_time
        self.trace.end_time = end_time

    def record_predictions(self, predictions: torch.Tensor, confidences: torch.Tensor):
        """Legacy argmax path; prefer record_logits_metrics."""
        if not self.trace.step_traces or self.prompt_len is None or self.gen_length is None:
            return
        pl, gl = self.prompt_len, self.gen_length
        if self.save_predictions:
            self.trace.step_traces[-1].predictions = slice_gen_region(predictions, pl, gl).clone()
        if self.save_confidences:
            self.trace.step_traces[-1].confidences = slice_gen_region(confidences, pl, gl).clone()
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
        pl = int(self.prompt_len or 0)
        gl = int(self.gen_length or transfer_index.shape[1] - pl)
        transfer_gen = transfer_index[:, gen_region_slice(pl, pl + gl)]
        mask_gen = step_trace.mask_index

        modified_transfer = transfer_index.clone()
        modified_x0 = x0.clone()
        intervention_applied = False

        if not hook.active(self.current_step):
            return modified_transfer, modified_x0, False

        if mask_gen.device != transfer_gen.device:
            mask_gen = mask_gen.to(device=transfer_gen.device)

        S = torch.nonzero(transfer_gen[0], as_tuple=False).squeeze(-1)

        block_start = self._current_block_start
        block_end = self._current_block_end
        block_start_gen = block_start - pl if block_start is not None else None
        block_end_gen = block_end - pl if block_end is not None else None

        R_all = torch.nonzero(
            mask_gen[0] & (~transfer_gen[0]),
            as_tuple=False,
        ).squeeze(-1)

        if hook.mode == "swap" and block_start_gen is not None and block_end_gen is not None:
            R = R_all[(R_all >= block_start_gen) & (R_all < block_end_gen)]
        else:
            R = R_all

        if hook.mode == "drop":
            if S.numel() == 0:
                return modified_transfer, modified_x0, False

            k = min(hook.k, S.numel())
            perm = torch.randperm(S.numel(), device=S.device)
            dropped_gen = S[perm[:k]]
            dropped_abs = dropped_gen + pl

            modified_transfer[0, dropped_abs] = False

            intervention_applied = True
            step_trace.intervention_type = "drop"
            step_trace.metadata["dropped_positions"] = dropped_gen.tolist()
            step_trace.metadata["k"] = k
            step_trace.metadata["S_size"] = int(S.numel())

        elif hook.mode == "swap":
            if S.numel() == 0 or R.numel() == 0:
                return modified_transfer, modified_x0, False

            k = min(hook.k, S.numel(), R.numel())
            perm_S = torch.randperm(S.numel(), device=S.device)
            perm_R = torch.randperm(R.numel(), device=R.device)

            removed_gen = S[perm_S[:k]]
            added_gen = R[perm_R[:k]]
            removed_abs = removed_gen + pl
            added_abs = added_gen + pl

            modified_transfer[0, removed_abs] = False
            modified_transfer[0, added_abs] = True

            if self._current_predictions is not None:
                preds = self._current_predictions.to(device=modified_x0.device)
                modified_x0[0, added_abs] = preds[0, added_abs]
            elif self._current_logits is not None:
                predictions = torch.argmax(self._current_logits, dim=-1)
                modified_x0[0, added_abs] = predictions[0, added_abs]
            else:
                if x0.shape[1] > added_abs.max().item():
                    modified_x0[0, added_abs] = x0[0, added_abs]
                else:
                    modified_transfer[0, added_abs] = False
                    return modified_transfer, modified_x0, False

            intervention_applied = True
            step_trace.intervention_type = "swap"
            step_trace.metadata["swapped_out"] = removed_gen.tolist()
            step_trace.metadata["swapped_in"] = added_gen.tolist()
            step_trace.metadata["k"] = k
            step_trace.metadata["S_size"] = int(S.numel())
            step_trace.metadata["R_size"] = int(R.numel())
            if block_start_gen is not None and block_end_gen is not None:
                step_trace.metadata["block_start"] = block_start_gen
                step_trace.metadata["block_end"] = block_end_gen

        step_trace.intervention_applied = intervention_applied
        return modified_transfer, modified_x0, intervention_applied

    def record_transfer(self, transfer_index: torch.Tensor, x0: torch.Tensor):
        if not self.trace.step_traces:
            return

        self._ensure_generation_bounds(x0)
        pl = int(self.prompt_len or 0)
        gl = int(self.gen_length or transfer_index.shape[1] - pl)
        transfer_gen = transfer_index[:, gen_region_slice(pl, pl + gl)].detach().cpu()
        x0_gen = x0[:, gen_region_slice(pl, pl + gl)].detach().cpu()

        step_trace = self.trace.step_traces[-1]
        # DUS (and similar) write x[b, pos] in a loop — each write hits this hook.
        # OR into the step's selection instead of replacing, or only the last pos is kept.
        if step_trace.selected_positions is None:
            step_trace.selected_positions = transfer_gen.clone()
            selected_tokens = torch.zeros_like(x0_gen)
            selected_tokens[transfer_gen] = x0_gen[transfer_gen]
            step_trace.selected_tokens = selected_tokens.clone()
        else:
            step_trace.selected_positions = step_trace.selected_positions | transfer_gen
            if step_trace.selected_tokens is None:
                step_trace.selected_tokens = torch.zeros_like(x0_gen)
            step_trace.selected_tokens[transfer_gen] = x0_gen[transfer_gen]

        positions = torch.nonzero(transfer_gen[0], as_tuple=False).squeeze(-1)
        if positions.dim() == 0:
            positions = positions.unsqueeze(0)

        for pos in positions.tolist():
            if pos not in self.trace.unmasking_order:
                self.trace.unmasking_order[int(pos)] = self.current_step

    def clear_recorded_position(self, absolute_pos: int) -> None:
        """Drop one gen position selected earlier in the open step.

        DUS writes a token and may put it back to the mask in the same step.
        The remask must leave ``selected_positions`` equal to the tokens that
        stay unmasked, which is what the next step's mask diff records.
        """
        if not self.trace.step_traces or self.prompt_len is None:
            return
        pl = int(self.prompt_len)
        gen_pos = int(absolute_pos) - pl
        step_trace = self.trace.step_traces[-1]
        selected = step_trace.selected_positions
        if selected is None or gen_pos < 0 or gen_pos >= selected.shape[-1]:
            return
        selected[0, gen_pos] = False
        if step_trace.selected_tokens is not None and gen_pos < step_trace.selected_tokens.shape[-1]:
            step_trace.selected_tokens[0, gen_pos] = 0
        if self.trace.unmasking_order.get(int(gen_pos)) == self.current_step:
            del self.trace.unmasking_order[int(gen_pos)]
    
    def save_trace(self, name: Optional[str] = None) -> str:
        """Write the in-memory trace to JSON."""
        save_name = name or self.trace_name
        if save_name is None:
            raise ValueError("trace_name must be provided in __init__ or as argument to save_trace()")
        
        return self.trace.save_to_json(save_name, self.base_dir)
    
    def save_incremental(self, clear_memory: bool = True):
        """Flush new steps to disk; optionally drop them from memory."""
        if not self.save_to_disk or self.trace_name is None:
            return
        
        if not self.trace.step_traces:
            return
        
        # Save only new steps since last save
        steps_to_save = self.trace.step_traces[self._saved_steps:]
        if not steps_to_save:
            return
        
        # Create incremental save file — use absolute step count so filenames are unique across flushes
        n_new = len(steps_to_save)
        inc_filepath = os.path.join(self.base_dir, f"{self.trace_name}_steps_{self._total_flushed}_{self._total_flushed + n_new}.json")
        os.makedirs(self.base_dir, exist_ok=True)

        steps_data = [st.to_dict() for st in steps_to_save]
        with open(inc_filepath, 'w') as f:
            json.dump(steps_data, f, indent=2)

        self._total_flushed += n_new

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
    """Build a drop/swap intervention hook."""
    return InterventionHook(
        mode=mode,
        steps=set(steps) if steps is not None else set(),
        k=k
    )