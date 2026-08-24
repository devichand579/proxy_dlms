"""Pre-logit hidden capture and .pt storage for CST."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from cst.storage_dtype import map_float_tensors


def extract_pre_logits_hidden(outputs: Any) -> torch.Tensor:
    """Last layer from model hidden_states."""
    if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
        raise ValueError(
            "Model output has no hidden_states; call forward with output_hidden_states=True."
        )
    hs = outputs.hidden_states
    if len(hs) == 0:
        raise ValueError("hidden_states is empty.")
    return hs[-1]


def forward_with_pre_logits_hidden(
    model: nn.Module,
    *args: Any,
    **kwargs: Any,
) -> Tuple[Any, torch.Tensor]:
    """Forward pass; return outputs and last hidden as float32."""
    kwargs = dict(kwargs)
    kwargs["output_hidden_states"] = True
    kwargs["return_dict"] = True
    outputs = model(*args, **kwargs)
    h = extract_pre_logits_hidden(outputs).detach().float()
    return outputs, h


def _extract_input_ids_from_forward(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Optional[torch.Tensor]:
    if kwargs.get("input_ids") is not None:
        return kwargs["input_ids"]
    if args and isinstance(args[0], torch.Tensor):
        return args[0]
    return None


class PreLogitsCaptureWrapper(nn.Module):
    """Wrap the LM and record pre-logit hiddens each forward."""

    def __init__(
        self,
        model: nn.Module,
        *,
        enabled: bool = True,
        move_to_cpu: bool = True,
        capture_initial_embed: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.enabled = enabled
        self.move_to_cpu = move_to_cpu
        self.capture_initial_embed = capture_initial_embed
        self.records: List[torch.Tensor] = []

    def _embed_initial(self, input_ids: torch.Tensor) -> torch.Tensor:
        m = self.model
        if not hasattr(m, "get_input_embeddings"):
            raise RuntimeError("PreLogitsCaptureWrapper: inner model has no get_input_embeddings().")
        emb = m.get_input_embeddings()(input_ids)
        return emb.detach().float()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        kwargs = dict(kwargs)
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        input_ids = _extract_input_ids_from_forward(args, kwargs)
        outputs = self.model(*args, **kwargs)
        if self.enabled and input_ids is not None:
            if self.capture_initial_embed and len(self.records) == 0:
                h0 = self._embed_initial(input_ids)
                if self.move_to_cpu:
                    h0 = h0.cpu()
                self.records.append(h0)
            h = extract_pre_logits_hidden(outputs).detach().float()
            if self.move_to_cpu:
                h = h.cpu()
            self.records.append(h)
        return outputs

    def clear_records(self) -> None:
        self.records.clear()

    def unwrap(self) -> nn.Module:
        return self.model

    def __getattr__(self, name: str) -> Any:
        if name in (
            "model",
            "enabled",
            "move_to_cpu",
            "records",
            "capture_initial_embed",
        ):
            return super().__getattr__(name)
        return getattr(self.model, name)


def save_hidden_records(
    records: Sequence[torch.Tensor],
    path: Union[str, Path],
    *,
    prompt_len: int,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Save hidden_states and meta.prompt_len to a .pt file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec_list = [map_float_tensors(t.float(), to_storage=True) for t in records]
    meta: Dict[str, Any] = {"prompt_len": int(prompt_len)}
    if extra_meta:
        meta.update(extra_meta)
    payload: dict[str, Any] = {
        "hidden_states": rec_list,
        "meta": meta,
    }
    torch.save(payload, path)
