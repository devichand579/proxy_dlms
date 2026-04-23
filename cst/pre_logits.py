"""Pre-logit hiddens: last layer of ``hidden_states``; captured in fp32, written to disk as bfloat16 (see :mod:`cst.storage_dtype`)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from cst.storage_dtype import map_float_tensors


def extract_pre_logits_hidden(outputs: Any) -> torch.Tensor:
    """Last layer tensor from ``outputs.hidden_states`` (needs ``output_hidden_states=True``)."""
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
    """Forward with hiddens on; returns ``(outputs, h)`` with ``h`` shape ``(B, L, D)`` float32."""
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


def _extract_attention_mask_from_forward(kwargs: dict[str, Any]) -> Optional[torch.Tensor]:
    return kwargs.get("attention_mask")


class PreLogitsCaptureWrapper(nn.Module):
    """Wraps the LM: records are float32 ``(B, L, H)`` — optional initial row (token embeddings) then one per ``forward``."""

    def __init__(
        self,
        model: nn.Module,
        *,
        mask_id: Optional[int] = None,
        enabled: bool = True,
        move_to_cpu: bool = True,
        capture_initial_embed: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self._mask_id = mask_id
        self.enabled = enabled
        self.move_to_cpu = move_to_cpu
        self.capture_initial_embed = capture_initial_embed
        self.records: List[torch.Tensor] = []
        # Per record: bool [B, L], aligned 1:1 with ``records`` when mask_id is set
        self.unmasked_records: List[torch.Tensor] = []
        self.diffusion_masked_records: List[torch.Tensor] = []

    def _embed_initial(self, input_ids: torch.Tensor) -> torch.Tensor:
        m = self.model
        if not hasattr(m, "get_input_embeddings"):
            raise RuntimeError("PreLogitsCaptureWrapper: inner model has no get_input_embeddings().")
        emb = m.get_input_embeddings()(input_ids)
        return emb.detach().float()

    def _append_diffusion_seg_masks(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> None:
        if self._mask_id is None:
            return
        valid = (attention_mask != 0) if attention_mask is not None else torch.ones(
            (input_ids.shape[0], input_ids.shape[1]),
            dtype=torch.bool,
            device=input_ids.device,
        )
        m_id = int(self._mask_id)
        is_m = (input_ids == m_id) & valid
        is_u = (~is_m) & valid
        if self.move_to_cpu:
            is_u, is_m = is_u.cpu(), is_m.cpu()
        self.unmasked_records.append(is_u)
        self.diffusion_masked_records.append(is_m)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        kwargs = dict(kwargs)
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        input_ids = _extract_input_ids_from_forward(args, kwargs)
        attn = _extract_attention_mask_from_forward(kwargs)
        outputs = self.model(*args, **kwargs)
        if self.enabled and input_ids is not None:
            if self.capture_initial_embed and len(self.records) == 0:
                h0 = self._embed_initial(input_ids)
                if self.move_to_cpu:
                    h0 = h0.cpu()
                self.records.append(h0)
                self._append_diffusion_seg_masks(input_ids, attn)
            h = extract_pre_logits_hidden(outputs).detach().float()
            if self.move_to_cpu:
                h = h.cpu()
            self.records.append(h)
            self._append_diffusion_seg_masks(input_ids, attn)
        return outputs

    def clear_records(self) -> None:
        self.records.clear()
        self.unmasked_records.clear()
        self.diffusion_masked_records.clear()

    def unwrap(self) -> nn.Module:
        """Inner module."""
        return self.model

    def __getattr__(self, name: str) -> Any:
        if name in (
            "model",
            "enabled",
            "move_to_cpu",
            "records",
            "capture_initial_embed",
            "unmasked_records",
            "diffusion_masked_records",
            "_mask_id",
        ):
            return super().__getattr__(name)
        return getattr(self.model, name)


def save_hidden_records(
    records: Sequence[torch.Tensor],
    path: Union[str, Path],
    *,
    extra: Optional[dict] = None,
    token_unmasked: Optional[Sequence[torch.Tensor]] = None,
    token_diffusion_masked: Optional[Sequence[torch.Tensor]] = None,
) -> None:
    """``torch.save`` dict with ``hidden_states`` (bfloat16), optional bool masks, and ``meta``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec_list = [map_float_tensors(t.float(), to_storage=True) for t in records]
    meta_merged: dict[str, Any] = dict(extra) if extra else {}
    meta_merged.setdefault("hidden_storage_dtype", "bfloat16")
    payload: dict[str, Any] = {
        "hidden_states": rec_list,
        "meta": meta_merged,
    }
    n = len(rec_list)
    if (
        token_unmasked is not None
        and token_diffusion_masked is not None
        and len(token_unmasked) == n
        and len(token_diffusion_masked) == n
    ):
        payload["token_unmasked"] = [x.bool() if x.dtype != torch.bool else x for x in token_unmasked]
        payload["token_diffusion_masked"] = [
            x.bool() if x.dtype != torch.bool else x for x in token_diffusion_masked
        ]
    torch.save(payload, path)
