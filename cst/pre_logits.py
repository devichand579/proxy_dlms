"""
Pre-logit hidden extraction and per-step recording (bfloat16).

For **LLaDA** / **LLaDA 1.5**, the last ``hidden_states`` entry is post-``ln_f``, immediately
before ``ff_out`` / tied projection.

For **Dream**, the last ``hidden_states`` entry is post-``norm`` (RMSNorm), immediately before
``lm_head``.

All schedulers that call ``model(...)`` with the usual keyword arguments are supported as long
as the underlying model implements ``output_hidden_states`` and returns a final layer tensor
for the full sequence length (including masked positions).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn


def extract_pre_logits_hidden(outputs: Any) -> torch.Tensor:
    """
    Take the last hidden state from a model output (post final norm, pre vocab linear).

    Expects ``outputs.hidden_states`` to be a non-empty tuple/list of tensors.
    """
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
    """
    Run ``model(*args, **kwargs)`` with ``output_hidden_states=True`` and ``return_dict=True``,
    and return ``(outputs, pre_logits_hidden_bf16)``.

    ``pre_logits_hidden_bf16`` has shape ``(batch, seq_len, dim)``, dtype ``bfloat16``, and
    includes every token position.
    """
    kwargs = dict(kwargs)
    kwargs["output_hidden_states"] = True
    kwargs["return_dict"] = True
    outputs = model(*args, **kwargs)
    h = extract_pre_logits_hidden(outputs).detach().to(torch.bfloat16)
    return outputs, h


class PreLogitsCaptureWrapper(nn.Module):
    """
    Wraps a diffusion LM so each forward records pre-logit hidden states in bfloat16.

    - Forwards all attributes/methods to the inner model (``config``, ``generate``, …).
    - Each call to ``forward`` appends one tensor ``(B, L, H)`` in bf16 to ``records`` (CPU by default).
    - Set ``enabled=False`` to disable recording (still requests hidden states while enabled).
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        enabled: bool = True,
        move_to_cpu: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.enabled = enabled
        self.move_to_cpu = move_to_cpu
        self.records: List[torch.Tensor] = []

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        kwargs = dict(kwargs)
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        outputs = self.model(*args, **kwargs)
        if self.enabled:
            h = extract_pre_logits_hidden(outputs).detach().to(torch.bfloat16)
            if self.move_to_cpu:
                h = h.cpu()
            self.records.append(h)
        return outputs

    def clear_records(self) -> None:
        self.records.clear()

    def unwrap(self) -> nn.Module:
        """Return the inner model (for saving checkpoints or compile)."""
        return self.model

    def __getattr__(self, name: str) -> Any:
        if name in ("model", "enabled", "move_to_cpu", "records"):
            return super().__getattr__(name)
        return getattr(self.model, name)


def save_hidden_records(
    records: Sequence[torch.Tensor],
    path: Union[str, Path],
    *,
    extra: Optional[dict] = None,
) -> None:
    """
    Save a list of bf16 tensors (one per scheduler step / forward) with ``torch.save``.

    File contains a dict: ``{"hidden_states_bf16": [...], "meta": extra or {}}``.
    Each tensor is ``(batch, seq_len, hidden_dim)`` in ``bfloat16``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "hidden_states_bf16": list(records),
        "meta": extra or {},
    }
    torch.save(payload, path)
