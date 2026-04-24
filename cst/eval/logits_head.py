"""Pre-logit hidden states to vocab logits (same last linear as the backbone)."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def unwrap_inner(model: nn.Module) -> nn.Module:
    """Return inner module if ``model.unwrap`` exists, else ``model``."""
    m = model.unwrap() if hasattr(model, "unwrap") else model
    return m


def logits_from_prelogit_hidden(model: nn.Module, h: torch.Tensor) -> torch.Tensor:
    """Project ``h`` (post-final-norm, pre-vocab) to logits; Dream uses ``lm_head``, LLaDA uses ``wte``/``ff_out``."""
    m = unwrap_inner(model)
    p0 = next(m.parameters(), None)
    dev = p0.device if p0 is not None else h.device
    dt = p0.dtype if p0 is not None else torch.float32
    h = h.to(device=dev, dtype=dt)

    # Dream: explicit lm_head
    if hasattr(m, "lm_head") and isinstance(getattr(m, "lm_head"), nn.Module):
        return m.lm_head(h)

    # LLaDA-style: F.linear(h, wte) or ff_out
    if hasattr(m, "transformer") and hasattr(m, "config"):
        tr = m.transformer
        cfg = m.config
        if getattr(cfg, "weight_tying", True):
            logits = F.linear(h, tr.wte.weight, None)
        else:
            logits = tr.ff_out(h)
        if getattr(cfg, "scale_logits", False):
            logits = logits * (1.0 / math.sqrt(float(cfg.d_model)))
        return logits

    raise TypeError(
        f"Cannot build logits from pre-logit hiddens for model type {type(m).__name__}. "
        "Expected Dream (lm_head) or LLaDA (transformer + config)."
    )


def unembedding_weight_matrix(model: nn.Module) -> torch.Tensor:
    """``[V, d]`` unembed weights; same linear as :func:`logits_from_prelogit_hidden` (F.linear(h, W_U))."""
    m = unwrap_inner(model)
    if hasattr(m, "lm_head") and isinstance(getattr(m, "lm_head"), nn.Linear):
        return m.lm_head.weight.detach()
    if hasattr(m, "transformer") and hasattr(m, "config"):
        tr = m.transformer
        cfg = m.config
        if getattr(cfg, "weight_tying", True):
            return tr.wte.weight.detach()
        if hasattr(tr, "ff_out") and isinstance(tr.ff_out, nn.Linear):
            return tr.ff_out.weight.detach()
    raise TypeError(
        f"Cannot get unembedding weights for {type(m).__name__}. "
        "Expected Dream (lm_head) or LLaDA (wte / ff_out)."
    )
