"""LLaDA masked diffusion loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pcfg_data import LABEL_IGNORE


def llada_mask_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask_t: torch.Tensor,
    n_content: torch.Tensor,
) -> torch.Tensor:
    """Eq. (3): -(1/tL) sum_i 1[masked] log p_theta(x0_i | x_t)."""
    losses = []
    for b in range(logits.size(0)):
        masked = labels[b] != LABEL_IGNORE
        if int(masked.sum().item()) == 0:
            continue
        t = float(mask_t[b].item())
        if t <= 0.0:
            continue
        ce_sum = F.cross_entropy(
            logits[b],
            labels[b],
            ignore_index=LABEL_IGNORE,
            reduction="sum",
        )
        losses.append(ce_sum / (t * float(n_content[b].item())))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()
