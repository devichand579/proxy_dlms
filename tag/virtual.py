"""Optional full k×k feature-space V between CST steps (decoder at t × encoder at t+δ)."""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from cst.model.cst_bank import CSTBank, effective_window_size


def virtual_weight_matrix(
    bank: CSTBank,
    t: int,
    delta: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dense k×k V for (t, δ); row f, col g from decoder column f and encoder row g."""
    if delta < 1:
        raise ValueError("delta must be >= 1 for cross-step virtual weights")
    cst_t = bank.get(t)
    t2 = t + delta
    if t2 > bank.max_steps:
        raise ValueError(f"t+delta={t2} > max_steps={bank.max_steps}")
    w_t = effective_window_size(t, bank.w, bank.max_steps)
    if delta > w_t:
        raise ValueError(f"delta {delta} > effective window w_t={w_t} at step {t}")
    cst_t2 = bank.get(t2)
    w_dec = cst_t.W_dec[delta - 1].weight.to(device=device, dtype=dtype)
    w_enc = cst_t2.W_enc.weight.to(device=device, dtype=dtype)
    # w_dec: [d, k], w_enc: [k, d] → [k, k] = w_dec.T @ w_enc.T
    v = w_dec.T @ w_enc.T
    return v


def precompute_virtual_weights(
    bank: CSTBank,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
    max_t: int | None = None,
) -> Dict[Tuple[int, int], torch.Tensor]:
    """Map (t, δ) to V on ``device`` for all valid pairs up to max_t (default bank.max_steps)."""
    dev = device if isinstance(device, torch.device) else torch.device(device)
    t_max = int(max_t) if max_t is not None else bank.max_steps
    out: Dict[Tuple[int, int], torch.Tensor] = {}
    for t in range(1, min(bank.max_steps, t_max) + 1):
        w_t = effective_window_size(t, bank.w, bank.max_steps)
        for delta in range(1, w_t + 1):
            if t + delta > t_max or t + delta > bank.max_steps:
                break
            out[(t, delta)] = virtual_weight_matrix(bank, t, delta, device=dev, dtype=dtype)
    return out
