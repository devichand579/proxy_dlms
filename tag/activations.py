"""Trajectory activations a^t_{i,f} and optional CST reconstruction for residuals."""

from __future__ import annotations

from typing import List, Tuple

import torch

from cst.model.cst_bank import CSTBank, effective_window_size


@torch.no_grad()
def compute_activation_list(
    H: torch.Tensor,
    bank: CSTBank,
    *,
    max_step: int | None = None,
) -> List[torch.Tensor]:
    """
    Per-step activations a^t = encode(H[t-1]) after JumpReLU; H is [T,B,L,d], each entry [B,L,k].
    List index s is step t = s+1 (same as eval).
    """
    t_max = H.shape[0] - 1
    m = int(max_step) if max_step is not None else min(bank.max_steps, t_max)
    m = min(m, t_max, bank.max_steps)
    out: List[torch.Tensor] = []
    for t in range(1, m + 1):
        cst = bank.get(t)
        dev = next(cst.W_enc.parameters()).device
        H_in = H[t - 1].to(device=dev, dtype=cst.W_enc.weight.dtype)
        a = cst.encode(H_in)
        out.append(a.float().cpu())
    return out


@torch.no_grad()
def cst_reconstruct_hidden(
    H: torch.Tensor,
    bank: CSTBank,
    anchor: int,
    w: int,
) -> torch.Tensor:
    """
    Windowed CST sum for H[anchor-1] (offset rule matches training and run_eval_cst). CPU float [B,L,d].
    """
    T, B, L, d = H.shape
    if anchor < 1 or anchor > T:
        raise ValueError("anchor out of range")
    device = H.device
    dtyp = H.dtype
    H_hat = torch.zeros((B, L, d), device=device, dtype=dtyp)
    s_target = anchor - 1
    for enc_slot in range(w):
        enc_step = anchor - w + 1 + enc_slot
        if enc_step < 1:
            continue
        offset = s_target - enc_step + 1
        if offset < 1:
            continue
        cst = bank.get(enc_step)
        if offset > cst.w:
            continue
        H_enc = H[enc_step - 1]
        A = cst.encode(H_enc.to(dtype=cst.W_enc.weight.dtype, device=next(cst.parameters()).device))
        H_hat = H_hat + cst.decode(A, offset)
    return H_hat.float().cpu()


@torch.no_grad()
def per_step_residual_vectors(
    H: torch.Tensor,
    bank: CSTBank,
    w: int,
    *,
    max_anchor: int | None = None,
) -> torch.Tensor:
    """
    H[anchor-1] - H_hat(anchor) per position (batch 0), for anchor in w..T.
    Returns float tensor [A, L, d] on CPU, A = number of anchors; row index
    row corresponds to anchor = w + row.
    """
    T = H.shape[0]
    hi = min(T, max_anchor) if max_anchor is not None else T
    rows: list[torch.Tensor] = []
    for anchor in range(w, hi + 1):
        tgt = H[anchor - 1].float().cpu()
        pred = cst_reconstruct_hidden(H, bank, anchor, w)
        rows.append(tgt[0] - pred[0])  # [L, d]
    if not rows:
        return torch.empty(0, 0, 0)
    return torch.stack(rows, dim=0)  # [A, L, d]


@torch.no_grad()
def per_step_residual_norm(
    H: torch.Tensor,
    bank: CSTBank,
    w: int,
    *,
    max_anchor: int | None = None,
) -> torch.Tensor:
    """Per-token L2 residual norm; shape [A, B, L] on CPU for anchors w..T."""
    T = H.shape[0]
    hi = min(T, max_anchor) if max_anchor is not None else T
    rows: list[torch.Tensor] = []
    for anchor in range(w, hi + 1):
        tgt = H[anchor - 1].float().cpu()
        pred = cst_reconstruct_hidden(H, bank, anchor, w)
        rows.append((tgt[0] - pred[0]).norm(dim=-1))  # [L] batch 0
    if not rows:
        return torch.empty(0)
    return torch.stack(rows, dim=0)
