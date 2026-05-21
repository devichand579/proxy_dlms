"""CST activations and reconstruction residuals for TAG."""

from __future__ import annotations

from typing import List

import torch

from cst.model.cst_bank import CSTBank


def last_h_index(H: torch.Tensor) -> int:
    """Last valid row index in H."""
    return int(H.shape[0]) - 1


def cst_step_count(H: torch.Tensor, bank: CSTBank, *, max_step: int | None = None) -> int:
    """How many CST steps to run on H (step t encodes H[t-1])."""
    t_steps = int(H.shape[0])
    cap = int(max_step) if max_step is not None else min(bank.max_steps, t_steps)
    return min(t_steps, cap, bank.max_steps)


@torch.no_grad()
def compute_activation_list(
    H: torch.Tensor,
    bank: CSTBank,
    *,
    max_step: int | None = None,
) -> List[torch.Tensor]:
    """Per-step CST activations; list index s is step t = s + 1."""
    m = cst_step_count(H, bank, max_step=max_step)
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
    """CST reconstruction of H[anchor] (same window rule as eval)."""
    last_h = last_h_index(H)
    if anchor < 1 or anchor > last_h:
        raise ValueError(f"anchor {anchor} out of range [1, {last_h}]")
    T, B, L, d = H.shape
    device = H.device
    dtyp = H.dtype
    H_hat = torch.zeros((B, L, d), device=device, dtype=dtyp)
    for enc_slot in range(w):
        enc_step = anchor - w + 1 + enc_slot
        if enc_step < 1:
            continue
        offset = anchor - enc_step + 1
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
    """Residual vectors H[anchor] - H_hat(anchor); row r is anchor w + r."""
    last_h = last_h_index(H)
    hi = min(last_h, max_anchor) if max_anchor is not None else last_h
    rows: list[torch.Tensor] = []
    for anchor in range(w, hi + 1):
        tgt = H[anchor].float().cpu()
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
    """Per-token L2 norm of reconstruction residual."""
    last_h = last_h_index(H)
    hi = min(last_h, max_anchor) if max_anchor is not None else last_h
    rows: list[torch.Tensor] = []
    for anchor in range(w, hi + 1):
        tgt = H[anchor].float().cpu()
        pred = cst_reconstruct_hidden(H, bank, anchor, w)
        rows.append((tgt[0] - pred[0]).norm(dim=-1))  # [L]
    if not rows:
        return torch.empty(0)
    return torch.stack(rows, dim=0)
