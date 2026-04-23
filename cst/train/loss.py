"""CST training loss: reconstruction (MSE), sparsity, optional dead-feature penalty."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from cst.model.cst_bank import CSTBank


def windowed_cst_loss_full(
    cst_bank: CSTBank,
    anchor: int,
    h_window: torch.Tensor,
    H_full: torch.Tensor,
    lambda_sparsity: float,
    c_tanh: float,
    lambda_ramp: float,
    lambda_pre_dead: float = 0.0,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """
    For one anchor step: predict hidden states from CST decoders, compare to ``H_full`` (MSE),
    add sparsity on JumpReLU activations, optional ``ReLU(-pre_act)`` dead penalty. Returns loss + metrics.

    ``h_window`` is ``[B, w_eff, L, D]`` (encoder inputs per slot); ``H_full`` is ``[n, B, L, D]`` with row ``t`` at timestep ``t``.
    """
    if h_window.dim() != 4:
        raise ValueError(
            f"h_window must be [B, w, L, D], got {tuple(h_window.shape)}"
        )
    if H_full.dim() != 4:
        raise ValueError(
            f"H_full must be [n, B, L, D], got {tuple(H_full.shape)}"
        )

    B, w_eff, L, D = h_window.shape
    if w_eff < 1 or w_eff > cst_bank.w:
        raise ValueError(
            f"h_window dim 1 must be in [1, cst_bank.w={cst_bank.w}], got {w_eff}"
        )

    device = h_window.device
    total_sparsity = torch.zeros((), device=device, dtype=torch.float32)
    total_pre_dead = torch.zeros((), device=device, dtype=torch.float32)

    enc_cache: List[Tuple[int, int, Any, torch.Tensor]] = []
    l0_acc = 0.0
    n_enc = 0

    for enc_slot in range(w_eff):
        enc_step = anchor - w_eff + 1 + enc_slot
        if enc_step < 1:
            continue

        cst_enc = cst_bank.get(enc_step)
        H_enc = h_window[:, enc_slot]
        H_t = H_enc.to(cst_enc.W_enc.weight.dtype)
        pre_act = cst_enc.W_enc(H_t)
        A = cst_enc.jump(pre_act)

        enc_cache.append((enc_slot, enc_step, cst_enc, A))
        total_sparsity = total_sparsity + cst_enc.decoder_sparsity_penalty(A, c_tanh)
        if lambda_pre_dead > 0.0:
            total_pre_dead = total_pre_dead + F.relu(-pre_act.float()).mean()

        with torch.no_grad():
            l0_acc += float((A > 0).float().mean().item())
            n_enc += 1

    if n_enc == 0:
        eff_lambda = float(lambda_sparsity) * float(lambda_ramp)
        return torch.zeros((), device=device, requires_grad=True), {
            "loss": 0.0,
            "recon_loss": 0.0,
            "sparsity_loss": 0.0,
            "pre_dead_raw": 0.0,
            "pre_dead_term": 0.0,
            "nmse": 0.0,
            "l0": 0.0,
            "lambda_eff": eff_lambda,
            "n_enc_steps": 0,
            "n_target_terms": 0,
        }

    total_recon = torch.zeros((), device=device, dtype=torch.float32)
    n_target_terms = 0

    # One MSE term per target step in the window (paper: one squared error per layer index).
    # Sliding anchors (trainer loop) cover the rest of the trajectory so each t>=1 is a target in some window.
    for tau in range(w_eff):
        s_step = anchor - w_eff + 1 + tau
        t_row = s_step
        tgt = H_full[t_row].float()
        H_hat = torch.zeros_like(tgt)
        n_contrib = 0

        for _enc_slot, enc_step, cst_enc, A in enc_cache:
            offset = s_step - enc_step
            if offset < 1 or offset > cst_enc.w:
                continue
            H_hat = H_hat + cst_enc.decode(A, offset).float()
            n_contrib += 1

        if n_contrib == 0:
            continue

        diff = H_hat - tgt
        total_recon = total_recon + diff.pow(2).sum()
        n_target_terms += 1

    eff_lambda = float(lambda_sparsity) * float(lambda_ramp)
    pre_dead_term = float(lambda_pre_dead) * total_pre_dead
    loss = total_recon + eff_lambda * total_sparsity + pre_dead_term

    with torch.no_grad():
        if n_target_terms > 0:
            rms = (total_recon / max(n_target_terms * B * L * D, 1)).sqrt()
            ref = H_full[anchor].float().norm(dim=(1, 2)).clamp_min(1e-8).mean()
            nmse = float((rms / ref).item())
        else:
            nmse = 0.0

    metrics: Dict[str, Any] = {
        "loss": float(loss.detach().item()),
        "recon_loss": float(total_recon.detach().item()),
        "sparsity_loss": float(total_sparsity.detach().item()),
        "pre_dead_raw": float(total_pre_dead.detach().item()),
        "pre_dead_term": float(pre_dead_term.detach().item()),
        "nmse": nmse,
        "l0": l0_acc / max(n_enc, 1),
        "lambda_eff": eff_lambda,
        "n_enc_steps": n_enc,
        "n_target_terms": n_target_terms,
    }
    return loss, metrics
