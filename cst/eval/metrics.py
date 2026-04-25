"""NMSE, L0, NTF, cosine; optional token subset. TC is always over all positions."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch


def unmasked_masked_token_masks(
    B: int,
    L: int,
    split_at: int,
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prompt/response split at ``split_at`` (left vs right). Prefer per-step masks from the chunk when available."""
    pl = max(0, min(int(split_at), L))
    idx = torch.arange(L, device=device, dtype=torch.long)
    unmasked = (idx < pl).view(1, L).expand(B, L)
    masked = (idx >= pl).view(1, L).expand(B, L)
    return unmasked, masked


prompt_gen_token_masks = unmasked_masked_token_masks


def compute_nmse(
    Y: torch.Tensor,
    Y_hat: torch.Tensor,
    token_mask: Optional[torch.Tensor] = None,
) -> float:
    """Mean error norm over mean target norm; optional bool mask ``[B,L]``."""
    d = Y - Y_hat
    num = d.norm(dim=-1)
    den = Y.norm(dim=-1)
    if token_mask is None:
        return (num.mean() / den.mean().clamp_min(1e-8)).item()
    if bool(token_mask.sum() == 0):
        return float("nan")
    return (num[token_mask].mean() / den[token_mask].mean().clamp_min(1e-8)).item()


def compute_l0(A: torch.Tensor, token_mask: Optional[torch.Tensor] = None) -> float:
    """Mean active features per token; optional mask ``[B,L]``."""
    per = (A > 0).float().sum(dim=-1)
    if token_mask is None:
        return per.mean().item()
    if bool(token_mask.sum() == 0):
        return float("nan")
    return per[token_mask].mean().item()


def _argmax_excluding_token_ids(
    logits: torch.Tensor, exclude_ids: Optional[Sequence[int]]
) -> torch.Tensor:
    """Argmax over vocab, optionally setting given token ids to -inf (e.g. first non-EOS)."""
    if not exclude_ids:
        return logits.argmax(dim=-1)
    lf = logits.float().clone()
    for tid in exclude_ids:
        ti = int(tid)
        if 0 <= ti < lf.shape[-1]:
            lf[..., ti] = float("-inf")
    return lf.argmax(dim=-1)


def compute_ntf_logits(
    logits_teacher: torch.Tensor,
    logits_substitute: torch.Tensor,
    conf_low: float = 0.2,
    conf_high: float = 0.8,
    token_mask: Optional[torch.Tensor] = None,
    *,
    ntf_first_non_eos: bool = False,
    eos_id: Optional[int] = None,
) -> float:
    """Argmax match on positions where teacher max-prob is in (conf_low, conf_high). Uncertainty from full softmax.

    If ``ntf_first_non_eos`` and ``eos_id`` is set, compare argmax on logits with ``eos_id`` masked to -inf
    (same for teacher and substitute) so NTF is not driven only by agreement on EOS.
    """
    probs = torch.softmax(logits_teacher.float(), dim=-1)
    conf = probs.max(dim=-1).values
    uncertain = (conf > conf_low) & (conf < conf_high)
    if token_mask is not None:
        uncertain = uncertain & token_mask
    if uncertain.sum() == 0:
        return float("nan")
    excl = (int(eos_id),) if ntf_first_non_eos and eos_id is not None else None
    pred_t = _argmax_excluding_token_ids(logits_teacher, excl)
    pred_s = _argmax_excluding_token_ids(logits_substitute, excl)
    match = (pred_t == pred_s).float()
    return match[uncertain].mean().item()


def cosine_fidelity(
    Y: torch.Tensor, Y_hat: torch.Tensor, token_mask: Optional[torch.Tensor] = None
) -> float:
    """Mean cosine between last-dim vectors; optional mask ``[B,L]``."""
    Yf = Y.float()
    Hh = Y_hat.float()
    a = Yf / Yf.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b = Hh / Hh.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    cos = (a * b).sum(dim=-1)
    if token_mask is None:
        return cos.mean().item()
    if bool(token_mask.sum() == 0):
        return float("nan")
    return cos[token_mask].mean().item()


def temporal_coherence(A_t: torch.Tensor, A_t1: torch.Tensor) -> float:
    """Mean cosine similarity of normalized activations at each ``(B,L)`` position."""
    a = A_t.float() / A_t.float().norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b = A_t1.float() / A_t1.float().norm(dim=-1, keepdim=True).clamp_min(1e-8)
    cos = (a * b).sum(dim=-1)
    return cos.mean().item()
