"""Load hidden chunks and build CST training windows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch


def records_to_H(
    hidden_states: List[torch.Tensor],
) -> torch.Tensor:
    """Stack per-step tensors into [T, B, L, D] float32."""
    if not hidden_states:
        raise ValueError("empty hidden_states list")
    stacked = torch.stack([h.float() for h in hidden_states], dim=0)
    return stacked


def load_chunk_H(
    path: str | Path,
) -> Tuple[torch.Tensor, Dict[str, Any], Optional[Dict[str, torch.Tensor]]]:
    """Load a .pt chunk as H, meta, and optional legacy token masks."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    hs = payload.get("hidden_states")
    if hs is None:
        raise KeyError(f"{path}: missing 'hidden_states'")
    meta = payload.get("meta") or {}
    hs_list = list(hs)
    H = records_to_H(hs_list)
    seg: Optional[Dict[str, torch.Tensor]] = None
    ul = payload.get("token_unmasked")
    ml = payload.get("token_diffusion_masked")
    if (
        ul is not None
        and ml is not None
        and len(ul) == len(hs_list) == len(ml)
    ):
        seg = {
            "unmasked": torch.stack(
                [x.bool() if x.dtype != torch.bool else x for x in ul], dim=0
            ),
            "diffusion_masked": torch.stack(
                [x.bool() if x.dtype != torch.bool else x for x in ml], dim=0
            ),
        }
    return H, meta, seg


def iter_anchor_samples(
    H: torch.Tensor,
    w: int,
    *,
    device: torch.device | None = None,
) -> Iterator[Tuple[int, torch.Tensor]]:
    """Yield each training anchor and its [B, w, L, D] window."""
    n, B, L, D = H.shape
    if w < 1:
        raise ValueError("w must be >= 1")
    t_max = n - 1
    if t_max < 1:
        return
    w_eff = min(w, t_max)
    for anchor in range(w_eff, t_max + 1):
        slots = [H[anchor - w_eff + j] for j in range(w_eff)]
        h_win = torch.stack(slots, dim=1)
        if device is not None:
            h_win = h_win.to(device)
        yield anchor, h_win
