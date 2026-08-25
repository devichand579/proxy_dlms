"""Load hidden chunks and build CST training windows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch

from cst.pre_logits import is_cst_hidden_file, load_cst_hidden_tensor


def records_to_H(
    hidden_states: Union[torch.Tensor, Sequence[torch.Tensor]],
) -> torch.Tensor:
    """Stack per-step tensors into [T, B, L, D] (keeps storage dtype)."""
    if torch.is_tensor(hidden_states):
        if hidden_states.dim() != 4:
            raise ValueError(
                f"stacked hidden_states must be [T, B, L, D], got {tuple(hidden_states.shape)}"
            )
        return hidden_states
    if not hidden_states:
        raise ValueError("empty hidden_states list")
    stacked = torch.stack(list(hidden_states), dim=0)
    return stacked


def load_chunk_H(
    path: str | Path,
) -> Tuple[torch.Tensor, Dict[str, Any], Optional[Dict[str, torch.Tensor]]]:
    """Load a .pt chunk as H, meta, and optional legacy token masks.

    H stays in the on-disk dtype (bf16) and is memory-mapped when the file
    supports it. Callers should move slices to GPU rather than the full tensor.
    """
    if is_cst_hidden_file(path):
        H, meta = load_cst_hidden_tensor(path)
        return H, meta, None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError, OSError, ValueError):
        payload = torch.load(path, map_location="cpu", weights_only=False)
    hs = payload.get("hidden_states")
    if hs is None:
        raise KeyError(f"{path}: missing 'hidden_states'")
    meta = payload.get("meta") or {}
    H = records_to_H(hs)
    hs_list: Optional[List[Any]] = None if torch.is_tensor(hs) else list(hs)
    seg: Optional[Dict[str, torch.Tensor]] = None
    ul = payload.get("token_unmasked")
    ml = payload.get("token_diffusion_masked")
    if (
        ul is not None
        and ml is not None
        and hs_list is not None
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


def window_len(n_frames: int, w: int) -> int:
    """Effective CST window for a chunk with ``n_frames`` rows."""
    if w < 1:
        raise ValueError("w must be >= 1")
    t_max = int(n_frames) - 1
    if t_max < 1:
        return 0
    return min(int(w), t_max)


def iter_anchors(n_frames: int, w: int) -> List[int]:
    """Anchor indices matching :func:`iter_anchor_samples`."""
    w_eff = window_len(n_frames, w)
    if w_eff < 1:
        return []
    t_max = int(n_frames) - 1
    return list(range(w_eff, t_max + 1))


def stack_h_window(
    H: torch.Tensor,
    anchor: int,
    w: int,
    *,
    b0: int = 0,
    b1: Optional[int] = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build ``[mb, w_eff, L, D]`` for one anchor, optionally a row slice."""
    n = int(H.shape[0])
    w_eff = window_len(n, w)
    b1_i = int(H.shape[1]) if b1 is None else int(b1)
    slots = [H[anchor - w_eff + j, b0:b1_i] for j in range(w_eff)]
    h_win = torch.stack(slots, dim=1)
    if device is not None:
        h_win = h_win.to(device, non_blocking=True)
    return h_win


def iter_anchor_samples(
    H: torch.Tensor,
    w: int,
    *,
    device: torch.device | None = None,
) -> Iterator[Tuple[int, torch.Tensor]]:
    """Yield each training anchor and its [B, w, L, D] window."""
    for anchor in iter_anchors(int(H.shape[0]), w):
        yield anchor, stack_h_window(H, anchor, w, device=device)
