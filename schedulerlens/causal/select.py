"""Map one diffusion step onto the live scheduler's position set."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from schedulerlens.causal.scores import ScoreBundle
from schedulers.dus import dilated_unmask_levels, merge_last_level
from schedulers.utils import get_num_transfer_tokens

SUPPORTED_PHI = ("pd", "bd", "top_k_c", "top_k_m", "eb", "dus")

_BLOCK_DEFAULT = {
    "pd": None,  # full gen length
    "bd": 32,
    "top_k_c": None,
    "top_k_m": None,
    "eb": None,
    "dus": 16,
}


class SelectError(ValueError):
    """Unsupported scheduler rule or bad selection inputs."""


def require_finite(label: str, *arrays: np.ndarray) -> None:
    """Raise when a score that chooses positions or enters M̂ or Ĵ is not finite."""
    for values in arrays:
        if not np.isfinite(np.asarray(values)).all():
            raise SelectError(f"{label} has a non-finite score")


def default_block_length(phi: str, gen_length: int, block_length: Optional[int]) -> int:
    # A block that covers the whole canvas is the launcher default, not an override.
    if block_length is not None and 0 < int(block_length) < int(gen_length):
        return int(block_length)
    fallback = _BLOCK_DEFAULT.get(phi)
    if fallback is None or fallback > gen_length:
        return int(gen_length)
    return int(fallback)


def schedule_window(
    phi: str,
    *,
    gen_length: int,
    steps: int,
    step: int,
    block_length: Optional[int] = None,
) -> tuple[int, int, int, int]:
    """Return ``(block_start, block_end, local_step, steps_in_block)`` in gen coords.

    Assumes a uniform step budget per block (no early-stop remapping).
    """
    bl = default_block_length(phi, gen_length, block_length)
    if bl >= gen_length:
        return 0, gen_length, int(step), int(steps)
    if gen_length % bl != 0:
        raise SelectError(f"gen_length {gen_length} not divisible by block_length {bl}")
    n_blocks = gen_length // bl
    if steps % n_blocks != 0:
        raise SelectError(f"steps {steps} not divisible by n_blocks {n_blocks}")
    spb = steps // n_blocks
    block = min(int(step) // spb, n_blocks - 1)
    local = int(step) - block * spb
    return block * bl, (block + 1) * bl, local, spb


def _llada_step_k(block_length: int, steps_in_block: int, local_step: int) -> int:
    """Column of ``get_num_transfer_tokens`` for a block that starts fully masked."""
    if block_length <= 0 or steps_in_block <= 0 or local_step < 0:
        return 0
    if local_step >= steps_in_block:
        return 0
    mask = torch.ones(1, int(block_length), dtype=torch.bool)
    counts = get_num_transfer_tokens(mask, int(steps_in_block))
    return int(counts[0, int(local_step)].item())


def _confidence_row(values: np.ndarray, mask: np.ndarray, block_end: int) -> torch.Tensor:
    """Confidence with -inf past the block end and on positions that are not masked."""
    scores = torch.as_tensor(np.asarray(values), dtype=torch.float32).unsqueeze(0)
    masked = torch.as_tensor(np.asarray(mask), dtype=torch.bool).unsqueeze(0)
    scoped = scores.clone()
    scoped[:, int(block_end) :] = -torch.inf
    return torch.where(masked, scoped, torch.full_like(scoped, -torch.inf))[0]


def _topk_row(confidence_row: torch.Tensor, k: int) -> torch.Tensor:
    chosen = torch.zeros_like(confidence_row, dtype=torch.bool)
    if int(k) > 0:
        _, select_index = torch.topk(confidence_row, k=int(k), dim=-1)
        chosen[select_index] = True
    return chosen


def _commit(chosen: torch.Tensor, mask: np.ndarray) -> np.ndarray:
    """Keep positions that are still masked. An unmasked top-k hit is a no-op write."""
    return chosen.detach().cpu().numpy() & np.asarray(mask, dtype=bool)


def select_gen(
    phi: str,
    scores: ScoreBundle,
    mask_gen: np.ndarray,
    *,
    step: int,
    steps: int,
    block_length: Optional[int] = None,
    top_k: int = 2,
    gamma: float = 0.1,
    eb_rank: str = "confidence",
    dus_confidence_threshold: float = 0.1,
) -> np.ndarray:
    """Boolean gen-region selection for one sequence."""
    if phi not in SUPPORTED_PHI:
        raise SelectError(
            f"scheduler {phi!r} is not supported (v1: {', '.join(SUPPORTED_PHI)})"
        )
    mask = np.asarray(mask_gen, dtype=bool)
    n = int(mask.shape[0])
    if scores.conf.shape[0] != n:
        raise SelectError(
            f"score length {scores.conf.shape[0]} != gen mask {n}"
        )
    require_finite("confidence", scores.conf)
    if phi == "top_k_m" or (phi == "eb" and eb_rank == "margin"):
        require_finite("margin", scores.margin)
    if phi == "eb":
        require_finite("EB entropy", scores.eb_h)
        if eb_rank == "entropy":
            require_finite("rank entropy", scores.h_rank)
    b0, b1, local, spb = schedule_window(
        phi, gen_length=n, steps=steps, step=step, block_length=block_length
    )
    eligible = mask.copy()
    eligible[:b0] = False
    eligible[b1:] = False
    chosen = np.zeros(n, dtype=bool)

    if phi in ("pd", "bd"):
        k = _llada_step_k(b1 - b0, spb, local)
        return _commit(_topk_row(_confidence_row(scores.conf, mask, b1), k), mask)

    if phi in ("top_k_c", "top_k_m"):
        rank = scores.conf if phi == "top_k_c" else scores.margin
        k = min(int(top_k), int(mask[b0:b1].sum()))
        return _commit(_topk_row(_confidence_row(rank, mask, b1), k), mask)

    if phi == "eb":
        if eb_rank == "confidence":
            rank = scores.conf
        elif eb_rank == "margin":
            rank = scores.margin
        elif eb_rank == "entropy":
            rank = -scores.h_rank
        else:
            raise SelectError(f"unknown EB rank {eb_rank!r}")
        block_positions = [pos for pos in range(b0, b1) if mask[pos]]
        if not block_positions:
            return chosen
        block_positions = sorted(
            block_positions,
            key=lambda pos: float(rank[pos]),
            reverse=True,
        )
        accumulated: list[float] = []
        for pos in block_positions:
            accumulated.append(float(scores.eb_h[pos]))
            if sum(accumulated) - max(accumulated) <= float(gamma):
                chosen[pos] = True
            else:
                break
        # generate_EB keeps the highest-ranked token when the constraint admits none.
        if not chosen.any():
            chosen[max(block_positions, key=lambda pos: float(rank[pos]))] = True
        return chosen

    # Levels come from the live dilation schedule. A longer step budget repeats the last one.
    levels = merge_last_level(dilated_unmask_levels(b0, b1 - 1))
    if not levels or local < 0:
        return chosen
    index = min(local, len(levels) - 1)
    final = local >= len(levels) - 1
    for pos in levels[index]:
        if not (0 <= pos < n and eligible[pos]):
            continue
        if not final and float(scores.conf[pos]) < float(dus_confidence_threshold):
            continue
        chosen[pos] = True
    return chosen


def select_set(
    phi: str,
    scores: ScoreBundle,
    *,
    prompt_len: int,
    mask_full: np.ndarray,
    step: int,
    steps: int,
    block_length: Optional[int] = None,
    top_k: int = 2,
    gamma: float = 0.1,
    eb_rank: str = "confidence",
    dus_confidence_threshold: float = 0.1,
) -> np.ndarray:
    """Boolean mask over the full sequence (prompt positions stay False)."""
    full = np.asarray(mask_full, dtype=bool)
    gen_mask = full[prompt_len:]
    gen_scores = scores.gen(prompt_len)
    gen_sel = select_gen(
        phi,
        gen_scores,
        gen_mask,
        step=step,
        steps=steps,
        block_length=block_length,
        top_k=top_k,
        gamma=gamma,
        eb_rank=eb_rank,
        dus_confidence_threshold=dus_confidence_threshold,
    )
    out = np.zeros(full.shape[0], dtype=bool)
    out[prompt_len:] = gen_sel
    return out


def candidate_sets(
    phis: Sequence[str],
    scores: ScoreBundle,
    mask_gen: np.ndarray,
    **kwargs,
) -> dict[str, np.ndarray]:
    return {phi: select_gen(phi, scores, mask_gen, **kwargs) for phi in phis}
