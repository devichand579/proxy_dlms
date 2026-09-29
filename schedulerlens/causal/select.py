"""Scheduler rules as pure set selectors (no token writes)."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from schedulerlens.causal.scores import ScoreBundle
from schedulers.dus import dilated_unmask_levels, merge_last_level

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


def _transfer_count(n_masked_block: int, local_step: int, steps_in_block: int) -> int:
    if n_masked_block <= 0 or steps_in_block <= 0:
        return 0
    base = n_masked_block // steps_in_block
    rem = n_masked_block % steps_in_block
    k = base + (1 if local_step < rem else 0)
    return int(min(max(k, 0), n_masked_block))


def _topk_masked(score: np.ndarray, eligible: np.ndarray, k: int) -> np.ndarray:
    chosen = np.zeros(score.shape[0], dtype=bool)
    idx = np.flatnonzero(eligible)
    if k <= 0 or idx.size == 0:
        return chosen
    k = min(int(k), int(idx.size))
    order = idx[np.argsort(-score[idx], kind="stable")]
    chosen[order[:k]] = True
    return chosen


def _dus_level(
    block_start: int,
    block_end: int,
    local_step: int,
) -> list[int]:
    levels = merge_last_level(dilated_unmask_levels(block_start, block_end - 1))
    if not levels:
        return []
    if local_step < 0:
        return []
    if local_step >= len(levels):
        return list(levels[-1])
    return list(levels[local_step])


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
    b0, b1, local, spb = schedule_window(
        phi, gen_length=n, steps=steps, step=step, block_length=block_length
    )
    eligible = mask.copy()
    eligible[:b0] = False
    eligible[b1:] = False
    chosen = np.zeros(n, dtype=bool)

    if phi in ("pd", "bd"):
        n_block = int(eligible.sum())
        k = _transfer_count(n_block, local, spb)
        return _topk_masked(scores.conf, eligible, k)

    if phi == "top_k_c":
        k = min(int(top_k), int(eligible.sum()))
        return _topk_masked(scores.conf, eligible, k)

    if phi == "top_k_m":
        k = min(int(top_k), int(eligible.sum()))
        return _topk_masked(scores.margin, eligible, k)

    if phi == "eb":
        if eb_rank == "confidence":
            rank = scores.conf
        elif eb_rank == "margin":
            rank = scores.margin
        elif eb_rank == "entropy":
            rank = -scores.entropy
        else:
            raise SelectError(f"unknown EB rank {eb_rank!r}")
        idx = np.flatnonzero(eligible)
        if idx.size == 0:
            return chosen
        order = idx[np.argsort(-rank[idx], kind="stable")]
        acc: list[float] = []
        for pos in order:
            acc.append(float(scores.entropy[pos]))
            if sum(acc) - max(acc) <= float(gamma):
                chosen[pos] = True
            else:
                break
        if not chosen.any():
            chosen[order[0]] = True
        return chosen

    # dus: schedule positions inside the block (absolute gen indices)
    level = _dus_level(b0, b1, local)
    for pos in level:
        if 0 <= pos < n and eligible[pos]:
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
