"""Dataset splitting utilities."""

from __future__ import annotations

import random


def split_train_val(
    token_seqs: list,
    *,
    val_fraction: float = 0.05,
    seed: int = 0,
) -> tuple[list, list]:
    """Hold out a random fraction of strings for validation."""
    if not token_seqs:
        return [], []
    if val_fraction <= 0.0:
        return list(token_seqs), []
    n_val = max(1, int(round(len(token_seqs) * val_fraction)))
    if n_val >= len(token_seqs):
        n_val = max(1, len(token_seqs) // 20)
    rng = random.Random(seed)
    indices = list(range(len(token_seqs)))
    rng.shuffle(indices)
    val_idx = set(indices[:n_val])
    train = [token_seqs[i] for i in range(len(token_seqs)) if i not in val_idx]
    val = [token_seqs[i] for i in val_idx]
    return train, val
