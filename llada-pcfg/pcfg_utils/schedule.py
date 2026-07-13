"""Learning-rate schedule utilities."""

from __future__ import annotations


def wsd_lr(
    step: int,
    *,
    warmup_steps: int,
    stable_end_step: int,
    decay_mid_step: int,
    total_steps: int,
    peak_lr: float,
    mid_lr: float,
    min_lr: float,
) -> float:
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(warmup_steps, 1)
    if step < stable_end_step:
        return peak_lr
    if step < decay_mid_step:
        frac = (step - stable_end_step) / max(decay_mid_step - stable_end_step, 1)
        return peak_lr + (mid_lr - peak_lr) * frac
    if step < total_steps:
        frac = (step - decay_mid_step) / max(total_steps - decay_mid_step, 1)
        return mid_lr + (min_lr - mid_lr) * frac
    return min_lr


def resolve_warmup_steps(
    total_steps: int,
    *,
    warmup_steps: int | None,
    warmup_fraction: float,
    warmup_max: int,
) -> int:
    if warmup_steps is not None:
        return max(1, warmup_steps)
    return max(1, min(warmup_max, int(total_steps * warmup_fraction)))
