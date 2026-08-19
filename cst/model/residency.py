"""Explicit GPU residency for a :class:`~cst.model.cst_bank.CSTBank`.

Replaces the former LRU cache in ``cst_bank.py``. An LRU is a policy for an
*unknown* future; here the future is known exactly — training sweeps anchors in
increasing order and anchor ``t`` touches steps ``t-w+1 .. t``, so a step is
dead the moment the sweep passes it. A sliding window over that schedule is
optimal and needs no bookkeeping.

Two tiers:

* **pinned** — a prefix of steps that stays on device for the whole run, sized
  from a byte budget. These never cross the bus.
* **streamed** — the remainder, brought in by :meth:`ensure` when a block needs
  them and pushed back to host by :meth:`release` when their window closes.

Parameters are migrated by assigning ``p.data`` rather than calling
``module.to(device)``. ``nn.Module._apply`` builds a *new* ``Parameter`` when
the device changes, which would orphan optimizer state keyed by parameter
identity; assigning ``.data`` keeps the object and lets
:meth:`~cst.train.adam_bf16.AdamBF16.state_to` follow it.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Sequence, Set

import torch
from torch import nn

from .cst_bank import CSTBank

logger = logging.getLogger("cst.residency")

__all__ = ["CSTResidency", "module_param_bytes"]


def module_param_bytes(mod: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in mod.parameters())


def _move_module(mod: nn.Module, device: torch.device, *, drop_grad: bool) -> None:
    """Move ``mod``'s tensors to ``device`` preserving ``Parameter`` identity."""
    with torch.no_grad():
        for p in mod.parameters(recurse=True):
            if p.data.device != device:
                p.data = p.data.to(device, non_blocking=True)
            if drop_grad:
                p.grad = None
            elif p.grad is not None and p.grad.device != device:
                p.grad = p.grad.to(device, non_blocking=True)
        for name, buf in list(mod.named_buffers(recurse=True)):
            if buf is not None and buf.device != device:
                owner = mod
                *path, leaf = name.split(".")
                for part in path:
                    owner = getattr(owner, part)
                setattr(owner, leaf, buf.to(device, non_blocking=True))


class CSTResidency:
    """Streams CST modules (and their optimizer moments) between host and device."""

    def __init__(
        self,
        bank: CSTBank,
        device: torch.device,
        *,
        optimizer: Optional[torch.optim.Optimizer] = None,
        pinned_bytes: int = 0,
        host_device: Optional[torch.device] = None,
    ) -> None:
        self.bank = bank
        self.device = torch.device(device)
        self.host = torch.device(host_device) if host_device is not None else torch.device("cpu")
        self.optimizer = optimizer
        self.bank.set_device(self.device)

        self._resident: Set[int] = set()
        self.pinned: Set[int] = set()

        per_step = module_param_bytes(bank.get(1)) if bank.max_steps >= 1 else 0
        # Moments are bf16 (half of fp32 params) for both exp_avg and exp_avg_sq,
        # i.e. one params-worth of extra bytes travelling with each step.
        self._bytes_per_step = per_step * 2 if optimizer is not None else per_step
        n_pinned = 0
        if self._bytes_per_step > 0 and pinned_bytes > 0:
            n_pinned = min(bank.max_steps, int(pinned_bytes // self._bytes_per_step))

        for t in range(1, n_pinned + 1):
            self.pinned.add(t)

        # Everything starts on host; pinned steps go up front and stay.
        for t in bank.steps():
            _move_module(bank.get(t), self.host, drop_grad=True)
        if self.pinned:
            self.ensure(sorted(self.pinned))

        logger.info(
            "residency: %d/%d steps pinned on %s (%.1f GB), %d streamed, %.3f GB per step",
            len(self.pinned),
            bank.max_steps,
            self.device,
            len(self.pinned) * self._bytes_per_step / 1e9,
            bank.max_steps - len(self.pinned),
            self._bytes_per_step / 1e9,
        )

    # -- residency ---------------------------------------------------------

    def ensure(self, steps: Iterable[int]) -> None:
        """Make ``steps`` resident on the training device."""
        for t in steps:
            if t in self._resident:
                continue
            mod = self.bank.get(t)
            _move_module(mod, self.device, drop_grad=False)
            if self.optimizer is not None:
                self._state_to(mod, self.device)
            self._resident.add(t)

    def release(self, step: int) -> None:
        """Push ``step`` back to host and drop its gradient. Pinned steps only drop the grad."""
        if step not in self._resident:
            return
        mod = self.bank.get(step)
        if step in self.pinned:
            for p in mod.parameters():
                p.grad = None
            return
        _move_module(mod, self.host, drop_grad=True)
        if self.optimizer is not None:
            self._state_to(mod, self.host)
        self._resident.discard(step)

    def release_all(self) -> None:
        for t in list(self._resident):
            self.release(t)

    def _state_to(self, mod: nn.Module, device: torch.device) -> None:
        assert self.optimizer is not None
        mover = getattr(self.optimizer, "state_to", None)
        if mover is None:
            return
        for p in mod.parameters():
            mover(p, device)

    # -- introspection -----------------------------------------------------

    @property
    def resident_steps(self) -> List[int]:
        return sorted(self._resident)

    @property
    def bytes_per_step(self) -> int:
        return self._bytes_per_step

    def resident_bytes(self) -> int:
        return len(self._resident) * self._bytes_per_step

    def describe(self) -> Dict[str, float]:
        return {
            "pinned_steps": float(len(self.pinned)),
            "resident_steps": float(len(self._resident)),
            "bytes_per_step_gb": self._bytes_per_step / 1e9,
            "resident_gb": self.resident_bytes() / 1e9,
        }


def steps_for_anchors(anchors: Sequence[int], w: int) -> List[int]:
    """Distinct bank steps touched by ``anchors`` under the ``w``-wide window rule."""
    need: Set[int] = set()
    for a in anchors:
        for slot in range(w):
            s = a - w + 1 + slot
            if s >= 1:
                need.add(s)
    return sorted(need)


def budget_report(
    *,
    max_steps: int,
    d: int,
    k: int,
    w: int,
    anchor_block: int,
    pinned_gb: float,
    batch: int,
    seq_len: int,
    has_moments: bool = True,
) -> Dict[str, float]:
    """Predicted GPU / host footprint for a training config, in GB.

    Cheap to compute and worth logging before the first chunk: an
    under-provisioned ``gpu_pinned_gb`` fails as a *host* OOM part-way through a
    run, because the parked remainder of the bank plus the chunk tensor has to
    fit in system RAM.
    """
    per = d * k * (1 + w) + k
    params = per * 4 / 1e9
    moments = (per * 2 * 2 / 1e9) if has_moments else 0.0
    persistent = params + moments
    total = persistent * max_steps
    pinned = min(float(pinned_gb), total)

    live = min(max_steps, anchor_block + w - 1)
    row = batch * seq_len * d * 4 / 1e9
    # Retained per anchor with the checkpointed sparsity penalty: 8 [B,L,k] and
    # 36 [B,L,d] tensors (see CST.decoder_sparsity_penalty).
    act = (8 * k + 36 * d) * batch * seq_len * 4 / 1e9
    gpu = pinned + live * params + act * anchor_block + 2 * live * row + 1.5
    host = max(0.0, total - pinned) + (max_steps + 1) * row + 6.0
    return {
        "per_cst_gb": persistent,
        "bank_total_gb": total,
        "pinned_gb": pinned,
        "live_csts": float(live),
        "gpu_gb": gpu,
        "host_gb": host,
        "act_per_anchor_gb": act,
    }
