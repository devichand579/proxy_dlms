"""One CST per diffusion step with optional LRU GPU residency."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List

import torch
from torch import nn

from .cst import CST


def effective_window_size(step: int, w: int, max_steps: int) -> int:
    """How many ``decode(1)…`` slots are valid for bank key ``step`` (cap near trajectory end)."""
    return min(w, max(0, max_steps - step))


class CSTBank(nn.Module):
    def __init__(
        self,
        max_steps: int,
        d: int,
        k: int,
        w: int,
        max_on_gpu: int = 32,
        prefetch: int = 8,
        initial_threshold: float = 0.03,
        bandwidth: float = 1.0,
    ) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.d = d
        self.k = k
        self.w = w
        self.max_on_gpu = max_on_gpu
        self.prefetch = prefetch

        self.csts = nn.ModuleDict()
        for t in range(1, max_steps + 1):
            w_t = effective_window_size(t, w, max_steps)
            self.csts[str(t)] = CST(d, k, w_t, initial_threshold, bandwidth)

        self._on_gpu: "OrderedDict[str, bool]" = OrderedDict()
        self._device = torch.device("cpu")

    def set_device(self, device: torch.device) -> None:
        self._device = device

    def _evict_lru(self) -> None:
        if len(self._on_gpu) < self.max_on_gpu:
            return
        lru_key, _ = self._on_gpu.popitem(last=False)
        self.csts[lru_key].cpu()

    def bring_to_gpu(self, step: int) -> None:
        key = str(step)
        if key in self._on_gpu:
            self._on_gpu.move_to_end(key)
            return
        self._evict_lru()
        self.csts[key].to(self._device)
        self._on_gpu[key] = True

    def prefetch_steps(self, steps: List[int]) -> None:
        for t in steps:
            if len(self._on_gpu) < self.max_on_gpu:
                self.bring_to_gpu(t)

    def get(self, step: int) -> CST:
        self.bring_to_gpu(step)
        return self.csts[str(step)]

    def state_dict_for_save(self) -> Dict[str, object]:
        return {str(t): self.csts[str(t)].state_dict() for t in range(1, self.max_steps + 1)}

    def load_state_dict_from_save(self, state: Dict[str, object]) -> None:
        for key, sd in state.items():
            self.csts[str(key)].load_state_dict(sd)

    def to_device_all(self, device: torch.device) -> None:
        """Place every CST on ``device`` and relax LRU so training does not evict."""
        self.set_device(device)
        self.max_on_gpu = max(self.max_on_gpu, self.max_steps)
        self._on_gpu.clear()
        for t in range(1, self.max_steps + 1):
            self.csts[str(t)].to(device)
            self._on_gpu[str(t)] = True
