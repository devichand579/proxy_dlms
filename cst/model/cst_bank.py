"""One CST per diffusion step.

GPU residency is not managed here: :class:`CSTBank` is a plain container and
``get`` is a direct lookup. Training streams CSTs between host and device via
:class:`cst.model.residency.CSTResidency`, which knows the anchor sweep order;
eval and TAG place the whole bank with :meth:`CSTBank.to_device_all`.
"""

from __future__ import annotations

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
        lazy: bool = False,
    ) -> None:
        """``max_on_gpu`` and ``prefetch`` are accepted for call-site compatibility and ignored.

        They configured a former LRU cache that never evicted (every consumer
        called :meth:`to_device_all` first, and ``max_on_gpu == max_steps`` in
        ``config/cst.yaml``). Residency is now explicit; see the module docstring.

        ``lazy=True`` skips eagerly constructing all ``max_steps`` CSTs (each is
        ~pure-random-init, no I/O, so this is cheap to defer) — see
        :meth:`get_or_create` / :meth:`evict`, used by the lazy-bank training path
        that materializes/frees a step exactly when its sliding window opens/closes.
        """
        super().__init__()
        self.max_steps = max_steps
        self.d = d
        self.k = k
        self.w = w
        self.lazy = bool(lazy)
        self._initial_threshold = initial_threshold
        self._bandwidth = bandwidth

        self.csts = nn.ModuleDict()
        if not self.lazy:
            for t in range(1, max_steps + 1):
                w_t = effective_window_size(t, w, max_steps)
                self.csts[str(t)] = CST(d, k, w_t, initial_threshold, bandwidth)

        self._device = torch.device("cpu")

    def set_device(self, device: torch.device) -> None:
        """Record the device that :meth:`to_device_all` and the residency manager target."""
        self._device = device

    @property
    def device(self) -> torch.device:
        return self._device

    def get(self, step: int) -> CST:
        """The CST for ``step``. Caller is responsible for residency."""
        return self.csts[str(step)]

    def has(self, step: int) -> bool:
        """Whether ``step`` currently exists in the bank.

        Always ``True`` for an eager (``lazy=False``) bank. For a lazy bank,
        ``False`` before first touch or after :meth:`evict`.
        """
        return str(step) in self.csts

    def get_or_create(self, step: int, device: torch.device | None = None) -> CST:
        """Lazy-bank only: return ``step``'s CST, constructing it on first touch.

        Construction is pure random init (no I/O), so this is cheap to call
        synchronously right before a step's window is needed.
        """
        key = str(step)
        if key in self.csts:
            return self.csts[key]
        w_t = effective_window_size(step, self.w, self.max_steps)
        mod = CST(self.d, self.k, w_t, self._initial_threshold, self._bandwidth)
        if device is not None:
            mod.to(device)
        self.csts[key] = mod
        return mod

    def evict(self, step: int) -> CST:
        """Lazy-bank only: remove and return ``step``'s CST (its window has closed for good).

        Raises ``KeyError`` if ``step`` was never created — evicting an absent
        step is a bug at the call site, not a silent no-op.
        """
        return self.csts.pop(str(step))

    def step_device(self, step: int) -> torch.device:
        """Device the parameters of ``step`` currently live on."""
        return self.csts[str(step)].W_enc.weight.device

    def steps(self) -> List[int]:
        return list(range(1, self.max_steps + 1))

    def state_dict_for_save(self) -> Dict[str, object]:
        return {str(t): self.csts[str(t)].state_dict() for t in range(1, self.max_steps + 1)}

    def load_state_dict_from_save(self, state: Dict[str, object]) -> None:
        for key, sd in state.items():
            self.csts[str(key)].load_state_dict(sd)

    def to_device_all(self, device: torch.device) -> None:
        """Place every CST on ``device``.

        Fine for small banks and for eval/TAG when the whole bank fits. At
        ``max_steps=512, k=4096`` this is ~172 GB of fp32 parameters; use
        :class:`cst.model.residency.CSTResidency` when it does not fit.
        """
        self.set_device(device)
        for t in range(1, self.max_steps + 1):
            self.csts[str(t)].to(device)
