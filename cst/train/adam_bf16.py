"""Adam with bf16 moment buffers.

Halves optimizer state versus ``torch.optim.Adam`` (172 GB -> 86 GB for a
512-step k=4096 bank) so the whole CST bank fits across GPU + host RAM.

Parameters and gradients stay fp32; only ``exp_avg`` / ``exp_avg_sq`` are
stored bf16 and upcast inside the update. bf16 keeps fp32's exponent range but
has 8 mantissa bits, so a naive ``v = b2*v + (1-b2)*g^2`` with ``b2=0.999``
rounds the increment away entirely once ``v`` stabilises and the second moment
stagnates. Stochastic rounding on the store makes the EMA unbiased and is on by
default; ``stochastic_rounding=False`` selects plain (biased) truncation.

State is keyed by ``Parameter`` object, so callers that migrate a parameter
between devices must preserve its identity (assign ``p.data``, never
``module.to(device)``) and call :meth:`AdamBF16.state_to`.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer

__all__ = ["AdamBF16"]


def _store_bf16(x: Tensor, *, stochastic: bool) -> Tensor:
    """fp32 -> bf16. With ``stochastic``, round by perturbing the truncated mantissa bits."""
    if not stochastic:
        return x.to(torch.bfloat16)
    # bf16 is the high 16 bits of fp32; add uniform noise over the discarded low
    # 16 bits before truncating so the expected value is preserved.
    xi = x.contiguous().view(torch.int32)
    noise = torch.randint(0, 1 << 16, xi.shape, device=x.device, dtype=torch.int32)
    rounded = (xi + noise) & ~0xFFFF
    return rounded.view(torch.float32).to(torch.bfloat16)


class AdamBF16(Optimizer):
    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        *,
        stochastic_rounding: bool = True,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"lr must be >= 0, got {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"betas must be in [0, 1), got {betas}")
        if eps < 0.0:
            raise ValueError(f"eps must be >= 0, got {eps}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.stochastic_rounding = bool(stochastic_rounding)

    def state_to(self, param: Tensor, device: torch.device) -> None:
        """Move ``param``'s moment buffers to ``device`` (no-op if it has none yet)."""
        st = self.state.get(param)
        if not st:
            return
        for key in ("exp_avg", "exp_avg_sq"):
            buf = st.get(key)
            if isinstance(buf, Tensor) and buf.device != device:
                st[key] = buf.to(device, non_blocking=True)

    def drop_state(self, params: Iterable[Tensor]) -> None:
        """Discard moment buffers for ``params`` outright — no target device.

        For the lazy-bank training path: called when a step's window has
        closed for good and its parameters are about to be deleted, so its
        Adam moments don't linger in ``self.state`` after the ``Parameter``
        objects themselves are gone.
        """
        for p in params:
            self.state.pop(p, None)

    def state_bytes(self, param: Tensor) -> int:
        st = self.state.get(param)
        if not st:
            return 0
        return sum(
            buf.numel() * buf.element_size()
            for buf in (st.get("exp_avg"), st.get("exp_avg_sq"))
            if isinstance(buf, Tensor)
        )

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[Tensor]:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            wd = float(group["weight_decay"])

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamBF16 does not support sparse gradients")

                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros_like(p, dtype=torch.bfloat16)
                    st["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.bfloat16)
                if st["exp_avg"].device != p.device:
                    # Parameter was migrated without state_to; follow it rather than fail.
                    self.state_to(p, p.device)

                st["step"] += 1
                t = st["step"]
                g = grad.float()

                m = st["exp_avg"].float().lerp_(g, 1.0 - beta1)
                v = st["exp_avg_sq"].float().mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                st["exp_avg"] = _store_bf16(m, stochastic=self.stochastic_rounding)
                st["exp_avg_sq"] = _store_bf16(v, stochastic=self.stochastic_rounding)

                bc1 = 1.0 - beta1**t
                bc2 = 1.0 - beta2**t
                denom = (v / bc2).sqrt_().add_(eps)

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.addcdiv_(m, denom, value=-lr / bc1)

        return loss

    def state_dict(self) -> Dict[str, Any]:
        sd = super().state_dict()
        sd["stochastic_rounding"] = self.stochastic_rounding
        return sd

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        sr = state_dict.pop("stochastic_rounding", None)
        super().load_state_dict(state_dict)
        if sr is not None:
            self.stochastic_rounding = bool(sr)
        # Optimizer.load_state_dict casts float state to the param dtype (fp32);
        # put the moments back to bf16 so memory matches the design budget.
        for st in self.state.values():
            for key in ("exp_avg", "exp_avg_sq"):
                buf = st.get(key)
                if isinstance(buf, Tensor) and buf.dtype != torch.bfloat16:
                    st[key] = buf.to(torch.bfloat16)
