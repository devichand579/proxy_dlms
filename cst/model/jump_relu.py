"""JumpReLU with straight-through estimator on the threshold."""

from __future__ import annotations

import torch
from torch import nn


class JumpReLUSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, threshold: torch.Tensor, bandwidth: float) -> torch.Tensor:
        assert x.shape[-1] == threshold.shape[0], (
            f"Last dim of x {x.shape[-1]} must match threshold size {threshold.shape[0]}"
        )
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return x * (x > threshold).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, threshold = ctx.saved_tensors
        beta = ctx.bandwidth
        mask = (x > threshold).to(grad_output.dtype)
        grad_x = grad_output * mask
        in_window = ((x - threshold).abs() < beta / 2).to(grad_output.dtype)
        grad_thr = -(x / beta) * in_window * grad_output
        grad_thr = grad_thr.sum(dim=tuple(range(grad_thr.dim() - 1)))
        return grad_x, grad_thr, None


class JumpReLU(nn.Module):
    def __init__(
        self,
        num_features: int,
        initial_threshold: float = 0.03,
        bandwidth: float = 1.0,
    ) -> None:
        super().__init__()
        self.bandwidth = bandwidth
        self.threshold = nn.Parameter(torch.full((num_features,), float(initial_threshold)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return JumpReLUSTE.apply(x, self.threshold, self.bandwidth)
