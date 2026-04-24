"""One CST module: linear encoder -> JumpReLU -> ``w`` linear decoders (offsets 1…w)."""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn

from .jump_relu import JumpReLU


class CST(nn.Module):
    """Encoder maps hidden ``d→k``; decoder ``offset`` (1…w) maps ``k→d`` for that lookahead step."""

    def __init__(
        self,
        d: int,
        k: int,
        w: int,
        initial_threshold: float = 0.03,
        bandwidth: float = 1.0,
    ) -> None:
        super().__init__()
        self.d = d
        self.k = k
        self.w = w

        self.W_enc = nn.Linear(d, k, bias=False)
        self.W_dec = nn.ModuleList([nn.Linear(k, d, bias=False) for _ in range(w)])
        self.jump = JumpReLU(k, initial_threshold, bandwidth)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.uniform_(self.W_enc.weight, -1.0 / self.k, 1.0 / self.k)
        if self.w > 0:
            bound = 1.0 / (self.w * self.d)
            for dec in self.W_dec:
                nn.init.uniform_(dec.weight, -bound, bound)

    def encode(self, H: torch.Tensor) -> torch.Tensor:
        return self.jump(self.W_enc(H))

    def decode(self, A: torch.Tensor, offset: int) -> torch.Tensor:
        """``offset`` in ``1..w`` picks ``W_dec[offset-1]``."""
        if self.w == 0:
            raise ValueError("CST has no decoders (w=0).")
        if not (1 <= offset <= self.w):
            raise ValueError(f"decode offset must be in 1..{self.w}, got {offset}")
        return self.W_dec[offset - 1](A)

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        A = self.encode(H)
        y_hats = [self.decode(A, j) for j in range(1, self.w + 1)]
        return A, y_hats

    def decoder_sparsity_penalty(self, A: torch.Tensor, c: float = 1.0) -> torch.Tensor:
        """Sparsity via tanh on column-weighted A; A is [B,L,k]."""
        if A.shape[-1] != self.k:
            raise ValueError(f"A last dim must be k={self.k}, got {A.shape}")
        penalty = torch.zeros((), device=A.device, dtype=torch.float32)
        Af = A.float()
        c_f = float(c)
        for dec in self.W_dec:
            col_norms = torch.linalg.norm(dec.weight.float(), dim=0)
            inner = c_f * col_norms.view(1, 1, -1) * Af
            penalty = penalty + torch.tanh(inner).sum()
        return penalty
