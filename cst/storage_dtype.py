"""On-disk float format for CST artifacts (hiddens ``.pt`` and training checkpoints)."""

from __future__ import annotations

from typing import Any

import torch

STORAGE_FLOAT_DTYPE = torch.bfloat16


def _to_storage_tensor(t: torch.Tensor) -> torch.Tensor:
    if not t.is_floating_point():
        return t
    return t.to(STORAGE_FLOAT_DTYPE)


def _from_storage_tensor(t: torch.Tensor) -> torch.Tensor:
    if not t.is_floating_point():
        return t
    return t.float()


def map_float_tensors(obj: Any, *, to_storage: bool) -> Any:
    """Recursively cast float tensors: to bf16 for save, to fp32 for load into modules/optimizers."""
    if isinstance(obj, torch.Tensor):
        return _to_storage_tensor(obj) if to_storage else _from_storage_tensor(obj)
    if isinstance(obj, dict):
        return {k: map_float_tensors(v, to_storage=to_storage) for k, v in obj.items()}
    if isinstance(obj, list):
        return [map_float_tensors(v, to_storage=to_storage) for v in obj]
    if isinstance(obj, tuple):
        return tuple(map_float_tensors(v, to_storage=to_storage) for v in obj)
    return obj
