"""Distributed training helpers."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 0, 1
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def is_main(rank: int) -> bool:
    return rank == 0


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model
