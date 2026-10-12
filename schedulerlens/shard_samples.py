"""Build lm-eval ``--samples`` files for deterministic dataset shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from traces.trace_metrics import TASK_SAMPLE_COUNTS


def sample_indices(
    task_name: str,
    num_shards: int,
    shard_index: int,
    *,
    limit: int | None = None,
) -> list[int]:
    if task_name not in TASK_SAMPLE_COUNTS:
        raise KeyError(f"unknown task sample count: {task_name}")
    if int(num_shards) < 1:
        raise ValueError("num_shards must be >= 1")
    if not 0 <= int(shard_index) < int(num_shards):
        raise ValueError("shard_index must satisfy 0 <= index < num_shards")
    count = int(TASK_SAMPLE_COUNTS[task_name])
    if limit is not None:
        if int(limit) < 1:
            raise ValueError("limit must be >= 1")
        count = min(count, int(limit))
    return list(range(int(shard_index), count, int(num_shards)))


def write_samples_file(
    path: Path,
    task_name: str,
    num_shards: int,
    shard_index: int,
    *,
    limit: int | None = None,
) -> Path:
    indices = sample_indices(
        task_name,
        num_shards,
        shard_index,
        limit=limit,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({task_name: indices}) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_samples_file(
        args.output,
        args.task,
        args.num_shards,
        args.shard_index,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
