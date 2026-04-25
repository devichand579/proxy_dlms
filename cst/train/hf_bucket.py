"""Upload CST ``best`` (+ small ``state`` json) to a Hugging Face bucket; optionally delete local checkpoints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from cst.naming import train_checkpoint_names

logger = logging.getLogger("cst.hf_bucket")


def _bucket_dest(path_prefix: str, basename: str) -> str:
    p = (path_prefix or "").strip().replace("\\", "/").strip("/")
    if not p:
        return basename
    return f"{p}/{basename}"


def push_cst_best_to_hf_bucket(
    ckpt_dir: str | Path,
    *,
    name_token: Optional[str],
    bucket_id: str,
    path_prefix: str = "",
    token: bool | str | None = True,
) -> List[Tuple[str, str]]:
    """
    Upload ``cst_*_train_best.pt`` and ``cst_*_train_state.json`` (if present) to ``bucket_id``.

    Returns the list of (local_path, bucket_path) uploaded.
    """
    from huggingface_hub import HfApi

    bid = str(bucket_id).strip()
    if not bid:
        raise ValueError("bucket_id must be non-empty")

    ckpt_dir = Path(ckpt_dir)
    latest_n, best_n, state_n = train_checkpoint_names(name_token)
    best_p = ckpt_dir / best_n
    if not best_p.is_file():
        raise FileNotFoundError(f"missing best checkpoint: {best_p}")

    add: List[Tuple[str, str]] = [
        (str(best_p), _bucket_dest(path_prefix, best_n)),
    ]
    state_p = ckpt_dir / state_n
    if state_p.is_file():
        add.append((str(state_p), _bucket_dest(path_prefix, state_n)))

    api = HfApi()
    for local_path, repo_path in add:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=bid,
            token=token,
        )
    logger.info("HF bucket %s: uploaded %s", bid, [d for _, d in add])
    return add


def remove_local_train_checkpoints(
    ckpt_dir: str | Path,
    *,
    name_token: Optional[str],
) -> None:
    """Remove local ``latest``, ``best``, and ``state`` files for this train naming (ignore missing)."""
    ckpt_dir = Path(ckpt_dir)
    latest_n, best_n, state_n = train_checkpoint_names(name_token)
    for name in (latest_n, best_n, state_n):
        p = ckpt_dir / name
        if p.is_file():
            p.unlink()
            logger.info("removed local checkpoint file %s", p)


def finalize_cst_hf_bucket(
    ckpt_dir: str | Path,
    *,
    name_token: Optional[str],
    bucket_id: str,
    path_prefix: str = "",
    delete_local_after_push: bool = True,
    token: bool | str | None = True,
) -> bool:
    """
    Push best (+ state) to the HF bucket; optionally delete local latest/best/state.

    Returns True if a push was performed, False if skipped (no best file).
    """
    ckpt_dir = Path(ckpt_dir)
    _, best_n, _ = train_checkpoint_names(name_token)
    if not (ckpt_dir / best_n).is_file():
        logger.warning(
            "No best checkpoint in %s; skip HF bucket push and local cleanup.",
            ckpt_dir,
        )
        return False
    push_cst_best_to_hf_bucket(
        ckpt_dir,
        name_token=name_token,
        bucket_id=bucket_id,
        path_prefix=path_prefix,
        token=token,
    )
    if delete_local_after_push:
        remove_local_train_checkpoints(ckpt_dir, name_token=name_token)
    return True
