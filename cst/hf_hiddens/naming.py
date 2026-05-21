"""HF Hub repo ids and paths for hidden .pt chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from cst.naming import fs_token

TRAIN_REPO_SUFFIX = "hiddens-train"
EVAL_REPO_SUFFIX = "hiddens-eval"


def _repo_owner(org: str) -> str:
    o = fs_token(str(org).strip(), max_len=64)
    if not o:
        raise ValueError("org is required to build a Hub repo id")
    return o


def train_hf_repo_id(
    *,
    org: str,
    repo_id: Optional[str] = None,
) -> str:
    if repo_id and str(repo_id).strip():
        return str(repo_id).strip()
    return f"{_repo_owner(org)}/{TRAIN_REPO_SUFFIX}"


def eval_hf_repo_id(
    *,
    org: str,
    repo_id: Optional[str] = None,
) -> str:
    if repo_id and str(repo_id).strip():
        return str(repo_id).strip()
    return f"{_repo_owner(org)}/{EVAL_REPO_SUFFIX}"


def manifest_hub_path(name_token: str) -> str:
    return f"{fs_token(name_token, max_len=200)}/manifest.json"


def chunks_dir_hub(name_token: str) -> str:
    return f"{fs_token(name_token, max_len=200)}/chunks"


def chunk_hub_path(name_token: str, filename: str) -> str:
    base = Path(filename).name
    if base != filename or not base.endswith(".pt"):
        raise ValueError(f"invalid chunk filename: {filename!r}")
    return f"{chunks_dir_hub(name_token)}/{base}"
