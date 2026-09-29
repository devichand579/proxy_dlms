"""Azure container names and paths for hidden .pt chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

TRAIN_CONTAINER = "hiddens-train"
EVAL_CONTAINER = "hiddens-eval"


def train_azure_container(container: Optional[str] = None) -> str:
    if container and str(container).strip():
        return str(container).strip()
    return TRAIN_CONTAINER


def eval_azure_container(container: Optional[str] = None) -> str:
    if container and str(container).strip():
        return str(container).strip()
    return EVAL_CONTAINER


def manifest_blob_path(name_token: str) -> str:
    return f"{name_token}/manifest.json"


def chunks_dir_blob(name_token: str) -> str:
    return f"{name_token}/chunks"


def chunk_blob_path(name_token: str, filename: str) -> str:
    base = Path(filename).name
    if base != filename or not base.endswith(".pt"):
        raise ValueError(f"invalid chunk filename: {filename!r}")
    return f"{chunks_dir_blob(name_token)}/{base}"
