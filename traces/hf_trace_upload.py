"""HF Hub upload/download for eval traces (default repo: SchedulerLens)."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_DATASET_REPO = "schedulerlens"


def _to_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "yes", "on")
    return bool(val)


def tracing_enabled_from_env() -> bool:
    return os.environ.get("TRACING_ENABLED", "false").strip().lower() == "true"


def trace_hf_upload_enabled() -> bool:
    """Hub upload on when TRACE_HF_UPLOAD is set, else follows tracing."""
    raw = os.environ.get("TRACE_HF_UPLOAD")
    if raw is None or not str(raw).strip():
        return tracing_enabled_from_env()
    return _to_bool(raw)


def trace_hf_dataset_repo() -> str:
    return os.environ.get("TRACE_HF_DATASET_REPO", DEFAULT_DATASET_REPO).strip() or DEFAULT_DATASET_REPO


def trace_hf_dataset_private() -> bool:
    return _to_bool(os.environ.get("TRACE_HF_DATASET_PRIVATE", "false"))


def trace_hf_revision() -> str:
    return os.environ.get("TRACE_HF_REVISION", "main").strip() or "main"


def normalize_dataset_repo_id(repo: str, namespace: Optional[str] = None) -> str:
    """Return ``namespace/repo``; infer namespace from env or HF whoami."""
    repo = (repo or "").strip()
    if not repo:
        raise ValueError("trace HF dataset repo id is empty")
    if "/" in repo:
        return repo
    ns = (namespace or os.environ.get("TRACE_HF_NAMESPACE") or os.environ.get("HF_HUB_USERNAME") or "").strip()
    if not ns:
        try:
            from huggingface_hub import HfApi

            info = HfApi().whoami()
            ns = str(info.get("name") or info.get("preferred_username") or "").strip()
        except Exception as e:
            logger.debug("HF whoami for trace repo namespace failed: %s", e)
    if not ns:
        raise ValueError(
            f"TRACE_HF_DATASET_REPO={repo!r} has no namespace; set org/repo or TRACE_HF_NAMESPACE"
        )
    return f"{ns}/{repo}"


def path_in_repo_for_trace(local_path: str | Path) -> str:
    return Path(local_path).name


def upload_trace_file_to_hf_dataset(
    local_path: str | Path,
    *,
    repo_id: Optional[str] = None,
    path_in_repo: Optional[str] = None,
    namespace: Optional[str] = None,
    revision: Optional[str] = None,
    private: Optional[bool] = None,
    token: bool | str | None = True,
    commit_message: Optional[str] = None,
    create_repo: bool = True,
) -> str:
    """Upload a trace file; returns hub path."""
    from cst.hf_hiddens.store import ensure_dataset_repo, upload_file

    local_path = Path(local_path)
    if not local_path.is_file():
        raise FileNotFoundError(local_path)

    full_repo = normalize_dataset_repo_id(repo_id or trace_hf_dataset_repo(), namespace)
    hub_path = path_in_repo or path_in_repo_for_trace(local_path)
    rev = revision if revision is not None else trace_hf_revision()
    priv = private if private is not None else trace_hf_dataset_private()

    if create_repo:
        ensure_dataset_repo(full_repo, token=token, private=priv)

    msg = commit_message or f"eval trace {hub_path} ({local_path.stat().st_size} bytes)"
    upload_file(
        full_repo,
        local_path,
        hub_path,
        revision=rev,
        token=token,
        commit_message=msg,
    )
    logger.info("Uploaded trace to https://huggingface.co/datasets/%s/tree/%s/%s", full_repo, rev, hub_path)
    return hub_path


def download_trace_file_from_hf_dataset(
    path_in_repo: str,
    local_path: str | Path | None = None,
    *,
    repo_id: Optional[str] = None,
    namespace: Optional[str] = None,
    revision: Optional[str] = None,
    token: bool | str | None = True,
) -> Path:
    """Download a trace by hub filename; writes to ``local_path`` or cwd."""
    from cst.hf_hiddens.store import download_hub_file

    hub_path = Path(path_in_repo).name
    dest = Path(local_path) if local_path is not None else Path(hub_path)
    full_repo = normalize_dataset_repo_id(repo_id or trace_hf_dataset_repo(), namespace)
    rev = revision if revision is not None else trace_hf_revision()

    download_hub_file(
        full_repo,
        hub_path,
        dest,
        revision=rev,
        token=token,
    )
    logger.info(
        "Downloaded trace from https://huggingface.co/datasets/%s/tree/%s/%s -> %s",
        full_repo,
        rev,
        hub_path,
        dest,
    )
    return dest


def maybe_upload_eval_trace_file(
    local_path: str | Path,
    *,
    trace_name: Optional[str] = None,
    task_name: Optional[str] = None,
    num_samples: int = 0,
    log: Optional[logging.Logger] = None,
) -> Optional[str]:
    """Upload if tracing is on and num_samples > 0; None if skipped."""
    log = log or logger
    if num_samples <= 0 or not tracing_enabled_from_env() or not trace_hf_upload_enabled():
        return None
    try:
        safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(trace_name or Path(local_path).stem))
        task = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(task_name or "unknown"))
        msg = f"SchedulerLens trace {safe} task={task} n={num_samples}"
        hub_path = upload_trace_file_to_hf_dataset(
            local_path,
            commit_message=msg,
        )
        full_repo = normalize_dataset_repo_id(trace_hf_dataset_repo())
        log.info(
            "Pushed trace to HF dataset %s (%s, %d samples)",
            full_repo,
            hub_path,
            num_samples,
        )
        return hub_path
    except Exception as e:
        log.warning("Failed to upload trace to HF dataset: %s", e)
        return None
