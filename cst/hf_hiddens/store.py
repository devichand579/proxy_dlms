"""HF Hub upload and download for manifests and chunk files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from cst.hf_hiddens.manifest import HiddenManifest
from cst.hf_hiddens.naming import manifest_hub_path

logger = logging.getLogger("cst.hf_hiddens.store")


def _api(token: bool | str | None = True):
    from huggingface_hub import HfApi

    return HfApi(token=token)


def ensure_dataset_repo(repo_id: str, *, token: bool | str | None = True, private: bool = False) -> None:
    """Create the dataset repo if it does not exist."""
    api = _api(token)
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=bool(private),
            exist_ok=True,
        )
    except Exception as e:
        logger.warning("create_repo(%s) note: %s", repo_id, e)


def download_hub_file(
    repo_id: str,
    path_in_repo: str,
    local_path: str | Path,
    *,
    revision: str = "main",
    token: bool | str | None = True,
) -> Path:
    from huggingface_hub import hf_hub_download

    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(
        repo_id=repo_id,
        filename=path_in_repo,
        repo_type="dataset",
        revision=revision,
        token=token,
    )
    src = Path(cached)
    if src.resolve() != local_path.resolve():
        local_path.write_bytes(src.read_bytes())
    return local_path


def upload_file(
    repo_id: str,
    local_path: str | Path,
    path_in_repo: str,
    *,
    revision: str = "main",
    token: bool | str | None = True,
    commit_message: Optional[str] = None,
) -> str:
    api = _api(token)
    local_path = Path(local_path)
    if not local_path.is_file():
        raise FileNotFoundError(local_path)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        commit_message=commit_message or f"upload {path_in_repo}",
        token=token,
    )
    return path_in_repo


def fetch_manifest(
    repo_id: str,
    name_token: str,
    *,
    cache_dir: str | Path,
    revision: str = "main",
    token: bool | str | None = True,
    create_if_missing: bool = False,
) -> HiddenManifest:
    """Download manifest from Hub, or start empty when create_if_missing is true."""
    cache_dir = Path(cache_dir)
    local = cache_dir / name_token / "manifest.json"
    hub_path = manifest_hub_path(name_token)
    try:
        download_hub_file(
            repo_id,
            hub_path,
            local,
            revision=revision,
            token=token,
        )
        return HiddenManifest.load_json(local)
    except Exception as e:
        if create_if_missing:
            logger.info("no manifest at %s/%s (%s); starting empty", repo_id, hub_path, e)
            return HiddenManifest.empty(scope="train", name_token=name_token)
        raise


def push_manifest(
    manifest: HiddenManifest,
    repo_id: str,
    *,
    revision: str = "main",
    token: bool | str | None = True,
    cache_dir: Optional[str | Path] = None,
) -> str:
    cache_dir = Path(cache_dir or ".")
    local = cache_dir / manifest.name_token / "manifest.json"
    manifest.save_json(local)
    hub_path = manifest_hub_path(manifest.name_token)
    upload_file(
        repo_id,
        local,
        hub_path,
        revision=revision,
        token=token,
        commit_message=f"update manifest {manifest.name_token}",
    )
    return hub_path
