"""HF Hub upload and download for manifests and chunk files."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from huggingface_hub.errors import HfHubHTTPError

from cst.hf_hiddens.manifest import MANIFEST_VERSION, ChunkRecord, HiddenManifest
from cst.hf_hiddens.naming import chunk_hub_path, chunks_dir_hub, manifest_hub_path
from cst.naming import parse_hidden_chunk_batch_idx

logger = logging.getLogger("cst.hf_hiddens.store")

# huggingface_hub has no StorageQuotaError. The Hub returns HTTP 403 with
# X-Error-Message / HfHubHTTPError.server_message. Documented limit policy:
# https://huggingface.co/docs/hub/en/storage-limits
# Exact server messages observed from Hub / huggingface_hub issues:
#   "403 Forbidden: Private repository storage limit reached, please upgrade your plan..."
#   "403 Forbidden: Your storage patterns tripped our internal systems! ..."
#   LFS batch: "Repository storage limit reached (Max: 1 GB)"
#   quota key: "private-storage-gbs exhausted"
_HUB_STORAGE_LIMIT_SERVER_MESSAGES = (
    "private repository storage limit reached",
    "repository storage limit reached",
    "storage patterns tripped our internal systems",
    "private-storage-gbs exhausted",
)


class HubStorageFullError(RuntimeError):
    """Hub HTTP 403 storage-limit response; store should stop early."""


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        cur = cur.__cause__ or cur.__context__


def is_hub_storage_full_error(exc: BaseException) -> bool:
    """True iff Hub rejected the upload with a documented storage-limit 403."""
    if isinstance(exc, HubStorageFullError):
        return True
    for cur in _exception_chain(exc):
        if isinstance(cur, HubStorageFullError):
            return True
        if not isinstance(cur, HfHubHTTPError):
            continue
        resp = getattr(cur, "response", None)
        status = getattr(resp, "status_code", None)
        if status != 403:
            continue
        text = " ".join(
            part
            for part in (getattr(cur, "server_message", None), str(cur))
            if part
        ).lower()
        if any(msg in text for msg in _HUB_STORAGE_LIMIT_SERVER_MESSAGES):
            return True
    return False


def _raise_if_hub_storage_full(exc: BaseException) -> None:
    if is_hub_storage_full_error(exc):
        raise HubStorageFullError(str(exc)) from exc


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
        shutil.copyfile(src, local_path)
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
    try:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            commit_message=commit_message or f"upload {path_in_repo}",
            token=token,
        )
    except Exception as e:
        _raise_if_hub_storage_full(e)
        raise
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


def hub_chunk_exists(
    repo_id: str,
    name_token: str,
    filename: str,
    *,
    revision: str = "main",
    token: bool | str | None = True,
) -> bool:
    """True if ``{name_token}/chunks/{filename}`` is already on the Hub."""
    api = _api(token)
    path = chunk_hub_path(name_token, filename)
    try:
        infos = api.get_paths_info(
            repo_id, [path], repo_type="dataset", revision=revision
        )
    except Exception as e:
        logger.info("get_paths_info(%s) note: %s", path, e)
        return False
    return bool(infos)


def list_hub_chunks(
    repo_id: str,
    name_token: str,
    *,
    revision: str = "main",
    token: bool | str | None = True,
) -> list[ChunkRecord]:
    """Chunk files currently under ``{name_token}/chunks/`` on the Hub (may be ahead of the manifest)."""
    prefix = chunks_dir_hub(name_token).rstrip("/") + "/"
    api = _api(token)
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
    except Exception as e:
        logger.info("list_repo_files(%s) note: %s", repo_id, e)
        return []
    paths = [
        p
        for p in files
        if p.startswith(prefix)
        and p.endswith(".pt")
        and parse_hidden_chunk_batch_idx(Path(p).name) is not None
    ]
    if not paths:
        return []
    sizes: Dict[str, int] = {}
    for i in range(0, len(paths), 50):
        batch = paths[i : i + 50]
        try:
            infos = api.get_paths_info(
                repo_id, batch, repo_type="dataset", revision=revision
            )
        except Exception as e:
            logger.warning("get_paths_info for Hub chunks failed: %s", e)
            continue
        for info in infos:
            sizes[str(info.path)] = int(getattr(info, "size", 0) or 0)
    out: list[ChunkRecord] = []
    for p in paths:
        base = Path(p).name
        bi = parse_hidden_chunk_batch_idx(base)
        if bi is None:
            continue
        out.append(ChunkRecord(batch_idx=bi, filename=base, bytes=int(sizes.get(p, 0))))
    out.sort(key=lambda c: int(c.batch_idx))
    return out


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


def upload_hidden_chunk(
    repo_id: str,
    name_token: str,
    local_path: str | Path,
    *,
    manifest: HiddenManifest,
    sampling_fingerprint: Optional[Dict[str, Any]] = None,
    cache_dir: str | Path = "outputs/cst_hf_cache",
    revision: str = "main",
    token: bool | str | None = True,
    private_repo: bool = False,
    delete_local_after: bool = True,
    scope: str = "train",
    skip_if_exists: bool = False,
) -> ChunkRecord:
    """Upload one hidden .pt and update the Hub manifest.

    With ``skip_if_exists``, a .pt already at that Hub path is not re-uploaded
    (avoids a second LFS blob after a crash between chunk and manifest commits).
    """
    ensure_dataset_repo(repo_id, token=token, private=private_repo)
    local_path = Path(local_path)
    if not local_path.is_file():
        raise FileNotFoundError(local_path)
    bi = parse_hidden_chunk_batch_idx(local_path.name)
    if bi is None:
        raise ValueError(f"chunk filename has no batch index: {local_path.name}")
    rec = ChunkRecord(batch_idx=bi, filename=local_path.name, bytes=local_path.stat().st_size)
    if skip_if_exists and hub_chunk_exists(
        repo_id, name_token, rec.filename, revision=revision, token=token
    ):
        logger.info("Hub already has %s; skipping re-upload, updating manifest only", rec.filename)
    else:
        upload_file(
            repo_id,
            local_path,
            chunk_hub_path(name_token, rec.filename),
            revision=revision,
            token=token,
            commit_message=f"chunk {rec.filename}",
        )
    manifest.scope = str(scope)
    manifest.name_token = name_token
    manifest.schema_version = MANIFEST_VERSION
    if sampling_fingerprint is not None:
        manifest.sampling_fingerprint = dict(sampling_fingerprint)
    prev_chunks = list(manifest.chunks)
    manifest.upsert_chunk(rec)
    try:
        push_manifest(manifest, repo_id, revision=revision, token=token, cache_dir=cache_dir)
    except Exception:
        manifest.chunks = prev_chunks
        raise
    if delete_local_after:
        try:
            local_path.unlink(missing_ok=True)
        except OSError as e:
            logger.warning("could not delete %s: %s", local_path, e)
    return rec


def download_hidden_chunk(
    repo_id: str,
    name_token: str,
    filename: str,
    local_dir: str | Path,
    *,
    expected_bytes: Optional[int] = None,
    revision: str = "main",
    token: bool | str | None = True,
) -> Path:
    """Download one hidden chunk into local_dir."""
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    dest = local_dir / Path(filename).name
    if dest.is_file() and (expected_bytes is None or dest.stat().st_size == int(expected_bytes)):
        return dest
    download_hub_file(
        repo_id,
        chunk_hub_path(name_token, dest.name),
        dest,
        revision=revision,
        token=token,
    )
    return dest


def download_all_chunks(
    repo_id: str,
    name_token: str,
    *,
    local_dir: str | Path,
    cache_dir: str | Path = "outputs/cst_hf_cache",
    revision: str = "main",
    token: bool | str | None = True,
) -> Path:
    """Download all chunks listed in the manifest for name_token."""
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    manifest = fetch_manifest(
        repo_id,
        name_token,
        cache_dir=Path(cache_dir),
        revision=revision,
        token=token,
        create_if_missing=False,
    )
    all_chunks = manifest.all_chunks_sorted()
    if not all_chunks:
        raise FileNotFoundError(f"manifest for {name_token} @ {repo_id} has no chunks")
    seen: set[str] = set()
    for ch in all_chunks:
        if ch.filename in seen:
            continue
        seen.add(ch.filename)
        download_hidden_chunk(
            repo_id,
            name_token,
            ch.filename,
            local_dir,
            expected_bytes=ch.bytes,
            revision=revision,
            token=token,
        )
    return local_dir
