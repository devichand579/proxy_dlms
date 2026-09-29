"""Azure Blob Storage upload and download for manifests and chunk files."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from azure.core.pipeline.transport import RequestsTransport
from azure.core.pipeline.transport._bigger_block_size_http_adapters import BiggerBlockSizeHTTPAdapter
from azure.storage.blob import BlobServiceClient
from urllib3.util.retry import Retry
from azure.core.exceptions import (
    ResourceNotFoundError,
    HttpResponseError,
    ServiceRequestError,
    ServiceResponseError,
    AzureError,
)

from cst.hf_hiddens.manifest import MANIFEST_VERSION, ChunkRecord, HiddenManifest
from cst.azure_hiddens.naming import chunk_blob_path, chunks_dir_blob, manifest_blob_path
from cst.naming import parse_hidden_chunk_batch_idx

logger = logging.getLogger("cst.azure_hiddens.store")


class HubStorageFullError(RuntimeError):
    """Parallel to HF HubStorageFullError. Azure equivalent might be OutOfRange or quota errors."""


_POOL_SIZE = 32


def _pooled_transport() -> RequestsTransport:
    # azure-core hardcodes its HTTPAdapter (pool_maxsize=10) and has no kwarg to change it,
    # so uploads/downloads with max_concurrency=16 thrash the pool ("Connection pool is
    # full, discarding connection"). Supply our own session with a larger pool.
    session = requests.Session()
    adapter = BiggerBlockSizeHTTPAdapter(
        pool_connections=_POOL_SIZE,
        pool_maxsize=_POOL_SIZE,
        max_retries=Retry(total=False, redirect=False, raise_on_status=False),
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return RequestsTransport(session=session)


def _get_client(token: str, max_block_size: int = 100 * 1024 * 1024) -> BlobServiceClient:
    return BlobServiceClient.from_connection_string(
        token,
        max_block_size=max_block_size,
        retry_total=10,
        retry_connect=10,
        retry_read=10,
        retry_status=10,
        retry_backoff_factor=2,
        transport=_pooled_transport(),
    )


def ensure_container(repo_id: str, token: str) -> None:
    """Create the blob container if it does not exist (repo_id is the container)."""
    try:
        client = _get_client(token)
        container = client.get_container_client(repo_id)
        if not container.exists():
            container.create_container()
    except Exception as e:
        logger.warning("ensure_container(%s) note: %s", repo_id, e)


def download_blob_file(
    repo_id: str,
    blob_path: str,
    local_path: str | Path,
    token: str,
    *,
    max_concurrency: int = 16,
    max_retries: int = 5,
) -> Path:
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, max_retries + 1):
        try:
            client = _get_client(token)
            blob_client = client.get_blob_client(container=repo_id, blob=blob_path)
            with open(local_path, "wb") as f:
                blob_client.download_blob(max_concurrency=max_concurrency).readinto(f)
            return local_path
        except (ServiceRequestError, ServiceResponseError, AzureError, ConnectionError, OSError) as e:
            if attempt == max_retries:
                logger.error("Azure download_blob_file failed after %d attempts: %s", max_retries, e)
                raise
            backoff = min(60, 5 * (2 ** (attempt - 1)))
            logger.warning(
                "Azure download_blob_file network error (attempt %d/%d): %s; retrying in %ds...",
                attempt, max_retries, e, backoff,
            )
            time.sleep(backoff)
    return local_path


def upload_file(
    repo_id: str,
    local_path: str | Path,
    blob_path: str,
    token: str,
    *,
    max_concurrency: int = 16,
    max_retries: int = 5,
) -> str:
    local_path = Path(local_path)
    if not local_path.is_file():
        raise FileNotFoundError(local_path)
    for attempt in range(1, max_retries + 1):
        try:
            client = _get_client(token)
            blob_client = client.get_blob_client(container=repo_id, blob=blob_path)
            try:
                blob_client.delete_blob()
            except Exception:
                pass
            with open(local_path, "rb") as data:
                blob_client.upload_blob(
                    data,
                    overwrite=True,
                    max_concurrency=max_concurrency,
                )
            return blob_path
        except HttpResponseError as e:
            if "QuotaExceeded" in str(e):
                raise HubStorageFullError(str(e)) from e
            if attempt == max_retries:
                raise
            backoff = min(60, 5 * (2 ** (attempt - 1)))
            logger.warning(
                "Azure upload_file failed (attempt %d/%d) with HttpResponseError: %s; retrying in %ds...",
                attempt, max_retries, e, backoff,
            )
            time.sleep(backoff)
        except (ServiceRequestError, ServiceResponseError, AzureError, ConnectionError, OSError) as e:
            if attempt == max_retries:
                logger.error("Azure upload_file failed after %d attempts: %s", max_retries, e)
                raise
            backoff = min(60, 5 * (2 ** (attempt - 1)))
            logger.warning(
                "Azure upload_file network error (attempt %d/%d): %s; retrying in %ds...",
                attempt, max_retries, e, backoff,
            )
            time.sleep(backoff)
    return blob_path


def fetch_manifest(
    repo_id: str,
    name_token: str,
    *,
    cache_dir: str | Path,
    revision: str = "main",
    token: bool | str | None = True,
    create_if_missing: bool = False,
) -> HiddenManifest:
    """Download manifest from Azure, or start empty when create_if_missing is true."""
    if not isinstance(token, str):
        raise ValueError("Azure requires a connection string passed as token.")
    cache_dir = Path(cache_dir)
    local = cache_dir / name_token / "manifest.json"
    blob_path = manifest_blob_path(name_token)
    try:
        download_blob_file(repo_id, blob_path, local, token=token)
        return HiddenManifest.load_json(local)
    except ResourceNotFoundError as e:
        if create_if_missing:
            logger.info("no manifest at %s/%s (%s); starting empty", repo_id, blob_path, e)
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
    if not isinstance(token, str):
        return False
    try:
        client = _get_client(token)
        blob_client = client.get_blob_client(container=repo_id, blob=chunk_blob_path(name_token, filename))
        return blob_client.exists()
    except Exception as e:
        logger.info("chunk_exists(%s) note: %s", filename, e)
        return False


def list_hub_chunks(
    repo_id: str,
    name_token: str,
    *,
    revision: str = "main",
    token: bool | str | None = True,
) -> list[ChunkRecord]:
    """Chunk files currently on Azure (may be ahead of the manifest)."""
    if not isinstance(token, str):
        return []
    prefix = chunks_dir_blob(name_token).rstrip("/") + "/"
    try:
        client = _get_client(token)
        container = client.get_container_client(repo_id)
        if not container.exists():
            return []
        blobs = list(container.list_blobs(name_starts_with=prefix))
    except Exception as e:
        logger.info("list_blob_chunks(%s) note: %s", repo_id, e)
        return []

    out: list[ChunkRecord] = []
    for b in blobs:
        if not b.name.endswith(".pt"):
            continue
        filename = Path(b.name).name
        bi = parse_hidden_chunk_batch_idx(filename)
        if bi is None:
            continue
        out.append(ChunkRecord(batch_idx=bi, filename=filename, bytes=b.size))
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
    if not isinstance(token, str):
        raise ValueError("Azure requires a connection string passed as token.")
    cache_dir = Path(cache_dir or ".")
    local = cache_dir / manifest.name_token / "manifest.json"
    manifest.save_json(local)
    blob_path = manifest_blob_path(manifest.name_token)
    upload_file(repo_id, local, blob_path, token=token)
    return blob_path


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
    n_rows: Optional[int] = None,
    skip_manifest_push: bool = False,
) -> ChunkRecord:
    """Upload one hidden .pt to Azure and update the manifest."""
    if not isinstance(token, str):
        raise ValueError("Azure requires a connection string passed as token.")
    ensure_container(repo_id, token)
    local_path = Path(local_path)
    if not local_path.is_file():
        raise FileNotFoundError(local_path)
    
    bi = parse_hidden_chunk_batch_idx(local_path.name)
    if bi is None:
        raise ValueError(f"chunk filename has no batch index: {local_path.name}")
    
    rec = ChunkRecord(
        batch_idx=bi,
        filename=local_path.name,
        bytes=local_path.stat().st_size,
        n_rows=int(n_rows) if n_rows is not None else None,
    )
    
    if skip_if_exists and hub_chunk_exists(repo_id, name_token, rec.filename, token=token):
        logger.info("Azure already has %s; skipping re-upload", rec.filename)
    else:
        upload_file(repo_id, local_path, chunk_blob_path(name_token, rec.filename), token=token)

    manifest.scope = str(scope)
    manifest.name_token = name_token
    manifest.schema_version = MANIFEST_VERSION
    if sampling_fingerprint is not None:
        manifest.sampling_fingerprint = dict(sampling_fingerprint)
    
    prev_chunks = list(manifest.chunks)
    manifest.upsert_chunk(rec)
    try:
        if not skip_manifest_push:
            push_manifest(manifest, repo_id, token=token, cache_dir=cache_dir)
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
    if not isinstance(token, str):
        raise ValueError("Azure requires a connection string passed as token.")
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    dest = local_dir / Path(filename).name
    if dest.is_file() and (expected_bytes is None or dest.stat().st_size == int(expected_bytes)):
        return dest
    download_blob_file(repo_id, chunk_blob_path(name_token, dest.name), dest, token=token)
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
    if not isinstance(token, str):
        raise ValueError("Azure requires a connection string passed as token.")
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    manifest = fetch_manifest(
        repo_id, name_token, cache_dir=cache_dir, token=token, create_if_missing=False
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
            repo_id, name_token, ch.filename, local_dir, expected_bytes=ch.bytes, token=token
        )
    return local_dir
