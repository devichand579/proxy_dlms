"""Upload and download hidden segments on the Hub."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from cst.hf_hiddens.manifest import ChunkRecord, SegmentRecord
from cst.hf_hiddens.naming import chunk_hub_path
from cst.naming import parse_hidden_chunk_batch_idx
from cst.hf_hiddens.store import (
    download_hub_file,
    ensure_dataset_repo,
    fetch_manifest,
    push_manifest,
    upload_file,
)

logger = logging.getLogger("cst.hf_hiddens.segment")


def _bytes_from_gb(gb: float) -> int:
    return max(0, int(float(gb) * 1e9))


def _parse_batch_idx(filename: str) -> Optional[int]:
    return parse_hidden_chunk_batch_idx(filename)


def _segment_record_from_paths(
    paths: Sequence[str | Path],
    *,
    segment: int,
    prefix: str,
    next_batch_idx: int,
    sampling_fingerprint: Optional[Dict[str, Any]] = None,
) -> SegmentRecord:
    chunks: List[ChunkRecord] = []
    total = 0
    for p in paths:
        fp = Path(p)
        if not fp.is_file():
            continue
        bi = _parse_batch_idx(fp.name)
        if bi is None:
            logger.warning("skip chunk without batch index in name: %s", fp.name)
            continue
        nbytes = fp.stat().st_size
        chunks.append(ChunkRecord(batch_idx=bi, filename=fp.name, bytes=nbytes))
        total += nbytes
    chunks.sort(key=lambda c: c.batch_idx)
    return SegmentRecord(
        segment=int(segment),
        prefix=str(prefix),
        chunks=chunks,
        total_bytes=int(total),
        next_batch_idx=int(next_batch_idx),
        sampling_fingerprint=sampling_fingerprint,
    )


def upload_hidden_segment_from_local(
    repo_id: str,
    name_token: str,
    local_paths: Sequence[str | Path],
    *,
    segment: int,
    prefix: str,
    next_batch_idx: int,
    sampling_fingerprint: Optional[Dict[str, Any]] = None,
    cache_dir: str | Path = "outputs/cst_hf_cache",
    revision: str = "main",
    token: bool | str | None = True,
    private_repo: bool = False,
    delete_local_after: bool = False,
    scope: str = "train",
) -> SegmentRecord:
    ensure_dataset_repo(repo_id, token=token, private=private_repo)
    cache_dir = Path(cache_dir)
    rec = _segment_record_from_paths(
        local_paths,
        segment=segment,
        prefix=prefix,
        next_batch_idx=next_batch_idx,
        sampling_fingerprint=sampling_fingerprint,
    )
    by_name = {Path(p).name: Path(p) for p in local_paths}
    for ch in rec.chunks:
        local = by_name.get(ch.filename)
        if local is None or not local.is_file():
            raise FileNotFoundError(f"local chunk missing for {ch.filename}")
        hub_p = chunk_hub_path(name_token, ch.filename)
        upload_file(
            repo_id,
            local,
            hub_p,
            revision=revision,
            token=token,
            commit_message=f"chunk {ch.filename} seg{segment}",
        )

    manifest = fetch_manifest(
        repo_id,
        name_token,
        cache_dir=cache_dir,
        revision=revision,
        token=token,
        create_if_missing=True,
    )
    manifest.scope = str(scope)
    manifest.name_token = name_token
    manifest.upsert_segment(rec)
    push_manifest(manifest, repo_id, revision=revision, token=token, cache_dir=cache_dir)

    if delete_local_after:
        for p in local_paths:
            try:
                Path(p).unlink(missing_ok=True)
            except OSError as e:
                logger.warning("could not delete %s: %s", p, e)
    return rec


def download_hiddens_until_gb(
    repo_id: str,
    name_token: str,
    *,
    start_batch_idx: int,
    target_gb: float,
    local_dir: str | Path,
    cache_dir: str | Path = "outputs/cst_hf_cache",
    revision: str = "main",
    token: bool | str | None = True,
) -> Tuple[List[str], int, Dict[str, Any]]:
    """Download chunks from start_batch_idx until about target_gb."""
    target_b = _bytes_from_gb(target_gb)
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
    candidates = manifest.chunks_from_batch_idx(int(start_batch_idx))
    if not candidates:
        raise FileNotFoundError(
            f"no chunks with batch_idx >= {start_batch_idx} for {name_token} @ {repo_id}"
        )

    paths: List[str] = []
    total_bytes = 0
    next_bi = int(start_batch_idx)
    n_dl = 0

    for ch in candidates:
        if total_bytes >= target_b and n_dl > 0:
            break
        dest = local_dir / ch.filename
        if not (dest.is_file() and dest.stat().st_size == ch.bytes):
            download_hub_file(
                repo_id,
                chunk_hub_path(name_token, ch.filename),
                dest,
                revision=revision,
                token=token,
            )
        paths.append(str(dest))
        total_bytes += dest.stat().st_size
        next_bi = int(ch.batch_idx) + 1
        n_dl += 1

    summary = {
        "target_gb": float(target_gb),
        "written_bytes": total_bytes,
        "written_gb": total_bytes / 1e9,
        "n_batches": n_dl,
        "next_idx": next_bi,
        "source": "hf_hub",
        "repo_id": repo_id,
        "name_token": name_token,
    }
    return paths, next_bi, summary


def download_all_manifest_chunks(
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
        dest = local_dir / ch.filename
        if not (dest.is_file() and dest.stat().st_size == ch.bytes):
            download_hub_file(
                repo_id,
                chunk_hub_path(name_token, ch.filename),
                dest,
                revision=revision,
                token=token,
            )
    return local_dir
