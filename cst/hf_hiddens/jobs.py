"""File job queue for Hub upload and download on a worker machine."""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from cst.hf_hiddens.config import hf_token

logger = logging.getLogger("cst.hf_hiddens.jobs")


def _queue_dirs(queue_dir: Path) -> tuple[Path, Path, Path, Path]:
    pending = queue_dir / "pending"
    running = queue_dir / "running"
    done = queue_dir / "done"
    failed = queue_dir / "failed"
    for d in (pending, running, done, failed):
        d.mkdir(parents=True, exist_ok=True)
    return pending, running, done, failed


def submit_job(queue_dir: str | Path, job: Dict[str, Any]) -> str:
    q = Path(queue_dir)
    pending, _, _, _ = _queue_dirs(q)
    job_id = str(job.get("job_id") or uuid.uuid4().hex)
    job = dict(job, job_id=job_id, submitted_at=time.time())
    (pending / f"{job_id}.json").write_text(json.dumps(job, indent=0), encoding="utf-8")
    logger.info("submitted job %s type=%s", job_id, job.get("type"))
    return job_id


def wait_for_job(
    queue_dir: str | Path,
    job_id: str,
    *,
    timeout_sec: float = 86400.0,
    poll_sec: float = 5.0,
) -> Dict[str, Any]:
    q = Path(queue_dir)
    _, _, done, failed = _queue_dirs(q)
    deadline = time.time() + float(timeout_sec)
    while time.time() < deadline:
        if (done / f"{job_id}.json").is_file():
            return json.loads((done / f"{job_id}.json").read_text(encoding="utf-8"))
        if (failed / f"{job_id}.json").is_file():
            payload = json.loads((failed / f"{job_id}.json").read_text(encoding="utf-8"))
            raise RuntimeError(f"HF worker job {job_id} failed: {payload.get('error', payload)}")
        time.sleep(poll_sec)
    raise TimeoutError(f"HF worker job {job_id} did not finish within {timeout_sec}s")


def list_pending(queue_dir: str | Path) -> List[Path]:
    pending, _, _, _ = _queue_dirs(Path(queue_dir))
    return sorted(pending.glob("*.json"))


def _run_job(job: Dict[str, Any]) -> Dict[str, Any]:
    jtype = str(job.get("type", "")).strip().lower()
    token = job.get("hf_token", True)

    if jtype in ("upload_hidden_segment", "upload_train_segment"):
        from cst.hf_hiddens.segment import upload_hidden_segment_from_local

        paths = [str(p) for p in job.get("local_paths") or []]
        rec = upload_hidden_segment_from_local(
            str(job["repo_id"]),
            str(job["name_token"]),
            paths,
            segment=int(job["segment"]),
            prefix=str(job["prefix"]),
            next_batch_idx=int(job["next_batch_idx"]),
            sampling_fingerprint=job.get("sampling_fingerprint"),
            cache_dir=str(job.get("cache_dir", "outputs/cst_hf_cache")),
            revision=str(job.get("revision", "main")),
            token=token,
            private_repo=bool(job.get("private_repo", False)),
            delete_local_after=bool(job.get("delete_local_after", False)),
            scope=str(job.get("scope", "train")),
        )
        return {"segment": rec.to_dict(), "n_uploaded": len(paths)}

    if jtype == "download_hiddens_until_gb":
        from cst.hf_hiddens.segment import download_hiddens_until_gb

        local_paths, next_bi, summary = download_hiddens_until_gb(
            str(job["repo_id"]),
            str(job["name_token"]),
            start_batch_idx=int(job["start_batch_idx"]),
            target_gb=float(job["target_gb"]),
            local_dir=str(job["local_dir"]),
            cache_dir=str(job.get("cache_dir", "outputs/cst_hf_cache")),
            revision=str(job.get("revision", "main")),
            token=token,
        )
        return {
            "local_paths": local_paths,
            "next_batch_idx": next_bi,
            "summary": summary,
        }

    if jtype == "download_train_segment":
        from cst.hf_hiddens.segment import download_hiddens_until_gb

        local_paths, next_bi, summary = download_hiddens_until_gb(
            str(job["repo_id"]),
            str(job["name_token"]),
            start_batch_idx=int(job["start_batch_idx"]),
            target_gb=float(job["target_gb"]),
            local_dir=str(job["local_dir"]),
            cache_dir=str(job.get("cache_dir", "outputs/cst_hf_cache")),
            revision=str(job.get("revision", "main")),
            token=token,
        )
        return {
            "local_paths": local_paths,
            "next_batch_idx": next_bi,
            "summary": summary,
        }

    raise ValueError(f"unknown job type: {jtype!r}")


def process_one_job(queue_dir: str | Path) -> Optional[str]:
    pending = list_pending(queue_dir)
    if not pending:
        return None
    q = Path(queue_dir)
    pending_dir, running_dir, done_dir, failed_dir = _queue_dirs(q)
    job_path = pending[0]
    job = json.loads(job_path.read_text(encoding="utf-8"))
    job_id = str(job.get("job_id", job_path.stem))
    running_path = running_dir / job_path.name
    job_path.rename(running_path)
    try:
        result = _run_job(job)
        (done_dir / job_path.name).write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "type": job.get("type"),
                    "ok": True,
                    "result": result,
                    "finished_at": time.time(),
                },
                indent=0,
            ),
            encoding="utf-8",
        )
    except Exception as e:
        logger.exception("job %s failed", job_id)
        (failed_dir / job_path.name).write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "type": job.get("type"),
                    "ok": False,
                    "error": str(e),
                    "finished_at": time.time(),
                },
                indent=0,
            ),
            encoding="utf-8",
        )
    finally:
        running_path.unlink(missing_ok=True)
    return job_id


def dispatch_or_run(
    queue_dir: Optional[str | Path],
    job: Dict[str, Any],
    *,
    use_queue: bool,
    wait_timeout_sec: float = 86400.0,
) -> Dict[str, Any]:
    if use_queue and queue_dir:
        job_id = submit_job(queue_dir, job)
        logger.info(
            "enqueued HF job %s — run: python hidden_hf_worker.py poll --queue-dir %s",
            job_id,
            queue_dir,
        )
        done = wait_for_job(queue_dir, job_id, timeout_sec=wait_timeout_sec)
        return done.get("result") or {}
    return _run_job(job)


def dispatch_from_config(tr: Any, job: Dict[str, Any]) -> Dict[str, Any]:
    job = {
        **job,
        "cache_dir": str(tr.get("hf_local_cache_dir", "outputs/cst_hf_cache")),
        "revision": str(tr.get("hf_hiddens_revision", "main")),
        "hf_token": hf_token(tr),
        "private_repo": bool(tr.get("hf_hiddens_private", False)),
    }
    use_q = bool(tr.get("hf_use_worker_queue", False))
    qdir = str(tr.get("hf_worker_queue_dir", "outputs/hf_worker_queue"))
    timeout = float(tr.get("hf_worker_wait_timeout_sec", 86400.0))
    return dispatch_or_run(qdir, job, use_queue=use_q, wait_timeout_sec=timeout)
