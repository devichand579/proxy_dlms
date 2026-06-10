"""HF Hub upload/download for eval traces (default repo: SchedulerLens)."""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
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


def _hf_api(token: bool | str | None = True):
    from huggingface_hub import HfApi

    return HfApi(token=token)


def ensure_dataset_repo(
    repo_id: str,
    *,
    token: bool | str | None = True,
    private: bool = False,
) -> None:
    """Create the dataset repo if it does not exist."""
    api = _hf_api(token)
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=bool(private),
            exist_ok=True,
        )
    except Exception as e:
        logger.warning("create_repo(%s) note: %s", repo_id, e)


def upload_file_to_dataset(
    repo_id: str,
    local_path: str | Path,
    path_in_repo: str,
    *,
    revision: str = "main",
    token: bool | str | None = True,
    commit_message: Optional[str] = None,
) -> str:
    api = _hf_api(token)
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


def download_file_from_dataset(
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
    upload_file_to_dataset(
        full_repo,
        local_path,
        hub_path,
        revision=rev,
        token=token,
        commit_message=msg,
    )
    logger.info("Uploaded trace to https://huggingface.co/datasets/%s/tree/%s/%s", full_repo, rev, hub_path)
    return hub_path


def iter_local_upload_targets(
    path: str | Path,
    *,
    recursive: bool = False,
) -> list[tuple[Path, str]]:
    """Expand a file or directory into ``(local_path, path_in_repo)`` pairs."""
    root = Path(path).expanduser().resolve()
    if root.is_file():
        return [(root, root.name)]
    if root.is_dir():
        if recursive:
            files = sorted(f for f in root.rglob("*") if f.is_file())
            return [(f, str(f.relative_to(root))) for f in files]
        return [(f, f.name) for f in sorted(root.iterdir()) if f.is_file()]
    raise FileNotFoundError(root)


def upload_paths_to_hf_dataset(
    paths: list[str | Path],
    *,
    repo_id: Optional[str] = None,
    namespace: Optional[str] = None,
    revision: Optional[str] = None,
    private: Optional[bool] = None,
    token: bool | str | None = True,
    create_repo: bool = True,
    recursive: bool = False,
    dry_run: bool = False,
) -> list[str]:
    """Upload files and/or all files in directories. Returns hub paths (flat names or relative paths)."""
    targets: list[tuple[Path, str]] = []
    for path in paths:
        targets.extend(iter_local_upload_targets(path, recursive=recursive))
    if not targets:
        raise ValueError("no files to upload (paths must be files or non-empty directories)")

    full_repo = normalize_dataset_repo_id(repo_id or trace_hf_dataset_repo(), namespace)
    rev = revision if revision is not None else trace_hf_revision()
    priv = private if private is not None else trace_hf_dataset_private()

    if create_repo and not dry_run:
        ensure_dataset_repo(full_repo, token=token, private=priv)

    hub_paths: list[str] = []
    for local_path, hub_path in targets:
        if dry_run:
            logger.info("[dry-run] would upload %s -> %s (revision=%s)", local_path, hub_path, rev)
            hub_paths.append(hub_path)
            continue
        upload_file_to_dataset(
            full_repo,
            local_path,
            hub_path,
            revision=rev,
            token=token,
            commit_message=f"eval trace {hub_path} ({local_path.stat().st_size} bytes)",
        )
        logger.info(
            "Uploaded %s -> https://huggingface.co/datasets/%s/tree/%s/%s",
            local_path,
            full_repo,
            rev,
            hub_path,
        )
        hub_paths.append(hub_path)
    return hub_paths


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
    hub_path = Path(path_in_repo).name
    dest = Path(local_path) if local_path is not None else Path(hub_path)
    full_repo = normalize_dataset_repo_id(repo_id or trace_hf_dataset_repo(), namespace)
    rev = revision if revision is not None else trace_hf_revision()

    download_file_from_dataset(
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


def _add_repo_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--repo",
        "--repo-id",
        dest="repo_id",
        default=None,
        help=f"Hub dataset repo (default: TRACE_HF_DATASET_REPO or {DEFAULT_DATASET_REPO})",
    )
    p.add_argument(
        "--namespace",
        default=None,
        help="HF user/org when --repo has no slash (else TRACE_HF_NAMESPACE / whoami)",
    )
    p.add_argument("--revision", default=None, help=f"Branch/tag (default: {trace_hf_revision()})")
    p.add_argument(
        "--private",
        action="store_true",
        help="Create/use a private dataset repo",
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload or download eval trace files on the SchedulerLens HF dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python traces/hf_trace_upload.py upload trace_outputs/llada_gsm8k_pd_s512.jsonl
  python traces/hf_trace_upload.py upload trace_outputs/
  python traces/hf_trace_upload.py upload trace_outputs/ --recursive
  python traces/hf_trace_upload.py upload trace_outputs/*.jsonl --repo org/schedulerlens
  python traces/hf_trace_upload.py download llada_gsm8k_pd_s512.jsonl -o trace_outputs/llada_gsm8k_pd_s512.jsonl
""".strip(),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_up = sub.add_parser(
        "upload",
        help="Upload one or more files, or every file in each directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python traces/hf_trace_upload.py upload trace_outputs/foo.jsonl
  python traces/hf_trace_upload.py upload trace_outputs/
  python traces/hf_trace_upload.py upload trace_outputs/ --recursive --dry-run
""".strip(),
    )
    p_up.add_argument(
        "paths",
        nargs="+",
        help="File(s) and/or directory(ies) to upload",
    )
    p_up.add_argument(
        "--recursive",
        action="store_true",
        help="When a path is a directory, include files in subdirectories (repo paths keep relative paths)",
    )
    p_up.add_argument(
        "--dry-run",
        action="store_true",
        help="List uploads without calling the Hub API",
    )
    p_up.add_argument(
        "--no-create-repo",
        action="store_true",
        help="Do not call create_repo before uploading",
    )
    _add_repo_args(p_up)
    p_up.set_defaults(func=cmd_upload)

    p_dl = sub.add_parser(
        "download",
        help="Download a trace file from the dataset by hub filename",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python traces/hf_trace_upload.py download llada_gsm8k_pd_s512.jsonl
  python traces/hf_trace_upload.py download llada_gsm8k_pd_s512.jsonl -o /tmp/trace.jsonl
""".strip(),
    )
    p_dl.add_argument("path_in_repo", help="Filename as stored on the Hub (basename)")
    p_dl.add_argument(
        "-o",
        "--output",
        default=None,
        help="Local destination path (default: ./<path_in_repo>)",
    )
    _add_repo_args(p_dl)
    p_dl.set_defaults(func=cmd_download)

    return parser


def cmd_upload(args: argparse.Namespace) -> int:
    hub_paths = upload_paths_to_hf_dataset(
        args.paths,
        repo_id=args.repo_id,
        namespace=args.namespace,
        revision=args.revision,
        private=True if args.private else None,
        create_repo=not args.no_create_repo,
        recursive=args.recursive,
        dry_run=args.dry_run,
    )
    repo = normalize_dataset_repo_id(args.repo_id or trace_hf_dataset_repo(), args.namespace)
    rev = args.revision or trace_hf_revision()
    print(f"repo: https://huggingface.co/datasets/{repo}/tree/{rev}")
    print(f"uploaded: {len(hub_paths)}")
    for hub_path in hub_paths:
        print(f"  {hub_path}")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    dest = download_trace_file_from_hf_dataset(
        args.path_in_repo,
        local_path=args.output,
        repo_id=args.repo_id,
        namespace=args.namespace,
        revision=args.revision,
    )
    print(f"downloaded: {dest}")
    return 0


def main(argv: list[str] | None = None) -> int:
    _repo_root = Path(__file__).resolve().parent.parent
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    from repo_env_defaults import apply_repo_env_defaults

    apply_repo_env_defaults()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    args = build_argparser().parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
