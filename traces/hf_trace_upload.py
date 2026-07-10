"""HF Hub upload/download for eval traces (default repo: SchedulerLens)."""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Optional

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


def resolve_repo(
    repo_id: Optional[str] = None,
    namespace: Optional[str] = None,
) -> str:
    return normalize_dataset_repo_id(repo_id or trace_hf_dataset_repo(), namespace)


def list_repo_files(
    *,
    repo_id: Optional[str] = None,
    namespace: Optional[str] = None,
    revision: Optional[str] = None,
    token: bool | str | None = True,
) -> list[str]:
    full_repo = resolve_repo(repo_id, namespace)
    rev = revision if revision is not None else trace_hf_revision()
    api = _hf_api(token)
    return list(api.list_repo_files(repo_id=full_repo, repo_type="dataset", revision=rev))


def file_sizes(
    filenames: list[str],
    *,
    repo_id: Optional[str] = None,
    namespace: Optional[str] = None,
    revision: Optional[str] = None,
    token: bool | str | None = True,
) -> dict[str, int]:
    if not filenames:
        return {}
    full_repo = resolve_repo(repo_id, namespace)
    rev = revision if revision is not None else trace_hf_revision()
    api = _hf_api(token)
    sizes: dict[str, int] = {}
    for i in range(0, len(filenames), 50):
        infos = api.get_paths_info(
            full_repo, filenames[i : i + 50], repo_type="dataset", revision=rev
        )
        for info in infos:
            sizes[info.path] = int(info.size or 0)
    return sizes


def download_file(
    filename: str,
    *,
    dest: str | Path | None = None,
    repo_id: Optional[str] = None,
    namespace: Optional[str] = None,
    revision: Optional[str] = None,
    token: bool | str | None = True,
) -> Path:
    """Download a hub trace file.

    Without ``dest``, returns the HF cache path (audit/convert pipelines).
    With ``dest``, copies there and returns that path (CLI downloads).
    """
    from huggingface_hub import hf_hub_download

    hub_path = Path(filename).name
    full_repo = resolve_repo(repo_id, namespace)
    rev = revision if revision is not None else trace_hf_revision()
    cached = Path(
        hf_hub_download(
            full_repo,
            hub_path,
            repo_type="dataset",
            revision=rev,
            token=token,
        )
    )
    if dest is None:
        return cached

    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if cached.resolve() != dest_path.resolve():
        dest_path.write_bytes(cached.read_bytes())
    logger.info(
        "Downloaded trace from https://huggingface.co/datasets/%s/tree/%s/%s -> %s",
        full_repo,
        rev,
        hub_path,
        dest_path,
    )
    return dest_path


def cleanup_cache(
    repo_id: str,
    filename: str,
    *,
    revision: Optional[str] = None,
) -> None:
    from huggingface_hub import try_to_load_from_cache

    rev = revision or trace_hf_revision()
    cached = try_to_load_from_cache(repo_id, filename, repo_type="dataset", revision=rev)
    if cached is None:
        return
    p = Path(cached)
    if p.is_file():
        p.unlink()
    for parent in [p.parent, p.parent.parent, p.parent.parent.parent]:
        try:
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
        except OSError:
            pass


def upload_file(
    local_path: str | Path,
    path_in_repo: str,
    *,
    repo_id: Optional[str] = None,
    namespace: Optional[str] = None,
    revision: Optional[str] = None,
    token: bool | str | None = True,
    commit_message: Optional[str] = None,
) -> str:
    full_repo = resolve_repo(repo_id, namespace)
    rev = revision if revision is not None else trace_hf_revision()
    return upload_file_to_dataset(
        full_repo,
        local_path,
        path_in_repo,
        revision=rev,
        token=token,
        commit_message=commit_message,
    )


def delete_file(
    path_in_repo: str,
    *,
    repo_id: Optional[str] = None,
    namespace: Optional[str] = None,
    revision: Optional[str] = None,
    token: bool | str | None = True,
    commit_message: Optional[str] = None,
) -> None:
    full_repo = resolve_repo(repo_id, namespace)
    rev = revision if revision is not None else trace_hf_revision()
    api = _hf_api(token)
    api.delete_file(
        path_in_repo=path_in_repo,
        repo_id=full_repo,
        repo_type="dataset",
        revision=rev,
        commit_message=commit_message or f"remove {path_in_repo}",
        token=token,
    )
    logger.info("Deleted %s from %s@%s", path_in_repo, full_repo, rev)


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


def _upload_targets_to_hf_dataset(
    targets: list[tuple[Path, str]],
    *,
    repo_id: Optional[str] = None,
    namespace: Optional[str] = None,
    revision: Optional[str] = None,
    private: Optional[bool] = None,
    token: bool | str | None = True,
    create_repo: bool = True,
    dry_run: bool = False,
    commit_message: Optional[str] = None,
) -> list[str]:
    if not targets:
        raise ValueError("no files to upload")

    full_repo = resolve_repo(repo_id, namespace)
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
        msg = commit_message or f"eval trace {hub_path} ({local_path.stat().st_size} bytes)"
        upload_file_to_dataset(
            full_repo,
            local_path,
            hub_path,
            revision=rev,
            token=token,
            commit_message=msg,
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
    return _upload_targets_to_hf_dataset(
        targets,
        repo_id=repo_id,
        namespace=namespace,
        revision=revision,
        private=private,
        token=token,
        create_repo=create_repo,
        dry_run=dry_run,
    )


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
    """Upload a single trace file; returns hub path."""
    local_path = Path(local_path)
    if not local_path.is_file():
        raise FileNotFoundError(local_path)
    hub_paths = _upload_targets_to_hf_dataset(
        [(local_path, path_in_repo or local_path.name)],
        repo_id=repo_id,
        namespace=namespace,
        revision=revision,
        private=private,
        token=token,
        create_repo=create_repo,
        commit_message=commit_message,
    )
    return hub_paths[0]


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
        full_repo = resolve_repo()
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


def convert_paths_to_npz(
    paths: Iterable[str | Path],
    *,
    output_dir: str | Path | None = None,
    delete_jsonl: bool = False,
) -> list[Path]:
    from traces.trace_npz import compression_ratio, convert_jsonl_to_npz

    outputs: list[Path] = []
    for path in paths:
        src = Path(path)
        if not src.is_file():
            raise FileNotFoundError(src)
        if src.suffix.lower() not in (".jsonl", ".json"):
            raise ValueError(f"expected .jsonl or .json, got {src}")
        if output_dir is not None:
            dst = Path(output_dir) / f"{src.stem}.npz"
        else:
            dst = src.with_suffix(".npz")
        out = convert_jsonl_to_npz(src, dst)
        ratio = compression_ratio(src, out)
        logger.info(
            "Converted %s -> %s (%.2fx smaller, %d -> %d bytes)",
            src,
            out,
            ratio,
            src.stat().st_size,
            out.stat().st_size,
        )
        outputs.append(out)
        if delete_jsonl:
            src.unlink()
    return outputs


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
  python traces/trace_audit.py inventory
  python traces/trace_audit.py run-all
  python traces/hf_trace_upload.py convert trace_outputs/foo.jsonl
  python traces/hf_trace_upload.py convert trace_outputs/*.jsonl --output-dir trace_outputs_npz
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

    p_conv = sub.add_parser(
        "convert",
        help="Convert JSONL/JSON trace files to compressed NPZ",
    )
    p_conv.add_argument("paths", nargs="+", help="Trace file(s) to convert")
    p_conv.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Write .npz files here (default: next to each input file)",
    )
    p_conv.add_argument(
        "--delete-jsonl",
        action="store_true",
        help="Delete source JSONL/JSON after successful conversion",
    )
    p_conv.set_defaults(func=cmd_convert)

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
    repo = resolve_repo(args.repo_id, args.namespace)
    rev = args.revision or trace_hf_revision()
    print(f"repo: https://huggingface.co/datasets/{repo}/tree/{rev}")
    print(f"uploaded: {len(hub_paths)}")
    for hub_path in hub_paths:
        print(f"  {hub_path}")
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    outputs = convert_paths_to_npz(
        args.paths,
        output_dir=args.output_dir,
        delete_jsonl=args.delete_jsonl,
    )
    print(f"converted: {len(outputs)}")
    for path in outputs:
        print(f"  {path}")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    dest = download_file(
        args.path_in_repo,
        dest=args.output or Path(args.path_in_repo).name,
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
