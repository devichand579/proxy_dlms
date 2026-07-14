#!/usr/bin/env python3
"""CLI worker for Hub hidden uploads, downloads, and job polling."""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from repo_env_defaults import apply_repo_env_defaults

apply_repo_env_defaults()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("hidden_hf_worker")


def _add_hf_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--repo-id", required=True, help="Hub dataset repo id")
    p.add_argument("--revision", default="main")
    p.add_argument("--cache-dir", default="outputs/cst_hf_cache")
    p.add_argument("--private-repo", action="store_true")


def cmd_poll(args: argparse.Namespace) -> int:
    from cst.hf_hiddens.jobs import list_pending, process_one_job

    q = Path(args.queue_dir)
    while True:
        if not list_pending(q):
            if not args.watch:
                logger.info("no pending jobs in %s", q)
                return 0
            time.sleep(args.poll_sec)
            continue
        process_one_job(q)
        if not args.watch:
            continue
        time.sleep(args.poll_sec)


def cmd_upload_hidden_segment(args: argparse.Namespace) -> int:
    from cst.hf_hiddens.segment import upload_hidden_segment_from_local

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"no files match glob: {args.glob}")
    rec = upload_hidden_segment_from_local(
        args.repo_id,
        args.name_token,
        paths,
        segment=args.segment,
        prefix=args.prefix,
        next_batch_idx=args.next_batch_idx,
        revision=args.revision,
        cache_dir=args.cache_dir,
        private_repo=args.private_repo,
        delete_local_after=args.delete_local_after,
        scope=args.scope,
    )
    print(json.dumps(rec.to_dict(), indent=2))
    return 0


def cmd_download_hiddens(args: argparse.Namespace) -> int:
    from cst.hf_hiddens.segment import download_hiddens_until_gb

    paths, next_bi, summary = download_hiddens_until_gb(
        args.repo_id,
        args.name_token,
        start_batch_idx=args.start_batch_idx,
        target_gb=args.target_gb,
        local_dir=args.local_dir,
        cache_dir=args.cache_dir,
        revision=args.revision,
    )
    print(json.dumps({"local_paths": paths, "next_batch_idx": next_bi, "summary": summary}, indent=2))
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    p_poll = sub.add_parser("poll")
    p_poll.add_argument("--queue-dir", default="outputs/hf_worker_queue")
    p_poll.add_argument("--watch", action="store_true")
    p_poll.add_argument("--poll-sec", type=float, default=5.0)
    p_poll.set_defaults(func=cmd_poll)

    p_up = sub.add_parser("upload-hidden-segment")
    _add_hf_args(p_up)
    p_up.add_argument("--name-token", required=True)
    p_up.add_argument("--segment", type=int, required=True)
    p_up.add_argument("--prefix", required=True)
    p_up.add_argument("--next-batch-idx", type=int, required=True)
    p_up.add_argument("--glob", required=True)
    p_up.add_argument("--scope", default="train", choices=("train", "eval"))
    p_up.add_argument("--delete-local-after", action="store_true")
    p_up.set_defaults(func=cmd_upload_hidden_segment)

    p_dl = sub.add_parser("download-hiddens")
    _add_hf_args(p_dl)
    p_dl.add_argument("--name-token", required=True)
    p_dl.add_argument("--start-batch-idx", type=int, default=0)
    p_dl.add_argument("--target-gb", type=float, default=1.0)
    p_dl.add_argument("--local-dir", required=True)
    p_dl.set_defaults(func=cmd_download_hiddens)

    args = ap.parse_args()
    sys.exit(int(args.func(args)))


if __name__ == "__main__":
    main()
