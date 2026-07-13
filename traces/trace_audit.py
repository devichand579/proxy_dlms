"""CLI for trace audit and NPZ conversion pipelines."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from traces.trace_checks import check_trace_file
from traces.trace_pipeline import (
    DEFAULT_AUDIT_CSV,
    DEFAULT_CONVERT_CSV,
    audit_one_hub_file,
    convert_one_hub_file,
    hub_inventory,
    print_check,
    print_inventory,
    run_audit_all,
    run_convert_all,
)


def _add_repo_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--repo", "--repo-id", dest="repo_id", default=None)
    p.add_argument("--namespace", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--metrics-csv", default=None)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit and convert SchedulerLens traces (one file at a time).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python traces/trace_audit.py inventory
  python traces/trace_audit.py local trace_outputs/foo.jsonl
  python traces/trace_audit.py audit foo.jsonl
  python traces/trace_audit.py audit-all
  python traces/trace_audit.py convert foo.jsonl --delete-jsonl-on-hub
  python traces/trace_audit.py convert-all --delete-jsonl-on-hub
  python test_tracing.py
""".strip(),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_inv = sub.add_parser("inventory", help="Hub filename inventory (no downloads)")
    _add_repo_args(p_inv)
    p_inv.set_defaults(func=cmd_inventory)

    p_local = sub.add_parser("local", help="Audit a local trace file")
    p_local.add_argument("path")
    p_local.add_argument("--max-samples", type=int, default=3)
    p_local.add_argument("--all-samples", action="store_true")
    p_local.set_defaults(func=cmd_local)

    p_audit = sub.add_parser("audit", help="Download and audit one Hub JSONL")
    p_audit.add_argument("filename")
    p_audit.add_argument("--max-samples", type=int, default=3)
    p_audit.add_argument(
        "--no-accuracy",
        action="store_true",
        help="Skip lm-eval accuracy recompute (pass@1, strict-match, …)",
    )
    _add_repo_args(p_audit)
    p_audit.set_defaults(func=cmd_audit)

    p_audit_all = sub.add_parser("audit-all", help="Audit all Hub JSONL files one-by-one -> CSV")
    p_audit_all.add_argument("-o", "--output", default=None)
    p_audit_all.add_argument("--max-samples", type=int, default=3)
    p_audit_all.add_argument(
        "--no-accuracy",
        action="store_true",
        help="Skip lm-eval accuracy recompute",
    )
    p_audit_all.add_argument("--limit", type=int, default=None)
    p_audit_all.add_argument("--no-resume", action="store_true")
    p_audit_all.add_argument(
        "--no-retry-errors",
        action="store_true",
        help="Do not re-audit rows whose previous run failed with transient errors",
    )
    _add_repo_args(p_audit_all)
    p_audit_all.set_defaults(func=cmd_audit_all)

    p_conv = sub.add_parser("convert", help="Download JSONL -> repair -> NPZ -> upload")
    p_conv.add_argument("filename")
    p_conv.add_argument("--no-repair", action="store_true")
    p_conv.add_argument("--delete-jsonl-on-hub", action="store_true")
    _add_repo_args(p_conv)
    p_conv.set_defaults(func=cmd_convert)

    p_conv_all = sub.add_parser("convert-all", help="Convert all Hub JSONL files one-by-one -> CSV")
    p_conv_all.add_argument("-o", "--output", default=None)
    p_conv_all.add_argument("--no-repair", action="store_true")
    p_conv_all.add_argument("--delete-jsonl-on-hub", action="store_true")
    p_conv_all.add_argument("--limit", type=int, default=None)
    p_conv_all.add_argument("--no-resume", action="store_true")
    p_conv_all.add_argument("--force", action="store_true", help="Re-convert even if NPZ exists")
    _add_repo_args(p_conv_all)
    p_conv_all.set_defaults(func=cmd_convert_all)

    return parser


def cmd_inventory(args: argparse.Namespace) -> int:
    return print_inventory(
        hub_inventory(
            repo_id=args.repo_id,
            namespace=args.namespace,
            revision=args.revision,
            metrics_csv=args.metrics_csv,
        )
    )


def cmd_local(args: argparse.Namespace) -> int:
    max_samples = None if args.all_samples else args.max_samples
    print_check(check_trace_file(args.path, max_samples=max_samples))
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    row = audit_one_hub_file(
        args.filename,
        repo_id=args.repo_id,
        namespace=args.namespace,
        revision=args.revision,
        metrics_csv=args.metrics_csv,
        max_samples=args.max_samples,
        compute_accuracy=not args.no_accuracy,
    )
    for k, v in row.items():
        print(f"{k}: {v}")
    return 0


def cmd_audit_all(args: argparse.Namespace) -> int:
    rows = run_audit_all(
        repo_id=args.repo_id,
        namespace=args.namespace,
        revision=args.revision,
        metrics_csv=args.metrics_csv,
        output_csv=args.output or DEFAULT_AUDIT_CSV,
        max_samples=args.max_samples,
        compute_accuracy=not args.no_accuracy,
        limit=args.limit,
        resume=not args.no_resume,
        retry_errors=not args.no_retry_errors,
    )
    out = args.output or DEFAULT_AUDIT_CSV
    bad = sum(1 for row in rows if row.get("is_bad"))
    print(f"wrote {len(rows)} rows to {out} ({bad} flagged)")
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    row = convert_one_hub_file(
        args.filename,
        repo_id=args.repo_id,
        namespace=args.namespace,
        revision=args.revision,
        repair=not args.no_repair,
        delete_jsonl_on_hub=args.delete_jsonl_on_hub,
    )
    for k, v in row.items():
        print(f"{k}: {v}")
    return 0 if row.get("status") == "ok" else 2


def cmd_convert_all(args: argparse.Namespace) -> int:
    rows = run_convert_all(
        repo_id=args.repo_id,
        namespace=args.namespace,
        revision=args.revision,
        output_csv=args.output or DEFAULT_CONVERT_CSV,
        repair=not args.no_repair,
        delete_jsonl_on_hub=args.delete_jsonl_on_hub,
        skip_existing_npz=not args.force,
        limit=args.limit,
        resume=not args.no_resume,
    )
    out = args.output or DEFAULT_CONVERT_CSV
    print(f"wrote {len(rows)} rows to {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
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
