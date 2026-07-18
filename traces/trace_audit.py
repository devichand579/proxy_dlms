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
    DEFAULT_REPAIR_NPZ_CSV,
    audit_one_hub_file,
    convert_one_hub_file,
    hub_inventory,
    print_check,
    print_inventory,
    repair_one_hub_npz,
    run_audit_all,
    run_convert_all,
    run_repair_npz_all,
)


def _add_pcfg_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--pcfg-dataset",
        default=None,
        help="PCFG Hub dataset for test-split sample count (default: PCFG_HF_DATASET or pcfg-code)",
    )
    p.add_argument(
        "--pcfg-namespace",
        default=None,
        help="Org/user for --pcfg-dataset when the repo id has no prefix",
    )


def _add_repo_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--repo", "--repo-id", dest="repo_id", default=None)
    p.add_argument("--namespace", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--metrics-csv", default=None)


def _add_output_repo_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--output-repo",
        "--output-repo-id",
        dest="output_repo_id",
        default=None,
        help="Destination dataset repo for converted NPZ (default: same as --repo)",
    )
    p.add_argument(
        "--output-namespace",
        default=None,
        help="Namespace for --output-repo when it has no org prefix",
    )


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
  python traces/trace_audit.py audit-all --npz-only
  python traces/trace_audit.py convert foo.jsonl --output-repo schedulerlens-npz
  python traces/trace_audit.py convert-all --output-repo schedulerlens-npz
  python traces/trace_audit.py repair-npz llada-100m-pcfg-pcfg_dus_s128.npz --repo llada-pcfg-traces
  python traces/trace_audit.py repair-npz-all --repo llada-pcfg-traces -o fetch_evals/trace_repair_npz.csv
  python tests/test_tracing.py
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

    p_audit = sub.add_parser("audit", help="Download and audit one Hub JSONL or NPZ")
    p_audit.add_argument("filename")
    p_audit.add_argument("--max-samples", type=int, default=3)
    p_audit.add_argument(
        "--no-accuracy",
        action="store_true",
        help="Skip lm-eval accuracy recompute (pass@1, strict-match, …)",
    )
    _add_repo_args(p_audit)
    _add_pcfg_args(p_audit)
    p_audit.set_defaults(func=cmd_audit)

    p_audit_all = sub.add_parser("audit-all", help="Audit Hub JSONL and NPZ files one-by-one -> CSV")
    p_audit_all.add_argument("-o", "--output", default=None)
    p_audit_all.add_argument("--max-samples", type=int, default=3)
    p_audit_all.add_argument(
        "--no-accuracy",
        action="store_true",
        help="Skip lm-eval accuracy recompute",
    )
    p_audit_all.add_argument(
        "--jsonl-only",
        action="store_true",
        help="Audit only .jsonl files",
    )
    p_audit_all.add_argument(
        "--npz-only",
        action="store_true",
        help="Audit only .npz files",
    )
    p_audit_all.add_argument("--limit", type=int, default=None)
    p_audit_all.add_argument("--no-resume", action="store_true")
    p_audit_all.add_argument(
        "--no-retry-errors",
        action="store_true",
        help="Do not re-audit rows whose previous run failed with transient errors",
    )
    _add_repo_args(p_audit_all)
    _add_pcfg_args(p_audit_all)
    p_audit_all.set_defaults(func=cmd_audit_all)

    p_conv = sub.add_parser(
        "convert",
        help="Download JSONL → repair+pack NPZ → audit NPZ → upload",
    )
    p_conv.add_argument("filename")
    p_conv.add_argument("--no-repair", action="store_true")
    p_conv.add_argument("--delete-jsonl-on-hub", action="store_true")
    p_conv.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Process-pool workers for per-sample repair (default: 1)",
    )
    _add_repo_args(p_conv)
    _add_output_repo_args(p_conv)
    p_conv.set_defaults(func=cmd_convert)

    p_conv_all = sub.add_parser(
        "convert-all",
        help="Convert all Hub JSONL (repair+pack → audit NPZ → upload) → CSV",
    )
    p_conv_all.add_argument("-o", "--output", default=None)
    p_conv_all.add_argument("--no-repair", action="store_true")
    p_conv_all.add_argument("--delete-jsonl-on-hub", action="store_true")
    p_conv_all.add_argument("--limit", type=int, default=None)
    p_conv_all.add_argument("--no-resume", action="store_true")
    p_conv_all.add_argument(
        "--no-retry-errors",
        action="store_true",
        help="Do not re-convert rows whose previous run failed",
    )
    p_conv_all.add_argument("--force", action="store_true", help="Re-convert even if NPZ exists")
    p_conv_all.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Process-pool workers for per-sample repair (default: 1)",
    )
    _add_repo_args(p_conv_all)
    _add_output_repo_args(p_conv_all)
    p_conv_all.set_defaults(func=cmd_convert_all)

    p_rnpz = sub.add_parser(
        "repair-npz",
        help="Download NPZ → repair (phantoms/mask-diff/trim) → re-upload",
    )
    p_rnpz.add_argument("filename")
    p_rnpz.add_argument(
        "--force",
        action="store_true",
        help="Repair even when audit does not flag a mismatch",
    )
    p_rnpz.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Process-pool workers for per-sample repair (default: 1)",
    )
    _add_repo_args(p_rnpz)
    _add_output_repo_args(p_rnpz)
    p_rnpz.set_defaults(func=cmd_repair_npz)

    p_rnpz_all = sub.add_parser(
        "repair-npz-all",
        help="Repair Hub NPZs (phantoms/mask-diff/trim) → re-upload → CSV",
    )
    p_rnpz_all.add_argument("-o", "--output", default=None)
    p_rnpz_all.add_argument("--force", action="store_true")
    p_rnpz_all.add_argument("--limit", type=int, default=None)
    p_rnpz_all.add_argument("--no-resume", action="store_true")
    p_rnpz_all.add_argument("--no-retry-errors", action="store_true")
    p_rnpz_all.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Process-pool workers for per-sample repair (default: 1)",
    )
    _add_repo_args(p_rnpz_all)
    _add_output_repo_args(p_rnpz_all)
    p_rnpz_all.set_defaults(func=cmd_repair_npz_all)

    p_rlocal = sub.add_parser(
        "repair-npz-local",
        help="Repair a local NPZ in place (or write to --output)",
    )
    p_rlocal.add_argument("path")
    p_rlocal.add_argument("-o", "--output", default=None, help="Output NPZ (default: overwrite)")
    p_rlocal.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Process-pool workers for per-sample repair (default: 1)",
    )
    p_rlocal.set_defaults(func=cmd_repair_npz_local)

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
        pcfg_hf_dataset=args.pcfg_dataset,
        pcfg_hf_namespace=args.pcfg_namespace,
    )
    for k, v in row.items():
        print(f"{k}: {v}")
    return 0


def cmd_audit_all(args: argparse.Namespace) -> int:
    if args.jsonl_only and args.npz_only:
        print("Error: --jsonl-only and --npz-only are mutually exclusive", file=sys.stderr)
        return 1
    rows = run_audit_all(
        repo_id=args.repo_id,
        namespace=args.namespace,
        revision=args.revision,
        metrics_csv=args.metrics_csv,
        output_csv=args.output or DEFAULT_AUDIT_CSV,
        max_samples=args.max_samples,
        compute_accuracy=not args.no_accuracy,
        include_jsonl=not args.npz_only,
        include_npz=not args.jsonl_only,
        limit=args.limit,
        resume=not args.no_resume,
        retry_errors=not args.no_retry_errors,
        pcfg_hf_dataset=args.pcfg_dataset,
        pcfg_hf_namespace=args.pcfg_namespace,
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
        output_repo_id=args.output_repo_id,
        output_namespace=args.output_namespace,
        revision=args.revision,
        repair=not args.no_repair,
        delete_jsonl_on_hub=args.delete_jsonl_on_hub,
        workers=args.workers,
    )
    for k, v in row.items():
        print(f"{k}: {v}")
    return 0 if row.get("status") == "ok" else 2


def cmd_convert_all(args: argparse.Namespace) -> int:
    rows = run_convert_all(
        repo_id=args.repo_id,
        namespace=args.namespace,
        output_repo_id=args.output_repo_id,
        output_namespace=args.output_namespace,
        revision=args.revision,
        output_csv=args.output or DEFAULT_CONVERT_CSV,
        repair=not args.no_repair,
        delete_jsonl_on_hub=args.delete_jsonl_on_hub,
        skip_existing_npz=not args.force,
        limit=args.limit,
        resume=not args.no_resume,
        retry_errors=not args.no_retry_errors,
        workers=args.workers,
    )
    out = args.output or DEFAULT_CONVERT_CSV
    print(f"wrote {len(rows)} rows to {out}")
    return 0


def cmd_repair_npz(args: argparse.Namespace) -> int:
    row = repair_one_hub_npz(
        args.filename,
        repo_id=args.repo_id,
        namespace=args.namespace,
        output_repo_id=args.output_repo_id,
        output_namespace=args.output_namespace,
        revision=args.revision,
        force=args.force,
        workers=args.workers,
    )
    for k, v in row.items():
        print(f"{k}: {v}")
    return 0 if row.get("status") in ("ok", "skipped_clean") else 2


def cmd_repair_npz_all(args: argparse.Namespace) -> int:
    rows = run_repair_npz_all(
        repo_id=args.repo_id,
        namespace=args.namespace,
        output_repo_id=args.output_repo_id,
        output_namespace=args.output_namespace,
        revision=args.revision,
        output_csv=args.output or DEFAULT_REPAIR_NPZ_CSV,
        force=args.force,
        limit=args.limit,
        resume=not args.no_resume,
        retry_errors=not args.no_retry_errors,
        workers=args.workers,
    )
    out = args.output or DEFAULT_REPAIR_NPZ_CSV
    n_ok = sum(1 for r in rows if r.get("status") in ("ok", "skipped_clean"))
    print(f"wrote {len(rows)} rows to {out} ({n_ok} ok/skipped)")
    return 0


def cmd_repair_npz_local(args: argparse.Namespace) -> int:
    from traces.trace_checks import (
        check_trace_file,
        stream_repair_npz,
        validate_npz_roundtrip,
    )

    src = Path(args.path)
    dst = Path(args.output) if args.output else src
    check = check_trace_file(src, max_samples=3)
    print_check(check)
    n, repairs, _task, samples = stream_repair_npz(
        src, dst, workers=args.workers
    )
    issues = validate_npz_roundtrip(samples, dst)
    print(f"repaired_samples: {n}")
    print(f"repairs: {len(repairs)}")
    for r in repairs[:10]:
        print(f"  {r}")
    if len(repairs) > 10:
        print(f"  ... and {len(repairs) - 10} more")
    print(f"roundtrip_ok: {not issues}")
    if issues:
        for i in issues[:10]:
            print(f"  {i}")
        return 2
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
