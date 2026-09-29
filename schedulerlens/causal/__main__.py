"""CLI: python -m schedulerlens.causal audit|swap|adaptive"""

from __future__ import annotations

import argparse
import fnmatch
import logging
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from schedulerlens.causal.audit import ( 
    audit_npz,
    iter_npz_paths,
    keep_stem,
    load_audit_csv,
    write_audit_csv,
)
from load_model import infer_b44_eos_unmask_safety  
from schedulerlens.causal.models import place_fleet  
from schedulerlens.causal.select import SUPPORTED_PHI 
from traces.hf_trace_upload import (  
    download_file_to_temp,
    list_repo_files,
    remove_temp_download,
)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--predictor", default="llada", help="Predictor model")
    p.add_argument("--controller", default="llada1_5", help="Controller model")
    p.add_argument("--device-map", choices=("split", "share"), default="split")
    p.add_argument("--block-length", type=int, default=None)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--gamma", type=float, default=0.1)


def _cmd_audit(args: argparse.Namespace) -> int:
    paths = list(iter_npz_paths(args.local_dir, pattern=args.pattern))
    if args.repo:
        names = list_repo_files(repo_id=args.repo, namespace=args.namespace)
        paths = [
            Path(n) for n in names if fnmatch.fnmatch(Path(n).name, args.pattern)
        ]
    kept = []
    for path in paths:
        name = path.name
        if keep_stem(Path(name), steps=args.steps, exclude_task=args.exclude_task):
            kept.append(path)
    if args.limit is not None:
        kept = kept[: args.limit]
    if not kept:
        logging.error("No NPZs matched filters")
        return 1
    out = Path(args.output_dir) / "causal_audit.csv"
    rows = load_audit_csv(out)
    done = {
        (row.get("filename"), row.get("predictor"), row.get("controller"))
        for row in rows
    }
    pending = [
        path
        for path in kept
        if (path.name, args.predictor, args.controller) not in done
    ]
    logging.info(
        "Auditing %d file(s), %d already in %s",
        len(pending),
        len(kept) - len(pending),
        out,
    )
    if not pending:
        print(f"CSV: {out} ({len(rows)} rows, nothing new)")
        return 0
    loaded = place_fleet(args.predictor, args.controller, device_map=args.device_map)
    logging.info(
        "predictor on %s, controller on %s",
        loaded["predictor_devices"],
        loaded["controller_devices"],
    )
    for path in pending:
        temp = None
        local = path
        try:
            if args.repo and not path.is_file():
                local, temp = download_file_to_temp(
                    path.name, repo_id=args.repo, namespace=args.namespace
                )
            logging.info("Audit %s", local.name)
            rows.append(
                audit_npz(
                    local,
                    predictors=loaded["predictors"],
                    controllers=loaded["controllers"],
                    mask_id=loaded["mask_id"],
                    predictor_name=args.predictor,
                    controller_name=args.controller,
                    eos_id=loaded["eos_id"],
                    eot_id=loaded["eot_id"],
                    controller_drop_eos=infer_b44_eos_unmask_safety(loaded["controller_dir"]),
                    block_length=args.block_length,
                    top_k=args.top_k,
                    gamma=args.gamma,
                    max_samples=args.max_samples,
                    max_steps=args.max_steps,
                    batch_size=args.batch_size,
                )
            )
            write_audit_csv(out, rows)
        finally:
            if temp is not None:
                remove_temp_download(temp)
    print(f"CSV: {out} ({len(rows)} rows)")
    return 0


def _cmd_roles(args: argparse.Namespace) -> int:
    """Print the four swap cells and the adaptive arms for a Hydra sweep."""
    os.environ["PC_PHI"] = args.phi
    os.environ["PC_ARM"] = args.arm
    os.environ["PC_LAMBDA"] = str(args.lam)
    os.environ["PC_CONTROLLER"] = args.controller
    os.environ["PC_DEVICE_MAP"] = args.device_map
    print(
        f"mode={args.mode} predictor={args.predictor} controller={args.controller} "
        f"phi={args.phi} arm={args.arm} lambda={args.lam}"
    )
    print(
        "Launch lm-eval with the harness model as the predictor and "
        "PC_CONTROLLER set to the controller. schedulers.generate routes "
        "mode pc_swap / pc_adaptive into schedulerlens.causal.generate."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SchedulerLens causal predictor–controller experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="Exp1: fixed-history Δ_pred / Δ_ctrl on NPZs")
    src = audit.add_mutually_exclusive_group(required=True)
    src.add_argument("--local-dir", type=Path)
    src.add_argument("--repo")
    audit.add_argument("--namespace", default="devichand")
    audit.add_argument("--pattern", default="*.npz")
    audit.add_argument("--steps", type=int, default=256)
    audit.add_argument("--exclude-task", default="arc_challenge_chat")
    audit.add_argument("--max-samples", type=int, default=None)
    audit.add_argument("--max-steps", type=int, default=None)
    audit.add_argument("--batch-size", type=int, default=8)
    audit.add_argument("--limit", type=int, default=None)
    audit.add_argument("-o", "--output-dir", type=Path, required=True)
    _add_common(audit)

    for name, help_text in (
        ("swap", "Exp2: full predictor–controller swap (sets env for eval)"),
        ("adaptive", "Exp3: proxy-guided adaptive selection (sets env for eval)"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--phi", default="eb", choices=SUPPORTED_PHI)
        p.add_argument("--arm", default="joint", choices=("m", "j", "joint", "random"))
        p.add_argument("--lam", type=float, default=1.0)
        _add_common(p)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        return _cmd_audit(args)
    args.mode = "pc_swap" if args.command == "swap" else "pc_adaptive"
    return _cmd_roles(args)


if __name__ == "__main__":
    raise SystemExit(main())
