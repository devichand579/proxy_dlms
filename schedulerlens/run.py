#!/usr/bin/env python3
"""Single SchedulerLens CLI.

Subcommands:
  file         Analyze one local NPZ (metrics CSV/JSON)
  inventory    List Hub NPZs (no download)
  compare      Batch Hub/local compare → CSV
  causal       Predictor–controller audit / swap / adaptive
  prefix-sweep      Prefix M-hat, J-hat, and G-hat/f for each run
  prefix-sweep-pcfg Same sweep on PCFG traces against oracle M
  prefix-selection  Scheduler with the lowest prefix G-hat/f
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from repo_env_defaults import apply_repo_env_defaults  # noqa: E402
from schedulerlens.completion import (  # noqa: E402
    compute_completion_metrics,
    join_rows_with_strace,
)
from schedulerlens.metrics import (  # noqa: E402
    TaskMetrics,
    CORR_PREFIXES,
    build_metrics_row,
    compute_task_metrics,
    expand_prefix_metric_rows,
    parse_prefix_steps,
    parse_run_stem,
    prefix_pearsons,
    write_metrics_csv,
)
from schedulerlens.oracle import (  # noqa: E402
    _ensure_pcfg_path,
    compute_task_perplexity,
    save_oracle_artifacts,
)
from schedulerlens.proxies import TraceProxyError  # noqa: E402

logger = logging.getLogger("schedulerlens")


def run_schedulerlens(
    trace_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    max_samples: Optional[int] = None,
    bucket_size: Optional[int] = None,
    verbose: bool = True,
    compute_perplexity: bool = True,
    replay_oracle: bool = False,
    force_pcfg: bool = False,
    model_path: Optional[Union[str, Path]] = None,
    hf_model: Optional[str] = None,
    oracle_workers: int = 1,
    step_workers: Optional[int] = None,
    forward_batch_size: int = 8,
    jt_exact_max: Optional[int] = None,
    jt_markov_order: Optional[int] = None,
    jt_large_set: Optional[int] = None,
    strace_metrics_path: Optional[Union[str, Path]] = None,
    oracle: Optional[Union[str, Path]] = None,
    prefix_steps: Optional[Sequence[int]] = None,
    metrics_only: bool = False,
    csv_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute metrics and write metrics.json / metrics.csv to output_dir."""
    from schedulerlens.oracle import (
        load_oracle_artifacts,
        resolve_oracle_npz,
        task_perplexity_from_oracle_npz,
    )

    _ensure_pcfg_path()
    trace_path = Path(trace_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = parse_run_stem(trace_path)
    prefix_ks = parse_prefix_steps(prefix_steps)
    if (meta.is_pcfg or force_pcfg) and not prefix_ks:
        prefix_ks = CORR_PREFIXES
    if metrics_only:
        compute_perplexity = False

    cached_oracle = resolve_oracle_npz(oracle, meta.stem) if oracle else None

    metrics = compute_task_metrics(
        trace_path,
        max_samples=max_samples,
        bucket_size=bucket_size,
        verbose=verbose,
        prefix_steps=prefix_ks or None,
    )

    completion = None
    if not metrics_only:
        steps_budget = int(meta.steps) if meta.steps is not None else int(metrics.n_steps)
        completion = compute_completion_metrics(
            trace_path,
            model=meta.model,
            steps_budget=steps_budget,
            max_samples=max_samples,
        )

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "trace_path": str(trace_path.resolve()),
                "meta": meta.to_dict(),
                **metrics.to_dict(),
                "completion": completion,
            },
            f,
            indent=2,
        )

    result: Dict[str, Any] = {
        "M_hat": metrics.M_hat,
        "J_hat": metrics.J_hat,
        "G_hat": float(metrics.M_hat) + float(metrics.J_hat),
        "n_samples": metrics.n_samples,
        "n_steps": metrics.n_steps,
        "metrics_path": str(metrics_path.resolve()),
        "metrics": metrics,
        "meta": meta,
        "completion": completion,
    }

    ppl = None
    prefix_pearson = None
    do_replay = (
        cached_oracle is None
        and replay_oracle
        and (meta.is_pcfg or force_pcfg)
    )

    if compute_perplexity:
        if cached_oracle is not None:
            logger.info("Using cached oracle %s", cached_oracle)
            ppl = task_perplexity_from_oracle_npz(cached_oracle)
            result["oracle_artifacts_path"] = str(cached_oracle.resolve())
        else:
            engine = vocab = model = device = None
            if do_replay:
                try:
                    from pcfg_data import build_stack

                    _g, _cnf, engine, vocab = build_stack()
                except Exception as e:
                    logger.warning("PCFG stack unavailable: %s", e)
                from schedulerlens.hub import _load_oracle_stack

                _e, vocab2, model, device = _load_oracle_stack(
                    replay_oracle=True,
                    model_path=model_path,
                    hf_model=hf_model,
                )
                if engine is None:
                    engine = _e
                if vocab2 is not None:
                    vocab = vocab2

            ppl = compute_task_perplexity(
                trace_path,
                engine=engine,
                vocab=vocab,
                model=model,
                device=device,
                replay_oracle=do_replay and model is not None,
                max_samples=max_samples,
                verbose=verbose,
                oracle_workers=oracle_workers,
                step_workers=step_workers,
                forward_batch_size=forward_batch_size,
                jt_exact_max=jt_exact_max,
                jt_markov_order=jt_markov_order,
                jt_large_set=jt_large_set,
            )
            tables = getattr(ppl, "oracle_tables", None) or {}
            if do_replay and (
                tables.get("step_sample_idx")
                or any(s is not None for s in ppl.sample_oracle)
            ):
                oracle_path = save_oracle_artifacts(
                    output_dir / "oracle.npz",
                    ppl,
                    stem=meta.stem,
                )
                result["oracle_artifacts_path"] = str(oracle_path.resolve())
                step_sample_idx = list(ppl.oracle_tables.get("step_sample_idx", []))
                step = list(ppl.oracle_tables.get("step", []))
                step_mt = list(ppl.oracle_tables.get("step_Mt", []))
                step_jt = list(ppl.oracle_tables.get("step_Jt", []))
                if len(step):
                    prefix_pearson = prefix_pearsons(
                        metrics,
                        step_sample_idx=step_sample_idx,
                        step=step,
                        step_mt=step_mt,
                        step_jt=step_jt,
                    )
                ppl.oracle_tables = {}
                ppl.sample_oracle = []

        ppl_path = output_dir / "perplexity.json"
        with ppl_path.open("w", encoding="utf-8") as f:
            json.dump(ppl.to_dict(), f, indent=2)
        result["perplexity"] = ppl
        result["perplexity_path"] = str(ppl_path.resolve())

        if cached_oracle is not None:
            arts = load_oracle_artifacts(cached_oracle)
            step = arts.get("step", [])
            if len(step):
                prefix_pearson = prefix_pearsons(
                    metrics,
                    step_sample_idx=arts.get("step_sample_idx", []),
                    step=step,
                    step_mt=arts.get("step_Mt", []),
                    step_jt=arts.get("step_Jt", []),
                )
        if prefix_pearson:
            result["prefix_pearson"] = prefix_pearson

        with metrics_path.open("r", encoding="utf-8") as f:
            merged = json.load(f)
        merged["perplexity"] = ppl.to_dict()
        if prefix_pearson:
            merged["prefix_pearson"] = {
                label: {
                    "pearson_M": pair[0],
                    "pearson_J": pair[1],
                    "pearson_G": pair[2],
                }
                for label, pair in prefix_pearson.items()
            }
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)

    row = build_metrics_row(
        trace_path,
        metrics=metrics,
        perplexity=ppl,
        prefix_pearson=prefix_pearson,
        meta=meta,
        completion=completion,
    )
    row = join_rows_with_strace([row], strace_path=strace_metrics_path)[0]
    rows = expand_prefix_metric_rows(row, metrics, prefix_pearson)
    csv_path = write_metrics_csv(_placed_csv(output_dir, csv_name, "metrics.csv"), rows)
    result["csv_path"] = str(csv_path.resolve())
    result["row"] = row
    result["rows"] = rows
    return result


def _placed_csv(output_dir: Union[str, Path], csv_name: Optional[str], default_name: str) -> Path:
    """CSV path: an absolute ``--csv`` as given, otherwise ``output_dir / name``."""
    if csv_name:
        given = Path(csv_name)
        if given.is_absolute():
            given.parent.mkdir(parents=True, exist_ok=True)
            return given
        name = given
    else:
        name = Path(default_name)
    path = Path(output_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


_REPO = Path(__file__).resolve().parents[1]
COMPARE_DEFAULT = _REPO / "outputs/schedulerlens/compare"
SWEEP_DEFAULT = _REPO / "outputs/schedulerlens/prefix_sweep/prefix_sweep.csv"
SWEEP_PCFG_DEFAULT = _REPO / "outputs/schedulerlens/prefix_sweep_pcfg/prefix_sweep_pcfg.csv"


def _sweep_csv(args: argparse.Namespace, default: Path) -> Path:
    directory = args.output_dir if args.output_dir is not None else default.parent
    return _placed_csv(directory, args.csv, default.name)


def _selection_csv(args: argparse.Namespace) -> Path:
    runs = Path(args.runs) if args.runs is not None else SWEEP_DEFAULT
    directory = args.output_dir if args.output_dir is not None else runs.parent
    return _placed_csv(directory, args.csv, f"prefix_selection_tau_{args.tau}.csv")


def _add_csv_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--csv",
        default=None,
        help="Output CSV filename or absolute path (default depends on the command)",
    )


def _add_repo_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--repo", "--repo-id", dest="repo_id", default=None)
    p.add_argument(
        "--namespace",
        default=None,
        help="HF user/org for short --repo names and oracle model ids "
        "({ns}/{stem_model}). Default: TRACE_HF_NAMESPACE / whoami",
    )
    p.add_argument("--revision", default=None)


def _add_exclude_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--exclude-scheduler",
        default=None,
        help="Comma-separated schedulers to skip (e.g. fhs,fdm)",
    )
    p.add_argument(
        "--exclude-task",
        default=None,
        help="Comma-separated tasks to skip (e.g. arc_challenge_chat)",
    )


def _add_filter_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--pattern", default="*.npz")
    p.add_argument("--model", dest="model_filter", default=None)
    p.add_argument("--scheduler", dest="scheduler_filter", default=None)
    p.add_argument("--steps", dest="steps_filter", type=int, default=None)
    p.add_argument("--pcfg-only", action="store_true")
    _add_exclude_args(p)


def _add_prefix_steps_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--prefix-steps",
        default=None,
        metavar="Ks",
        help=(
            "Comma-separated step budgets (e.g. 8,16,32,64). Emits extra CSV rows "
            "with cumulative M_hat/J_hat/G_hat over step < K (prefix_steps column)."
        ),
    )


def _add_fast_metrics_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--metrics-only",
        action="store_true",
        help=(
            "Fast path: proxy caps (+ prefix rows) and strace join only. "
            "Skips completion diagnostics and the cheap non-oracle PPL re-stream. "
            "--oracle / --replay-oracle still run when set."
        ),
    )


def _add_oracle_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--replay-oracle", action="store_true")
    p.add_argument("--pcfg", action="store_true", help="Force PCFG oracle gating")
    p.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Local PCFG checkpoint dir (or HF repo id if not a local path)",
    )
    p.add_argument(
        "--hf-model",
        default=None,
        help="Fixed HF model repo for all traces (overrides per-stem lazy load)",
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--oracle-workers",
        type=int,
        default=1,
        metavar="N",
        help="Parallel samples during oracle replay (shared GPU model + engine pool)",
    )
    p.add_argument(
        "--oracle-step-workers",
        type=int,
        default=None,
        metavar="N",
        help="Parallel PCFG Mt/Jt threads per sample (default: min(8, CPUs))",
    )
    p.add_argument(
        "--oracle-forward-batch",
        type=int,
        default=8,
        metavar="B",
        help="Batched model forwards across steps within a sample (default: 8)",
    )
    p.add_argument(
        "--oracle-jt-exact-max",
        type=int,
        default=None,
        metavar="K",
        help=(
            "Exact PCFG Jt only for |S|≤K (default 3). Larger simultaneous "
            "unmasks (DUS/EB) use Markov Jt to avoid OOM. "
            "Env: SCHEDULERLENS_JT_EXACT_MAX"
        ),
    )
    p.add_argument(
        "--oracle-jt-markov-order",
        type=int,
        default=None,
        metavar="M",
        help=(
            "Markov order for large-|S| Jt (default 2 → exact joints of size ≤3). "
            "Env: SCHEDULERLENS_JT_MARKOV_ORDER"
        ),
    )
    p.add_argument(
        "--oracle-jt-large-set",
        type=int,
        default=None,
        metavar="N",
        help=(
            "If |S|>N, force pairwise Jt (Markov order 1) for speed (default 6). "
            "Set 0 to disable. Env: SCHEDULERLENS_JT_LARGE_SET"
        ),
    )
    p.add_argument(
        "--strace-metrics",
        type=Path,
        default=None,
        help="Path to strace_metrics.csv for accuracy/NFE join "
        "(default: fetch_evals/strace_metrics.csv)",
    )
    p.add_argument(
        "--oracle",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Reuse saved oracle.npz (file or directory). Matched to traces by "
            "stem; fills Mt/Jt/PPL + corr_* without live replay. With "
            "--replay-oracle, missing stems still replay."
        ),
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SchedulerLens — single entrypoint for metrics and Hub analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- file ---
    p_file = sub.add_parser("file", help="Analyze one local NPZ")
    p_file.add_argument("trace_path", type=Path)
    p_file.add_argument("-o", "--output-dir", type=Path, required=True)
    _add_csv_arg(p_file)
    p_file.add_argument("--bucket-size", type=int, default=None, metavar="N")
    p_file.add_argument("--no-perplexity", action="store_true")
    _add_oracle_args(p_file)
    _add_prefix_steps_arg(p_file)
    _add_fast_metrics_arg(p_file)
    p_file.add_argument("-q", "--quiet", action="store_true")

    # --- inventory ---
    p_inv = sub.add_parser("inventory", help="List Hub NPZs (no downloads)")
    _add_repo_args(p_inv)
    _add_filter_args(p_inv)

    # --- compare (batch) ---
    p_cmp = sub.add_parser(
        "compare",
        help="Batch Hub/local compare → CSV",
    )
    src = p_cmp.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo", "--repo-id", dest="repo_id", default=None)
    src.add_argument("--local-dir", type=Path, default=None)
    p_cmp.add_argument(
        "--namespace",
        default=None,
        help="HF user/org for short --repo names and oracle model ids "
        "({ns}/{stem_model}). Default: TRACE_HF_NAMESPACE / whoami",
    )
    p_cmp.add_argument("--revision", default=None)
    p_cmp.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=COMPARE_DEFAULT,
        help="Output directory (default: outputs/schedulerlens/compare)",
    )
    _add_csv_arg(p_cmp)
    _add_filter_args(p_cmp)
    _add_oracle_args(p_cmp)
    _add_prefix_steps_arg(p_cmp)
    _add_fast_metrics_arg(p_cmp)
    p_cmp.add_argument(
        "--include-duplicate-budgets",
        action="store_true",
        help="Also analyze step budgets whose NFE repeats a lower budget (DUS, top-k)",
    )
    p_cmp.add_argument("--limit", type=int, default=None)
    p_cmp.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Skip a filename only when every requested prefix is already in the CSV (default). "
            "A missing prefix redoes that file. --no-resume rewrites the sheet."
        ),
    )
    p_cmp.add_argument("--no-purge-cache", action="store_true")
    p_cmp.add_argument("-q", "--quiet", action="store_true")

    p_causal = sub.add_parser(
        "causal",
        help="Predictor–controller causal experiments (audit, swap, adaptive)",
    )
    p_causal.add_argument("causal_argv", nargs=argparse.REMAINDER)

    p_sweep = sub.add_parser(
        "prefix-sweep",
        help="Prefix M-hat, J-hat, and G-hat/f for each run",
    )
    _add_exclude_args(p_sweep)
    p_sweep.add_argument("-o", "--output-dir", type=Path, default=None)
    _add_csv_arg(p_sweep)
    p_sweep.add_argument("-q", "--quiet", action="store_true")
    p_sweep_pcfg = sub.add_parser(
        "prefix-sweep-pcfg",
        help="Same prefix sweep on PCFG traces against oracle M",
    )
    _add_exclude_args(p_sweep_pcfg)
    p_sweep_pcfg.add_argument("-o", "--output-dir", type=Path, default=None)
    _add_csv_arg(p_sweep_pcfg)
    p_sweep_pcfg.add_argument("-q", "--quiet", action="store_true")
    p_prefix = sub.add_parser(
        "prefix-selection",
        help="Accuracy of the scheduler with the lowest prefix G-hat/f",
    )
    _add_exclude_args(p_prefix)
    p_prefix.add_argument("-o", "--output-dir", type=Path, default=None)
    _add_csv_arg(p_prefix)
    p_prefix.add_argument("-q", "--quiet", action="store_true")
    p_prefix.add_argument(
        "--runs",
        type=Path,
        default=None,
        help="Prefix sweep CSV (default: outputs/schedulerlens/prefix_sweep/prefix_sweep.csv)",
    )
    p_prefix.add_argument("--tau", default="1.0", help="Entropy budget column to rank (default 1.0)")

    return parser


def _cmd_file(args: argparse.Namespace) -> int:
    if not args.trace_path.exists():
        logger.error("Trace file not found: %s", args.trace_path)
        return 1
    if args.trace_path.suffix.lower() != ".npz":
        logger.error(
            "SchedulerLens expects an NPZ trace archive, got %s",
            args.trace_path.name,
        )
        return 1
    try:
        result = run_schedulerlens(
            args.trace_path,
            args.output_dir,
            max_samples=args.max_samples,
            bucket_size=args.bucket_size,
            verbose=not args.quiet,
            compute_perplexity=not args.no_perplexity,
            replay_oracle=args.replay_oracle,
            force_pcfg=args.pcfg,
            model_path=args.model_path,
            hf_model=args.hf_model,
            oracle_workers=args.oracle_workers,
            step_workers=args.oracle_step_workers,
            forward_batch_size=args.oracle_forward_batch,
            jt_exact_max=args.oracle_jt_exact_max,
            jt_markov_order=args.oracle_jt_markov_order,
            jt_large_set=args.oracle_jt_large_set,
            strace_metrics_path=args.strace_metrics,
            oracle=args.oracle,
            prefix_steps=parse_prefix_steps(args.prefix_steps) or None,
            metrics_only=bool(args.metrics_only),
            csv_name=args.csv,
        )
    except TraceProxyError as e:
        logger.error("Aborted (bad trace data): %s", e)
        return 1
    except Exception as e:
        logger.error("Failed: %s", e)
        return 1

    metrics: TaskMetrics = result["metrics"]
    print(metrics.summary())
    if result.get("perplexity") is not None:
        ppl = result["perplexity"]
        print(
            f"\nPerplexity ({ppl.n_samples} samples): "
            f"model_commit_ppl={ppl.model_commit_ppl:.2f} "
            f"oracle_ppl={ppl.oracle_ppl:.2f} "
            f"Mt={ppl.Mt_total:.4f} Jt={ppl.Jt_total:.4f}"
        )
    if result.get("prefix_pearson"):
        parts = [
            f"{label} M {pair[0]:.3f} J {pair[1]:.3f} G {pair[2]:.3f}"
            for label, pair in result["prefix_pearson"].items()
        ]
        print("\nSample Pearson (M̂, Mt), (Ĵ, Jt), (Ĝ, G): " + "  ".join(parts))
    print(f"\nOutput dir: {args.output_dir.resolve()}")
    print(f"  metrics.json : {result['metrics_path']}")
    print(f"  metrics.csv  : {result['csv_path']}")
    if result.get("oracle_artifacts_path"):
        print(f"  oracle.npz   : {result['oracle_artifacts_path']}")
    return 0


def _cmd_inventory(args: argparse.Namespace) -> int:
    from schedulerlens.hub import hub_inventory, print_inventory

    inv = hub_inventory(
        repo_id=args.repo_id,
        namespace=args.namespace,
        revision=args.revision,
        pattern=args.pattern,
        model=args.model_filter,
        scheduler=args.scheduler_filter,
        steps=args.steps_filter,
        pcfg_only=args.pcfg_only,
        exclude_scheduler=args.exclude_scheduler,
        exclude_task=args.exclude_task,
    )
    print_inventory(inv)
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from schedulerlens.hub import run_compare

    try:
        result = run_compare(
            output_dir=args.output_dir,
            repo_id=args.repo_id,
            namespace=args.namespace,
            revision=args.revision,
            local_dir=args.local_dir,
            pattern=args.pattern,
            model_filter=args.model_filter,
            scheduler_filter=args.scheduler_filter,
            steps_filter=args.steps_filter,
            pcfg_only=args.pcfg_only,
            exclude_scheduler=args.exclude_scheduler,
            exclude_task=args.exclude_task,
            force_pcfg=args.pcfg,
            replay_oracle=args.replay_oracle,
            model_path=args.model_path,
            hf_model=args.hf_model,
            max_samples=args.max_samples,
            limit=args.limit,
            resume=args.resume,
            purge_cache=not args.no_purge_cache,
            verbose=not args.quiet,
            oracle_workers=args.oracle_workers,
            step_workers=args.oracle_step_workers,
            forward_batch_size=args.oracle_forward_batch,
            jt_exact_max=args.oracle_jt_exact_max,
            jt_markov_order=args.oracle_jt_markov_order,
            jt_large_set=args.oracle_jt_large_set,
            strace_metrics_path=args.strace_metrics,
            skip_duplicate_budgets=not args.include_duplicate_budgets,
            oracle_dir=args.oracle,
            prefix_steps=parse_prefix_steps(args.prefix_steps) or None,
            metrics_only=bool(args.metrics_only),
            csv_name=args.csv,
        )
    except TraceProxyError as e:
        logger.error("%s", e)
        return 1
    except Exception as e:
        logger.error("Compare failed: %s", e)
        return 1

    print(f"Runs: {len(result['rows'])}")
    print(f"CSV : {result['csv_path']}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    apply_repo_env_defaults()
    _ensure_pcfg_path()
    args = build_argparser().parse_args(argv)
    quiet = bool(getattr(args, "quiet", False))
    logging.basicConfig(
        level=logging.WARNING if quiet else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    cmd = args.command
    if cmd == "file":
        return _cmd_file(args)
    if cmd == "inventory":
        return _cmd_inventory(args)
    if cmd == "compare":
        return _cmd_compare(args)
    if cmd == "causal":
        from schedulerlens.causal.__main__ import main as causal_main

        return causal_main(args.causal_argv)
    if cmd == "prefix-sweep":
        from schedulerlens.prefix_sweep import main as prefix_sweep_main

        return prefix_sweep_main(
            exclude_task=args.exclude_task,
            exclude_scheduler=args.exclude_scheduler,
            quiet=args.quiet,
            output=_sweep_csv(args, SWEEP_DEFAULT),
        )
    if cmd == "prefix-sweep-pcfg":
        from schedulerlens.prefix_sweep import main_pcfg as prefix_sweep_pcfg_main

        return prefix_sweep_pcfg_main(
            exclude_task=args.exclude_task,
            exclude_scheduler=args.exclude_scheduler,
            quiet=args.quiet,
            output=_sweep_csv(args, SWEEP_PCFG_DEFAULT),
        )
    if cmd == "prefix-selection":
        from schedulerlens.prefix_selection import main as prefix_selection_main

        return prefix_selection_main(
            args.runs,
            args.tau,
            exclude_task=args.exclude_task,
            exclude_scheduler=args.exclude_scheduler,
            quiet=args.quiet,
            output=_selection_csv(args),
        )
    logger.error("Unknown command: %s", cmd)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
