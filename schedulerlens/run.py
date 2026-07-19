#!/usr/bin/env python3
"""Single SchedulerLens CLI.

Subcommands:
  file         Analyze one local NPZ (metrics + plots)
  inventory    List Hub NPZs (no download)
  analyze      Download/analyze one Hub NPZ (or --local path)
  compare      Batch Hub/local compare → CSV + plots
  pos          POS-stratified unmask analysis
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from repo_env_defaults import apply_repo_env_defaults  # noqa: E402
from schedulerlens.metrics import (  # noqa: E402
    TaskMetrics,
    build_metrics_row,
    compute_task_metrics,
    correlations_from_trends,
    parse_run_stem,
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
) -> Dict[str, Any]:
    """Compute metrics, write metrics.json / metrics.csv and plots to output_dir."""
    _ensure_pcfg_path()
    trace_path = Path(trace_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = parse_run_stem(trace_path)

    metrics = compute_task_metrics(
        trace_path,
        max_samples=max_samples,
        bucket_size=bucket_size,
        verbose=verbose,
    )

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "trace_path": str(trace_path.resolve()),
                "meta": meta.to_dict(),
                **metrics.to_dict(),
            },
            f,
            indent=2,
        )

    result: Dict[str, Any] = {
        "M_hat": metrics.M_hat,
        "J_hat": metrics.J_hat,
        "M_hat_uniform": metrics.M_hat_uniform,
        "J_hat_uniform": metrics.J_hat_uniform,
        "n_samples": metrics.n_samples,
        "n_steps": metrics.n_steps,
        "metrics_path": str(metrics_path.resolve()),
        "metrics": metrics,
        "meta": meta,
    }

    ppl = None
    correlations = None
    fidelity = None
    do_replay = replay_oracle and (meta.is_pcfg or force_pcfg)

    if compute_perplexity:
        engine = vocab = model = device = None
        if meta.is_pcfg or force_pcfg:
            try:
                from pcfg_data import build_stack

                _g, _cnf, engine, vocab = build_stack()
            except Exception as e:
                logger.warning("PCFG stack unavailable: %s", e)
            if do_replay:
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
        )
        ppl_path = output_dir / "perplexity.json"
        with ppl_path.open("w", encoding="utf-8") as f:
            json.dump(ppl.to_dict(), f, indent=2)
        result["perplexity"] = ppl
        result["perplexity_path"] = str(ppl_path.resolve())
        tables = getattr(ppl, "oracle_tables", None) or {}
        if do_replay and (
            tables.get("step_sample_idx")
            or any(s is not None for s in ppl.sample_oracle)
        ):
            oracle_path = save_oracle_artifacts(
                output_dir / "plot_artifacts" / meta.stem / "oracle.npz",
                ppl,
                stem=meta.stem,
            )
            result["oracle_artifacts_path"] = str(oracle_path.resolve())
            ppl.oracle_tables = {}
            ppl.sample_oracle = []

        if do_replay and ppl.mt_trend and ppl.jt_trend:
            m_hat = metrics.proxies["M_hat_t"].trend
            j_hat = metrics.proxies["J_hat_t"].trend
            n = min(len(ppl.mt_trend), len(m_hat), len(ppl.jt_trend), len(j_hat))
            correlations = correlations_from_trends(
                mt_trend=ppl.mt_trend[:n],
                m_hat_trend=m_hat[:n],
                jt_trend=ppl.jt_trend[:n],
                j_hat_trend=j_hat[:n],
                sample_oracle_nll=ppl.sample_oracle_nll,
                sample_model_nll=ppl.sample_model_nll,
            )
            fidelity = {
                "mt": np.asarray(ppl.mt_trend[:n], dtype=float),
                "m_hat": np.asarray(m_hat[:n], dtype=float),
                "jt": np.asarray(ppl.jt_trend[:n], dtype=float),
                "j_hat": np.asarray(j_hat[:n], dtype=float),
            }
            result["correlations"] = correlations

        with metrics_path.open("r", encoding="utf-8") as f:
            merged = json.load(f)
        merged["perplexity"] = ppl.to_dict()
        if correlations is not None:
            merged["correlations"] = correlations.to_dict()
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)

    row = build_metrics_row(
        trace_path,
        metrics=metrics,
        perplexity=ppl,
        correlations=correlations,
        meta=meta,
    )
    csv_path = write_metrics_csv(output_dir / "metrics.csv", [row])
    result["csv_path"] = str(csv_path.resolve())

    try:
        from schedulerlens.plots import compute_trace_surfaces, save_compare_plots

        surfaces = compute_trace_surfaces(trace_path, max_samples=max_samples)
    except ImportError:
        surfaces = None
    if surfaces is not None and surfaces.n_samples > 0:
        surface_paths = save_compare_plots(
            [row],
            output_dir,
            surfaces_by_stem={meta.stem: surfaces},
            fidelity_points={meta.stem: fidelity} if fidelity else None,
        )
        result["surface_plots"] = {k: str(v) for k, v in surface_paths.items()}

    if metrics.n_steps > 0:
        try:
            from schedulerlens.plots import save_bucket_plots, save_trend_plots

            prefix = trace_path.stem
            m_plot, j_plot, unmasked_plot = save_trend_plots(
                metrics, output_dir, prefix=prefix
            )
            result["M_t_plot"] = str(m_plot.resolve())
            result["J_t_plot"] = str(j_plot.resolve())
            result["unmasked_count_plot"] = str(unmasked_plot.resolve())
            if bucket_size is not None:
                bucket_all, bucket_unmasked = save_bucket_plots(
                    metrics, output_dir, prefix=prefix
                )
                result["bucket_all_trends_plot"] = str(bucket_all.resolve())
                result["bucket_unmasked_plot"] = str(bucket_unmasked.resolve())
            else:
                result["bucket_all_trends_plot"] = None
                result["bucket_unmasked_plot"] = None
        except ImportError as e:
            logger.warning("Skipping trend plots (missing dependency): %s", e)
            result["M_t_plot"] = None
            result["J_t_plot"] = None
            result["unmasked_count_plot"] = None
            result["bucket_all_trends_plot"] = None
            result["bucket_unmasked_plot"] = None
    else:
        result["M_t_plot"] = None
        result["J_t_plot"] = None
        result["unmasked_count_plot"] = None
        result["bucket_all_trends_plot"] = None
        result["bucket_unmasked_plot"] = None
        logger.warning("Skipped trend plots: no valid steps in trace.")

    return result


def _add_repo_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--repo", "--repo-id", dest="repo_id", default=None)
    p.add_argument(
        "--namespace",
        default=None,
        help="HF user/org for short --repo names and oracle model ids "
        "({ns}/{stem_model}). Default: TRACE_HF_NAMESPACE / whoami",
    )
    p.add_argument("--revision", default=None)


def _add_filter_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--pattern", default="*.npz")
    p.add_argument("--model", dest="model_filter", default=None)
    p.add_argument("--scheduler", dest="scheduler_filter", default=None)
    p.add_argument("--steps", dest="steps_filter", type=int, default=None)
    p.add_argument("--pcfg-only", action="store_true")


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


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SchedulerLens — single entrypoint for metrics, Hub, and POS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- file ---
    p_file = sub.add_parser("file", help="Analyze one local NPZ")
    p_file.add_argument("trace_path", type=Path)
    p_file.add_argument("-o", "--output-dir", type=Path, required=True)
    p_file.add_argument("--bucket-size", type=int, default=None, metavar="N")
    p_file.add_argument("--no-perplexity", action="store_true")
    _add_oracle_args(p_file)
    p_file.add_argument("-q", "--quiet", action="store_true")

    # --- inventory ---
    p_inv = sub.add_parser("inventory", help="List Hub NPZs (no downloads)")
    _add_repo_args(p_inv)
    _add_filter_args(p_inv)

    # --- analyze (one Hub/local) ---
    p_one = sub.add_parser("analyze", help="Analyze one Hub NPZ (or --local path)")
    p_one.add_argument("filename")
    p_one.add_argument("-o", "--output-dir", type=Path, required=True)
    p_one.add_argument("--local", action="store_true")
    _add_repo_args(p_one)
    _add_oracle_args(p_one)
    p_one.add_argument("-q", "--quiet", action="store_true")

    # --- compare (batch) ---
    p_cmp = sub.add_parser(
        "compare",
        aliases=["analyze-all"],
        help="Batch Hub/local compare → CSV + plots",
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
    p_cmp.add_argument("-o", "--output-dir", type=Path, required=True)
    _add_filter_args(p_cmp)
    _add_oracle_args(p_cmp)
    p_cmp.add_argument("--strace-metrics", type=Path, default=None)
    p_cmp.add_argument("--limit", type=int, default=None)
    p_cmp.add_argument("--resume", action="store_true")
    p_cmp.add_argument("--no-purge-cache", action="store_true")
    p_cmp.add_argument("-q", "--quiet", action="store_true")

    # --- pos ---
    p_pos = sub.add_parser("pos", help="POS-stratified unmask analysis")
    p_pos.add_argument("trace_path", type=Path)
    p_pos.add_argument("-o", "--output-dir", type=Path, required=True)
    p_pos.add_argument("--max-samples", type=int, default=None)
    p_pos.add_argument("--bucket-size", type=int, default=None, metavar="N")
    p_pos.add_argument("--top-k", type=int, default=10)
    p_pos.add_argument("--model-path", type=Path, default=None)
    p_pos.add_argument("--models-root", type=Path, default=None)
    p_pos.add_argument("-q", "--quiet", action="store_true")

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
    if result.get("correlations") is not None:
        c = result["correlations"]
        print(
            f"\nProxy fidelity: "
            f"corr_M(full/early/mid/late)="
            f"{c.corr_M_full:.3f}/{c.corr_M_early:.3f}/{c.corr_M_mid:.3f}/{c.corr_M_late:.3f}  "
            f"corr_J="
            f"{c.corr_J_full:.3f}/{c.corr_J_early:.3f}/{c.corr_J_mid:.3f}/{c.corr_J_late:.3f}"
        )
    print(f"\nOutput dir: {args.output_dir.resolve()}")
    print(f"  metrics.json : {result['metrics_path']}")
    print(f"  metrics.csv  : {result['csv_path']}")
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
    )
    print_inventory(inv)
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    from schedulerlens.hub import (
        _load_oracle_stack,
        analyze_hub_file,
        analyze_local_file,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    engine = vocab = model = device = None
    if args.replay_oracle:
        from schedulerlens.hub import hf_model_id_for_meta
        from schedulerlens.metrics import parse_run_stem

        meta = parse_run_stem(args.filename)
        resolved = args.hf_model or (
            None
            if args.model_path is not None
            else hf_model_id_for_meta(meta, namespace=args.namespace)
        )
        engine, vocab, model, device = _load_oracle_stack(
            replay_oracle=True,
            model_path=args.model_path,
            hf_model=resolved,
        )
    try:
        oracle_kw = dict(
            oracle_workers=args.oracle_workers,
            step_workers=args.oracle_step_workers,
            forward_batch_size=args.oracle_forward_batch,
            jt_exact_max=args.oracle_jt_exact_max,
            jt_markov_order=args.oracle_jt_markov_order,
        )
        if args.local:
            result = analyze_local_file(
                args.filename,
                repo=str(Path(args.filename).resolve().parent),
                max_samples=args.max_samples,
                replay_oracle=args.replay_oracle,
                force_pcfg=args.pcfg,
                model=model,
                device=device,
                engine=engine,
                vocab=vocab,
                verbose=not args.quiet,
                **oracle_kw,
            )
        else:
            result = analyze_hub_file(
                args.filename,
                repo_id=args.repo_id,
                namespace=args.namespace,
                revision=args.revision,
                max_samples=args.max_samples,
                replay_oracle=args.replay_oracle,
                force_pcfg=args.pcfg,
                model=model,
                device=device,
                engine=engine,
                vocab=vocab,
                verbose=not args.quiet,
                **oracle_kw,
            )
    except TraceProxyError as e:
        logger.error("%s", e)
        return 1

    csv_path = write_metrics_csv(out / "metrics.csv", [result["row"]])
    with (out / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result["row"], f, indent=2)
    print(f"CSV: {csv_path}")
    print(
        f"M_hat={result['row'].get('M_hat')}  J_hat={result['row'].get('J_hat')}  "
        f"Mt={result['row'].get('Mt_total')}  Jt={result['row'].get('Jt_total')}"
    )
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
            force_pcfg=args.pcfg,
            replay_oracle=args.replay_oracle,
            model_path=args.model_path,
            hf_model=args.hf_model,
            strace_metrics_path=args.strace_metrics,
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
        )
    except TraceProxyError as e:
        logger.error("%s", e)
        return 1
    except Exception as e:
        logger.error("Compare failed: %s", e)
        return 1

    print(f"Runs: {len(result['rows'])}")
    print(f"CSV : {result['csv_path']}")
    for name, path in result.get("plot_paths", {}).items():
        print(f"  {name}: {path}")
    return 0


def _cmd_pos(args: argparse.Namespace) -> int:
    from schedulerlens.pos import run_pos_analysis

    if not args.trace_path.exists():
        logger.error("Trace file not found: %s", args.trace_path)
        return 1
    if args.trace_path.suffix.lower() != ".npz":
        logger.error("Expected NPZ, got %s", args.trace_path.name)
        return 1
    try:
        result = run_pos_analysis(
            args.trace_path,
            args.output_dir,
            max_samples=args.max_samples,
            bucket_size=args.bucket_size,
            top_k=args.top_k,
            model_path=args.model_path,
            models_root=args.models_root,
            verbose=not args.quiet,
        )
    except Exception as e:
        logger.error("POS analysis failed: %s", e)
        return 1

    print(result["metrics"].summary())
    print(f"\nOutput dir: {args.output_dir.resolve()}")
    print(f"  pos_metrics.json : {result['metrics_path']}")
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
    if cmd == "analyze":
        return _cmd_analyze(args)
    if cmd in ("compare", "analyze-all"):
        return _cmd_compare(args)
    if cmd == "pos":
        return _cmd_pos(args)
    logger.error("Unknown command: %s", cmd)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
