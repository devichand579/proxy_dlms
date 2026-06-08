#!/usr/bin/env python3
"""CLI: POS-stratified unmasking analysis on a trace file."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from schedulerlens.pos_analysis import PosTaskMetrics, compute_pos_metrics
from schedulerlens.pos_plots import save_pos_plots

logger = logging.getLogger("schedulerlens.pos_run")


def run_pos_analysis(
    trace_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    max_samples: Optional[int] = None,
    bucket_size: Optional[int] = None,
    top_k: int = 10,
    model_path: Optional[Union[str, Path]] = None,
    models_root: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Compute POS metrics, write pos_metrics.json and plots."""
    trace_path = Path(trace_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_pos_metrics(
        trace_path,
        max_samples=max_samples,
        bucket_size=bucket_size,
        top_k=top_k,
        model_path=model_path,
        models_root=models_root,
        verbose=verbose,
    )

    metrics_path = output_dir / "pos_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "trace_path": str(trace_path.resolve()),
                **metrics.to_dict(),
            },
            f,
            indent=2,
        )

    result: Dict[str, Any] = {
        "n_samples": metrics.n_samples,
        "n_steps": metrics.n_steps,
        "model_name": metrics.model_name,
        "model_dir": metrics.model_dir,
        "metrics_path": str(metrics_path.resolve()),
        "metrics": metrics,
        "pos_mean_unmask_plot": None,
        "pos_unmask_trend_plot": None,
        "pos_bucket_trend_plot": None,
        "pos_bucket_mean_unmask_plot": None,
    }

    if metrics.n_steps > 0 and metrics.mean_unmask_step_by_pos:
        prefix = trace_path.stem
        mean_plot, trend_plot, bucket_plot, bucket_mean_plot = save_pos_plots(
            metrics, output_dir, prefix=prefix
        )
        result["pos_mean_unmask_plot"] = str(mean_plot.resolve())
        result["pos_unmask_trend_plot"] = str(trend_plot.resolve())
        result["pos_bucket_trend_plot"] = (
            str(bucket_plot.resolve()) if bucket_plot else None
        )
        result["pos_bucket_mean_unmask_plot"] = (
            str(bucket_mean_plot.resolve()) if bucket_mean_plot else None
        )
    else:
        logger.warning("Skipped POS plots: no unmask events found.")

    return result


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "POS-stratified unmasking analysis: mean unmask step by POS tag, "
            "top-k tokens per step, optional bucket trends."
        ),
        epilog=(
            "Example: python -m schedulerlens.pos_run "
            "trace_outputs/llada_humaneval_pure_diffusion_s256.jsonl "
            "-o outputs/schedulerlens/pos1 --bucket-size 16"
        ),
    )
    parser.add_argument(
        "trace_path",
        type=Path,
        help="Path to trace .jsonl or .json from the eval harness",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for pos_metrics.json and plots",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Process only the first N samples (debug)",
    )
    parser.add_argument(
        "--bucket-size",
        type=int,
        default=None,
        metavar="N",
        help="Aggregate POS unmask counts into buckets of N steps",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-k (token, POS) pairs to store per step (default: 10)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Override model directory for tokenizer (default: infer from trace filename)",
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=None,
        help="Root directory containing llada/ llada1_5/ dream/ (default: ./models)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Less logging",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.trace_path.exists():
        logger.error("Trace file not found: %s", args.trace_path)
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

    metrics: PosTaskMetrics = result["metrics"]
    print(metrics.summary())
    print(f"\nOutput dir: {args.output_dir.resolve()}")
    print(f"  pos_metrics.json : {result['metrics_path']}")
    if result["pos_mean_unmask_plot"]:
        print(f"  Mean unmask step : {result['pos_mean_unmask_plot']}")
        print(f"  POS step trend   : {result['pos_unmask_trend_plot']}")
    if result.get("pos_bucket_trend_plot"):
        print(f"  POS bucket trend : {result['pos_bucket_trend_plot']}")
    if result.get("pos_bucket_mean_unmask_plot"):
        print(f"  POS bucket mean  : {result['pos_bucket_mean_unmask_plot']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
