#!/usr/bin/env python3
"""CLI: run SchedulerLens on a trace file."""

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

from schedulerlens.metrics import TaskMetrics, compute_task_metrics
from schedulerlens.plots import save_bucket_plots, save_trend_plots

logger = logging.getLogger("schedulerlens.run")


def run_schedulerlens(
    trace_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    max_samples: Optional[int] = None,
    bucket_size: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Compute metrics, write metrics.json and trend plots to output_dir."""
    trace_path = Path(trace_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    }

    if metrics.n_steps > 0:
        prefix = trace_path.stem
        m_plot, j_plot, unmasked_plot = save_trend_plots(
            metrics,
            output_dir,
            prefix=prefix,
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
    else:
        result["M_t_plot"] = None
        result["J_t_plot"] = None
        result["unmasked_count_plot"] = None
        result["bucket_all_trends_plot"] = None
        result["bucket_unmasked_plot"] = None
        logger.warning("Skipped trend plots: no valid steps in trace.")

    return result


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SchedulerLens: M̂/Ĵ caps and per-step trend plots from a trace file.",
        epilog="Example: python -m schedulerlens.run trace_outputs/run.jsonl -o outputs/schedulerlens/run1",
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
        help="Directory for metrics.json and trend plots",
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
        help="Aggregate trends into buckets of N steps; writes bucket plots",
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
        result = run_schedulerlens(
            args.trace_path,
            args.output_dir,
            max_samples=args.max_samples,
            bucket_size=args.bucket_size,
            verbose=not args.quiet,
        )
    except Exception as e:
        logger.error("SchedulerLens failed: %s", e)
        return 1

    metrics: TaskMetrics = result["metrics"]
    print(metrics.summary())
    print(f"\nOutput dir: {args.output_dir.resolve()}")
    print(f"  metrics.json : {result['metrics_path']}")
    if result["M_t_plot"]:
        print(f"  M trends     : {result['M_t_plot']}")
        print(f"  J trends     : {result['J_t_plot']}")
        print(f"  Unmasked     : {result['unmasked_count_plot']}")
    if result.get("bucket_all_trends_plot"):
        print(f"  Bucket trends: {result['bucket_all_trends_plot']}")
        print(f"  Bucket unmask: {result['bucket_unmasked_plot']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
