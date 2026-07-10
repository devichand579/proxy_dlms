"""Export Strace eval metrics from Weights & Biases to CSV. See docs/evaluation.md."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
import wandb

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "strace_metrics.csv"

DEFAULT_PROJECT = os.environ.get("STRACE_WANDB_PROJECT", "strace")
NFE_KEYS = (
    "global/avg_nfe_per_sample",
    "avg_nfe_per_sample",
)

KNOWN_TASKS = (
    "arc_challenge_chat",
    "minerva_math500",
    "openai_humaneval",
    "humaneval",
    "gsm8k",
    "mbpp",
)

TASK_SCORE_KEYS: dict[str, tuple[str, ...]] = {
    "gsm8k": ("strict_match", "gsm8k/strict_match"),
    "minerva_math500": (
        "exact_match",
        "minerva_math500/exact_match",
        "math_verify",
        "minerva_math500/math_verify",
    ),
    "arc_challenge_chat": ("exact_match", "arc_challenge_chat/exact_match"),
    "mbpp": ("pass_at_1", "pass@1", "mbpp/pass_at_1", "mbpp/pass@1"),
    "humaneval": ("pass_at_1", "pass@1", "humaneval/pass_at_1", "openai_humaneval/pass_at_1"),
    "openai_humaneval": ("pass_at_1", "pass@1", "openai_humaneval/pass_at_1", "humaneval/pass_at_1"),
}

MetricMatcher = Callable[[str], bool]


def _strict_match_key(key: str) -> bool:
    lower = key.lower()
    return "strict_match" in lower or "strict-match" in lower


def _exact_match_key(key: str) -> bool:
    lower = key.lower()
    return ("exact_match" in lower or "exact-match" in lower) and "strict" not in lower


def _math_verify_key(key: str) -> bool:
    return "math_verify" in key.lower()


def _minerva_score_key(key: str) -> bool:
    return _exact_match_key(key) or _math_verify_key(key)


def _pass_at_1_key(key: str) -> bool:
    lower = key.lower()
    return "pass_at_1" in lower or "pass@1" in lower


def _minerva_score_rank(key: str) -> tuple[int, int]:
    lower = key.lower()
    if "exact_match" in lower or "exact-match" in lower:
        return (0, len(key))
    if "math_verify" in lower:
        return (1, len(key))
    return (99, len(key))


TASK_FUZZY_MATCHERS: dict[str, MetricMatcher] = {
    "gsm8k": _strict_match_key,
    "minerva_math500": _minerva_score_key,
    "arc_challenge_chat": _exact_match_key,
    "mbpp": _pass_at_1_key,
    "humaneval": _pass_at_1_key,
    "openai_humaneval": _pass_at_1_key,
}


def _first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> tuple[Any, Optional[str]]:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key], key
    return None, None


def _fuzzy_metric(mapping: dict[str, Any], task: str) -> tuple[Any, Optional[str]]:
    matcher = TASK_FUZZY_MATCHERS.get(task)
    if matcher is None:
        return None, None

    task_prefix = f"{task}/".lower()
    candidates: list[tuple[tuple[int, int, int], str, Any]] = []
    for key, value in mapping.items():
        if value is None or not matcher(key):
            continue
        lower = key.lower()
        prefix_rank = 0 if lower.startswith(task_prefix) or lower.startswith(task) else 1
        if task == "minerva_math500":
            metric_rank = _minerva_score_rank(key)
            sort_key = (prefix_rank, metric_rank[0], metric_rank[1])
        else:
            sort_key = (prefix_rank, 0, len(key))
        candidates.append((sort_key, key, value))

    if not candidates:
        return None, None

    candidates.sort(key=lambda item: item[0])
    _, best_key, best_val = candidates[0]
    return best_val, best_key


def _flatten_summary(summary: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in summary.items():
        full_key = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_summary(value, full_key))
        else:
            flat[full_key] = value
    return flat


def _parse_run_name(name: str) -> dict[str, Optional[str]]:
    out: dict[str, Optional[str]] = {
        "model": None,
        "task": None,
        "mode": None,
        "steps": None,
    }
    if not name:
        return out

    steps_match = re.search(r"_s(\d+)(?:_|$)", name)
    if steps_match:
        out["steps"] = steps_match.group(1)

    task = None
    task_start = -1
    for candidate in KNOWN_TASKS:
        marker = f"_{candidate}_"
        idx = name.find(marker)
        if idx >= 0:
            task = candidate
            task_start = idx + 1
            break

    if task is None:
        return out

    out["task"] = task
    out["model"] = name[: task_start - 1] if task_start > 0 else None

    after_task = name[task_start + len(task) + 1 :]
    mode_part = after_task
    if steps_match:
        mode_part = after_task[: steps_match.start() - len(after_task) + len(after_task)]
        s_idx = after_task.rfind(f"_s{steps_match.group(1)}")
        if s_idx >= 0:
            mode_part = after_task[:s_idx]
    mode_part = mode_part.rstrip("_")
    if mode_part.endswith("_intv"):
        mode_part = mode_part[: -len("_intv")]
    out["mode"] = mode_part or None
    return out


def _task_from_run(run: Any, parsed: dict[str, Optional[str]]) -> str:
    if parsed.get("task"):
        return str(parsed["task"])
    cfg = getattr(run, "config", None) or {}
    for key in ("task_name", "task", "tasks"):
        if key in cfg and cfg[key]:
            val = cfg[key]
            if isinstance(val, (list, tuple)) and val:
                return str(val[0])
            return str(val)
    return ""


def _metric_for_task(
    flat_summary: dict[str, Any], task: str
) -> tuple[Any, Optional[str], str]:
    keys = TASK_SCORE_KEYS.get(task, ())
    value, key = _first_present(flat_summary, keys)
    if value is not None:
        return value, key, "wandb_summary"

    value, key = _fuzzy_metric(flat_summary, task)
    if value is not None:
        return value, key, "wandb_fuzzy"

    return None, None, "missing"


def _nfe_from_summary(flat_summary: dict[str, Any]) -> tuple[Any, str]:
    value, _ = _first_present(flat_summary, NFE_KEYS)
    if value is not None:
        return value, "wandb_summary"
    return None, "missing"


def _nfe_from_history(run: Any) -> tuple[Any, str]:
    try:
        history = run.history(keys=list(NFE_KEYS), pandas=True)
    except Exception:
        return None, "missing"
    if history is None or history.empty:
        return None, "missing"
    for col in NFE_KEYS:
        if col in history.columns:
            series = history[col].dropna()
            if not series.empty:
                return float(series.iloc[-1]), "wandb_history"
    return None, "missing"


def _sanitize_wandb_url(url: Optional[str]) -> Optional[str]:
    """Drop the W&B entity from run URLs (``wandb.ai/{entity}/{project}/...`` → ``wandb.ai/{project}/...``)."""
    if not url:
        return url
    return re.sub(r"https://wandb\.ai/[^/]+/", "https://wandb.ai/", url)


    project: str,
    include_unfinished: bool = False,
) -> pd.DataFrame:
    api = wandb.Api()
    runs = api.runs(project)

    rows: list[dict[str, Any]] = []
    for run in runs:
        if not include_unfinished and run.state != "finished":
            continue

        name = run.name or ""
        parsed = _parse_run_name(name)
        task = _task_from_run(run, parsed)
        flat_summary = _flatten_summary(dict(run.summary or {}))

        metric, metric_key, metric_source = _metric_for_task(flat_summary, task)

        nfe, nfe_source = _nfe_from_summary(flat_summary)
        if nfe is None:
            nfe, nfe_source = _nfe_from_history(run)

        rows.append(
            {
                "name": name,
                "run_id": run.id,
                "state": run.state,
                "model": parsed.get("model"),
                "task": task or parsed.get("task"),
                "mode": parsed.get("mode"),
                "steps": parsed.get("steps"),
                "avg_nfe_per_sample": nfe,
                "avg_nfe_source": nfe_source,
                "metric": metric,
                "metric_key": metric_key,
                "metric_source": metric_source,
                "url": _sanitize_wandb_url(run.url),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Strace eval metrics from W&B to CSV."
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT),
        help=f"W&B project (default: {DEFAULT_PROJECT})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path (default: fetch_evals/strace_metrics.csv)",
    )
    parser.add_argument(
        "--include-unfinished",
        action="store_true",
        help="Include crashed/running runs, not just finished",
    )
    args = parser.parse_args()

    if not os.environ.get("WANDB_API_KEY"):
        raise SystemExit(
            "WANDB_API_KEY is not set. Get a key at https://wandb.ai/authorize and export it."
        )

    df = fetch_metrics(
        project=args.project,
        include_unfinished=args.include_unfinished,
    )
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
