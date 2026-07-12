"""Trace accuracy metrics vs fetch_evals/strace_metrics.csv (W&B export)."""

from __future__ import annotations

import csv
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from load_model import infer_diffusion_token_ids, is_dream_model_dir, load_tokenizer, prepend_bos_for_dream
from traces.trace_doc_match import build_doc_index, match_trace_to_instance

logger = logging.getLogger(__name__)

DEFAULT_METRICS_CSV = Path(__file__).resolve().parent.parent / "fetch_evals/strace_metrics.csv"

TASK_SAMPLE_COUNTS: dict[str, int] = {
    "humaneval": 164,
    "openai_humaneval": 164,
    "mbpp": 500,
    "gsm8k": 1319,
    "minerva_math500": 500,
    "arc_challenge_chat": 1172,
}

LM_EVAL_TASK_ALIASES: dict[str, str] = {
    "openai_humaneval": "humaneval",
}

# Primary metric substrings per task (matches fetch_wandb_metrics.py)
TASK_PRIMARY_PATTERNS: dict[str, tuple[str, ...]] = {
    "gsm8k": ("strict-match", "strict_match"),
    "minerva_math500": ("exact_match", "math_verify"),
    "arc_challenge_chat": ("exact_match",),
    "mbpp": ("pass@1", "pass_at_1"),
    "humaneval": ("pass@1", "pass_at_1"),
    "openai_humaneval": ("pass@1", "pass_at_1"),
}

MODEL_DIR_NAMES: dict[str, str] = {
    "llada": "llada",
    "llada1_5": "llada1_5",
    "dream": "dream",
}


def load_metrics_lookup(metrics_csv: str | Path | None = None) -> dict[str, dict[str, str]]:
    path = Path(metrics_csv or DEFAULT_METRICS_CSV)
    if not path.is_file():
        raise FileNotFoundError(path)
    with open(path, newline="", encoding="utf-8") as f:
        return {row["name"].strip(): dict(row) for row in csv.DictReader(f) if row.get("name")}


def expected_trace_names(metrics_csv: str | Path | None = None) -> set[str]:
    return set(load_metrics_lookup(metrics_csv).keys())


def is_pcfg_trace(run_name: str, samples: list[dict]) -> bool:
    """PCFG lladalens traces are not in strace_metrics.csv / lm-eval tasks."""
    if str(run_name).startswith("pcfg_"):
        return True
    if samples and (samples[0].get("metadata") or {}).get("task") == "pcfg_full":
        return True
    return False


def pcfg_metrics_skip(*, n_samples: int) -> dict[str, Any]:
    """Audit row fragment: hook checks only, no W&B / lm-eval scoring."""
    return {
        "metric_wandb": None,
        "metric_key_wandb": "",
        "metric_trace": None,
        "metric_key_trace": "",
        "metric_delta": None,
        "n_scored": 0,
        "n_unmatched": 0,
        "n_samples_expected": None,
        "sample_count_delta": None,
        "metrics_flags": f"pcfg trace ({n_samples} samples): skipped lm-eval / wandb metrics",
    }


def _safe_float(val: Any) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def model_dir_for_name(model: str, repo_root: Optional[Path] = None) -> Path:
    root = repo_root or Path(__file__).resolve().parent.parent
    sub = MODEL_DIR_NAMES.get(model.strip(), model.strip())
    return root / "models" / sub


def lm_eval_task_name(task: str) -> str:
    return LM_EVAL_TASK_ALIASES.get(task.strip(), task.strip())


def eval_num_fewshot(task_name: str) -> int:
    
    t = str(task_name).strip().lower()
    if "humaneval" in t:
        return 0
    if "gsm8k" in t:
        return 4
    if "math500" in t or "minerva_math500" in t:
        return 4
    if "mbpp3" in t:
        return 3
    if "mbpp" in t:
        return 3
    if "arc_challenge_chat" in t:
        return 0
    return 0


def decode_generation_from_trace(
    sample: dict,
    tokenizer,
    *,
    eos_id: int,
    stop_tokens: Optional[list[str]] = None,
    use_eos_harness: bool = True,
) -> Optional[str]:
    final = sample.get("final_sequence")
    if final is None:
        return None
    prompt_len = int(sample.get("prompt_len") or 0)
    seq = final[0] if final and isinstance(final[0], list) else final
    if not seq:
        return None
    tail = list(seq[prompt_len:])
    if use_eos_harness:
        for i, tok in enumerate(tail):
            if int(tok) == int(eos_id):
                tail = tail[:i]
                break
    text = tokenizer.decode(tail, skip_special_tokens=True)
    if stop_tokens:
        for stop in stop_tokens:
            if stop and stop in text:
                text = text.split(stop)[0]
    return text


def _prompt_token_key(token_ids: list[int], pad_id: Optional[int]) -> tuple[int, ...]:
    if pad_id is None:
        return tuple(int(t) for t in token_ids)
    return tuple(int(t) for t in token_ids if int(t) != int(pad_id))


def _instance_prompt_key(inst, tokenizer, model_dir: Path) -> tuple[int, ...]:
    text = str(inst.args[0])
    if is_dream_model_dir(model_dir):
        text = prepend_bos_for_dream(tokenizer, text)
    return tuple(tokenizer.encode(text, add_special_tokens=False))


def _trace_prompt_key(sample: dict, tokenizer) -> Optional[tuple[int, ...]]:
    prompt = sample.get("prompt")
    if prompt is None:
        return None
    ids = prompt[0] if isinstance(prompt[0], list) else prompt
    return _prompt_token_key(list(ids), tokenizer.pad_token_id)


def _load_scoring_task(task_name: str, num_fewshot: int):
    from lm_eval.tasks import TaskManager

    lm_task = lm_eval_task_name(task_name)
    tm = TaskManager()
    task_dict = tm.load_task_or_group(lm_task)
    if lm_task in task_dict:
        task = task_dict[lm_task]
    elif len(task_dict) == 1:
        task = next(iter(task_dict.values()))
    else:
        raise ValueError(f"ambiguous lm-eval task {lm_task!r}: {list(task_dict.keys())}")

    if hasattr(task, "config") and hasattr(task.config, "num_fewshot"):
        task.config.num_fewshot = int(num_fewshot)
    if hasattr(task, "set_fewshot_seed"):
        task.set_fewshot_seed(0)
    task.build_all_requests(limit=None, rank=0, world_size=1)
    return task


def _task_stop_tokens(task) -> list[str]:
    try:
        gen_kw = task.config.generation_kwargs or {}
        until = gen_kw.get("until") or []
        return [str(u) for u in until]
    except Exception:
        return []


def _apply_filters(inst, task) -> None:
    inst.resps = list(inst.resps)
    if hasattr(task, "_filters"):
        for filt in task._filters:
            filt.apply([inst])


def _score_instance(task, inst, gen_text: str) -> dict[str, float]:
    inst.resps = [gen_text]
    _apply_filters(inst, task)
    per_filter: dict[str, float] = {}
    for filter_key, filtered in inst.filtered_resps.items():
        metrics = task.process_results(inst.doc, [filtered])
        for metric_name, value in metrics.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    composite = f"{k},{filter_key}"
                    per_filter[composite] = float(v)
            else:
                composite = f"{metric_name},{filter_key}"
                per_filter[composite] = float(value)
    return per_filter


def _aggregate_metrics(per_doc: list[dict[str, float]], task) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for doc_metrics in per_doc:
        for key, val in doc_metrics.items():
            buckets[key].append(val)

    agg_fns = task.aggregation() if hasattr(task, "aggregation") else {}
    out: dict[str, float] = {}
    for key, values in buckets.items():
        metric_name = key.split(",")[0]
        fn = agg_fns.get(metric_name) or agg_fns.get(key)
        if fn is None:
            fn = sum(values) / len(values) if values else (lambda x: 0.0)
        try:
            out[key] = float(fn(values))
        except TypeError:
            out[key] = float(sum(values) / len(values)) if values else 0.0
    return out


def _pick_primary_metric(
    aggregated: dict[str, float],
    task: str,
    wandb_metric_key: str = "",
) -> tuple[Optional[float], str]:
    if wandb_metric_key:
        for key, val in aggregated.items():
            if key == wandb_metric_key or key.endswith(wandb_metric_key.split("/")[-1]):
                return val, key
        if wandb_metric_key in aggregated:
            return aggregated[wandb_metric_key], wandb_metric_key

    patterns = TASK_PRIMARY_PATTERNS.get(task, ())
    candidates: list[tuple[int, str, float]] = []
    for key, val in aggregated.items():
        if "stderr" in key.lower():
            continue
        lower = key.lower()
        for i, pat in enumerate(patterns):
            if pat.lower() in lower:
                candidates.append((i, key, val))
                break
    if not candidates:
        return None, ""
    candidates.sort(key=lambda x: (x[0], len(x[1])))
    return candidates[0][2], candidates[0][1]


def compute_trace_accuracy(
    *,
    task_name: str,
    model_name: str,
    samples: list[dict],
    repo_root: Optional[Path] = None,
    num_fewshot: Optional[int] = None,
) -> dict[str, Any]:
    """
    Score trace generations with lm-eval task metrics (pass@1, exact_match, strict-match, …).
    Matches samples to eval instances by decoded prompt doc keys (few-shot safe),
    with token-id fallback.
    """
    root = repo_root or Path(__file__).resolve().parent.parent
    model_dir = model_dir_for_name(model_name, root)
    flags: list[str] = []

    t_lower = str(task_name).strip().lower()
    if "mbpp" in t_lower or "humaneval" in t_lower:
        os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")

    if not model_dir.is_dir():
        return {
            "metric_trace": None,
            "metric_key_trace": "",
            "n_scored": 0,
            "n_unmatched": len(samples),
            "accuracy_flags": f"model dir missing: {model_dir}",
        }

    try:
        tokenizer = load_tokenizer(str(model_dir))
    except Exception as e:
        return {
            "metric_trace": None,
            "metric_key_trace": "",
            "n_scored": 0,
            "n_unmatched": len(samples),
            "accuracy_flags": f"tokenizer load failed: {e}",
        }

    _, eos_id, _ = infer_diffusion_token_ids(model_dir)
    ns = int(num_fewshot) if num_fewshot is not None else eval_num_fewshot(task_name)

    try:
        task = _load_scoring_task(task_name, ns)
    except Exception as e:
        return {
            "metric_trace": None,
            "metric_key_trace": "",
            "n_scored": 0,
            "n_unmatched": len(samples),
            "accuracy_flags": f"lm-eval task load failed: {e}",
        }

    stop_tokens = _task_stop_tokens(task)
    doc_index = build_doc_index(task, task_name)
    token_index: dict[tuple[int, ...], Any] = {}
    for inst in task.instances:
        key = _instance_prompt_key(inst, tokenizer, model_dir)
        token_index.setdefault(key, inst)

    per_doc_metrics: list[dict[str, float]] = []
    n_unmatched = 0
    n_no_final = 0
    n_matched_doc = 0
    n_matched_token = 0

    for sample in samples:
        inst, method = match_trace_to_instance(
            sample,
            task_name=task_name,
            tokenizer=tokenizer,
            doc_index=doc_index,
            token_index=token_index,
        )
        if inst is None:
            n_unmatched += 1
            continue
        if method == "doc_key":
            n_matched_doc += 1
        elif method == "token_key":
            n_matched_token += 1
        gen_text = decode_generation_from_trace(
            sample,
            tokenizer,
            eos_id=eos_id,
            stop_tokens=stop_tokens,
            use_eos_harness=True,
        )
        if gen_text is None:
            n_no_final += 1
            continue
        try:
            per_doc_metrics.append(_score_instance(task, inst, gen_text))
        except Exception as e:
            flags.append(f"score failed doc {getattr(inst, 'doc_id', '?')}: {e}")

    if n_matched_doc:
        flags.append(f"matched by doc_key: {n_matched_doc}")
    if n_matched_token:
        flags.append(f"matched by token_key: {n_matched_token}")
    if n_unmatched:
        flags.append(f"unmatched samples: {n_unmatched}")
    if n_no_final:
        flags.append(f"missing final_sequence: {n_no_final}")

    if not per_doc_metrics:
        return {
            "metric_trace": None,
            "metric_key_trace": "",
            "n_scored": 0,
            "n_unmatched": n_unmatched,
            "accuracy_flags": "; ".join(flags) if flags else "no scorable samples",
        }

    aggregated = _aggregate_metrics(per_doc_metrics, task)
    metric_trace, metric_key_trace = _pick_primary_metric(aggregated, task_name)

    return {
        "metric_trace": metric_trace,
        "metric_key_trace": metric_key_trace,
        "n_scored": len(per_doc_metrics),
        "n_unmatched": n_unmatched,
        "accuracy_flags": "; ".join(flags) if flags else "",
        "all_metrics": aggregated,
    }


def compare_with_metrics_row(
    *,
    run_name: str,
    samples: list[dict],
    metrics_lookup: dict[str, dict[str, str]],
    task: str = "",
    model: str = "",
    compute_accuracy: bool = True,
    repo_root: Optional[Path] = None,
) -> dict[str, Any]:
    """Compare trace aggregates and recomputed accuracy vs the W&B CSV row."""
    row = metrics_lookup.get(run_name, {})
    task = task or row.get("task", "")
    model = model or row.get("model", "")
    n_samples = len(samples)
    expected_n = TASK_SAMPLE_COUNTS.get(task)
    metric_wandb = _safe_float(row.get("metric"))
    metric_key_wandb = row.get("metric_key", "")

    sample_delta: Optional[int] = None
    if expected_n is not None:
        sample_delta = n_samples - expected_n

    flags: list[str] = []
    if expected_n is not None and n_samples != expected_n:
        flags.append(f"sample_count {n_samples} != expected {expected_n}")
    if not row:
        flags.append("missing metrics.csv row")
    if metric_wandb is None and row:
        flags.append("missing wandb task metric")

    result: dict[str, Any] = {
        "metric_wandb": metric_wandb,
        "metric_key_wandb": metric_key_wandb,
        "n_samples_trace": n_samples,
        "n_samples_expected": expected_n,
        "sample_count_delta": sample_delta,
        "metric_trace": None,
        "metric_key_trace": "",
        "metric_delta": None,
        "n_scored": 0,
        "n_unmatched": 0,
        "metrics_flags": "",
    }

    if compute_accuracy and task and model and samples:
        acc = compute_trace_accuracy(
            task_name=task,
            model_name=model,
            samples=samples,
            repo_root=repo_root,
        )
        result["metric_trace"] = acc.get("metric_trace")
        result["metric_key_trace"] = acc.get("metric_key_trace", "")
        result["n_scored"] = acc.get("n_scored", 0)
        result["n_unmatched"] = acc.get("n_unmatched", 0)
        acc_flags = acc.get("accuracy_flags", "")
        if acc_flags:
            flags.append(acc_flags)

        if result["metric_trace"] is not None and metric_wandb is not None:
            result["metric_delta"] = abs(result["metric_trace"] - metric_wandb)
            if result["metric_delta"] > 0.02:
                flags.append(
                    f"metric_delta {result['metric_delta']:.4f} "
                    f"(trace={result['metric_trace']:.4f} wandb={metric_wandb:.4f})"
                )

    result["metrics_flags"] = "; ".join(flags) if flags else ""
    return result
