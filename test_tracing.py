"""Smoke-test tracing: 3 models × 8 schedulers × 1 sample, with audit."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

import torch

from load_model import (
    infer_b44_eos_unmask_safety,
    infer_diffusion_token_ids,
    is_dream_model_dir,
    load_model_and_tokenizer,
    prepend_bos_for_dream,
)
from schedulers.dus import generate_dus
from schedulers.eb_sampler import generate_EB
from schedulers.fdm import generate_fdm
from schedulers.fhs import generate_fhs
from schedulers.llada_schedulers import generate_llada
from schedulers.top_k import generate_topk
from traces.trace_checks import check_trace_file
from traces.trace_wrapper import _task_trace_list_ctx

logger = logging.getLogger(__name__)

DEFAULT_MODELS = ("./models/llada", "./models/llada1_5", "./models/dream")
SMOKE_PROMPT = (
    "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?"
)
GEN_LENGTH = 64
STEPS = 64


def _model_label(path: str) -> str:
    return Path(path).name


def _run_scheduler(
    name: str,
    fn: Callable,
    model,
    input_ids,
    attention_mask,
    *,
    mask_id: int,
    eos_id: int,
    eot_id: Optional[int],
    eos_unmask_b44: bool,
    batch_size: int,
) -> list[dict]:
    traces: list[dict] = []
    token = _task_trace_list_ctx.set(traces)
    trace_name = f"tracing_{name}_b{batch_size}"

    kwargs: dict = {
        "steps": STEPS,
        "gen_length": GEN_LENGTH,
        "trace_name": trace_name,
        "mask_id": mask_id,
        "eos_id": eos_id,
        "eot_id": eot_id,
        "use_eos_harness": True,
    }

    if name == "pure_diffusion":
        fn(
            model,
            input_ids,
            attention_mask,
            block_length=GEN_LENGTH,
            temperature=0.0,
            remasking="confidence",
            logits_eos_inf=eos_unmask_b44,
            confidence_eos_eot_inf=eos_unmask_b44,
            **kwargs,
        )
    elif name == "block_diffusion":
        fn(
            model,
            input_ids,
            attention_mask,
            block_length=32,
            temperature=0.0,
            remasking="confidence",
            logits_eos_inf=eos_unmask_b44,
            confidence_eos_eot_inf=eos_unmask_b44,
            **kwargs,
        )
    elif name in ("top_k_confidence", "top_k_margin"):
        strategy = "confidence" if name == "top_k_confidence" else "margin"
        fn(
            model,
            input_ids,
            attention_mask,
            block_length=GEN_LENGTH,
            top_k=1,
            strategy=strategy,
            logits_eos_inf=eos_unmask_b44,
            confidence_eos_eot_inf=eos_unmask_b44,
            **kwargs,
        )
    elif name == "fhs":
        fn(model, input_ids, attention_mask, block_length=32, temperature=1.0, **kwargs)
    elif name == "eb_sampler":
        fn(model, input_ids, attention_mask, block_length=GEN_LENGTH, **kwargs)
    elif name == "dus":
        fn(model, input_ids, attention_mask, block_length=16, **kwargs)
    elif name == "fdm":
        fn(model, input_ids, attention_mask, block_length=64, **kwargs)
    else:
        raise ValueError(name)

    _task_trace_list_ctx.reset(token)
    return traces


SCHEDULERS: dict[str, Callable] = {
    "pure_diffusion": generate_llada,
    "block_diffusion": generate_llada,
    "top_k_confidence": generate_topk,
    "top_k_margin": generate_topk,
    "fhs": generate_fhs,
    "eb_sampler": generate_EB,
    "dus": generate_dus,
    "fdm": generate_fdm,
}

BOOLEAN_SCHEDULERS = frozenset({
    "pure_diffusion",
    "block_diffusion",
    "eb_sampler",
    "top_k_confidence",
    "top_k_margin",
})


def main(
    *,
    model_paths: tuple[str, ...] = DEFAULT_MODELS,
    schedulers: tuple[str, ...] = tuple(SCHEDULERS.keys()),
    batch_size: int = 2,
    device: str = "cuda",
) -> list[dict]:
    """Run tracing smoke tests; return audit result rows."""
    results: list[dict] = []

    for model_dir in model_paths:
        if not Path(model_dir).is_dir():
            results.append({
                "model": _model_label(model_dir),
                "scheduler": "*",
                "status": "skip",
                "reason": "model dir missing",
            })
            continue

        model, tokenizer = load_model_and_tokenizer(model_dir, device=device, compile_model=False)
        mask_id, eos_id, eot_id = infer_diffusion_token_ids(model_dir)
        b44 = infer_b44_eos_unmask_safety(model_dir)
        if tokenizer.padding_side != "left":
            tokenizer.padding_side = "left"

        prompt = SMOKE_PROMPT
        if is_dream_model_dir(model_dir):
            prompt = prepend_bos_for_dream(tokenizer, prompt)

        encoded = tokenizer(prompt, add_special_tokens=False, padding=True, return_tensors="pt")
        base_ids = encoded["input_ids"].to(device)
        base_mask = encoded["attention_mask"].to(device)

        if batch_size > 1:
            input_ids = base_ids.repeat(batch_size, 1)
            attention_mask = base_mask.repeat(batch_size, 1)
        else:
            input_ids, attention_mask = base_ids, base_mask

        for sched_name in schedulers:
            if sched_name not in SCHEDULERS:
                results.append({
                    "model": _model_label(model_dir),
                    "scheduler": sched_name,
                    "status": "skip",
                    "reason": "unknown scheduler",
                })
                continue
            try:
                traces = _run_scheduler(
                    sched_name,
                    SCHEDULERS[sched_name],
                    model,
                    input_ids,
                    attention_mask,
                    mask_id=mask_id,
                    eos_id=eos_id,
                    eot_id=eot_id,
                    eos_unmask_b44=b44,
                    batch_size=batch_size,
                )
                if len(traces) != batch_size:
                    raise AssertionError(f"expected {batch_size} traces, got {len(traces)}")

                with tempfile.TemporaryDirectory(prefix="test_tracing_") as tmp:
                    path = Path(tmp) / f"{_model_label(model_dir)}_{sched_name}.jsonl"
                    with open(path, "w", encoding="utf-8") as f:
                        for t in traces:
                            f.write(json.dumps(t, ensure_ascii=False) + "\n")
                    check = check_trace_file(path, max_samples=batch_size)

                hooks_ok = check.hooks_ok and check.samples_missing_required == 0
                if sched_name in BOOLEAN_SCHEDULERS and batch_size > 1:
                    hooks_ok = (
                        check.samples_with_any_selected == check.samples_checked
                        and check.samples_missing_required == 0
                    )

                status = "ok" if hooks_ok else "fail"
                results.append({
                    "model": _model_label(model_dir),
                    "scheduler": sched_name,
                    "batch_size": batch_size,
                    "n_traces": len(traces),
                    "transfer_status": check.transfer_status,
                    "hooks_ok": hooks_ok,
                    "status": status,
                })
                logger.info(
                    "%s / %s: %s transfer=%s",
                    _model_label(model_dir),
                    sched_name,
                    status,
                    check.transfer_status,
                )
            except Exception as e:
                logger.exception("%s / %s failed", _model_label(model_dir), sched_name)
                results.append({
                    "model": _model_label(model_dir),
                    "scheduler": sched_name,
                    "status": "error",
                    "reason": str(e),
                })

        del model, tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    from repo_env_defaults import apply_repo_env_defaults

    apply_repo_env_defaults()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Smoke-test trace hooks on all models × schedulers (like test_schedulers.py).",
    )
    parser.add_argument(
        "--model-path",
        action="append",
        default=None,
        help="Model directory (repeat for multiple; default: llada, llada1_5, dream)",
    )
    parser.add_argument("--scheduler", default=None, choices=list(SCHEDULERS.keys()))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    models = tuple(args.model_path) if args.model_path else DEFAULT_MODELS
    schedulers = (args.scheduler,) if args.scheduler else tuple(SCHEDULERS.keys())

    rows = main(
        model_paths=models,
        schedulers=schedulers,
        batch_size=args.batch_size,
        device=args.device,
    )
    ok = sum(1 for r in rows if r.get("status") == "ok")
    fail = sum(1 for r in rows if r.get("status") in ("fail", "error"))
    print(f"\n{ok} ok, {fail} fail/error, {len(rows)} total")
    for r in rows:
        print(
            f"  {r.get('model')} / {r.get('scheduler')}: {r.get('status')} "
            f"transfer={r.get('transfer_status', '-')} hooks_ok={r.get('hooks_ok', '-')}"
        )
    sys.exit(0 if fail == 0 else 2)
