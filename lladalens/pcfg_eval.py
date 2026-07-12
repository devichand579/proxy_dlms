#!/usr/bin/env python3
"""
Evaluate a trained PCFG LLaDA model on the test split with all eight schedulers.

Enables tracing, uploads traces to HF (optional), and writes generation + oracle metrics.

Example:
  cd lladalens
  TRACING_ENABLED=true python pcfg_eval.py \\
    --model-path checkpoints/llada-100m-pcfg \\
    --corpus data/pcfg --schedulers all --max-samples 32

  TRACING_ENABLED=true python pcfg_eval.py \\
    --hf-model username/llada-100m-pcfg --hf-dataset pcfg-code \\
    --schedulers pd,bd,eb --steps 128
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LADALENS = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_LADALENS) not in sys.path:
    sys.path.insert(0, str(_LADALENS))

# Tracing hooks are applied at scheduler import time.
os.environ.setdefault("TRACING_ENABLED", "true")

import torch
from hf_hub import apply_pcfg_hf_env, resolve_hf_token
from models.pcfg_models import load_pcfg_llada_checkpoint, load_pcfg_tokenizer
from pcfg_data import ARCH_LEN, build_stack, load_corpus_from_disk, load_corpus_from_hub
from pcfg_metrics import (
    build_full_mask_inputs,
    exact_match,
    masked_perplexity,
    replay_oracle_mt_jt,
    structural_accuracy,
    trace_model_commit_nll,
    trace_oracle_nll,
)
from repo_env_defaults import apply_repo_env_defaults
from schedulers.generate import generate
from traces.hf_trace_upload import maybe_upload_eval_trace_file
from traces.trace_wrapper import _task_trace_list_ctx

SCHEDULER_MAP = {
    "pd": ("pure_diffusion", "confidence"),
    "bd": ("block_diffusion", "confidence"),
    "eb": ("eb_sampler", "confidence"),
    "fdm": ("fdm", None),
    "fhs": ("fhs", None),
    "top_k_c": ("top_k", "confidence"),
    "top_k_m": ("top_k", "top_margin"),
    "dus": ("dus", "confidence"),
}
ALL_SCHEDULERS = list(SCHEDULER_MAP.keys())


def resolve_block_length(mode: str, gen_length: int, block_length: int | None) -> int | None:
    if block_length is not None:
        return int(block_length)
    if mode == "fhs":
        return None
    if mode == "block_diffusion":
        return min(32, gen_length)
    if mode == "fdm":
        return min(64, gen_length)
    if mode == "dus":
        return min(16, gen_length)
    return gen_length


def resolve_model_path(
    *,
    model_path: Path | None,
    hf_model: str | None,
    token: str | None,
) -> Path:
    if model_path is not None:
        path = Path(model_path)
        if not path.is_dir():
            raise FileNotFoundError(f"--model-path not found: {path}")
        return path
    if hf_model is None:
        raise ValueError("pass --model-path or --hf-model")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(hf_model, token=token))


def load_test_corpus(args) -> list[list[str]]:
    if args.corpus is not None:
        _train, test, _ = load_corpus_from_disk(args.corpus)
    elif args.hf_dataset is not None:
        _train, test, _ = load_corpus_from_hub(
            args.hf_dataset,
            namespace=args.hf_namespace,
            token=resolve_hf_token(None),
        )
    else:
        raise ValueError("pass --corpus or --hf-dataset for test split")
    if not test:
        raise ValueError("corpus has no test split; regenerate with pcfg_generate holdout")
    return test


def parse_schedulers(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return ALL_SCHEDULERS
    keys = [s.strip().lower() for s in raw.split(",") if s.strip()]
    bad = [k for k in keys if k not in SCHEDULER_MAP]
    if bad:
        raise ValueError(f"unknown schedulers {bad}; choose from {ALL_SCHEDULERS}")
    return keys


def eval_scheduler(
    scheduler_key: str,
    *,
    model,
    vocab,
    engine,
    token_seqs: list[list[str]],
    device: torch.device,
    steps: int,
    batch_size: int,
    out_dir: Path,
    variant: str,
    upload_traces: bool,
    replay_oracle: bool,
) -> dict:
    mode, strategy = SCHEDULER_MAP[scheduler_key]
    strategy_kw = {} if strategy is None else {"strategy": strategy}

    metrics = {
        "scheduler": scheduler_key,
        "mode": mode,
        "strategy": strategy,
        "n_samples": 0,
        "exact_match": 0.0,
        "structural_accuracy": 0.0,
        "model_commit_nll": 0.0,
        "oracle_nll": 0.0,
        "Mt_total": 0.0,
        "Jt_total": 0.0,
    }
    trace_records: list[dict] = []
    n_done = 0

    for start in range(0, len(token_seqs), batch_size):
        batch_tokens = token_seqs[start : start + batch_size]
        input_ids, attention_mask, gen_lens, ref_ids = build_full_mask_inputs(batch_tokens, vocab)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        gen_length = max(gen_lens)
        bl = resolve_block_length(mode, gen_length, None)

        trace_name = f"pcfg_{variant}_{scheduler_key}_s{steps}"
        traces: list[dict] = []
        token = _task_trace_list_ctx.set(traces)
        try:
            out, _steps_taken, _blocks = generate(
                model,
                input_ids,
                attention_mask,
                steps=steps,
                gen_length=gen_length,
                block_length=bl if bl is not None else gen_length,
                mode=mode,
                top_k=1,
                trace_name=trace_name,
                upload_to_wandb=False,
                mask_id=vocab.mask_id,
                eos_id=vocab.eos_id,
                use_eos_harness=False,
                **strategy_kw,
            )
        finally:
            _task_trace_list_ctx.reset(token)

        if not traces:
            # Single-sample path: synthesize from output only (no trace fields)
            traces = [{"final_sequence": out[b : b + 1].detach().cpu().tolist(),
                       "prompt_len": 0, "gen_length": gen_lens[b],
                       "step_traces": []} for b in range(out.shape[0])]

        for b, tokens in enumerate(batch_tokens):
            n_content = len(tokens)
            tr = traces[b] if b < len(traces) else traces[-1]
            tr = dict(tr)
            ref = ref_ids[b]
            gen_region = out[b, : gen_lens[b]].detach().cpu().tolist()
            em = exact_match(gen_region, ref, n_content=n_content)
            struct_acc = structural_accuracy(gen_region, ref, vocab, n_content=n_content)

            tr["metadata"] = {
                "task": "pcfg_full",
                "variant": variant,
                "scheduler": scheduler_key,
                "reference_tokens": tokens,
                "reference_token_ids": ref,
                "n_content": n_content,
            }
            trace_records.append(tr)

            m_nll = trace_model_commit_nll(tr, ref)
            o_nll = trace_oracle_nll(engine, tr, tokens)
            metrics["exact_match"] += float(em)
            metrics["structural_accuracy"] += struct_acc
            metrics["model_commit_nll"] += m_nll
            metrics["oracle_nll"] += o_nll

            if replay_oracle and tr.get("step_traces"):
                oracle = replay_oracle_mt_jt(model, tr, engine, vocab, device)
                metrics["Mt_total"] += oracle.Mt_total
                metrics["Jt_total"] += oracle.Jt_total

            n_done += 1

    if n_done == 0:
        return metrics

    for key in ("exact_match", "structural_accuracy",
                "model_commit_nll", "oracle_nll", "Mt_total", "Jt_total"):
        metrics[key] = metrics[key] / n_done
    metrics["n_samples"] = n_done
    metrics["model_commit_ppl"] = masked_perplexity(metrics["model_commit_nll"])
    metrics["oracle_ppl"] = masked_perplexity(metrics["oracle_nll"])

    trace_dir = out_dir / "trace_outputs"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"pcfg_{variant}_{scheduler_key}_s{steps}.jsonl"
    with trace_path.open("w", encoding="utf-8") as f:
        for tr in trace_records:
            f.write(json.dumps(tr) + "\n")
    metrics["trace_path"] = str(trace_path)

    if upload_traces:
        uploaded = maybe_upload_eval_trace_file(
            str(trace_path),
            trace_name=trace_path.stem,
            task_name="pcfg",
            num_samples=n_done,
        )
        metrics["trace_hf_upload"] = uploaded

    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path, default=None)
    p.add_argument("--hf-model", type=str, default=None, help="HF model repo id")
    p.add_argument("--variant", type=str, default="100m", help="Label for trace filenames")
    p.add_argument("--corpus", type=Path, default=None)
    p.add_argument("--hf-dataset", type=str, default=None)
    p.add_argument("--hf-namespace", type=str, default=None)
    p.add_argument("--schedulers", type=str, default="all", help="Comma list or 'all'")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", type=Path, default=Path("eval_outputs/pcfg"))
    p.add_argument("--upload-traces", action="store_true", default=True)
    p.add_argument("--no-upload-traces", dest="upload_traces", action="store_false")
    p.add_argument("--replay-oracle", action="store_true", default=True,
                   help="Re-forward model for exact Mt/Jt (slower)")
    p.add_argument("--no-replay-oracle", dest="replay_oracle", action="store_false")
    args = p.parse_args()

    apply_repo_env_defaults()
    apply_pcfg_hf_env()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    schedulers = parse_schedulers(args.schedulers)
    model_dir = resolve_model_path(
        model_path=args.model_path,
        hf_model=args.hf_model,
        token=resolve_hf_token(None),
    )

    model, config, vocab = load_pcfg_llada_checkpoint(model_dir, device=str(device))
    model.eval()
    _g, _cnf, engine, _vocab = build_stack()

    test = load_test_corpus(args)
    if args.max_samples is not None:
        test = test[: args.max_samples]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model_path": str(model_dir),
        "variant": args.variant,
        "n_test": len(test),
        "steps": args.steps,
        "schedulers": {},
    }

    print(f"model={model_dir} test={len(test)} schedulers={schedulers} device={device}", flush=True)

    for key in schedulers:
        print(f"eval scheduler={key} ...", flush=True)
        m = eval_scheduler(
            key,
            model=model,
            vocab=vocab,
            engine=engine,
            token_seqs=test,
            device=device,
            steps=args.steps,
            batch_size=args.batch_size,
            out_dir=args.out_dir,
            variant=args.variant,
            upload_traces=args.upload_traces,
            replay_oracle=args.replay_oracle,
        )
        summary["schedulers"][key] = m
        print(
            f"  {key}: em={m['exact_match']:.3f} struct={m['structural_accuracy']:.3f} "
            f"oracle_ppl={m.get('oracle_ppl', float('nan')):.2f} "
            f"trace={m.get('trace_path')}",
            flush=True,
        )

    summary_path = args.out_dir / f"pcfg_eval_{args.variant}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
