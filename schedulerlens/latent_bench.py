"""Run matched latent schedulers on a fixed, locally cached HumanEval subset.

One model load, per-run artifacts, fixed token proposals/cardinality, and actual
code tests. Repeated searches on this subset are explicitly in-sample tuning.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from load_model import load_model_and_tokenizer, infer_diffusion_token_ids
from schedulers.llada_schedulers import generate_llada
from schedulers.eb_sampler import generate_EB
import schedulers.latent as latent

LATENT_SOURCE_SNAPSHOT = Path(latent.__file__).read_bytes()

STOPS = ("\nclass", "\ndef", "\n#", "\nif", "\nprint")


def load_documents(path, count):
    from datasets import Dataset
    if path is None:
        candidates = sorted(Path("/root/.cache/huggingface/datasets/openai___openai_humaneval").glob("**/openai_humaneval-test.arrow"))
        if len(candidates) != 1:
            raise ValueError("provide --dataset-arrow to select a local HumanEval dataset")
        path = candidates[0]
    data = Dataset.from_file(str(path))
    if not 1 <= count <= len(data):
        raise ValueError("invalid sample count")
    return [dict(data[i]) for i in range(count)], Path(path)


def code_results(documents, completions):
    """Use the existing cached HF/OpenAI correctness executor, in fresh children."""
    candidates = sorted(Path("/root/.cache/huggingface/modules/evaluate_modules/metrics/evaluate-metric--code_eval").glob("*/execute.py"))
    if len(candidates) != 1:
        raise ValueError("expected one cached code_eval executor")
    spec = importlib.util.spec_from_file_location("strace_code_executor", candidates[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    results = []
    for index, (doc, text) in enumerate(zip(documents, completions)):
        program = doc["prompt"] + text + "\n" + doc["test"] + f"\ncheck({doc['entry_point']})"
        result = module.check_correctness(program, 3., index, 0)
        results.append({"task_id": doc["task_id"], "passed": bool(result["passed"]), "result": result["result"]})
    return results


def decode(tokenizer, generated, prompt_len, eos_id):
    texts = []
    for tail in generated[:, prompt_len:]:
        eos = (tail == eos_id).nonzero().flatten()
        if len(eos):
            tail = tail[:int(eos[0])]
        text = tokenizer.decode(tail, skip_special_tokens=True)
        # Match eval.py's stop-token order for the installed HumanEval task.
        text = next((text.split(stop)[0] for stop in STOPS if stop in text), text)
        texts.append(text)
    return texts


def summarize(out):
    payloads = [json.loads(p.read_text()) for p in sorted(out.glob("*/results.json"))]
    baselines = [p for p in payloads if p["config"]["variant"] == "eb_sampler"]
    confidence_controls = [p for p in payloads if p["config"]["variant"] == "confidence"]
    rows = []
    for p in payloads:
        full_canvas_eb = [b for b in baselines if b["config"]["block_length"] == b["config"]["gen_length"]]
        baseline = full_canvas_eb[0] if full_canvas_eb else (baselines[0] if baselines else None)
        score = p["metrics"]
        row = {"run": p["run"], "variant": p["config"]["variant"], "gen_length": p["config"]["gen_length"], "block_length": p["config"]["block_length"],
               "gain_beta": p["config"].get("gain_beta", .25),
               "diversity_beta": p["config"]["diversity_beta"], "stability_beta": p["config"]["stability_beta"],
               "passed": score["passed"], "samples": score["samples"], "pass_at_1": score["pass_at_1"],
               "generation_seconds": score["generation_seconds"], "forward_evaluations_per_sample": score["sequence_evaluations"] / score["samples"],
               "backward_evaluations_per_sample": score["backward_sequence_evaluations"] / score["samples"],
               "peak_allocated_GiB": score["peak_allocated_bytes"] / 2**30, "unfinished_masks": score["unfinished_masks"]}
        row.update(baseline_run=baseline["run"] if baseline else None, wins_vs_eb=None, losses_vs_eb=None)
        if baseline:
            a = {r["task_id"]: r["passed"] for r in baseline["samples"]}
            b = {r["task_id"]: r["passed"] for r in p["samples"]}
            if set(a) != set(b):
                raise ValueError("comparison has mismatched task ids")
            row["wins_vs_eb"] = sum(b[k] and not a[k] for k in a)
            row["losses_vs_eb"] = sum(a[k] and not b[k] for k in a)
        matched_confidence = [b for b in confidence_controls if b["config"]["block_length"] == p["config"]["block_length"]]
        confidence = matched_confidence[0] if matched_confidence else (confidence_controls[0] if confidence_controls else None)
        row.update(confidence_control=confidence["run"] if confidence else None, wins_vs_confidence=None, losses_vs_confidence=None)
        if confidence:
            a = {r["task_id"]: r["passed"] for r in confidence["samples"]}
            b = {r["task_id"]: r["passed"] for r in p["samples"]}
            if set(a) != set(b):
                raise ValueError("confidence control has mismatched task ids")
            row["wins_vs_confidence"] = sum(b[k] and not a[k] for k in a)
            row["losses_vs_confidence"] = sum(a[k] and not b[k] for k in a)
        rows.append(row)
    if not rows:
        return []
    with (out / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return rows


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def run(args, model_bundle=None):
    gradient_pool_min = getattr(args,'gradient_pool_min',0)
    gradient_temporal = getattr(args,'gradient_temporal',False)
    gradient_probe_batch_size = getattr(args,'gradient_probe_batch_size',0)
    gradient_risk_eta = getattr(args,'gradient_risk_eta',0.)
    gradient_readiness_gate = getattr(args,'gradient_readiness_gate',0.)
    documents, dataset_path = load_documents(args.dataset_arrow, args.samples)
    variants = args.variants.split(",")
    if any(v not in ("confidence", "eb_sampler", *latent.VARIANTS) for v in variants):
        raise ValueError("unknown variant")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"task_ids": [d["task_id"] for d in documents], "dataset_arrow": str(dataset_path),
                "selection": "first N canonical HumanEval test rows; reused for exploratory tuning",
                "seed": args.seed, "steps": args.steps, "gen_length": args.gen_length,
                "token_sampler": "argmax", "early_stop": False, "decode": "truncate first EOS, then installed task stop tokens",
                "prompt_format": "raw HumanEval code completion, zero shot", "model": str(args.model.resolve())}
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("output directory contains a different experiment manifest")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    pending = []
    for variant in variants:
        config = {"variant": variant, "steps": args.steps, "gen_length": args.gen_length,
                  "block_length": args.block_length, "diversity_beta": args.diversity_beta,
                  "stability_beta": args.stability_beta, "probe_every": args.probe_every,
                  "fisher_vocab": args.fisher_vocab, "batch_size": args.batch_size, "seed": args.seed}
        name = f"{variant}_b{args.block_length}_d{args.diversity_beta:g}_s{args.stability_beta:g}_p{args.probe_every}_v{args.fisher_vocab}"
        if variant.endswith(("gradient_gain", "latent_support", "temporal_support")):
            config["gain_beta"] = args.gain_beta
            if args.gain_beta != .25:
                name += f"_g{args.gain_beta:g}"
        if variant.endswith('gradient_gain'):
            if gradient_pool_min:
                config['gradient_pool_min'] = gradient_pool_min; name += f'_cp{gradient_pool_min}'
            if gradient_temporal:
                config['gradient_temporal'] = True; name += '_ft'
            if gradient_probe_batch_size:
                config['gradient_probe_batch_size'] = gradient_probe_batch_size; name += f'_mb{gradient_probe_batch_size}'
            if gradient_risk_eta:
                config['gradient_risk_eta'] = gradient_risk_eta; name += f'_risk{gradient_risk_eta:g}'
            if gradient_readiness_gate:
                config['gradient_readiness_gate'] = gradient_readiness_gate; name += f'_gate{gradient_readiness_gate:g}'
        if variant.endswith("response_ready"):
            config["response_incoming_beta"] = args.response_incoming_beta
            name += f"_ri{args.response_incoming_beta:g}"
        result_path = args.out / name / "results.json"
        if result_path.exists():
            previous_config = json.loads(result_path.read_text())["config"]
            if variant.endswith(("gradient_gain", "latent_support", "temporal_support")):
                previous_config.setdefault("gain_beta", .25)
            if previous_config != config:
                raise ValueError("existing run configuration differs")
            print(f"Using completed run {name}", flush=True)
        else:
            pending.append((variant, name, config, result_path))
    if not pending:
        summarize(args.out); return model_bundle
    if not torch.cuda.is_available():
        raise RuntimeError("HumanEval GPU runner requires GPU access")
    set_seed(args.seed)
    if model_bundle is None:
        model, tokenizer = load_model_and_tokenizer(str(args.model), device="cuda", compile_model=False)
    else:
        model, tokenizer = model_bundle
    tokenizer.padding_side = "left"
    mask_id, eos_id, eot_id = infer_diffusion_token_ids(args.model)
    encoded = [tokenizer(d["prompt"], add_special_tokens=False)["input_ids"] for d in documents]
    # Keep exact batch membership/order consistent across all arms.
    batches = [list(range(i, min(i + args.batch_size, len(documents)))) for i in range(0, len(documents), args.batch_size)]
    code_hash = hashlib.sha256(LATENT_SOURCE_SNAPSHOT).hexdigest()
    for variant, name, config, result_path in pending:
        print(f"START {name}", flush=True)
        set_seed(args.seed)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        completions = [None] * len(documents)
        masks = [None] * len(documents)
        totals = {"sequence_evaluations": 0, "forward_calls": 0, "backward_sequence_evaluations": 0,
                  "backward_passes": 0, "probe_steps": 0, "probe_switches": 0, "unfinished_masks": 0}
        batch_metrics = []
        torch.cuda.synchronize(); start = time.perf_counter()
        for chunk, indices in enumerate(batches):
            length = max(len(encoded[i]) for i in indices)
            ids = torch.full((len(indices), length), tokenizer.pad_token_id, device="cuda", dtype=torch.long)
            attention = torch.zeros_like(ids)
            for row, index in enumerate(indices):
                tokens = torch.tensor(encoded[index], device="cuda")
                ids[row, -len(tokens):] = tokens
                attention[row, -len(tokens):] = 1
            common = dict(steps=args.steps, gen_length=args.gen_length, block_length=args.block_length,
                          temperature=0., mask_id=mask_id, eos_id=eos_id, eot_id=eot_id, use_eos_harness=False)
            if variant in ("confidence", "eb_sampler"):
                if variant == "confidence":
                    generated, steps_taken, _ = generate_llada(model, ids, attention, remasking="confidence", upload_to_wandb=False, **common)
                else:
                    generated, steps_taken, _ = generate_EB(model, ids, attention, gamma=.1, E="confidence", upload_to_wandb=False, **common)
                metric = {"sequence_evaluations": steps_taken * len(indices), "forward_calls": steps_taken,
                          "unfinished_masks": int((generated[:, length:] == mask_id).sum())}
            else:
                def progress(metric):
                    print(f"  {name}: batch {chunk + 1}/{len(batches)}, step {metric['steps_taken']}/{args.steps}, forward work {metric['sequence_evaluations'] / len(indices):.0f}/sample", flush=True)
                generated, _, _ = latent.generate_latent(model, ids, attention, variant=variant,
                    diversity_beta=args.diversity_beta, stability_beta=args.stability_beta, probe_every=args.probe_every,
                    fisher_vocab=args.fisher_vocab, gain_beta=args.gain_beta,
                    gradient_pool_min=gradient_pool_min if variant.endswith('gradient_gain') else 0,
                    gradient_temporal=gradient_temporal if variant.endswith('gradient_gain') else False,
                    gradient_probe_batch_size=gradient_probe_batch_size if variant.endswith('gradient_gain') else 0,
                    gradient_risk_eta=gradient_risk_eta if variant.endswith('gradient_gain') else 0.,
                    gradient_readiness_gate=gradient_readiness_gate if variant.endswith('gradient_gain') else 0.,
                    response_incoming_beta=args.response_incoming_beta, progress=progress, **common)
                metric = dict(latent.last_metrics)
            torch.cuda.synchronize()
            texts = decode(tokenizer, generated, length, eos_id)
            for row, index in enumerate(indices):
                completions[index] = texts[row]
                masks[index] = int((generated[row, length:] == mask_id).sum())
            for key in totals:
                totals[key] += metric.get(key, 0)
            batch_metrics.append(metric)
            print(f"  Finished batch {chunk + 1}/{len(batches)} in {time.perf_counter() - start:.1f}s cumulative", flush=True)
            del generated, ids, attention
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        peak_allocated, peak_reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        # Preserve generated outputs before testing them, including on interruption.
        result_path.parent.mkdir(parents=True, exist_ok=True)
        (result_path.parent / "scheduler_source.py").write_bytes(LATENT_SOURCE_SNAPSHOT)
        samples = [{"task_id": d["task_id"], "completion": text, "unfinished_masks": mask} for d, text, mask in zip(documents, completions, masks)]
        (result_path.parent / "completions.json").write_text(json.dumps(samples, indent=2) + "\n")
        outcomes = code_results(documents, completions)
        for sample, outcome in zip(samples, outcomes):
            sample.update(outcome)
        passed = sum(s["passed"] for s in samples)
        metrics = {**totals, "samples": len(samples), "passed": passed, "pass_at_1": passed / len(samples),
                   "generation_seconds": seconds, "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved}
        payload = {"run": name, "config": config, "metrics": metrics, "samples": samples, "batch_metrics": batch_metrics,
                   "metadata": {"latent_code_sha256": code_hash, "torch": torch.__version__, "hip": torch.version.hip,
                                "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
                                "model_config_sha256": hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest()}}
        result_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        summarize(args.out)
        print(f"RESULT {name}: {passed}/{len(samples)} passed, {seconds:.1f}s, forward evaluations/sample {totals['sequence_evaluations'] / len(samples):.1f}, backward evaluations/sample {totals['backward_sequence_evaluations'] / len(samples):.1f}", flush=True)
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/llada"))
    parser.add_argument("--dataset-arrow", type=Path)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=512)
    parser.add_argument("--block-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--variants", default="eb_sampler,confidence,entropy,raw_diversity,fisher_diversity,temporal,gradient_gain,response_gain,hybrid_probe")
    parser.add_argument("--diversity-beta", type=float, default=.25)
    parser.add_argument("--stability-beta", type=float, default=.15)
    parser.add_argument("--probe-every", type=int, default=8)
    parser.add_argument("--fisher-vocab", type=int, default=128)
    parser.add_argument("--gain-beta", type=float, default=.25)
    parser.add_argument("--response-incoming-beta", type=float, default=.2)
    parser.add_argument("--gradient-pool-min", type=int, default=0)
    parser.add_argument("--gradient-temporal", action='store_true')
    parser.add_argument("--gradient-probe-batch-size", type=int, default=0)
    parser.add_argument("--gradient-risk-eta", type=float, default=0.)
    parser.add_argument("--gradient-readiness-gate", type=float, default=0.)
    parser.add_argument("--out", type=Path, default=Path("outputs/latent_humaneval20"))
    args = parser.parse_args()
    if args.batch_size < 1 or args.fisher_vocab < 1 or args.probe_every < 1:
        parser.error("batch size, Fisher vocabulary and probe interval must be positive")
    # Explicitly keep the GPU runner local: no tracing or remote experiment logs.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    run(args)


if __name__ == "__main__":
    main()
