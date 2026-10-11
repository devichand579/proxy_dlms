"""Orthogonal count/rank ablations packed into a large GPU batch."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from load_model import load_model_and_tokenizer, infer_diffusion_token_ids
from schedulerlens.latent_bench import load_documents, code_results, decode, set_seed
from schedulers import latent_control


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--budgets", default="0.015,0.03,0.06")
    parser.add_argument("--coupling-beta", type=float, default=.25)
    parser.add_argument("--rank-beta", type=float, default=.15)
    parser.add_argument("--rank-betas", help="One temporal penalty weight per configured cohort")
    parser.add_argument("--rank-mode", choices=("risk", "support"), default="risk")
    parser.add_argument("--support-beta", type=float, default=.1)
    parser.add_argument("--risk-head", type=Path)
    parser.add_argument("--motion-beta", type=float, default=.25)
    parser.add_argument("--content-beta", type=float, default=.25)
    parser.add_argument("--value-head", type=Path)
    parser.add_argument("--value-beta", type=float, default=.25)
    parser.add_argument("--value-betas", help="One gain weight per configured cohort")
    parser.add_argument("--switch-step", type=int, default=64)
    parser.add_argument("--out", type=Path, default=Path("outputs/latent_factorial"))
    parser.add_argument("--configurations", help="Explicit cell:budget pairs, comma-separated")
    args = parser.parse_args()
    configurations = [("eb_eb", .03), ("eb_latent", .03)]
    for budget in map(float, args.budgets.split(",")):
        configurations += [("latent_eb", budget), ("latent_latent", budget)]
    if args.configurations:
        configurations = [(item.split(":")[0], float(item.split(":")[1])) for item in args.configurations.split(",")]
    if len(configurations) > 8:
        parser.error("at most eight cohorts per GPU batch")
    value_betas = list(map(float, args.value_betas.split(','))) if args.value_betas else [args.value_beta] * len(configurations)
    if len(value_betas) != len(configurations):
        parser.error("--value-betas requires one weight per cohort")
    rank_betas = list(map(float,args.rank_betas.split(','))) if args.rank_betas else [args.rank_beta] * len(configurations)
    if len(rank_betas) != len(configurations):
        parser.error("--rank-betas requires one weight per cohort")
    documents, dataset_path = load_documents(None, args.samples)
    set_seed(1234)
    model, tokenizer = load_model_and_tokenizer("models/llada", device="cuda", compile_model=False)
    mask, eos, _ = infer_diffusion_token_ids("models/llada")
    encoded = [tokenizer(d["prompt"], add_special_tokens=False)["input_ids"] for d in documents]
    width = max(map(len, encoded))
    ids = torch.full((len(documents), width), tokenizer.pad_token_id, dtype=torch.long, device="cuda")
    attention = torch.zeros_like(ids)
    for row, tokens in enumerate(encoded):
        ids[row, -len(tokens):] = torch.tensor(tokens, device="cuda")
        attention[row, -len(tokens):] = 1
    ids, attention = ids.repeat(len(configurations), 1), attention.repeat(len(configurations), 1)
    cells = [cell for cell, _ in configurations for _ in documents]
    budgets = [budget for _, budget in configurations for _ in documents]
    args.out.mkdir(parents=True, exist_ok=True)
    source = Path(latent_control.__file__).read_bytes()
    (args.out / "scheduler_source.py").write_bytes(source)
    from schedulers import latent
    (args.out / "latent_helpers_source.py").write_bytes(Path(latent.__file__).read_bytes())
    from schedulers import temporal_divergence
    (args.out / "temporal_divergence_source.py").write_bytes(Path(temporal_divergence.__file__).read_bytes())
    (args.out / "benchmark_source.py").write_bytes(Path(__file__).read_bytes())
    readout = None
    if args.risk_head:
        from schedulers.latent_risk import RiskReadout
        readout = RiskReadout(args.risk_head)
        (args.out / "risk_head.pt").write_bytes(args.risk_head.read_bytes())
        from schedulers import latent_risk
        (args.out / "risk_readout_source.py").write_bytes(Path(latent_risk.__file__).read_bytes())
    value_readout = None
    if args.value_head:
        from schedulers import latent_value
        value_readout = latent_value.ValueReadout(args.value_head)
        (args.out / "value_head.pt").write_bytes(args.value_head.read_bytes())
        (args.out / "value_readout_source.py").write_bytes(Path(latent_value.__file__).read_bytes())
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    start = time.perf_counter()
    def progress(metrics):
        print(f"Batch {len(ids)}: step {metrics['steps_taken']}/128", flush=True)
    output = latent_control.generate_factorized(model, ids, attention, cells=cells, risk_budgets=budgets,
        mask_id=mask, eos_id=eos, rank_beta=[weight for weight in rank_betas for _ in documents], coupling_beta=args.coupling_beta,
        steps=128, gen_length=512, progress=progress, rank_mode=args.rank_mode, support_beta=args.support_beta,
        risk_readout=readout, motion_beta=args.motion_beta, content_beta=args.content_beta,
        value_readout=value_readout, value_beta=[weight for weight in value_betas for _ in documents],
        switch_step=args.switch_step)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    metrics = latent_control.last_metrics
    summaries = []
    for cohort, (cell, budget) in enumerate(configurations):
        lo, hi = cohort * len(documents), (cohort + 1) * len(documents)
        generated = output[lo:hi]
        texts = decode(tokenizer, generated, width, eos)
        outcomes = code_results(documents, texts)
        samples = [{**result, "completion": text, "unfinished_masks": int((tokens[width:] == mask).sum())}
                   for result, text, tokens in zip(outcomes, texts, generated)]
        counts = [row[lo:hi] for row in metrics["selected_counts"]]
        summary = {"cell": cell, "risk_budget": budget, "passed": sum(s["passed"] for s in samples),
                   "value_beta": value_betas[cohort],
                   "rank_beta": rank_betas[cohort],
                   "samples": len(documents), "mean_tokens_per_step": sum(map(sum, counts)) / (len(counts) * len(documents)),
                   "forced_singletons": sum(metrics["forced_singletons"][lo:hi]),
                   "unfinished_masks": sum(s["unfinished_masks"] for s in samples)}
        summaries.append(summary)
        suffix = f"_value{value_betas[cohort]:g}" if args.value_betas else ""
        if args.rank_betas:
            suffix += f"_rank{rank_betas[cohort]:g}"
        (args.out / f"{cell}_risk{budget:g}{suffix}.json").write_text(json.dumps({"summary": summary,
            "samples": samples, "selected_counts": counts}, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
    (args.out / "results.json").write_text(json.dumps({"summaries": summaries,
        "dataset_arrow": str(dataset_path), "task_ids": [d["task_id"] for d in documents],
        "selection": "first canonical HumanEval rows reused for exploratory tuning",
        "prompt_format": "raw code completion, zero shot", "token_sampler": "argmax",
        "early_stop": False, "decode": "first EOS then installed HumanEval stop strings",
        "torch": torch.__version__, "hip": torch.version.hip,
        "total_batch_size": len(ids), "steps": 128, "gen_length": 512,
        "generation_seconds_all_cohorts": seconds, "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30,
        "rank_beta": args.rank_beta, "coupling_beta": args.coupling_beta,
        "rank_betas": rank_betas,
        "rank_mode": args.rank_mode, "support_beta": args.support_beta,
        "motion_beta": args.motion_beta,
        "content_beta": args.content_beta,
        "value_beta": args.value_beta, "value_head": str(args.value_head) if args.value_head else None,
        "value_betas": value_betas,
        "switch_step": args.switch_step,
        "value_head_sha256": hashlib.sha256(args.value_head.read_bytes()).hexdigest() if args.value_head else None,
        "risk_head": str(args.risk_head) if args.risk_head else None,
        "risk_head_sha256": hashlib.sha256(args.risk_head.read_bytes()).hexdigest() if args.risk_head else None,
        "code_sha256": hashlib.sha256(source).hexdigest(), "scheduler_metrics": metrics}, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
