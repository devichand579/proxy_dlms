"""Pack four stability settings into one 80-sequence GPU batch."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from load_model import load_model_and_tokenizer, infer_diffusion_token_ids
from schedulerlens.latent_bench import load_documents, code_results, decode, set_seed
from schedulers import latent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--betas", default="0,0.1,0.25,0.4")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("outputs/latent_stability_sweep80"))
    args = parser.parse_args()
    betas = [float(x) for x in args.betas.split(",")]
    if len(betas) > 4:
        parser.error("limit to four settings per batch")
    documents, dataset_path = load_documents(None, args.samples)
    set_seed(1234)
    model, tokenizer = load_model_and_tokenizer("models/llada", device="cuda", compile_model=False)
    mask, eos, eot = infer_diffusion_token_ids("models/llada")
    encoded = [tokenizer(d["prompt"], add_special_tokens=False)["input_ids"] for d in documents]
    width = max(map(len, encoded))
    ids = torch.full((len(documents), width), tokenizer.pad_token_id, dtype=torch.long, device="cuda")
    attention = torch.zeros_like(ids)
    for row, tokens in enumerate(encoded):
        ids[row, -len(tokens):] = torch.tensor(tokens, device="cuda")
        attention[row, -len(tokens):] = 1
    ids, attention = ids.repeat(len(betas), 1), attention.repeat(len(betas), 1)
    row_betas = [beta for beta in betas for _ in documents]
    args.out.mkdir(parents=True, exist_ok=True)
    source = Path(latent.__file__).read_bytes()
    (args.out / "scheduler_source.py").write_bytes(source)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    started = time.perf_counter()
    def progress(metrics):
        print(f"Batch {len(ids)}: step {metrics['steps_taken']}/128", flush=True)
    output, _, _ = latent.generate_latent(model, ids, attention, variant="eb_target_temporal",
        steps=128, gen_length=512, block_length=512, mask_id=mask, eos_id=eos, eot_id=eot,
        use_eos_harness=False, stability_beta=row_betas, progress=progress)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    summaries = []
    for cohort, beta in enumerate(betas):
        generated = output[cohort * len(documents):(cohort + 1) * len(documents)]
        completions = decode(tokenizer, generated, width, eos)
        outcomes = code_results(documents, completions)
        samples = [{**result, "completion": text, "unfinished_masks": int((tail[width:] == mask).sum())}
                   for result, text, tail in zip(outcomes, completions, generated)]
        passed = sum(s["passed"] for s in samples)
        summary = {"stability_beta": beta, "passed": passed, "samples": len(samples), "pass_at_1": passed / len(samples)}
        summaries.append(summary)
        (args.out / f"beta_{beta:g}.json").write_text(json.dumps({"summary": summary, "samples": samples}, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
    payload = {"summaries": summaries, "total_batch_size": len(ids), "steps": 128, "gen_length": 512,
               "task_ids": [d["task_id"] for d in documents], "dataset_arrow": str(dataset_path),
               "generation_seconds_all_cohorts": seconds, "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30,
               "scheduler_metrics": latent.last_metrics, "code_sha256": hashlib.sha256(source).hexdigest()}
    (args.out / "results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
