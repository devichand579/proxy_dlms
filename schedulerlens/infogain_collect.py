"""Collect short LLaDA arithmetic infilling branches for offline score ablations.

Three single-token slots, fixed visible text, singleton commitments, common
data-consistent histories. This is a mechanism audit, not free-text accuracy.
No downloads, external uploads, compile, training or production-scheduler edits.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from load_model import infer_diffusion_token_ids, load_model_and_tokenizer
from schedulerlens.infogain_lab import dependence_decomposition, effect_metrics


EXAMPLES = """Check the statement by adding the numbers, then answer Yes or No.
Statement: 2 + 1 + 3 = 6
First sum: 3
Final sum: 6
Answer: Yes

Statement: 1 + 2 + 1 = 5
First sum: 3
Final sum: 4
Answer: No

"""


def make_problems(count, seed):
    rng = np.random.default_rng(seed)
    triples = [(a, b, c) for a, b, c in itertools.product(range(1, 5), range(1, 5), range(1, 4)) if a + b + c <= 9]
    if not 1 <= count <= len(triples):
        raise ValueError(f"samples must be between 1 and {len(triples)} distinct arithmetic problems")
    rng.shuffle(triples)
    problems = []
    for i, (a, b, c) in enumerate(triples[:count]):
        correct = a + b + c
        claim = correct if i % 2 == 0 else (correct + 1 if correct < 9 else correct - 1)
        parts = [EXAMPLES, f"Statement: {a} + {b} + {c} = {claim}\nFirst sum: ", str(a + b), "\nFinal sum: ", str(correct), "\nAnswer: ", "Yes" if claim == correct else "No"]
        spans = []
        text = ""
        for part_index, part in enumerate(parts):
            start = len(text)
            text += part
            if part_index in (2, 4, 6):
                spans.append((start, len(text)))
        problems.append({"id": f"sum_{a}_{b}_{c}_{claim}", "text": text, "spans": spans,
                         "split": "test" if (i // 2) % 4 == 3 else "validation"})
    return problems


def encode_problem(tokenizer, problem):
    encoded = tokenizer(problem["text"], add_special_tokens=False, return_offsets_mapping=True)
    positions = []
    for start, end in problem["spans"]:
        hits = [i for i, (lo, hi) in enumerate(encoded["offset_mapping"]) if lo < end and hi > start]
        if len(hits) != 1:
            raise ValueError(f"{problem['id']}: audit requires a single token per slot, got {hits}")
        positions.append(hits[0])
    return encoded["input_ids"], positions


def stats(logits, positions, mask_id):
    local = logits[:, positions].float().clone()
    # Exclude re-committing MASK. Finite suppression avoids inf-inf in deltas.
    local[..., mask_id] = -1e9
    logp = torch.log_softmax(local, -1)
    p = logp.exp()
    ent = -(p * logp).sum(-1)
    return local.cpu().numpy(), logp.cpu().numpy(), p.cpu().numpy(), ent.cpu().numpy()


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def raw_gram_scores(hidden, candidates, context, lam=.5):
    """Legacy raw singleton quadratic geometry; not the removed spectral.py."""
    candidate = hidden[candidates].float()
    candidate -= candidate.mean(0, keepdim=True)
    candidate /= candidate.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    ctx = hidden[context].float()
    ctx /= ctx.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    aligned = lam * candidate.mean(0)
    if len(context):
        aligned += (1 - lam) * ctx.mean(0)
    return (candidate @ aligned - .5 * candidate.square().sum(-1)).cpu().tolist()


@torch.no_grad()
def collect(args):
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable in this shell; use a GPU-enabled execution context")
    torch.manual_seed(args.seed)
    model_path = args.model.resolve()
    cfg = json.loads((model_path / "config.json").read_text())
    if cfg.get("model_type") != "llada" or cfg.get("weight_tying"):
        raise ValueError("This pilot supports LLaDA checkpoints with a separate output head")
    mask_id, _, _ = infer_diffusion_token_ids(model_path)
    dtype = torch.bfloat16 if str(args.device).startswith("cuda") else torch.float32
    started = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer(str(model_path), torch_dtype=dtype, device=args.device, compile_model=False)
    sync(args.device)
    load_seconds = time.perf_counter() - started
    records, states = [], []
    sequence_evals = forward_calls = 0
    measured_seconds = 0.0
    if str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    histories = ("initial", "first_sum_visible", "answer_visible")
    problems = make_problems(args.samples, args.seed)
    for index, problem in enumerate(problems):
        token_ids, slots = encode_problem(tokenizer, problem)
        reference = np.asarray([token_ids[p] for p in slots])
        for history in histories:
            ids = torch.tensor([token_ids], device=args.device)
            for position in slots:
                ids[0, position] = mask_id
            visible = {"initial": None, "first_sum_visible": 0, "answer_visible": 2}[history]
            if visible is not None:
                ids[0, slots[visible]] = int(reference[visible])
            eligible = [i for i in range(3) if i != visible]
            positions = [slots[i] for i in eligible]
            refs = reference[eligible]
            attention = torch.ones_like(ids)
            captured = []
            hook = model.get_output_embeddings().register_forward_pre_hook(lambda _module, inputs: captured.append(inputs[0].detach()))
            sync(args.device)
            start = time.perf_counter()
            try:
                out = model(ids, attention_mask=attention)
            finally:
                hook.remove()
            if len(captured) != 1:
                raise RuntimeError("expected one output-head input capture")
            context = [i for i in range(ids.shape[1]) if i not in positions]
            raw_scores = raw_gram_scores(captured[0][0], positions, context)
            del captured
            base_logits, base_logp, base_p, base_entropy = stats(out.logits, positions, mask_id)
            del out
            sequence_evals += 1
            forward_calls += 1
            pred = base_logits[0].argmax(-1)
            alternatives = np.argsort(base_logits[0], axis=-1)[:, -args.alternatives:][:, ::-1]
            actions = [(i, int(token)) for i in range(len(eligible)) for token in alternatives[i]]
            branches = {}
            for offset in range(0, len(actions), args.microbatch):
                group = actions[offset:offset + args.microbatch]
                branch_ids = ids.repeat(len(group), 1)
                for row, (source, token) in enumerate(group):
                    branch_ids[row, positions[source]] = token
                out = model(branch_ids, attention_mask=attention.repeat(len(group), 1))
                lg, lp, p, ent = stats(out.logits, positions, mask_id)
                del out
                for row, action in enumerate(group):
                    branches[action] = (lg[row], lp[row], p[row], ent[row])
                sequence_evals += len(group)
                forward_calls += 1
            sync(args.device)
            elapsed = time.perf_counter() - start
            measured_seconds += elapsed
            for source, slot in enumerate(eligible):
                others = [i for i in range(len(eligible)) if i != source]
                tokens = alternatives[source]
                weights = base_p[0, source, tokens].astype(np.float64)
                retained = float(weights.sum())
                weights /= retained
                predicted_branch = branches[source, int(pred[source])]
                external, state_gain = effect_metrics(base_entropy[0], predicted_branch[3], source)
                expected_gain = influence = dependence = defect = fisher = 0.0
                for token, weight in zip(tokens, weights):
                    branch = branches[source, int(token)]
                    expected_gain += weight * effect_metrics(base_entropy[0], branch[3], source)[0]
                    delta = branch[0][others].astype(np.float64) - base_logits[0, others].astype(np.float64)
                    probabilities = base_p[0, others].astype(np.float64)
                    mean = (probabilities * delta).sum(-1, keepdims=True)
                    fisher += weight * .5 * (probabilities * (delta - mean) ** 2).sum()
                for target in others:
                    a, b, c = dependence_decomposition(base_p[0, target], [branches[source, int(t)][2][target] for t in tokens], weights)
                    influence += a
                    dependence += b
                    defect += c
                gain_nll = sum(float(predicted_branch[1][target, refs[target]] - base_logp[0, target, refs[target]]) for target in others)
                sorted_p = np.sort(base_p[0, source])
                records.append({
                    "sample_id": problem["id"], "history": history, "split": problem["split"],
                    "source": slot, "position": positions[source], "predicted_token": int(pred[source]),
                    "reference_token": int(refs[source]), "confidence": float(sorted_p[-1]),
                    "margin": float(sorted_p[-1] - sorted_p[-2]), "entropy": float(base_entropy[0, source]),
                    "commit_nll": float(-base_logp[0, source, pred[source]]),
                    "external_gain": external, "expected_external_gain": float(expected_gain),
                    "paper_ig": state_gain - float(base_entropy[0, source]),
                    "external_ig": external / len(others) - float(base_entropy[0, source]),
                    "response_kl": float(influence), "local_dependence": float(dependence),
                    "coherence_defect": float(defect), "response_fisher": float(fisher),
                    "raw_gram": raw_scores[source], "reference_nll_gain": gain_nll,
                    "source_error": int(pred[source] != refs[source]), "omitted_mass": max(0.0, 1 - retained),
                })
            states.append({"sample_id": problem["id"], "history": history, "input_ids": ids[0].cpu().tolist(),
                           "positions": positions, "reference_tokens": refs.tolist(), "seconds": elapsed})
        print(f"Collected {index + 1}/{len(problems)} problems", flush=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    config_digest = hashlib.sha256((model_path / "config.json").read_bytes()).hexdigest()
    payload = {"schema": 1, "kind": "llada_fixed_history_arithmetic_infilling", "records": records, "states": states, "problems": problems,
               "metadata": {"model_path": str(model_path), "config_sha256": config_digest, "git_commit": commit,
                            "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            "torch": torch.__version__, "hip": torch.version.hip, "python": platform.python_version(),
                            "seed": args.seed, "samples": args.samples, "alternatives": args.alternatives,
                            "microbatch": args.microbatch, "singleton_budget": 1, "token_sampler": "argmax",
                            "histories": "data-consistent gold reveals; no schedule rollouts",
                            "distribution": "full vocabulary except MASK; no domain vocabulary constraints",
                            "forward_calls": forward_calls, "sequence_evaluations": sequence_evals,
                            "collection_seconds": measured_seconds, "load_seconds": load_seconds,
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device) if str(args.device).startswith("cuda") else None,
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved(args.device) if str(args.device).startswith("cuda") else None}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload["metadata"], indent=2))
    print(f"Cache: {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/llada"))
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alternatives", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=Path("outputs/infogain_lab/arithmetic.json"))
    args = parser.parse_args()
    if args.microbatch < 1:
        parser.error("microbatch must be positive")
    if args.samples < 1:
        parser.error("samples must be positive")
    if args.out.exists():
        parser.error("cache already exists; choose another --out to preserve comparisons")
    collect(args)


if __name__ == "__main__":
    main()
