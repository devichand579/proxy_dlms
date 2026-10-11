"""Completed arithmetic infilling: 32 canvases, four worksheets each, 12 slots.

All schedulers share the same forward-step cap, model and visible text. Adaptive
counts can leave masks unfinished; those slots count as errors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from load_model import load_model_and_tokenizer, infer_diffusion_token_ids
from schedulerlens.infogain_collect import make_problems, encode_problem, EXAMPLES
from schedulerlens.latent_bench import set_seed
from schedulers import latent
from schedulers import latent_control
from schedulers import latent_finite
from schedulers import latent_information
from schedulers import latent_conditioning
from schedulers import latent_depth
from schedulers import latent_attention
from schedulers import latent_consistency
from schedulers.eb_sampler import generate_EB
from schedulers.utils import get_mask_by_confidence, get_num_transfer_tokens


def canvases(count, seed):
    pool = make_problems(32, seed)
    examples = []
    for i in range(count):
        text = EXAMPLES
        spans = []
        for offset in range(4):
            problem = pool[(i + 7 * offset) % len(pool)]
            body = problem["text"][len(EXAMPLES):]
            start = len(text)
            text += body + "\n\n"
            spans += [(lo - len(EXAMPLES) + start, hi - len(EXAMPLES) + start) for lo, hi in problem["spans"]]
        examples.append({"id": f"worksheet_canvas_{i}", "text": text, "spans": spans})
    return examples


@torch.no_grad()
def run(args):
    variants = args.variants.split(",")
    conditioning_variants = {'conditioning_support025':.25,'conditioning_support05':.5,'conditioning_support1':1.}
    depth_variants = {'depth_motion_risk':(.15,0.),'depth_signed_support':(0.,.15),'depth_support_risk':(.15,.15)}
    attention_variants = {'attention_mask_read':(.15,False),'attention_uncertain_read015':(.15,True),'attention_uncertain_read03':(.3,True)}
    consistency_variants={'consistency_filter':'filter','consistency_singleton':'singleton'}
    finite_variants = {'finite_all_targets':(False,False),
                       'finite_balanced_targets':(True,False),
                       'finite_balanced_sources_targets':(True,True)}
    information_variants = {
        'response_information':dict(risk_eta=0.,incoming_beta=0.),
        'response_information_risk':dict(risk_eta=1.,incoming_beta=0.),
        'response_information_readiness':dict(risk_eta=1.,incoming_beta=.15),
        'response_source_weighted':dict(risk_eta=1.,incoming_beta=.15,source_mass_weighting=True),
        'response_count_only':dict(gain_beta=0.,risk_eta=0.,incoming_beta=0.,count_mode='probe_surprise'),
        'response_weighted_count':dict(risk_eta=1.,incoming_beta=.15,source_mass_weighting=True,count_mode='probe_surprise'),
        'information_depth005':dict(risk_eta=1.,incoming_beta=.15,depth_risk_beta=.05),
        'information_depth015':dict(risk_eta=1.,incoming_beta=.15,depth_risk_beta=.15),
    }
    control_cells = {"control_count": "latent_eb", "control_both": "latent_latent",
                     "risk_count": "learned_eb", "risk_both": "learned_learned",
                     "content_rank": "eb_content", "content_budget": "content_entropy",
                     "value_rank": "eb_value", "value_budget": "value_entropy",
                     "posterior_kl": "kl_entropy", "posterior_js": "js_entropy",
                     "hybrid_both": "learned_latent", "late_hybrid_both": "late_learned_latent",
                     "motion_count": "motion_eb", "motion_both": "motion_latent"}
    if any(v not in ("eb_sampler", "confidence", *control_cells, *latent.VARIANTS, *finite_variants, *information_variants, *conditioning_variants, *depth_variants, *attention_variants, *consistency_variants) for v in variants):
        raise ValueError("unknown variant")
    if any(v.startswith(("risk_", "content_")) or v in ('hybrid_both','late_hybrid_both') for v in variants) and args.risk_head is None:
        raise ValueError("risk/content variants require --risk-head")
    if any(v.startswith("value_") for v in variants) and args.value_head is None:
        raise ValueError("value variants require --value-head")
    args.out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    model, tokenizer = load_model_and_tokenizer(str(args.model), device="cuda", compile_model=False)
    mask_id, eos_id, eot_id = infer_diffusion_token_ids(args.model)
    readout = None
    if args.risk_head:
        from schedulers.latent_risk import RiskReadout
        readout = RiskReadout(args.risk_head)
        (args.out / "risk_head.pt").write_bytes(args.risk_head.read_bytes())
    value_readout = None
    if args.value_head:
        from schedulers import latent_value
        value_readout = latent_value.ValueReadout(args.value_head)
        (args.out / "value_head.pt").write_bytes(args.value_head.read_bytes())
        (args.out / "value_readout_source.py").write_bytes(Path(latent_value.__file__).read_bytes())
    for module, name in ((latent, "latent_source.py"), (latent_control, "controller_source.py"),
                         (latent_finite, "finite_source.py"),(latent_information,"information_source.py"),
                         (latent_conditioning,"conditioning_source.py"),(latent_depth,"depth_source.py"),
                         (latent_attention,"attention_source.py"),(latent_consistency,"consistency_source.py")):
        (args.out / name).write_bytes(Path(module.__file__).read_bytes())
    examples = canvases(args.samples, args.seed)
    encoded = [encode_problem(tokenizer, ex) for ex in examples]
    width = max(len(ids) for ids, _ in encoded)
    # Right-pad the fixed scaffold; padding is removed through the attention mask.
    canvas = torch.full((len(examples), width), tokenizer.pad_token_id, device="cuda", dtype=torch.long)
    attention = torch.zeros_like(canvas)
    positions, references = [], []
    for row, (ids, slots) in enumerate(encoded):
        canvas[row, :len(ids)] = torch.tensor(ids, device="cuda")
        attention[row, :len(ids)] = 1
        positions.append(slots)
        references.append([ids[pos] for pos in slots])
        canvas[row, slots] = mask_id
    prompt = canvas[:, :0]
    summary = []
    for variant in variants:
        set_seed(args.seed)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        start = time.perf_counter()
        if variant == "confidence":
            output = canvas.clone()
            quotas = get_num_transfer_tokens(canvas == mask_id, args.steps)
            for step in range(args.steps):
                logits = model(output, attention_mask=attention).logits
                logits[..., mask_id] = -torch.inf
                predicted = logits.argmax(-1)
                confidence = get_mask_by_confidence(logits, predicted)
                for row in range(len(examples)):
                    eligible = (output[row] == mask_id).nonzero().flatten()
                    pick = eligible[confidence[row, eligible].topk(int(quotas[row, step])).indices]
                    output[row, pick] = predicted[row, pick]
                del logits, predicted, confidence
            metrics = {"sequence_evaluations": args.steps * len(examples), "backward_sequence_evaluations": 0}
        elif variant == "eb_sampler":
            output, steps_taken, _ = generate_EB(model, prompt, attention[:, :0], initial_canvas=canvas,
                canvas_attention_mask=attention, gen_length=width, block_length=width,
                steps=args.steps, gamma=.1, E="confidence", mask_id=mask_id, eos_id=eos_id,
                eot_id=eot_id, use_eos_harness=False, upload_to_wandb=False)
            metrics = {"sequence_evaluations": steps_taken * len(examples), "backward_sequence_evaluations": 0}
        elif variant in consistency_variants:
            output=latent_consistency.generate_consistency(model,prompt,attention[:,:0],
                initial_canvas=canvas,full_attention=attention,gen_length=width,
                steps=args.steps,mask_id=mask_id,eos_id=eos_id,mode=consistency_variants[variant],
                probe_every=args.finite_probe_every,branch_batch_size=args.finite_branch_batch_size,
                stability_beta=args.stability_beta)
            metrics=dict(latent_consistency.last_metrics)
        elif variant in attention_variants:
            read_beta,uncertainty_weighted=attention_variants[variant]
            output=latent_attention.generate_attention(model,prompt,attention[:,:0],
                initial_canvas=canvas,full_attention=attention,gen_length=width,
                steps=args.steps,mask_id=mask_id,eos_id=eos_id,
                read_beta=read_beta,uncertainty_weighted=uncertainty_weighted,
                attention_layers=args.attention_layers,stability_beta=args.stability_beta)
            metrics=dict(latent_attention.last_metrics)
        elif variant in depth_variants:
            depth_risk_beta,depth_support_beta = depth_variants[variant]
            output = latent_depth.generate_depth(model,prompt,attention[:,:0],
                initial_canvas=canvas,full_attention=attention,gen_length=width,
                steps=args.steps,mask_id=mask_id,eos_id=eos_id,depth_back=args.depth_back,
                depth_risk_beta=depth_risk_beta,depth_support_beta=depth_support_beta,
                stability_beta=args.stability_beta)
            metrics = dict(latent_depth.last_metrics)
        elif variant in conditioning_variants:
            output = latent_conditioning.generate_conditioning(model,prompt,attention[:,:0],
                initial_canvas=canvas,full_attention=attention,gen_length=width,
                steps=args.steps,mask_id=mask_id,eos_id=eos_id,
                context_beta=conditioning_variants[variant],stability_beta=args.stability_beta,
                branch_batch_size=args.finite_branch_batch_size)
            metrics = dict(latent_conditioning.last_metrics)
        elif variant in information_variants:
            settings = {'gain_beta':args.gain_beta,**information_variants[variant]}
            output = latent_information.generate_information(model,prompt,attention[:,:0],
                initial_canvas=canvas,full_attention=attention,gen_length=width,
                steps=args.steps,mask_id=mask_id,eos_id=eos_id,
                candidates=args.finite_candidates,probe_every=args.finite_probe_every,
                branch_batch_size=args.finite_branch_batch_size,stability_beta=args.stability_beta,depth_back=args.depth_back,**settings)
            metrics = dict(latent_information.last_metrics)
        elif variant in finite_variants:
            target_balance,pool_balance = finite_variants[variant]
            output = latent_finite.generate_finite(model,prompt,attention[:,:0],
                initial_canvas=canvas,full_attention=attention,gen_length=width,
                steps=args.steps,mask_id=mask_id,eos_id=eos_id,
                candidates=args.finite_candidates,probe_every=args.finite_probe_every,
                branch_batch_size=args.finite_branch_batch_size,gain_beta=args.gain_beta,
                risk_eta=args.finite_risk_eta,stability_beta=args.stability_beta,
                target_balance=target_balance,pool_balance=pool_balance,
                matched_probe_control=True)
            metrics = dict(latent_finite.last_metrics)
        elif variant in control_cells:
            cell = control_cells[variant]
            output = latent_control.generate_factorized(model, prompt, attention[:, :0],
                cells=[cell] * len(examples), risk_budgets=[args.risk_budget] * len(examples),
                initial_canvas=canvas, full_attention=attention, gen_length=width,
                steps=args.steps, mask_id=mask_id, eos_id=eos_id,
                rank_mode=args.rank_mode, rank_beta=args.rank_beta, support_beta=args.support_beta,
                risk_readout=readout, motion_beta=args.motion_beta, content_beta=args.content_beta,
                value_readout=value_readout, value_beta=args.value_beta,switch_step=args.switch_step)
            metrics = dict(latent_control.last_metrics)
        else:
            output, _, _ = latent.generate_latent(model, prompt, attention[:, :0], initial_canvas=canvas,
                gen_length=width, block_length=width, steps=args.steps, variant=variant,
                canvas_attention_mask=attention,
                mask_id=mask_id, eos_id=eos_id, eot_id=eot_id, use_eos_harness=False,
                probe_every=1, diversity_beta=args.diversity_beta, stability_beta=args.stability_beta,
                gain_beta=args.gain_beta,
                response_incoming_beta=args.response_incoming_beta,
                progress=None)
            metrics = dict(latent.last_metrics)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        samples = []
        for row, slots in enumerate(positions):
            tokens = output[row, slots].tolist()
            exact = [token == ref for token, ref in zip(tokens, references[row])]
            correct = [tokenizer.decode([token]).strip().casefold() == tokenizer.decode([ref]).strip().casefold()
                       for token, ref in zip(tokens, references[row])]
            samples.append({"id": examples[row]["id"], "tokens": tokens, "references": references[row],
                            "slot_correct": correct, "exact_slot_correct": exact, "correct_slots": sum(correct),
                            "correct_worksheets": sum(all(correct[i:i + 3]) for i in range(0, 12, 3)),
                            "all_correct": all(correct)})
        result = {"variant": variant, "samples": args.samples, "steps": args.steps, "slots_per_canvas": 12,
                  "unfinished_masks": int((output == mask_id).sum()),
                  "slot_accuracy": sum(s["correct_slots"] for s in samples) / (12 * len(samples)),
                  "exact_slot_accuracy": sum(sum(s["exact_slot_correct"]) for s in samples) / (12 * len(samples)),
                  "worksheet_accuracy": sum(s["correct_worksheets"] for s in samples) / (4 * len(samples)),
                  "canvas_accuracy": sum(s["all_correct"] for s in samples) / len(samples),
                  "generation_seconds": seconds, "forward_evaluations_per_canvas": metrics["sequence_evaluations"] / len(samples),
                  "backward_evaluations_per_canvas": metrics.get("backward_sequence_evaluations", 0) / len(samples),
                  "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30}
        summary.append(result)
        payload = {"metrics": result, "samples": samples, "scheduler_metrics": metrics,
                   "seed": args.seed, "diversity_beta": args.diversity_beta, "stability_beta": args.stability_beta,
                   "gain_beta": metrics.get('gain_beta',args.gain_beta),
                   "source_mass_weighting": metrics.get('source_mass_weighting',False),
                   "count_mode": metrics.get('count_mode','eb'),
                   "context_beta": metrics.get('context_beta',0.),
                   "depth_back": metrics.get('depth_back'),
                   "depth_risk_beta": metrics.get('depth_risk_beta',0.),
                   "depth_support_beta": metrics.get('depth_support_beta',0.),
                   "read_beta": metrics.get('read_beta',0.),
                   "attention_layers": metrics.get('attention_layers'),
                   "uncertainty_weighted": metrics.get('uncertainty_weighted'),
                   "consistency_mode": metrics.get('mode'),
                   "finite_candidates": args.finite_candidates,
                   "finite_probe_every": args.finite_probe_every,
                   "finite_branch_batch_size": args.finite_branch_batch_size,
                   "finite_risk_eta": args.finite_risk_eta,
                   "response_incoming_beta": args.response_incoming_beta,
                   "rank_mode": args.rank_mode, "rank_beta": args.rank_beta,
                   "support_beta": args.support_beta,
                   "risk_budget": args.risk_budget, "motion_beta": args.motion_beta,
                   "content_beta": args.content_beta,
                   "switch_step": args.switch_step,
                   "value_beta": args.value_beta, "value_head": str(args.value_head) if args.value_head else None,
                   "value_head_sha256": hashlib.sha256(args.value_head.read_bytes()).hexdigest() if args.value_head else None,
                   "risk_head": str(args.risk_head) if args.risk_head else None,
                   "risk_head_sha256": hashlib.sha256(args.risk_head.read_bytes()).hexdigest() if args.risk_head else None,
                   "code_sha256": hashlib.sha256(Path((latent_consistency if variant in consistency_variants else latent_attention if variant in attention_variants else latent_depth if variant in depth_variants else latent_conditioning if variant in conditioning_variants else latent_information if variant in information_variants else latent_finite if variant in finite_variants else latent).__file__).read_bytes()).hexdigest()}
        (args.out / f"{variant}.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/llada"))
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variants", default="eb_sampler,confidence,entropy,raw_diversity,fisher_diversity,temporal,gradient_gain,response_gain,hybrid_probe")
    parser.add_argument("--diversity-beta", type=float, default=.25)
    parser.add_argument("--stability-beta", type=float, default=.15)
    parser.add_argument("--depth-back", type=int, default=4)
    parser.add_argument("--attention-layers", type=int, default=1)
    parser.add_argument("--gain-beta", type=float, default=.25)
    parser.add_argument("--finite-candidates", type=int, default=4)
    parser.add_argument("--finite-probe-every", type=int, default=1)
    parser.add_argument("--finite-branch-batch-size", type=int, default=128)
    parser.add_argument("--finite-risk-eta", type=float, default=1.)
    parser.add_argument("--response-incoming-beta", type=float, default=.2)
    parser.add_argument("--rank-mode", choices=("risk", "support"), default="risk")
    parser.add_argument("--rank-beta", type=float, default=.15)
    parser.add_argument("--support-beta", type=float, default=.1)
    parser.add_argument("--risk-head", type=Path)
    parser.add_argument("--risk-budget", type=float, default=.03)
    parser.add_argument("--motion-beta", type=float, default=.25)
    parser.add_argument("--content-beta", type=float, default=.25)
    parser.add_argument("--value-head", type=Path)
    parser.add_argument("--value-beta", type=float, default=.25)
    parser.add_argument("--switch-step", type=int, default=6)
    parser.add_argument("--out", type=Path, default=Path("outputs/latent_arithmetic32"))
    args = parser.parse_args()
    if not 1 <= args.samples <= 32 or not 1 <= args.steps <= 12:
        parser.error("samples must be 1–32 and steps must be 1–12")
    run(args)


if __name__ == "__main__":
    main()
