"""Screening / full HumanEval bench for the DAS project.

Reuses the mentor's harness pieces (prompt format, decoding, 3 s executor) so
numbers line up with the mentor's latent_full_humaneval164 run. Run via das/run.sh
(copied Strace modules come from das/vendor; model weights from MODEL_ROOT).

Batches are formed from absolute task offsets that are multiples of the batch
size, so batch-20 runs on tasks 20-59 reproduce the mentor's batch composition.

Each method run writes <out>/<run_name>/results.json with per-task outcomes.
Generations are saved before any generated code is executed.
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from das import model_path
from load_model import load_model_and_tokenizer, infer_diffusion_token_ids
from schedulerlens.latent_bench import load_documents, code_results, decode, set_seed
from schedulers.eb_sampler import generate_EB
from schedulers import latent
from das import das_sampler


def save(path, payload):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2) + '\n')
    tmp.replace(path)


def parse_tasks(spec):
    lo, hi = spec.split('-')
    return list(range(int(lo), int(hi) + 1))


def run_method(method, cfg, model, ids, attention, steps, mask, eos, eot):
    common = dict(steps=steps, gen_length=512, mask_id=mask, eos_id=eos)
    n = ids.shape[0]
    if method == 'eb':
        out, taken, _ = generate_EB(model, ids, attention, block_length=512, temperature=0.,
            eot_id=eot, use_eos_harness=False, gamma=.1, E='confidence',
            upload_to_wandb=False, **common)
        return out, dict(steps_taken=taken, forward_work_by_row=[taken] * n)
    if method == 'ebf':
        out, _, _ = latent.generate_latent(model, ids, attention, variant='eb_target_temporal',
            block_length=cfg.get("block_length", 512), temperature=0., eot_id=eot, use_eos_harness=False,
            diversity_beta=.25, stability_beta=.15, probe_every=8, fisher_vocab=128,
            logits_eos_inf=cfg.get('b44', False), confidence_eos_eot_inf=cfg.get('b44', False), **common)
        m = dict(latent.last_metrics)
        m.setdefault('forward_work_by_row', [m['steps_taken']] * n)
        return out, m
    if method == 'das':
        out, m = das_sampler.generate_das(model, ids, attention, eot_id=eot, **common, **cfg)
        return out, m
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True, choices=['eb', 'ebf', 'das'])
    ap.add_argument('--name', required=True)
    ap.add_argument('--cfg', default='{}', help='JSON kwargs for das')
    ap.add_argument('--tasks', default='20-59')
    ap.add_argument('--steps', type=int, default=128)
    ap.add_argument('--batch-size', type=int, default=20)
    ap.add_argument('--out', type=Path, default=Path('/root/aditya/outputs/screen'))
    ap.add_argument('--model', default='models/llada')
    args = ap.parse_args()
    cfg = json.loads(args.cfg)
    folder = args.out / args.name
    folder.mkdir(parents=True, exist_ok=True)
    config = dict(method=args.method, cfg=cfg, tasks=args.tasks, steps=args.steps,
                  batch_size=args.batch_size, model=args.model, gen_length=512)
    cpath = folder / 'config.json'
    if cpath.exists() and json.loads(cpath.read_text()) != config:
        raise ValueError(f'{folder} holds a different config')
    save(cpath, config)

    documents, _ = load_documents(None, 164)
    tasks = parse_tasks(args.tasks)
    bs = args.batch_size
    batches = {}
    for t in tasks:
        batches.setdefault(t // bs, []).append(t)

    model, tokenizer = load_model_and_tokenizer(model_path(args.model), device='cuda', compile_model=False)
    mask, eos, eot = infer_diffusion_token_ids(model_path(args.model))
    if cfg.get('count') == 'pace':
        cfg['pad_ids'] = [i for i in range(len(tokenizer)) if i not in (mask,)
                          and tokenizer.decode([i]).strip() == '' and tokenizer.decode([i]) != '']
        print(f'pace: {len(cfg["pad_ids"])} whitespace-only token ids', flush=True)
    encoded = {t: tokenizer(documents[t]['prompt'], add_special_tokens=False)['input_ids'] for t in tasks}

    all_samples = []
    for key in sorted(batches):
        idx = batches[key]
        ck = folder / f'batch_{idx[0]:03d}.json'
        if ck.exists():
            payload = json.loads(ck.read_text())
            print(f'RESUME {args.name} {idx[0]}-{idx[-1]}', flush=True)
        else:
            set_seed(1234)
            width = max(len(encoded[i]) for i in idx)
            ids = torch.full((len(idx), width), tokenizer.pad_token_id, device='cuda', dtype=torch.long)
            att = torch.zeros_like(ids)
            for r, i in enumerate(idx):
                ids[r, -len(encoded[i]):] = torch.tensor(encoded[i], device='cuda')
                att[r, -len(encoded[i]):] = 1
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            t0 = time.perf_counter()
            out, metrics = run_method(args.method, cfg, model, ids, att, args.steps, mask, eos, eot)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            texts = decode(tokenizer, out, width, eos)
            fw = metrics.get('forward_work_by_row', [args.steps] * len(idx))
            samples = [dict(task=i, task_id=documents[i]['task_id'], completion=txt,
                            unfinished_masks=int((out[r, width:] == mask).sum()),
                            forward_evaluations=fw[r])
                       for r, (i, txt) in enumerate(zip(idx, texts))]
            payload = dict(tasks=idx, samples=samples, generation_seconds=elapsed,
                           peak_GiB=torch.cuda.max_memory_allocated() / 2**30,
                           metrics={k: v for k, v in metrics.items() if not isinstance(v, torch.Tensor)})
            save(folder / f'generated_{idx[0]:03d}.json', payload)
            for s, o in zip(samples, code_results([documents[i] for i in idx], texts)):
                s.update(passed=o['passed'], result=o['result'])
            save(ck, payload)
        all_samples += payload['samples']
        print(f'BATCH {args.name} {idx[0]}-{idx[-1]}: {sum(s["passed"] for s in payload["samples"])}/{len(idx)} '
              f'in {payload["generation_seconds"]:.0f}s', flush=True)
    n = len(all_samples)
    summary = dict(name=args.name, passed=sum(s['passed'] for s in all_samples), n=n,
                   nfe_mean=sum(s['forward_evaluations'] for s in all_samples) / n,
                   nfe_max=max(s['forward_evaluations'] for s in all_samples),
                   unfinished_mean=sum(s['unfinished_masks'] for s in all_samples) / n)
    save(folder / 'results.json', dict(summary=summary, config=config,
         passed={s['task']: s['passed'] for s in all_samples}))
    print('RESULT ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
