"""GSM8K bench (4-shot, greedy) for comparing count rules under a step budget.

Prompt: 4 train examples as "Question: ...\\nAnswer: ...\\n\\n", then the test
question.  Completion is cut at the next "Question:" / blank-line block; the
prediction is the number after "####", else the last number in the completion.
Same batching / checkpoint layout as das.bench, so das.compare works on the
output directory.
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import torch

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from datasets import Dataset
from das import model_path
from load_model import load_model_and_tokenizer, infer_diffusion_token_ids
from schedulerlens.latent_bench import set_seed
from das.bench import run_method, save, parse_tasks

ROOT = Path('/root/.cache/huggingface/datasets/gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866')
STOPS = ['Question:', '\n\n\n']


def number(s):
    s = s.replace(',', '').replace('$', '')
    nums = re.findall(r'-?\d+(?:\.\d+)?', s)
    return nums[-1] if nums else None


def gold(answer):
    return number(answer.split('####')[-1])


def predict(text):
    if '####' in text:
        n = number(text.split('####')[1].split('\n')[0])
        if n is not None:
            return n
    return number(text)


def same(a, b):
    try:
        return a is not None and abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True, choices=['eb', 'ebf', 'das'])
    ap.add_argument('--name', required=True)
    ap.add_argument('--cfg', default='{}')
    ap.add_argument('--tasks', default='0-299')
    ap.add_argument('--steps', type=int, default=128)
    ap.add_argument('--gen-length', type=int, default=256)
    ap.add_argument('--batch-size', type=int, default=20)
    ap.add_argument('--model', default='models/llada')
    ap.add_argument('--out', type=Path, default=Path('/root/aditya/outputs/gsm8k'))
    args = ap.parse_args()
    cfg = json.loads(args.cfg)
    folder = args.out / args.name
    folder.mkdir(parents=True, exist_ok=True)
    config = dict(method=args.method, cfg=cfg, tasks=args.tasks, steps=args.steps, batch_size=args.batch_size,
                  model=args.model, gen_length=args.gen_length, bench='gsm8k-4shot')
    cpath = folder / 'config.json'
    if cpath.exists() and json.loads(cpath.read_text()) != config:
        raise ValueError(f'{folder} holds a different config')
    save(cpath, config)

    train = Dataset.from_file(str(ROOT / 'gsm8k-train.arrow'))
    test = Dataset.from_file(str(ROOT / 'gsm8k-test.arrow'))
    shots = ''.join(f"Question: {train[i]['question']}\nAnswer: {train[i]['answer']}\n\n" for i in range(4))
    tasks = parse_tasks(args.tasks)
    bs = args.batch_size
    batches = {}
    for t in tasks:
        batches.setdefault(t // bs, []).append(t)

    model, tokenizer = load_model_and_tokenizer(model_path(args.model), device='cuda', compile_model=False)
    mask, eos, eot = infer_diffusion_token_ids(model_path(args.model))
    if cfg.get('count') == 'pace':
        cfg['pad_ids'] = [i for i in range(len(tokenizer)) if i != mask
                          and tokenizer.decode([i]).strip() == '' and tokenizer.decode([i]) != '']
    encoded = {t: tokenizer(shots + f"Question: {test[t]['question']}\nAnswer:", add_special_tokens=False)['input_ids']
               for t in tasks}

    all_samples = []
    for key in sorted(batches):
        idx = batches[key]
        ck = folder / f'batch_{idx[0]:04d}.json'
        if ck.exists():
            payload = json.loads(ck.read_text())
        else:
            set_seed(1234)
            width = max(len(encoded[i]) for i in idx)
            ids = torch.full((len(idx), width), tokenizer.pad_token_id, device='cuda', dtype=torch.long)
            att = torch.zeros_like(ids)
            for r, i in enumerate(idx):
                ids[r, -len(encoded[i]):] = torch.tensor(encoded[i], device='cuda')
                att[r, -len(encoded[i]):] = 1
            torch.cuda.synchronize(); t0 = time.perf_counter()
            # run_method hard-codes gen_length=512; GSM8K uses its own length.
            out, metrics = run_method_len(args.method, cfg, model, ids, att, args.steps, mask, eos, eot, args.gen_length)
            torch.cuda.synchronize(); elapsed = time.perf_counter() - t0
            samples = []
            fw = metrics.get('forward_work_by_row', [args.steps] * len(idx))
            for r, i in enumerate(idx):
                tail = out[r, width:]
                e = (tail == eos).nonzero().flatten()
                if len(e):
                    tail = tail[:int(e[0])]
                text = tokenizer.decode(tail, skip_special_tokens=True)
                for s in STOPS:
                    text = text.split(s)[0]
                p, g = predict(text), gold(test[i]['answer'])
                samples.append(dict(task=i, completion=text, pred=p, gold=g, passed=same(p, g),
                                    forward_evaluations=fw[r], unfinished_masks=int((out[r, width:] == mask).sum())))
            payload = dict(tasks=idx, samples=samples, generation_seconds=elapsed,
                           metrics={k: v for k, v in metrics.items() if not isinstance(v, torch.Tensor)})
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


def run_method_len(method, cfg, model, ids, att, steps, mask, eos, eot, gen_length):
    from schedulers import latent
    from das import das_sampler
    if method == 'ebf':
        out, _, _ = latent.generate_latent(model, ids, att, variant='eb_target_temporal',
            steps=steps, gen_length=gen_length, block_length=cfg.get("block_length", gen_length), temperature=0., mask_id=mask,
            eos_id=eos, eot_id=eot, use_eos_harness=False, diversity_beta=.25, stability_beta=.15,
            probe_every=8, fisher_vocab=128,
            logits_eos_inf=cfg.get('b44', False), confidence_eos_eot_inf=cfg.get('b44', False))
        m = dict(latent.last_metrics)
        m.setdefault('forward_work_by_row', [m['steps_taken']] * ids.shape[0])
        return out, m
    if method == 'das':
        return das_sampler.generate_das(model, ids, att, steps=steps, gen_length=gen_length,
                                        mask_id=mask, eos_id=eos, eot_id=eot, **cfg)
    return run_method(method, cfg, model, ids, att, steps, mask, eos, eot)


if __name__ == '__main__':
    main()
