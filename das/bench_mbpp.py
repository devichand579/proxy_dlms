"""MBPP bench (3-shot, greedy) for comparing count rules under a step budget.

Prompt: the standard 3-shot MBPP format (prompt-split tasks 2-4), i.e.
"You are an expert Python programmer, and here is your task: ... Your code should
pass these tests:\\n\\n<asserts>\\n[BEGIN]\\n<code>\\n[DONE]\\n\\n".  The completion is
cut at "[DONE]"; pass@1 = all three test asserts pass under the cached code_eval
executor (3 s timeout), same as the HumanEval bench.  Task index t = test row t
(task_id 11 + t).  Output layout matches bench_gsm8k, so das.compare works.
"""
import argparse
import importlib.util
import json
import os
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
from das.bench import save, parse_tasks
from das.bench_gsm8k import run_method_len

ROOT = Path('/root/.cache/huggingface/datasets/google-research-datasets___mbpp/full/0.0.0/'
            '4bb6404fdc6cacfda99d4ac4205087b89d32030c')
STOPS = ['[DONE]', '\n\n\n\n']


def block(doc, code=None):
    tests = '\n'.join(doc['test_list'])
    s = (f"You are an expert Python programmer, and here is your task: {doc['text']} "
         f"Your code should pass these tests:\n\n{tests}\n[BEGIN]\n")
    return s + (f"{code}\n[DONE]\n\n" if code is not None else '')


def executor():
    cands = sorted(Path('/root/.cache/huggingface/modules/evaluate_modules/metrics/evaluate-metric--code_eval').glob('*/execute.py'))
    spec = importlib.util.spec_from_file_location('mbpp_code_executor', cands[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True, choices=['eb', 'ebf', 'das'])
    ap.add_argument('--name', required=True)
    ap.add_argument('--cfg', default='{}')
    ap.add_argument('--tasks', default='0-499')
    ap.add_argument('--steps', type=int, default=128)
    ap.add_argument('--gen-length', type=int, default=256)
    ap.add_argument('--batch-size', type=int, default=20)
    ap.add_argument('--model', default='models/llada')
    ap.add_argument('--out', type=Path, default=Path('/root/aditya/outputs/mbpp'))
    args = ap.parse_args()
    cfg = json.loads(args.cfg)
    folder = args.out / args.name
    folder.mkdir(parents=True, exist_ok=True)
    config = dict(method=args.method, cfg=cfg, tasks=args.tasks, steps=args.steps, batch_size=args.batch_size,
                  model=args.model, gen_length=args.gen_length, bench='mbpp-3shot')
    cpath = folder / 'config.json'
    if cpath.exists() and json.loads(cpath.read_text()) != config:
        raise ValueError(f'{folder} holds a different config')
    save(cpath, config)

    shots_ds = Dataset.from_file(str(ROOT / 'mbpp-prompt.arrow'))
    test = Dataset.from_file(str(ROOT / 'mbpp-test.arrow'))
    by_id = {d['task_id']: d for d in shots_ds}
    shots = ''.join(block(by_id[i], by_id[i]['code'].replace('\r\n', '\n')) for i in (2, 3, 4))
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
    encoded = {t: tokenizer(shots + block(test[t]), add_special_tokens=False)['input_ids'] for t in tasks}
    run = executor()

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
            out, metrics = run_method_len(args.method, cfg, model, ids, att, args.steps, mask, eos, eot, args.gen_length)
            torch.cuda.synchronize(); elapsed = time.perf_counter() - t0
            fw = metrics.get('forward_work_by_row', [args.steps] * len(idx))
            texts = []
            for r in range(len(idx)):
                tail = out[r, width:]
                e = (tail == eos).nonzero().flatten()
                if len(e):
                    tail = tail[:int(e[0])]
                text = tokenizer.decode(tail, skip_special_tokens=True)
                for s in STOPS:
                    text = text.split(s)[0]
                texts.append(text)
            # Save generations before executing any generated code.
            save(folder / f'generated_{idx[0]:04d}.json', dict(tasks=idx, completions=texts))
            samples = []
            for r, i in enumerate(idx):
                doc = test[i]
                program = doc['test_setup_code'] + '\n' + texts[r] + '\n' + '\n'.join(doc['test_list'])
                res = run.check_correctness(program, 3., i, 0)
                samples.append(dict(task=i, task_id=doc['task_id'], completion=texts[r], passed=bool(res['passed']),
                                    result=res['result'], forward_evaluations=fw[r],
                                    unfinished_masks=int((out[r, width:] == mask).sum())))
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


if __name__ == '__main__':
    main()
