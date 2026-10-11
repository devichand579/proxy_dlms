"""Evaluate exact saved monotone canvases at earlier steps, without model passes."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from schedulerlens.latent_bench import code_results, decode, load_documents


def reconstruct(sample, positions, row, budget, mask=126336):
    canvas = [mask] * len(sample['token_ids'])
    seen = set()
    for group in positions[:budget]:
        for index in group[row]:
            if index in seen: raise ValueError('replay requires monotone unique commits')
            seen.add(index)
            canvas[index] = sample['token_ids'][index]
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--additional-roots', type=Path, nargs='*', default=[])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--budgets', type=int, nargs='+', default=[32, 48, 64, 96, 128])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained('models/llada', trust_remote_code=True, local_files_only=True)
    documents, _ = load_documents(None, 20)
    docs = {d['task_id']: d for d in documents}
    cache_path = args.out / 'correctness_cache.json'
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else []
    outcomes = {(r['task_id'], r['completion']): r['outcome'] for r in cache}
    results = []
    paths = []
    for root in [args.root] + args.additional_roots:
        source = (root / 'sources' / 'schedulers__latent_context_flow.py').read_text()
        for path in sorted(root.glob('seed*/*/results.json')):
            payload = json.loads(path.read_text())
            if 'context_flow' in payload['scheduler_metrics']['ranking'] and 'leaf=inputs[0].detach()' in source:
                continue
            paths.append(path)
    for path in paths:
        p = json.loads(path.read_text())
        samples, positions = p['samples'], p['scheduler_metrics']['reveal_positions']
        for sample in samples:
            outcomes[(sample['task_id'], sample['completion'])] = {
                k: sample[k] for k in ('task_id', 'passed', 'result')}
        for budget in args.budgets:
            canvas = torch.tensor([reconstruct(s, positions, row, budget) for row, s in enumerate(samples)])
            texts = decode(tokenizer, canvas, 0, 126081)
            records = []
            for sample, text in zip(samples, texts):
                key = (sample['task_id'], text)
                if key not in outcomes:
                    outcomes[key] = code_results([docs[sample['task_id']]], [text])[0]
                records.append(dict(completion=text, **outcomes[key]))
            if budget >= len(positions):
                assert texts == [s['completion'] for s in samples], path
            result = dict(method=p['metrics']['variant'], budget=budget,
                          passed=sum(r['passed'] for r in records), samples=records,
                          effective_steps=min(budget, len(positions)))
            results.append(result)
            print(f'{result["method"]} step {budget}: {result["passed"]}/20', flush=True)
        cache_path.write_text(json.dumps([
            dict(task_id=k[0], completion=k[1], outcome=v) for k, v in outcomes.items()], indent=2) + '\n')
    (args.out / 'analysis.json').write_text(json.dumps(dict(studies=results,
        note='Exact replay of monotone commits from a 128-cap trajectory. No new forwards, no test-driven scheduling or stopping. Lower-budget outcomes can differ despite identical final programs.'), indent=2) + '\n')
    names = sorted({r['method'] for r in results})
    index = {(r['method'], r['budget']): r['passed'] for r in results}
    lines = ['# Correctness at saved earlier steps', '',
        'These canvases replay recorded commits; reference answers never enter generation.',
        'This measures a fixed earlier-step exit on the saved trajectory, not an oracle',
        'stopping policy. Partial backward work remains necessary for context-flow arms.', '',
        '| Method | ' + ' | '.join(f'Step {b}' for b in args.budgets) + ' |',
        '|---|' + '---:|' * len(args.budgets)]
    for name in names:
        lines.append('| ' + name + ' | ' + ' | '.join(f'{index[name, b]}/20' for b in args.budgets) + ' |')
    (args.out / 'README.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__': main()
