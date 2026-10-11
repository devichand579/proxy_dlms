"""Paired accuracy and compute analysis for the full HumanEval run."""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from schedulerlens.latent_full_bench import METHODS
from schedulerlens.latent_bench import load_documents


def wilson(passed, total):
    z = 1.959963984540054
    p = passed/total
    center = (p+z*z/(2*total))/(1+z*z/total)
    radius = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/(1+z*z/total)
    return [center-radius,center+radius]


def paired(reference, candidate):
    wins = [k for k in reference if candidate[k] and not reference[k]]
    losses = [k for k in reference if reference[k] and not candidate[k]]
    disagreements = len(wins)+len(losses)
    p = min(1.,2*sum(math.comb(disagreements,i) for i in range(min(len(wins),len(losses))+1))/2**disagreements) if disagreements else 1.
    return dict(wins=wins,losses=losses,net_gain=len(wins)-len(losses),
                exact_mcnemar_p=p)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('outputs/latent_full_humaneval164'))
    args = parser.parse_args()
    manifest = json.loads((args.out/'manifest.json').read_text())
    payloads = {m:json.loads((args.out/m/'results.json').read_text()) for m in METHODS}
    expected = manifest['task_ids']
    documents,_ = load_documents(Path(manifest['dataset_arrow']),164)
    syntax = {}
    if len(expected)!=164 or len(set(expected))!=164 or manifest['steps']!=128:
        raise ValueError('invalid full benchmark manifest')
    for m,payload in payloads.items():
        if payload['config']!=manifest or [s['task_id'] for s in payload['samples']]!=expected:
            raise ValueError('mismatched experiment or task ordering')
        if any(type(s['passed']) is not bool for s in payload['samples']):
            raise ValueError('missing correctness result')
        if payload['summary']['passed'] != sum(s['passed'] for s in payload['samples']):
            raise ValueError('incorrect summary score')
        checkpoints = sorted((args.out/m).glob('batch_*.json'))
        if len(checkpoints)!=9:
            raise ValueError('missing batch checkpoints')
        checked_samples = []
        for checkpoint in checkpoints:
            batch = json.loads(checkpoint.read_text())
            metrics = batch['scheduler_metrics']
            if not 0<metrics['steps_taken']<=128:
                raise ValueError('outer step limit violated')
            if sum(s['forward_evaluations'] for s in batch['samples'])!=metrics['sequence_evaluations']:
                raise ValueError('forward work accounting mismatch')
            checked_samples.extend(batch['samples'])
        if checked_samples!=payload['samples']:
            raise ValueError('final results differ from saved batches')
        syntax[m] = {}
        for doc,sample in zip(documents,payload['samples']):
            try:
                ast.parse(doc['prompt']+sample['completion'])
                valid = True
            except (SyntaxError,ValueError):
                valid = False
            syntax[m][sample['task_id']] = valid
    rows = []; comparisons = {}
    names = dict(eb_sampler='EB',eb_target_temporal='EB + temporal Fisher',
                 response_information_readiness='Information + risk + incoming sensitivity')
    for split,indices in [('all_164',range(164)),('tuned_first_20',range(20)),('remaining_144',range(20,164))]:
        outcomes = {}
        for m,payload in payloads.items():
            selected = [payload['samples'][i] for i in indices]
            passed = sum(s['passed'] for s in selected)
            outcomes[m] = {s['task_id']:s['passed'] for s in selected}
            rows.append(dict(split=split,method=m,passed=passed,samples=len(selected),
                pass_at_1=passed/len(selected),wilson_95=wilson(passed,len(selected)),
                forward_evaluations_per_sample=sum(s['forward_evaluations'] for s in selected)/len(selected),
                unfinished_masks=sum(s['unfinished_masks'] for s in selected),
                mean_committed_positions=512-sum(s['unfinished_masks'] for s in selected)/len(selected),
                syntax_valid=sum(syntax[m][s['task_id']] for s in selected),
                syntax_valid_but_failed=sum(syntax[m][s['task_id']] and not s['passed'] for s in selected),
                failure_results=dict(Counter(s['result'] for s in selected if not s['passed']))))
        comparisons[split] = {m:paired(outcomes['eb_sampler'],outcomes[m]) for m in METHODS[1:]}
        comparisons[split]['information_vs_fisher'] = paired(outcomes['eb_target_temporal'],outcomes['response_information_readiness'])
        for m,c in comparisons[split].items():
            reference = 'eb_target_temporal' if m=='information_vs_fisher' else 'eb_sampler'
            c['reference_invalid_syntax_wins'] = [k for k in c['wins'] if not syntax[reference][k]]
            c['reference_valid_syntax_wins'] = [k for k in c['wins'] if syntax[reference][k]]
    validation_path = args.out/'evaluator_validation164.json'
    validation = json.loads(validation_path.read_text()) if validation_path.exists() else None
    (args.out/'analysis.json').write_text(json.dumps(dict(rows=rows,paired_comparisons=comparisons,
        evaluator_validation=validation,report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    with (args.out/'per_task.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=['task_id','split']+[f'{m}_{field}' for m in METHODS for field in ('passed','result','forward_evaluations','unfinished_masks','syntax_valid')])
        writer.writeheader()
        for i,task in enumerate(expected):
            row = dict(task_id=task,split='tuned_first_20' if i<20 else 'remaining_144')
            for m in METHODS:
                row.update({f'{m}_{k}':payloads[m]['samples'][i][k] for k in ('passed','result','forward_evaluations','unfinished_masks')})
                row[f'{m}_syntax_valid'] = syntax[m][task]
            writer.writerow(row)
    lines = ['# Full HumanEval comparison','',
        'Frozen LLaDA-8B, all 164 canonical problems, greedy raw code completion, 512 generation positions, '
        '128 outer denoising steps, matched batches of 20 (last batch 4). '
        'No auxiliary training or learned scheduler policy. Correctness uses the same 3-second evaluator as the earlier runs.', '',
        'The information scheduler uses additional model evaluations inside the 128 steps; compute is reported separately.', '',
        '| Method | Full 164 | First 20 used for tuning | Remaining 144 | Forward evaluations/problem | Generation seconds | Peak GPU GiB |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for m in METHODS:
        scores = [next(r for r in rows if r['method']==m and r['split']==s) for s in ('all_164','tuned_first_20','remaining_144')]
        summary = payloads[m]['summary']
        lines.append('| '+names[m]+' | '+' | '.join(f'{r["passed"]}/{r["samples"]} ({100*r["pass_at_1"]:.2f}%)' for r in scores)
                     +f' | {summary["forward_evaluations_per_sample"]:.2f} | {summary["generation_seconds"]:.1f} | {summary["peak_allocated_GiB"]:.2f} |')
    lines += ['', '## Paired outcomes', '', '| Comparison against EB | Split | Wins | Losses | Net | Exact paired p-value |', '|---|---|---:|---:|---:|---:|']
    for split in ('all_164','remaining_144','tuned_first_20'):
        for m in METHODS[1:]:
            comparison = comparisons[split][m]
            lines.append(f'| {names[m]} | {split} | {len(comparison["wins"])} | {len(comparison["losses"])} | {comparison["net_gain"]:+d} | {comparison["exact_mcnemar_p"]:.4f} |')
    lines += ['', '## Failure breakdown', '', '| Method | Invalid syntax | Valid syntax but failed tests | Passed |', '|---|---:|---:|---:|']
    for m in METHODS:
        r = next(r for r in rows if r['method']==m and r['split']=='all_164')
        lines.append(f'| {names[m]} | {164-r["syntax_valid"]} | {r["syntax_valid_but_failed"]} | {r["passed"]} |')
    lines += ['', '## Commitment counts', '',
              '| Method | Mean positions committed out of 512 | Mean per outer step |', '|---|---:|---:|']
    for m in METHODS:
        r = next(r for r in rows if r['method']==m and r['split']=='all_164')
        lines.append(f'| {names[m]} | {r["mean_committed_positions"]:.2f} | {r["mean_committed_positions"]/128:.2f} |')
    lines += ['', 'These counts include the full canvas, including positions after the decoded end of a function. They measure scheduler behavior, not completion quality.', '']
    full = {m:next(r for r in rows if r['method']==m and r['split']=='all_164') for m in METHODS}
    remaining = {m:next(r for r in rows if r['method']==m and r['split']=='remaining_144') for m in METHODS}
    best = max(METHODS,key=lambda m:(full[m]['passed'],-full[m]['forward_evaluations_per_sample']))
    lines += ['', '## Interpretation', '',
        f'Highest accuracy: **{names[best]}**, {full[best]["passed"]}/164. '
        'Ties are resolved by lower forward-evaluation cost.', '',
        f'Temporal Fisher changes the position ranking while retaining EB’s entropy-based count rule. '
        f'Its measured change against EB is {full["eb_target_temporal"]["passed"]-full["eb_sampler"]["passed"]:+d} problems '
        f'({100*(full["eb_target_temporal"]["pass_at_1"]-full["eb_sampler"]["pass_at_1"]):+.2f} percentage points) overall, '
        f'and {remaining["eb_target_temporal"]["passed"]-remaining["eb_sampler"]["passed"]:+d}/144 outside the tuning set. '
        'Both use 128 forward sequence evaluations per problem. The entropy rule is unchanged, but a different ranking can naturally yield different accepted prefix lengths.', '',
        f'Of Fisher’s {len(comparisons["all_164"]["eb_target_temporal"]["wins"])} wins against EB, '
        f'{len(comparisons["all_164"]["eb_target_temporal"]["reference_invalid_syntax_wins"])} repaired invalid syntax, '
        f'while {len(comparisons["all_164"]["eb_target_temporal"]["reference_valid_syntax_wins"])} had valid syntax under EB but failed tests. '
        'Syntax validity alone does not imply correct reasoning or successful execution.', '',
        f'The information method changes accuracy by '
        f'{full["response_information_readiness"]["passed"]-full["eb_target_temporal"]["passed"]:+d} problems relative to Fisher, '
        f'at {full["response_information_readiness"]["forward_evaluations_per_sample"]/full["eb_target_temporal"]["forward_evaluations_per_sample"]:.2f}× '
        'the forward sequence evaluations. Its extra work includes both intervention branches and unchanged controls; '
        'the number of outer denoising steps remains 128.', '',
        'The absolute scores on all 164 problems should replace the small-set scores when judging code performance. '
        'Inspect paired wins and losses before attributing gains to fewer unfinished masks: unused masked tail positions can coexist with a correct decoded function.', '']
    lines += ['', 'The first 20 problems were repeatedly used to choose these settings. The remaining 144 were not used in those earlier scheduler searches. '
        'This is one deterministic completion per problem; confidence intervals and paired tests describe uncertainty across problems, not variability across generation seeds. '
        'No multiple-comparison correction is applied. Batch shape can affect near-tied ROCm predictions, so all methods use the same normal-pass batching.', '',
        '[Per-problem outcomes](per_task.csv) · [Detailed analysis](analysis.json) · [Generation settings and source hashes](manifest.json)', '']
    if validation is not None:
        lines += [f'Evaluator controls: {validation["canonical_passed"]}/164 reference solutions passed; '
                  f'{validation["negative_passed"]}/164 deliberately failing solutions passed. '
                  'Control solutions were never provided to generation. [Validation](evaluator_validation164.json)', '']
    (args.out/'README.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(11,4),layout='constrained')
    labels = ['EB','EB + Fisher','Information + risk\n+ incoming sensitivity']
    x = np.arange(3)
    for j,(split,label) in enumerate([('all_164','All 164'),('remaining_144','Remaining 144')]):
        values = [next(r for r in rows if r['method']==m and r['split']==split) for m in METHODS]
        axes[0].bar(x+(j-.5)*.36,[100*r['pass_at_1'] for r in values],width=.36,label=label)
    axes[0].set(xticks=x,xticklabels=labels,ylabel='Problems passed (%)',ylim=(0,100),title='HumanEval at 128 denoising steps')
    axes[0].legend(); axes[0].grid(axis='y',alpha=.2)
    axes[1].bar(x,[payloads[m]['summary']['forward_evaluations_per_sample'] for m in METHODS],color=['#4c78a8','#f58518','#54a24b'])
    axes[1].set(xticks=x,xticklabels=labels,ylabel='Forward sequence evaluations / problem',title='Actual model computation')
    axes[1].grid(axis='y',alpha=.2)
    fig.savefig(args.out/'comparison.png',dpi=180)
    fig.savefig(args.out/'comparison.svg')
    print('\n'.join(lines),flush=True)


if __name__=='__main__': main()
