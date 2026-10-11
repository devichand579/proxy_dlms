"""Summarize exploratory one-pass scheduler results without pooling protocols."""
import argparse
import json
from pathlib import Path


def records(roots):
    values=[]
    for root in roots:
        for path in sorted(Path(root).glob('seed*/*/results.json')):
            data=json.loads(path.read_text())
            data['artifact']=str(path.resolve())
            values.append(data)
    return values


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arithmetic',nargs='+',required=True)
    parser.add_argument('--human',nargs='*',default=[])
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    grouped={}
    for item in records(args.arithmetic):
        m=item['metrics'];key=(m['variant'],m['seed'])
        if key in grouped:
            old=grouped[key]
            if old['samples']!=item['samples']:
                raise ValueError(f'nonreproducible duplicate arithmetic result: {key}')
        grouped[key]=item
    methods=sorted({key[0] for key in grouped})
    arithmetic=[]
    for name in methods:
        arms={seed:item for (method,seed),item in grouped.items() if method==name}
        if set(arms)!={0,1}:continue
        metrics=[arms[s]['metrics'] for s in (0,1)]
        arithmetic.append(dict(method=name,seed0=metrics[0]['slot_accuracy'],seed1=metrics[1]['slot_accuracy'],
            mean_accuracy=sum(m['slot_accuracy'] for m in metrics)/2,
            forwards=[m['forward_evaluations_per_sample'] for m in metrics],
            artifacts=[arms[s]['artifact'] for s in (0,1)]))
    arithmetic.sort(key=lambda r:r['mean_accuracy'],reverse=True)
    human_records=records(args.human)
    baseline=next((p for p in human_records if p['metrics']['variant']=='eb_sampler'),None)
    human=[]
    for item in human_records:
        m=item['metrics'];work=item['scheduler_metrics']
        row=dict(method=m['variant'],passed=m['passed'],samples=m['samples'],
            forwards=m['forward_evaluations_per_sample'],seconds=m['generation_seconds'],
            mean_count=m.get('mean_count'),committed_per_sample=512-m['unfinished_masks']/m['samples'],
            forced_singletons=m.get('forced_singletons'),singleton_control=work.get('config',{}).get('singleton',False),
            artifact=item['artifact'])
        if baseline:
            a={s['task_id']:s['passed'] for s in baseline['samples']}
            b={s['task_id']:s['passed'] for s in item['samples']}
            if set(a)!=set(b):raise ValueError('HumanEval task mismatch')
            row.update(wins_vs_eb=[k for k in a if b[k] and not a[k]],
                       losses_vs_eb=[k for k in a if a[k] and not b[k]])
        human.append(row)
    human.sort(key=lambda r:(-r['passed'],r['forwards'],r['seconds']))
    candidates=[r for r in human if r['method']!='eb_sampler' and not r['singleton_control']]
    best_candidate=candidates[0] if candidates else None
    payload=dict(arithmetic=arithmetic,humaneval=human,best_new_adaptive_candidate=best_candidate,
        limitations='Single greedy completions; reused exploratory tasks. Arithmetic slots are correlated. No fresh holdout or global-best claim.')
    (args.out/'analysis.json').write_text(json.dumps(payload,indent=2)+'\n')
    lines=['# One-pass latent objective experiments','',
        'Frozen LLaDA-8B Base on the local AMD GPU. Candidate scoring and count use activations,',
        'existing decoder-gradient directions, and latent Gaussian set objectives. No EB count,',
        'vocabulary confidence/entropy, trained heads, backward passes, or extra model forwards.',
        'Native EB is the baseline only.','',
        'These are exploratory results on reused tasks, not a claim of the best possible algorithm.','',
        '## Findings','']
    if human:
        lines.append(f'Completed {len(arithmetic)} arithmetic configurations on both seeds and {len(human)} HumanEval configurations.')
        if baseline and best_candidate:
            base=baseline['metrics']
            lines += ['',f'Native EB passed **{base["passed"]}/20**. The best new adaptive candidate,',
                f'`{best_candidate["method"]}`, passed **{best_candidate["passed"]}/20** using',
                f'{best_candidate["forwards"]:g} ordinary forwards/problem. Singleton controls are excluded from this candidate ranking.']
        arithmetic_leader=arithmetic[0] if arithmetic else None
        transfer=next((r for r in human if arithmetic_leader and r['method']==arithmetic_leader['method']),None)
        if arithmetic_leader and transfer:
            lines += ['',f'The arithmetic leader scored {arithmetic_leader["seed0"]:.2%} / {arithmetic_leader["seed1"]:.2%},',
                f'but passed **{transfer["passed"]}/20** on HumanEval. Its arithmetic gain did not transfer.']
        lines += ['', 'Counts alone did not repair weak selection: a temporal-risk variant committed the',
            'entire 512-position canvas in 57 steps and passed 0/20. The no-information',
            'context arm beat its matching singleton control 3/20 versus 0/20.',
            'Adding a small information weight reached 4/20; a singleton control with',
            'that exact weight was not run, so its additional gain cannot be assigned',
            'to count alone. None of these candidates warrants promotion to a',
            'full-164 run on the evidence here.', '',
            'The original error term remains an unvalidated commitment-risk proxy; the',
            'Gaussian information and dependence terms describe internal geometry rather',
            'than verified token interactions. Better algebra did not establish better generation.', '',
            'For context, the earlier activation-only singleton control passed 7/20 at',
            '128 forwards/problem, and its costly growth guard passed 8/20 at 1144',
            'forwards/problem. Those are historical controls, not new adaptive one-pass candidates.',
            '[Earlier count study](/root/Strace/outputs/latent_count_ablation_report/README.md).', '']
    lines += ['## Arithmetic: 32 canvases, 384 slots per seed, 12-step cap','',
        '| Method | Seed 0 | Seed 1 | Mean | Forwards/canvas (0 / 1) |',
        '|---|---:|---:|---:|---:|']
    for r in arithmetic:
        lines.append(f'| {r["method"]} | {r["seed0"]:.2%} | {r["seed1"]:.2%} | {r["mean_accuracy"]:.2%} | {r["forwards"][0]:g} / {r["forwards"][1]:g} |')
    lines += ['', 'Arithmetic slots share worksheets and are not independent observations. Duplicate',
              'settings across stages must reproduce their exact per-slot outcomes.','',
              '## HumanEval-20: 128-step cap, 512 positions, standalone batch 20','',
              '| Method | Passed | Forwards/problem | Mean committed | Mean batch count | Generation seconds |',
              '|---|---:|---:|---:|---:|---:|']
    for r in human:
        count=f'{r["mean_count"]:.2f}' if r['mean_count'] is not None else 'not traced'
        lines.append(f'| {r["method"]} | {r["passed"]}/{r["samples"]} | {r["forwards"]:g} | {r["committed_per_sample"]:.2f} | {count} | {r["seconds"]:.1f} |')
    if not human:lines += ['','HumanEval generation has not completed.']
    lines += ['','Scores include one greedy completion per problem. Decoder output is truncated at',
              'the first committed EOS and standard task stops, matching the existing protocol.',
              'Committed positions can include unused output tails; they are not a quality metric.','',
              '## Paired HumanEval outcomes against EB','']
    for r in human:
        if r['method']=='eb_sampler' or 'wins_vs_eb' not in r:continue
        lines.append(f'- {r["method"]}: wins {r["wins_vs_eb"]}; losses {r["losses_vs_eb"]}.')
    lines += ['','## Artifacts','']
    for root in args.arithmetic+args.human:
        p=Path(root).resolve()
        lines.append(f'- [{p.name}]({p})')
    (args.out/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(arithmetic_top=arithmetic[:4],humaneval=human),indent=2))


if __name__=='__main__':main()
