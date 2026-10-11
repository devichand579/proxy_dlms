"""Report fixed-ranking count experiments with diagnostic scopes explicit."""
import argparse
import json
from pathlib import Path


def collect(roots):
    result=[]
    for root in roots:
        for path in sorted(Path(root).glob('seed*/*/results.json')):
            item=json.loads(path.read_text());item['artifact']=str(path.resolve())
            result.append(item)
    return result


def traits(item):
    work=item['scheduler_metrics'];cfg=work.get('config',{})
    source=cfg.get('ranking','eb_confidence')
    controller=cfg.get('controller','eb')
    metric=cfg.get('metric','entropy')
    return dict(ranking=source,controller=controller,metric=metric,
        confidence_ranking=work.get('confidence_scoring',False),
        count_probability_weights=work.get('count_uses_vocabulary_probabilities',controller=='eb'),
        strict_candidate=source=='activation' and controller not in ('eb','singleton') and metric!='fisher_dispersion',
        count_candidate=controller not in ('eb','singleton'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arithmetic',nargs='+',required=True)
    parser.add_argument('--human',nargs='*',default=[])
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    groups={}
    for item in collect(args.arithmetic):
        m=item['metrics'];key=(m['variant'],m['seed'])
        if key in groups and groups[key]['samples']!=item['samples']:
            raise ValueError('duplicate setting changed outcomes')
        groups[key]=item
    arithmetic=[]
    for method in sorted({name for name,_ in groups}):
        if any((method,s) not in groups for s in (0,1)):continue
        arms=[groups[method,s] for s in (0,1)];ms=[p['metrics'] for p in arms]
        arithmetic.append(dict(method=method,seed0=ms[0]['slot_accuracy'],seed1=ms[1]['slot_accuracy'],
            mean_accuracy=sum(m['slot_accuracy'] for m in ms)/2,
            forwards=[m['forward_evaluations_per_sample'] for m in ms],**traits(arms[0])))
    arithmetic.sort(key=lambda r:(-r['mean_accuracy'],sum(r['forwards'])))
    records=collect(args.human)
    baselines={p['metrics']['variant']:p for p in records
               if p['metrics']['variant'] in ('eb_sampler','fisher_eb','fisher_singleton','activation_singleton')}
    human=[]
    for item in records:
        m=item['metrics'];w=item['scheduler_metrics'];row=dict(method=m['variant'],passed=m['passed'],
            forwards=m['forward_evaluations_per_sample'],seconds=m['generation_seconds'],
            mean_count=m.get('mean_count'),committed=512-m['unfinished_masks']/20,
            artifact=item['artifact'],**traits(item))
        counts=[k for step in w.get('reveal_counts',[]) for k in step]
        row['multi_reveal_steps']=sum(k>1 for k in counts)
        for label,baseline in baselines.items():
            a={s['task_id']:s for s in baseline['samples']}
            b={s['task_id']:s for s in item['samples']}
            if set(a)!=set(b):raise ValueError('task mismatch')
            row[label+'_wins']=[k for k in a if b[k]['passed'] and not a[k]['passed']]
            row[label+'_losses']=[k for k in a if a[k]['passed'] and not b[k]['passed']]
            row[label+'_identical_completions']=sum(a[k]['completion']==b[k]['completion'] for k in a)
        human.append(row)
    human.sort(key=lambda r:(-r['passed'],r['forwards'],r['seconds']))
    payload=dict(arithmetic=arithmetic,humaneval=human,
        note='Reused exploratory tasks. Confidence/Fisher rankings and probability-weighted count metrics are explicitly diagnostic.')
    (args.out/'analysis.json').write_text(json.dumps(payload,indent=2)+'\n')
    lines=['# Latent error allowances and response-feedback counts','',
        'One normal LLaDA forward per step. New controllers use actual cached-state changes,',
        'native last-block attention, and frozen decoder-gradient geometry. No policy training,',
        'counterfactual branch forwards, or backward passes. Fisher-geometry diagnostics use',
        'model probabilities as gradient-covariance weights; these are not strict probability-free candidates.', '',
        'The Fisher ranking includes confidence and is held fixed as a diagnostic. Activation',
        'ranking uses attention to observed context minus temporal movement. Source ranking',
        'and count eligibility are reported separately. Allowances are empirical proxies, not',
        'certified bounds on data KL or program error.', '']
    if human:
        lines += ['## Findings','',
            f'Completed {len(arithmetic)} arithmetic settings on two seeds and {len(human)} HumanEval-20 settings.','']
        for method in ('eb_sampler','fisher_eb','fisher_singleton','activation_singleton'):
            row=next((r for r in human if r['method']==method),None)
            if row:lines.append(f'- {method}: {row["passed"]}/20, {row["forwards"]:g} forwards/problem.')
        lines += ['',
            'The probability-weighted Fisher-geometry counts matched 14/20 with independent',
            'non-EB count rules, but all 20 decoded completions were identical to Fisher',
            'singleton. Their additional reveals did not change the decoded programs, so',
            'these results do not establish a quality benefit from adaptive count.', '',
            'The probability-free grounded feedback count also matched 14/20 under Fisher',
            'ranking, but committed only 128.25 positions/problem and made five two-token',
            'decisions across all 20 trajectories. It effectively reproduced singleton.', '',
            'The largest tested Fisher-ranked temporal allowance dropped to 11/20; relative',
            'feedback reached 13/20. Larger groups did not reliably improve code.', '',
            'Under activation-only ranking, grounded feedback retained 7/20 with nearly',
            'singleton counts. The other activation-ranked count arms scored 6/20. No',
            'strict probability-free adaptive candidate improved on the 7/20 scoring control.', '',
            'A full-164 promotion is not supported by this pilot: tied small-set results',
            'mostly reproduce the same programs or reveal more unused tail positions.', '',
            'The practical next question is whether an interaction estimate identifies',
            'safe parallelism among actual output-body decisions. Motion magnitude alone',
            'is not wrong-commitment risk, and last-block attention alone is not token TC.', '',
            'Exact oracle examples, separately from neural results, demonstrate entropy-bound',
            'conservatism for independent bits and the failure of pairwise MI on parity.',
            '[Toy sampling diagnostics](/root/Strace/outputs/latent_allowance_toys/diagnostic.json).','']
    lines += [
        '## Arithmetic, two seeds, 32 canvases / 384 answers per seed, 12-step cap','',
        '| Method | Seed 0 | Seed 1 | Mean | Forwards/canvas | Ranking | Count metric |',
        '|---|---:|---:|---:|---:|---|---|']
    for r in arithmetic:
        lines.append(f'| {r["method"]} | {r["seed0"]:.2%} | {r["seed1"]:.2%} | {r["mean_accuracy"]:.2%} | {r["forwards"][0]:g}/{r["forwards"][1]:g} | {r["ranking"]} | {r["metric"]} |')
    lines+=['','Arithmetic component worksheets are reused and correlated; small gains do not establish robustness.','',
        '## HumanEval-20, 512 positions, 128-step cap, matched standalone batch 20','',
        '| Method | Passed | Forwards/problem | Mean committed | Count >1 events | Scope |',
        '|---|---:|---:|---:|---:|---|']
    for r in human:
        scope=('strict candidate' if r['strict_candidate'] else
               'activation ranking / Fisher-weighted count diagnostic' if not r['confidence_ranking'] and r['metric']=='fisher_dispersion' else
               'activation singleton control' if r['controller']=='singleton' and not r['confidence_ranking'] else 'confidence/Fisher diagnostic or baseline')
        lines.append(f'| {r["method"]} | {r["passed"]}/20 | {r["forwards"]:g} | {r["committed"]:.2f} | {r["multi_reveal_steps"]} | {scope} |')
    if not human:lines+=['','No completed HumanEval runs yet.']
    lines+=['','## Fixed-ranking paired comparisons','']
    for r in human:
        if not r['count_candidate']:continue
        reference='fisher_singleton' if r['ranking']=='diagnostic_fisher' else 'activation_singleton'
        if reference+'_wins' not in r:continue
        lines.append(f'- {r["method"]} versus {reference}: wins {r[reference+"_wins"]}; losses {r[reference+"_losses"]}; identical completions {r[reference+"_identical_completions"]}/20.')
    lines+=['','## Artifacts and reproduction','']
    for root in args.arithmetic+args.human:
        p=Path(root).resolve();lines.append(f'- [{p.name}]({p})')
    lines+=['','Each experiment stores source hashes and source snapshots, fixed decoder geometry,',
        'per-step count decisions, state checkpoints, completions, and fresh-child correctness',
        'outcomes. No hidden tests or reference answers enter scheduling.','',
        'Run new directories with `python -m schedulerlens.latent_allowance_bench --task arithmetic`',
        'or `--task humaneval`, supplying `--out` and an explicit `--plans` list.','',
        'Feedback responds after a commit and cannot undo earlier errors. Attention is an',
        'interaction proxy, not measured token mutual information. Observed displacement can',
        'be beneficial learning from context; its magnitude alone is not an error certificate.']
    (args.out/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(arithmetic_top=arithmetic[:5],humaneval=human),indent=2))


if __name__=='__main__':main()
