"""Rebuild a protocol-aware scoreboard from completed scheduler experiments."""
import argparse
import csv
import json
from pathlib import Path


LEARNED_METHODS = {
    'eb_content','content_entropy','eb_learned','learned_eb','learned_learned',
    'learned_latent','late_learned_latent','eb_value','value_entropy',
    'risk_count','risk_both','content_rank','content_budget','value_rank',
    'value_budget','hybrid_both','late_hybrid_both',
}


def activation_candidate(work):
    return (all(work.get(key) is False for key in
                ('confidence_scoring','entropy_scoring','eb_count','auxiliary_training'))
            and work.get('controller') not in ('fixed','singleton')
            and not (work.get('config') or {}).get('singleton',False)
            and (work.get('config') or {}).get('candidate_region') != 'frontier'
            and work.get('count_uses_vocabulary_probabilities') is not True)


def outcomes(payload):
    samples = payload.get('samples')
    if not isinstance(samples,list) or not samples or 'task_id' not in samples[0]:
        return None
    if any(not isinstance(s.get('passed'),bool) for s in samples):
        raise ValueError('missing boolean code-test outcome')
    result = {s['task_id']:s['passed'] for s in samples}
    if len(result) != len(samples):
        raise ValueError('duplicate task identifiers')
    return result


def human_results(root):
    rows = []
    for path in sorted(root.rglob('results.json')):
        data = json.loads(path.read_text())
        arms = [(path,data)] if 'config' in data else [(p,json.loads(p.read_text())) for p in path.parent.glob('*.json') if p != path]
        for arm_path, arm in arms:
            result = outcomes(arm)
            if result is None: continue
            config, summary = arm.get('config'),arm.get('summary')
            if config and 'variant' in config:
                method = config['variant']; samples = len(result)
                steps, length = config['steps'],config['gen_length']
                batch = config['batch_size']
                score = arm['metrics']['passed']
                unfinished = arm['metrics'].get('unfinished_masks')
                forward = arm['metrics']['sequence_evaluations']/samples
                backward = arm['metrics'].get('backward_sequence_evaluations',0)/samples
                settings = {k:v for k,v in config.items() if k not in ('variant','steps','gen_length','batch_size','seed')}
            elif config and summary and 'method' in summary:
                method = summary['method']; samples = len(result)
                steps,length,batch = config['steps'],config['gen_length'],config['batch_size']
                score = summary['passed']; unfinished = summary.get('unfinished_masks')
                forward = summary['forward_evaluations_per_sample']
                backward = summary['backward_evaluations_per_sample']
                keys = ['block_length']
                if method=='eb_sampler': keys += ['gamma']
                else: keys += ['stability_beta','fisher_vocab']
                if method=='response_information_readiness':
                    keys += ['candidates','probe_every','branch_batch_size','gain_beta','risk_eta','incoming_beta','auxiliary_training']
                settings = {k:config[k] for k in keys}
            elif summary and 'passed' in summary:
                method = summary.get('cell',summary.get('name','eb_target_temporal'))
                samples, steps, length = len(result),data['steps'],data['gen_length']
                batch, score = data['total_batch_size'],summary['passed']
                unfinished = summary.get('unfinished_masks')
                work = data.get('scheduler_metrics',{})
                forward = summary.get('forward_evaluations_per_sample',work.get('sequence_evaluations',batch*steps)/batch)
                backward = summary.get('backward_evaluations_per_sample',work.get('backward_sequence_evaluations',0)/batch)
                settings = {k:v for k,v in summary.items() if k in ('risk_budget','stability_beta','rank_beta','value_beta','gain_beta','pool_min','temporal','risk_eta','readiness_gate','target_balance','pool_balance','incoming_beta','prefix_targets','prefix_pool','source_mass_weighting','count_mode','count_budget','context_beta','depth_risk_beta','depth_support_beta','read_beta','uncertainty_weighted','enabled','mode')}
                for k in ('rank_beta','rank_mode','coupling_beta','support_beta','content_beta','motion_beta','switch_step'):
                    if k in data and k not in settings: settings[k] = data[k]
                for k in ('candidates','probe_every','branch_batch_size','matched_probe_control','target_vocab','depth_back','attention_layers'):
                    if k in data: settings[k] = data[k]
            else: continue
            if score != sum(result.values()):
                raise ValueError(f'{arm_path}: reported pass count differs from outcomes')
            work = arm.get('scheduler_metrics',{})
            # Retain the faulty partial-gradient pilot on disk, not in rankings.
            if 'context_flow' in work.get('ranking',''):
                source = next((parent/'sources'/'schedulers__latent_context_flow.py'
                    for parent in arm_path.parents
                    if (parent/'sources'/'schedulers__latent_context_flow.py').exists()), None)
                if source and 'leaf=inputs[0].detach()' in source.read_text(): continue
            partial_backward = work.get('partial_backward_sequence_evaluations',0)/samples
            activation_only = activation_candidate(work)
            rows.append({'method':method,'eligible_intrinsic_only':method not in LEARNED_METHODS,'settings':json.dumps(settings,sort_keys=True),
                'eligible_activation_only':activation_only,
                'steps':steps,'gen_length':length,'samples':samples,'total_batch_size':batch,
                'standalone_batch20':batch == samples == 20,'passed':score,'pass_rate':score/samples,
                'forward_evaluations_per_sample':forward,'backward_evaluations_per_sample':backward,
                'partial_backward_evaluations_per_sample':partial_backward,
                'unfinished_masks':unfinished,'result_path':str(arm_path.resolve()),'outcomes':result,
                'modified':arm_path.stat().st_mtime})
    return rows


def arithmetic_results(root):
    rows = []
    for path in sorted(root.rglob('summary.json')):
        data = json.loads(path.read_text())
        if not isinstance(data,list) or not data or 'slot_accuracy' not in data[0]: continue
        baseline = next((a['slot_accuracy'] for a in data if a['variant'] == 'eb_sampler'),None)
        for arm in data:
            artifact = path.parent/(arm['variant']+'.json')
            if not artifact.exists(): continue
            detail = json.loads(artifact.read_text())
            work = detail.get('scheduler_metrics',{})
            activation_only = activation_candidate(work)
            rows.append({'method':arm['variant'],'eligible_intrinsic_only':arm['variant'] not in LEARNED_METHODS,
                'eligible_activation_only':activation_only,'seed':detail.get('seed',arm.get('seed')),'samples':arm['samples'],
                'probe_candidates':detail.get('finite_candidates',detail.get('candidates')),'probe_every':detail.get('finite_probe_every'),
                'branch_batch_size':detail.get('finite_branch_batch_size'),
                'source_mass_weighting':detail.get('source_mass_weighting',False),
                'count_mode':detail.get('count_mode','eb'),
                'depth_back':detail.get('depth_back'),
                'depth_risk_beta':detail.get('depth_risk_beta',0.),
                'depth_support_beta':detail.get('depth_support_beta',0.),
                'read_beta':detail.get('read_beta',0.),
                'attention_layers':detail.get('attention_layers'),
                'uncertainty_weighted':detail.get('uncertainty_weighted'),
                'steps':arm['steps'],'slot_accuracy':arm['slot_accuracy'],'worksheet_accuracy':arm['worksheet_accuracy'],
                'forward_evaluations_per_canvas':arm.get('forward_evaluations_per_canvas'),
                'backward_evaluations_per_canvas':arm.get('backward_evaluations_per_canvas'),
                'slot_delta_vs_eb':None if baseline is None else arm['slot_accuracy']-baseline,
                'unfinished_masks':arm.get('unfinished_masks'),'result_path':str(artifact.resolve())})
    return rows


def write_csv(path,rows):
    if not rows: return
    with path.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('outputs'))
    parser.add_argument('--out',type=Path,default=Path('outputs/scheduler_scoreboard'))
    args = parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    human, small = human_results(args.root),arithmetic_results(args.root)
    expected = {f'HumanEval/{i}' for i in range(20)}
    baselines = [r for r in human if r['method']=='eb_sampler' and r['standalone_batch20']
                 and r['steps']==128 and r['gen_length']==512 and set(r['outcomes'])==expected
                 and json.loads(r['settings'])['block_length']==512]
    baseline = max(baselines,key=lambda r:r['modified']) if baselines else None
    full_expected = {f'HumanEval/{i}' for i in range(164)}
    full_baselines = [r for r in human if r['method']=='eb_sampler' and r['steps']==128
                      and r['gen_length']==512 and set(r['outcomes'])==full_expected]
    full_baseline = max(full_baselines,key=lambda r:r['modified']) if full_baselines else None
    for row in human:
        reference = full_baseline if set(row['outcomes'])==full_expected else baseline
        matched = reference and row['steps']==128 and row['gen_length']==512 and set(row['outcomes'])==set(reference['outcomes'])
        row['wins_vs_native_eb'] = sum(row['outcomes'][k] and not reference['outcomes'][k] for k in reference['outcomes']) if matched else None
        row['losses_vs_native_eb'] = sum(reference['outcomes'][k] and not row['outcomes'][k] for k in reference['outcomes']) if matched else None
    human.sort(key=lambda r:(r['gen_length'],r['standalone_batch20'],r['passed'],
                           -r['forward_evaluations_per_sample'],-r['backward_evaluations_per_sample'],r['modified']),reverse=True)
    write_csv(args.out/'humaneval.csv',[{k:v for k,v in r.items() if k not in ('outcomes','modified')} for r in human])
    write_csv(args.out/'arithmetic.csv',small)
    lines = ['# Completed scheduler results','',
        'HumanEval results use greedy code completion. The first 20 tasks are reused for tuning; these scores do not establish generalization.',
        'Current constraint: no trained scheduler heads or policies. Tables below contain only intrinsic methods computed from the frozen model. Historical learned-head results remain in the CSV files with eligible_intrinsic_only=False.',
        'Batch-20 checks and packed exploratory runs are separate. The 256-position canvas is kept separate from the main 512-position comparison.','']
    objective_report = args.root/'latent_objective_report'/'README.md'
    if objective_report.exists():
        lines += ['## One-forward three-term study','',
                  'Activation and decoder-gradient risk, Gaussian information/dependence, and independent adaptive counts.',
                  f'[Complete arithmetic/HumanEval results and limitations]({objective_report.resolve()}).','']
    allowance_report = args.root/'latent_allowance_report'/'README.md'
    if allowance_report.exists():
        lines += ['## Cached-response and error-allowance count study','',
                  'Fixed-ranking diagnostics, probability-free count arms, and separately labelled Fisher-geometry controls.',
                  f'[Complete results, paired outcomes, and limitations]({allowance_report.resolve()}).','']
    latent_report = args.root/'latent_only_ablation'/'README.md'
    body_report = args.root/'latent_body_progress_report'/'README.md'
    if body_report.exists():
        lines += ['## Useful code-body progress and local gradient scoring','',
                  'Independent count diagnostics, corrected final-MLP gradients, and retrospective commitment timing.',
                  f'[Outcome, work accounting, and paired comparisons]({body_report.resolve()}).','']
    if latent_report.exists():
        lines += ['The latest requirement excludes EB counts and vocabulary confidence/entropy from both scheduler decisions. '
                  'Historical intrinsic results below do not meet that stricter requirement. '
                  f'[Activation-only pilot results]({latent_report.resolve()}). '
                  'The CSV records eligibility under both constraints.','']
    if full_baseline:
        lines += ['## Full HumanEval-164, 128 steps, 512 positions','',
                  '| Method | Passed | Accuracy | Wins / losses vs EB | Forward evaluations/problem |',
                  '|---|---:|---:|---:|---:|']
        for row in human:
            if not row['eligible_intrinsic_only'] or row['steps']!=128 or row['gen_length']!=512 or set(row['outcomes'])!=full_expected: continue
            lines.append(f"| {row['method']} | {row['passed']}/164 | {100*row['pass_rate']:.2f}% | {row['wins_vs_native_eb']} / {row['losses_vs_native_eb']} | {row['forward_evaluations_per_sample']:.2f} |")
        full_report = Path(full_baseline['result_path']).parent.parent/'README.md'
        lines += ['',f'[Full report, remaining-144 results, compute, and paired analysis]({full_report}).','']
    if baseline:
        lines += [f"Native EB reference: **{baseline['passed']}/20**, 128 steps, 512 positions. [Artifact]({baseline['result_path']}).",'']
    lines += ['## Standalone HumanEval-20, 128 steps, 512 positions','',
        '| Method | Passed | Wins / losses vs EB | Forward / full backward / partial backward evaluations | Remaining masks | Artifact |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    seen = set()
    for row in human:
        if not row['eligible_intrinsic_only']: continue
        if not row['standalone_batch20'] or row['steps']!=128 or row['gen_length']!=512 or set(row['outcomes'])!=expected: continue
        key = row['method'],row['settings']
        if key in seen: continue
        seen.add(key)
        settings = json.loads(row['settings']); labels = []
        if settings.get('block_length',512) != 512: labels.append(f"block={settings['block_length']}")
        if 'temporal' in row['method']: labels.append(f"beta={settings.get('stability_beta',settings.get('rank_beta'))}")
        if 'gradient' in row['method']: labels.append(f"gain={settings.get('gain_beta',.25)}")
        if settings.get('gradient_risk_eta'): labels.append(f"risk={settings['gradient_risk_eta']}")
        if 'risk_budget' in settings and row['method'].startswith(('latent_','learned_','late_')):
            labels.append(f"budget={settings['risk_budget']}")
        label = row['method'] + (' ('+', '.join(labels)+')' if labels else '')
        compute = f"{row['forward_evaluations_per_sample']:.1f} / {row['backward_evaluations_per_sample']:.1f} / {row['partial_backward_evaluations_per_sample']:.1f}"
        lines.append(f"| {label} | {row['passed']}/20 | {row['wins_vs_native_eb']} / {row['losses_vs_native_eb']} | {compute} | {row['unfinished_masks']} | [Result]({row['result_path']}) |")
    lines += ['',f"Settings and every completed packed/standalone trial are in [humaneval.csv]({(args.out/'humaneval.csv').resolve()}).",
              'Remaining masks include unused tail positions; they are not equivalent to incomplete code.','',
              '## Latest extended arithmetic checks','',
              '| Seed | Method | Answer accuracy | Delta vs EB | Worksheet accuracy |',
              '| ---: | --- | ---: | ---: | ---: |']
    for row in small:
        if not row['eligible_intrinsic_only']: continue
        if 'latent_arithmetic_extended_seed' not in row['result_path']: continue
        delta = row['slot_delta_vs_eb']
        lines.append(f"| {row['seed']} | {row['method']} | {100*row['slot_accuracy']:.2f}% | {100*delta:+.2f} pp | {100*row['worksheet_accuracy']:.2f}% |")
    lines += ['',f"All arithmetic studies: [arithmetic.csv]({(args.out/'arithmetic.csv').resolve()}). Canvases share component worksheets, so their outcomes are correlated.",
              '', '## Actual-reveal arithmetic checks', '',
              '| Seed | Method | Answer accuracy | Worksheet accuracy | Forward evaluations per canvas |',
              '| ---: | --- | ---: | ---: | ---: |']
    for row in small:
        if not row['eligible_intrinsic_only'] or 'latent_finite_arithmetic_seed' not in row['result_path']: continue
        lines.append(f"| {row['seed']} | {row['method']} | {100*row['slot_accuracy']:.2f}% | {100*row['worksheet_accuracy']:.2f}% | {row['forward_evaluations_per_canvas']:.2f} |")
    lines += ['',
              '## Multi-value information arithmetic checks','',
              '| Seed | Method | Answer accuracy | Worksheet accuracy | Forward evaluations per canvas |',
              '| ---: | --- | ---: | ---: | ---: |']
    for row in small:
        if not row['eligible_intrinsic_only'] or not any(study in row['result_path'] for study in ('latent_information_arithmetic_seed','latent_information_fullpool_seed')): continue
        label = row['method']
        if row['method'].startswith('response_information'): label += f" (candidates={row['probe_candidates']})"
        lines.append(f"| {row['seed']} | {label} | {100*row['slot_accuracy']:.2f}% | {100*row['worksheet_accuracy']:.2f}% | {row['forward_evaluations_per_canvas']:.2f} |")
    lines += ['',
              '## Source-support and count arithmetic checks','',
              '| Seed | Method | Answer accuracy | Worksheet accuracy | Forward evaluations per canvas |',
              '| ---: | --- | ---: | ---: | ---: |']
    for row in small:
        if not row['eligible_intrinsic_only'] or 'latent_support_count_arithmetic_seed' not in row['result_path']: continue
        lines.append(f"| {row['seed']} | {row['method']} | {100*row['slot_accuracy']:.2f}% | {100*row['worksheet_accuracy']:.2f}% | {row['forward_evaluations_per_canvas']:.2f} |")
    lines += ['',
              '## Within-forward depth arithmetic checks','',
              '| Seed | Method | Answer accuracy | Worksheet accuracy | Forward evaluations per canvas |',
              '| ---: | --- | ---: | ---: | ---: |']
    for row in small:
        if not row['eligible_intrinsic_only'] or 'latent_depth_arithmetic_seed' not in row['result_path']: continue
        lines.append(f"| {row['seed']} | {row['method']} | {100*row['slot_accuracy']:.2f}% | {100*row['worksheet_accuracy']:.2f}% | {row['forward_evaluations_per_canvas']:.2f} |")
    lines += ['',
              '## Combined information and depth arithmetic checks','',
              '| Seed | Method | Answer accuracy | Worksheet accuracy | Forward evaluations per canvas |',
              '| ---: | --- | ---: | ---: | ---: |']
    for row in small:
        if not row['eligible_intrinsic_only'] or 'latent_information_depth_arithmetic_seed' not in row['result_path']: continue
        lines.append(f"| {row['seed']} | {row['method']} | {100*row['slot_accuracy']:.2f}% | {100*row['worksheet_accuracy']:.2f}% | {row['forward_evaluations_per_canvas']:.2f} |")
    lines += ['',
              '## Unresolved-attention arithmetic checks','',
              '| Seed | Method | Answer accuracy | Worksheet accuracy | Forward evaluations per canvas |',
              '| ---: | --- | ---: | ---: | ---: |']
    for row in small:
        if not row['eligible_intrinsic_only'] or 'latent_attention_arithmetic_seed' not in row['result_path']: continue
        lines.append(f"| {row['seed']} | {row['method']} | {100*row['slot_accuracy']:.2f}% | {100*row['worksheet_accuracy']:.2f}% | {row['forward_evaluations_per_canvas']:.2f} |")
    lines += ['',
              'Only completed, scored artifacts are included; this report does not assert that pending jobs have finished.','']
    (args.out/'README.md').write_text('\n'.join(lines))
    print(json.dumps({'human_trials':len(human),'arithmetic_trials':len(small),'report':str((args.out/'README.md').resolve())}),flush=True)


if __name__ == '__main__': main()
