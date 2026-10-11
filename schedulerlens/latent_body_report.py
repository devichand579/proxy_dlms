"""Validate and compare the body-progress pilot without loading the GPU model."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def validate(root):
    manifest = read(root / 'manifest.json')
    for source, digest in manifest['source_sha256'].items():
        snapshot = root / 'sources' / source.replace('/', '__')
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == digest, source
    trajectories = []
    for path in sorted(root.glob('seed*/*/results.json')):
        payload = read(path)
        metrics, work = payload['metrics'], payload['scheduler_metrics']
        batch, steps = metrics['samples'], work['steps_taken']
        assert steps <= manifest['steps']
        assert work['sequence_evaluations'] == batch * steps
        assert work['forward_calls'] == steps
        assert work['backward_sequence_evaluations'] == 0
        uses_flow = 'context_flow' in work['ranking']
        assert work['partial_backward_sequence_evaluations'] == (batch * steps if uses_flow else 0)
        assert not work['auxiliary_training']
        assert len(work['reveal_counts']) == len(work['reveal_positions']) == steps
        assert len(work['count_records']) == len(work['candidate_counts']) == steps
        assert metrics['passed'] == sum(s['passed'] for s in payload['samples'])
        for row, sample in enumerate(payload['samples']):
            revealed = []
            for counts, positions, candidates, records in zip(
                    work['reveal_counts'], work['reveal_positions'],
                    work['candidate_counts'], work['count_records']):
                assert counts[row] == len(positions[row]) == records[row]['count']
                assert counts[row] <= candidates[row]
                revealed.extend(positions[row])
            assert len(revealed) == len(set(revealed))
            committed = {i for i, t in enumerate(sample['token_ids']) if t != 126336}
            assert committed == set(revealed)
            assert sample['unfinished_masks'] == len(sample['token_ids']) - len(committed)
        trajectories.append(payload)
    return trajectories


def paired(candidate, baseline):
    reference = {s['task_id']: s for s in baseline['samples']}
    assert set(reference) == {s['task_id'] for s in candidate['samples']}
    wins, losses, identical = [], [], 0
    for sample in candidate['samples']:
        old = reference[sample['task_id']]
        if sample['passed'] and not old['passed']: wins.append(sample['task_id'])
        if old['passed'] and not sample['passed']: losses.append(sample['task_id'])
        identical += sample['completion'] == old['completion']
    return dict(wins=wins, losses=losses, identical_completions=identical)


def scope(work):
    if work['config']['candidate_region'] == 'frontier':
        return 'position-window diagnostic; confidence ranking'
    if work['confidence_scoring']:
        return 'confidence-ranking diagnostic' + ('; probability-weighted count' if work['count_uses_vocabulary_probabilities'] else '')
    if work['controller'] == 'singleton': return 'confidence-free ranking control; fixed count'
    if work['count_uses_vocabulary_probabilities']: return 'probability-weighted count diagnostic'
    return 'confidence-free ranking and adaptive count'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--additional-roots', type=Path, nargs='*', default=[])
    parser.add_argument('--arithmetic-root', type=Path)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    trajectories = []
    excluded = []
    for root in [args.root] + args.additional_roots:
        checked = validate(root)
        source = (root / 'sources' / 'schedulers__latent_context_flow.py').read_text()
        for payload in checked:
            if 'context_flow' in payload['scheduler_metrics']['ranking'] and 'leaf=inputs[0].detach()' in source:
                excluded.append(dict(method=payload['metrics']['variant'],root=str(root),
                    reason='Gradient capture omitted the final residual identity derivative; rerun required.'))
            else: trajectories.append(payload)
    data = {p['metrics']['variant']: p for p in trajectories}
    audit = {s['method']: s for s in read(args.audit / 'analysis.json')['studies']}
    replays = {}
    for name, old in [('fisher_singleton', 'diagnostic_fisher_singleton'),
                      ('activation_singleton', 'activation_singleton')]:
        if name in data:
            replays[name] = paired(data[name], read(f'outputs/latent_count_singleton20/{old}/results.json'))
    comparisons = {}
    matched = dict(context_flow_allow01='context_flow_singleton',
        fisher_flow_dispersion05='fisher_flow_singleton005',
        fisher_dispersion05_eos='fisher_dispersion05',
        fisher_dispersion05_frontier32='fisher_singleton_frontier32')
    for name, payload in data.items():
        base = matched.get(name, 'activation_singleton' if payload['scheduler_metrics']['ranking'] in ('activation', 'context_flow') else 'fisher_singleton')
        comparisons[name] = dict(baseline=base, **paired(payload, data[base]))
        if comparisons[name]['identical_completions'] == len(payload['samples']):
            old = {s['task_id']: s for s in audit[base]['samples']}
            savings = [old[s['task_id']]['rendered_body_ready_step']-s['rendered_body_ready_step']
                       for s in audit[name]['samples']]
            comparisons[name]['body_timing'] = dict(earlier=sum(t>0 for t in savings),
                equal=sum(t==0 for t in savings),later=sum(t<0 for t in savings),
                mean_saved_steps=sum(savings)/len(savings))
    result = dict(validated_trajectories=len(trajectories), excluded_runs=excluded,source_snapshots_verified=True,
                  historical_control_replays=replays, paired_comparisons=comparisons,
                  metrics=[p['metrics'] for p in trajectories])
    (args.out / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# Latent scoring and useful code-body progress', '',
        'Exploratory HumanEval first 20 tasks, greedy LLaDA-8B Base, 512 positions,',
        '128-step cap, standalone batch 20. No training or extra backbone forwards.',
        'Context-flow methods require one partial backward through the final tokenwise',
        'MLP and final normalization per forward. Full-model backwards remain zero.', '',
        'Visible-body timing is retrospective: it cannot be used as an online stopping',
        'rule. It excludes unresolved mask holes. Identical programs can have different',
        'commit times; actual execution cost is shown separately.', '']
    if 'fisher_dispersion05' in comparisons:
        c = comparisons['fisher_dispersion05']
        a,b = audit['fisher_dispersion05'],audit['fisher_singleton']
        lines += ['## Main findings', '',
            f'Independent Fisher-dispersion count: {data["fisher_dispersion05"]["metrics"]["passed"]}/20 versus singleton {data["fisher_singleton"]["metrics"]["passed"]}/20, with {c["identical_completions"]}/20 identical final programs.',
            f'Mean visible-body commitment moved from step {b["mean_rendered_body_ready_step"]:.2f} to {a["mean_rendered_body_ready_step"]:.2f}. This is a progress gain under confidence ranking, not a final-quality or measured runtime gain.', '']
    if 'context_flow_allow01' in data:
        lines += [f'The corrected confidence-free context-flow scorer passed {data["context_flow_singleton"]["metrics"]["passed"]}/20 with singleton and {data["context_flow_allow01"]["metrics"]["passed"]}/20 with adaptive allowance, versus activation singleton {data["activation_singleton"]["metrics"]["passed"]}/20. It is not supported as a standalone readiness estimator.', '']
    lines += [
        '| Method | Passed | Forwards | Partial backwards | Batch seconds | Mean body-ready step | Body multi-reveal events | Mean mask holes | Scope |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for name, p in data.items():
        m, a, w = p['metrics'], audit[name], p['scheduler_metrics']
        lines.append(f'| {name} | {m["passed"]}/20 | {m["forward_evaluations_per_sample"]:g} | {m["partial_backward_evaluations_per_sample"]:g} | {m["generation_seconds"]:.1f} | {a["mean_rendered_body_ready_step"]:.2f} | {a["body_multi_steps"]} | {a["mean_body_mask_holes"]:.2f} | {scope(w)} |')
    lines += ['', '## Paired outcomes', '']
    for name, comparison in comparisons.items():
        lines.append(f'- {name} versus {comparison["baseline"]}: wins {comparison["wins"]}; losses {comparison["losses"]}; identical completions {comparison["identical_completions"]}/20.')
        if name != comparison['baseline'] and 'body_timing' in comparison:
            timing = comparison['body_timing']
            lines.append(f'  Identical-output body timing: {timing["earlier"]} earlier, {timing["equal"]} equal, {timing["later"]} later; mean {timing["mean_saved_steps"]:.2f} steps saved.')
    if args.arithmetic_root:
        root = args.arithmetic_root
        manifest = read(root / 'manifest.json')
        for source, digest in manifest['source_sha256'].items():
            assert hashlib.sha256((root/'sources'/source.replace('/', '__')).read_bytes()).hexdigest() == digest
        arithmetic = {}
        for path in sorted(root.glob('seed*/*/results.json')):
            payload = read(path);m,w = payload['metrics'],payload['scheduler_metrics']
            assert w['sequence_evaluations'] == m['samples']*w['steps_taken']
            assert w['backward_sequence_evaluations'] == 0
            assert w['partial_backward_sequence_evaluations'] == (w['sequence_evaluations'] if 'context_flow' in w['ranking'] else 0)
            assert m['slot_accuracy'] == sum(s['correct_slots'] for s in payload['samples'])/384
            arithmetic.setdefault(m['variant'], {})[m['seed']] = m
        lines += ['', '## Corrected arithmetic check', '',
            '32 canvases and 384 masked answers per seed, 12-step cap. Reused correlated',
            'worksheets; these results are exploratory and do not establish transfer to code.', '',
            '| Method | Seed 0 | Seed 1 | Forwards/canvas | Partial backwards/canvas |',
            '|---|---:|---:|---:|---:|']
        for name, seeds in arithmetic.items():
            a,b = seeds[0],seeds[1]
            lines.append(f'| {name} | {a["slot_accuracy"]:.2%} | {b["slot_accuracy"]:.2%} | {a["forward_evaluations_per_sample"]:g}/{b["forward_evaluations_per_sample"]:g} | {a["partial_backward_evaluations_per_sample"]:g}/{b["partial_backward_evaluations_per_sample"]:g} |')
        result['validated_arithmetic_trajectories'] = sum(len(s) for s in arithmetic.values())
        result['arithmetic_metrics'] = [m for seeds in arithmetic.values() for m in seeds.values()]
        (args.out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    lines += ['', '## Validation and interpretation', '',
        f'{len(trajectories)} trajectories passed source-snapshot, work-count, unique-reveal,',
        'final-token and correctness-total checks. Historical control replay comparisons:',
        '```json', json.dumps(replays, indent=2), '```', '',
        'The first context-flow run omitted the residual identity derivative. It is',
        'excluded from conclusions and retained with its original source snapshot.',
        'Corrected runs pass an independent complete-residual gradient check.', '',
        'Observed-key value attribution holds final-block attention fixed and measures',
        'a local readout derivative; observed value vectors already contain earlier-layer',
        'context. It is neither whole-model causal prompt attribution nor a data-error',
        'certificate. Fisher-dispersion count uses probabilities as covariance weights.',
        'Frontier windows use token position and are diagnostics rather than fully latent',
        'schedulers. Final program boundaries and test outcomes never enter generation.', '',
        f'[Per-task body audit]({(args.audit / "README.md").resolve()})',
        f'[Saved runs and source snapshots]({args.root.resolve()})']
    for root in args.additional_roots:
        lines += [f'[Corrected runs and source snapshots]({root.resolve()})']
    replay = args.out.parent/'latent_body_progress_budget_replay'/'README.md'
    if replay.exists(): lines += [f'[Earlier-step correctness replay]({replay.resolve()})']
    (args.out / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
