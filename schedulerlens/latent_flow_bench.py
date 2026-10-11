"""Resumable HumanEval evaluation of the activation-only latent-flow policy."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from load_model import load_model_and_tokenizer,infer_diffusion_token_ids
from schedulerlens.latent_bench import load_documents,code_results,decode,set_seed
from schedulerlens.latent_full_bench import save
from schedulers import latent_flow,latent_residual


def report(out,documents,result):
    manifest = json.loads((out/'manifest.json').read_text())
    reference = Path('outputs/latent_full_humaneval164')
    reference_manifest = json.loads((reference/'manifest.json').read_text())
    for key in ('dataset_sha256','model_config_sha256','batch_size','seed','gen_length','steps','prompt_format'):
        if manifest[key]!=reference_manifest[key]:raise ValueError('baseline protocol differs')
    results = {'latent_pca_flow':result['samples']}
    summaries = {'latent_pca_flow':result['metrics']}
    for method in ('eb_sampler','eb_target_temporal'):
        payload = json.loads((reference/method/'results.json').read_text())
        samples = payload['samples'][:len(documents)]
        if [s['task_id'] for s in samples]!=manifest['task_ids']:raise ValueError('baseline tasks differ')
        results[method] = samples
        summaries[method] = dict(passed=sum(s['passed'] for s in samples),
            forward_evaluations_per_sample=sum(s['forward_evaluations'] for s in samples)/len(samples),
            steps_taken=128)
    comparisons = {}
    for baseline in ('eb_sampler','eb_target_temporal'):
        a = {s['task_id']:s['passed'] for s in results[baseline]}
        b = {s['task_id']:s['passed'] for s in results['latent_pca_flow']}
        comparisons[baseline] = dict(wins=[k for k in a if b[k] and not a[k]],
                                    losses=[k for k in a if a[k] and not b[k]])
    save(out/'comparison.json',dict(summaries=summaries,paired_outcomes=comparisons))
    lines = ['# Activation-only latent flow on HumanEval','',
        f'Completed {len(documents)} canonical problems at a 128-step cap, 512 generation positions, greedy raw code completion, and matched normal batches of 20. '
        'Parameters were frozen before generation. Logits supply probe and committed values; no probabilities, confidence, entropy, or EB count enter the new scheduler.', '',
        '| Method | Passed | Accuracy | Forward evaluations/problem | Outer steps used |', '|---|---:|---:|---:|---:|']
    for method in ('eb_sampler','eb_target_temporal','latent_pca_flow'):
        summary = summaries[method]
        if method=='latent_pca_flow':
            forward = summary['sequence_evaluations']/len(documents)
            used = ' / '.join(str(m['steps_taken']) for m in result['batch_metrics'])
        else:forward = summary['forward_evaluations_per_sample'];used = '128'
        passed = summary['passed']
        lines.append(f'| {method} | {passed}/{len(documents)} | {100*passed/len(documents):.2f}% | {forward:.2f} | {used} |')
    lines += ['', 'EB and temporal Fisher are previously completed matched benchmark controls. They do not satisfy the activation-only policy constraint. '
        'The new policy stops if its entire canvas is committed before the cap; no-op forwards are not added to pretend that more denoising occurred.', '',
        f'Novel policy generation time: {result["metrics"]["generation_seconds"]:.1f}s; peak allocated memory: {result["metrics"]["peak_allocated_bytes"]/2**30:.2f} GiB. '
        'Every intervention and unchanged control counts toward sequence work. No backward passes or auxiliary training.', '']
    for baseline,c in comparisons.items():
        lines += [f'Against {baseline}: {len(c["wins"])} wins, {len(c["losses"])} losses. '
                  f'Wins: {", ".join(c["wins"]) or "none"}. Losses: {", ".join(c["losses"]) or "none"}.','']
    lines += ['The first 20 problems have been used for repeated exploratory tuning. This is one greedy completion per problem, not a generalization estimate.', '',
        f'[Settings and source hashes]({(out/"manifest.json").resolve()}) · '
        f'[Completions and test outcomes]({(out/"latent_pca_flow"/"results.json").resolve()}) · '
        f'[Paired comparison]({(out/"comparison.json").resolve()})', '']
    (out/'README.md').write_text('\n'.join(lines))
    print('\n'.join(lines),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=int,default=20)
    parser.add_argument('--branch-batch-size',type=int,default=480)
    parser.add_argument('--resume-from',type=Path,help='Preserve completed steps from the earlier controlled phase.')
    parser.add_argument('--controls',action='store_true',help='Evaluate unused unchanged controls for a numerical ablation.')
    parser.add_argument('--out',type=Path,default=Path('outputs/latent_pca_flow_humaneval20'))
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_DATASETS_OFFLINE']='1'
    args.out.mkdir(parents=True,exist_ok=True)
    documents,dataset = load_documents(None,args.samples)
    paths = [Path(__file__),Path(latent_flow.__file__),Path(latent_residual.__file__),
             Path('schedulerlens/latent_bench.py'),Path('load_model.py')]
    config = dict(variant='latent_pca_flow',steps=128,gen_length=512,block_length=512,
        batch_size=20,seed=1234,candidates=12,branch_batch_size=args.branch_batch_size,
        incoming_weight=.5,consistency_weight=0.,progress_reward=.05,geometry='pca')
    config['evaluate_controls'] = args.controls
    manifest = dict(**config,task_ids=[d['task_id'] for d in documents],dataset_arrow=str(dataset),
        dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
        model_config_sha256=hashlib.sha256(Path('models/llada/config.json').read_bytes()).hexdigest(),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        prompt_format='raw HumanEval code completion, zero shot',token_sampler='argmax',
        decode='first EOS, then installed HumanEval stop tokens',correctness_timeout_seconds=3.,
        confidence_scoring=False,entropy_scoring=False,eb_count=False,auxiliary_training=False,
        torch=torch.__version__,hip=torch.version.hip)
    if args.resume_from is not None:
        earlier_manifest = json.loads((args.resume_from/'manifest.json').read_text())
        for key in ('task_ids','dataset_sha256','model_config_sha256','steps','gen_length','batch_size',
                    'seed','candidates','branch_batch_size','incoming_weight','consistency_weight','progress_reward','geometry'):
            if earlier_manifest[key]!=manifest[key]:raise ValueError('earlier phase policy or protocol differs')
        manifest['earlier_phase'] = dict(directory=str(args.resume_from.resolve()),
            source_sha256=earlier_manifest['source_sha256'],optimization='Unused controls removed at zero consistency weight; policy-equivalence checks pass.')
    manifest_path = args.out/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:
        raise ValueError('existing experiment differs; use a new output directory')
    save(manifest_path,manifest)
    snapshots = args.out/'sources';snapshots.mkdir(exist_ok=True)
    for path in paths:(snapshots/str(path).replace('/','__')).write_bytes(path.read_bytes())
    folder = args.out/'latent_pca_flow';folder.mkdir(exist_ok=True)
    if args.resume_from is not None:
        for earlier in (args.resume_from/'latent_pca_flow').glob('state_*.pt'):
            target = folder/earlier.name
            if not target.exists():target.write_bytes(earlier.read_bytes())
    result_path = folder/'results.json'
    if result_path.exists():
        report(args.out,documents,json.loads(result_path.read_text()));return
    model,tokenizer = load_model_and_tokenizer('models/llada',device='cuda',compile_model=False)
    mask,eos,_ = infer_diffusion_token_ids('models/llada')
    encoded = [tokenizer(d['prompt'],add_special_tokens=False)['input_ids'] for d in documents]
    completed = []
    for offset in range(0,len(documents),20):
        checkpoint = folder/f'batch_{offset:03d}.json'
        generated_path = folder/f'generated_{offset:03d}.json'
        state_path = folder/f'state_{offset:03d}.pt'
        indices = list(range(offset,min(offset+20,len(documents))))
        task_ids = [documents[i]['task_id'] for i in indices]
        if checkpoint.exists():
            payload = json.loads(checkpoint.read_text())
        else:
            if generated_path.exists():
                payload = json.loads(generated_path.read_text())
            else:
                width = max(len(encoded[i]) for i in indices)
                ids = torch.full((len(indices),width),tokenizer.pad_token_id,device='cuda',dtype=torch.long)
                attention = torch.zeros_like(ids)
                for row,i in enumerate(indices):
                    ids[row,-len(encoded[i]):] = torch.tensor(encoded[i],device='cuda')
                    attention[row,-len(encoded[i]):] = 1
                state = torch.load(state_path,map_location='cpu',weights_only=True) if state_path.exists() else None
                prior_seconds = state.get('generation_seconds',0.) if state else 0.
                prior_peak = state.get('peak_allocated_bytes',0) if state else 0
                set_seed(1234);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
                started = time.perf_counter()
                def store_state(s):
                    s['generation_seconds'] = prior_seconds+time.perf_counter()-started
                    s['peak_allocated_bytes'] = max(prior_peak,torch.cuda.max_memory_allocated())
                    temporary = state_path.with_suffix('.tmp')
                    torch.save(s,temporary);temporary.replace(state_path)
                    print(f'Batch {offset//20+1}: step {s["next_step"]}/128, '
                          f'forward work {s["metrics"]["sequence_evaluations"]/len(indices):.1f}/problem, '
                          f'committed {sum(s["metrics"]["reveal_counts"][-1])} positions, '
                          f'{s["generation_seconds"]:.1f}s',flush=True)
                print(f'START tasks {task_ids[0]}..{task_ids[-1]}, prompt width {width}, '
                      f'probe batch {args.branch_batch_size}, resume step {state["next_step"] if state else 0}',flush=True)
                generated = latent_flow.generate_flow(model,ids,attention,steps=128,gen_length=512,
                    mask_id=mask,eos_id=eos,candidates=12,branch_batch_size=args.branch_batch_size,
                    incoming_weight=.5,consistency_weight=0.,progress_reward=.05,geometry='pca',
                    resume_state=state,step_checkpoint=store_state,evaluate_controls=args.controls)
                torch.cuda.synchronize()
                elapsed = prior_seconds+time.perf_counter()-started
                metrics = dict(latent_flow.last_metrics)
                texts = decode(tokenizer,generated,width,eos)
                samples = [dict(task_id=documents[i]['task_id'],completion=text,
                    unfinished_masks=int((generated[row,width:]==mask).sum()),
                    forward_evaluations=metrics['forward_work_by_row'][row])
                    for row,(i,text) in enumerate(zip(indices,texts))]
                payload = dict(task_ids=task_ids,samples=samples,scheduler_metrics=metrics,
                    generation_seconds=elapsed,peak_allocated_bytes=max(prior_peak,torch.cuda.max_memory_allocated()))
                save(generated_path,payload)
                del generated,ids,attention
            outcomes = code_results([documents[i] for i in indices],[s['completion'] for s in payload['samples']])
            for sample,outcome in zip(payload['samples'],outcomes):sample.update(outcome)
            save(checkpoint,payload)
        if payload['task_ids']!=task_ids:raise ValueError('checkpoint tasks differ')
        completed.append(payload)
        print(f'FINISHED batch {offset//20+1}: {sum(s["passed"] for s in payload["samples"])}/{len(indices)}',flush=True)
    samples = [s for p in completed for s in p['samples']]
    metrics = dict(samples=len(samples),passed=sum(s['passed'] for s in samples),
        pass_at_1=sum(s['passed'] for s in samples)/len(samples),
        sequence_evaluations=sum(s['forward_evaluations'] for s in samples),backward_sequence_evaluations=0,
        generation_seconds=sum(p['generation_seconds'] for p in completed),
        peak_allocated_bytes=max(p['peak_allocated_bytes'] for p in completed),
        unfinished_masks=sum(s['unfinished_masks'] for s in samples))
    payload = dict(config=config,run='latent_pca_flow',metrics=metrics,samples=samples,
        batch_metrics=[p['scheduler_metrics'] for p in completed])
    if len(completed)==1:payload['scheduler_metrics'] = completed[0]['scheduler_metrics']
    save(result_path,payload)
    report(args.out,documents,payload)


if __name__=='__main__':main()
