"""Audit frozen-model reconstruction of supplied prompts from generated code.

Correctness labels are used only after scoring, for the diagnostic report.
This retrospective selection audit is not a new 128-step scheduler result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from load_model import load_model_and_tokenizer,infer_diffusion_token_ids
from schedulerlens.latent_bench import load_documents,set_seed
from schedulers.utils import get_mask_by_confidence

DEFAULT_SOURCES = [
    'outputs/latent_current_verified20/eb_target_temporal_b512_d0.25_s0.15_p8_v128/results.json',
    'outputs/latent_information_readiness_verified20/response_information_readiness.json',
    'outputs/latent_information_every_step20/response_information_readiness.json',
    'outputs/latent_conditioning80/conditioning_support025.json',
]


@torch.no_grad()
def masked_context_score(model,inputs,attention,reference,targets,mask_id):
    if inputs.shape!=reference.shape or inputs.shape!=targets.shape or attention.shape!=inputs.shape:
        raise ValueError('score input shapes must match')
    if not bool(targets.any(-1).all()) or bool((targets&~attention.bool()).any()):
        raise ValueError('nonempty attended targets required')
    if not bool((inputs[targets]==mask_id).all()):
        raise ValueError('scored labels must be masked in model input')
    logits=model(inputs,attention_mask=attention).logits
    logits[...,mask_id]=-torch.inf
    logp=get_mask_by_confidence(logits,reference).clamp_min(1e-30).log()
    return (logp*targets).sum(-1),targets.sum(-1)


def candidates_from_sources(paths):
    candidates=[]; seen={}
    for path in paths:
        data=json.loads(path.read_text())
        for sample in data['samples']:
            key=(sample['task_id'],sample['completion'])
            if key in seen:
                entry=candidates[seen[key]]
                if entry['passed']!=sample['passed']: raise ValueError('duplicate code has inconsistent outcomes')
                entry['sources'].append(str(path)); continue
            seen[key]=len(candidates)
            candidates.append(dict(task_id=sample['task_id'],completion=sample['completion'],
                passed=sample['passed'],sources=[str(path)]))
    return candidates


def selection_report(candidates,field):
    groups={}
    for candidate in candidates: groups.setdefault(candidate['task_id'],[]).append(candidate)
    selected=[]; pair_wins=0.; pair_count=0
    for task_id,group in groups.items():
        # Select solely by the frozen-model score. Stable order resolves ties.
        winner=max(group,key=lambda c:c[field])
        selected.append(dict(task_id=task_id,passed=winner['passed'],score=winner[field],sources=winner['sources']))
        for correct in (c for c in group if c['passed']):
            for wrong in (c for c in group if not c['passed']):
                pair_wins += (correct[field]>wrong[field])+.5*(correct[field]==wrong[field]); pair_count+=1
    return dict(selected_passed=sum(s['passed'] for s in selected),tasks=len(groups),
        available_correct_tasks=sum(any(c['passed'] for c in group) for group in groups.values()),
        correct_vs_wrong_pairs=pair_count,pair_ranking_accuracy=pair_wins/pair_count if pair_count else None,
        selected=selected)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',type=Path,nargs='+',default=[Path(p) for p in DEFAULT_SOURCES])
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--out',type=Path,default=Path('outputs/context_reconstruction_audit'))
    args=parser.parse_args()
    if args.batch_size<1: parser.error('positive batch size required')
    args.out.mkdir(parents=True,exist_ok=True); set_seed(1234)
    documents,dataset=load_documents(None,20); prompts={d['task_id']:d['prompt'] for d in documents}
    candidates=candidates_from_sources(args.sources)
    if set(c['task_id'] for c in candidates)!=set(prompts): raise ValueError('expected the same first 20 task IDs')
    model,tokenizer=load_model_and_tokenizer('models/llada',device='cuda',compile_model=False)
    mask,_,_=infer_diffusion_token_ids('models/llada')
    encoded=[(tokenizer(prompts[c['task_id']],add_special_tokens=False)['input_ids'],
              tokenizer(c['completion'],add_special_tokens=False)['input_ids']) for c in candidates]
    prefix=max(len(p) for p,_ in encoded); tail=max(len(body) for _,body in encoded); n=len(candidates)
    reference=torch.full((n,prefix+tail),tokenizer.pad_token_id,dtype=torch.long,device='cuda')
    attention=torch.zeros_like(reference); context=torch.zeros_like(reference,dtype=torch.bool)
    position=torch.zeros_like(reference)
    for row,(prompt,body) in enumerate(encoded):
        reference[row,prefix-len(prompt):prefix]=torch.tensor(prompt,device='cuda')
        if body: reference[row,prefix:prefix+len(body)]=torch.tensor(body,device='cuda')
        attention[row,prefix-len(prompt):prefix+len(body)]=1
        context[row,prefix-len(prompt):prefix]=True
        position[row,prefix-len(prompt):prefix]=torch.arange(len(prompt),device='cuda')
    source=Path(__file__).read_bytes(); (args.out/'audit_source.py').write_bytes(source)
    torch.cuda.reset_peak_memory_stats(); start=time.perf_counter(); evaluations=0
    for scheme in ('all','interleaved'):
        sums=torch.zeros(n,device='cuda'); counts=torch.zeros(n,device='cuda')
        for group in range(1 if scheme=='all' else 4):
            targets=context if scheme=='all' else context&(position%4==group)
            for lo in range(0,n,args.batch_size):
                hi=min(lo+args.batch_size,n); inputs=reference[lo:hi].clone()
                inputs[targets[lo:hi]]=mask
                score,count=masked_context_score(model,inputs,attention[lo:hi],reference[lo:hi],targets[lo:hi],mask)
                sums[lo:hi]+=score; counts[lo:hi]+=count; evaluations+=hi-lo
        for candidate,score,count in zip(candidates,sums.tolist(),counts.tolist()):
            candidate[scheme+'_mean_logp']=score/count
        print(json.dumps(dict(scheme=scheme,candidates=n,evaluations=evaluations)),flush=True)
    torch.cuda.synchronize()
    reports={scheme:selection_report(candidates,scheme+'_mean_logp') for scheme in ('all','interleaved')}
    payload=dict(reports=reports,candidates=candidates,source_artifacts=[str(p.resolve()) for p in args.sources],
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in args.sources},
        dataset_arrow=str(dataset),forward_sequence_evaluations=evaluations,backward_sequence_evaluations=0,
        batch_size=args.batch_size,generation_performed=False,auxiliary_training=False,
        scored_targets='supplied prompt tokens only',hidden_tests_used_for_scoring=False,
        code_sha256=hashlib.sha256(source).hexdigest(),seconds=time.perf_counter()-start,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
    (args.out/'audit.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print(json.dumps({s:{k:v for k,v in r.items() if k!='selected'} for s,r in reports.items()}),flush=True)


if __name__=='__main__': main()
