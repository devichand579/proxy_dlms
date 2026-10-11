"""Retrospective decoded-body timing; final boundaries never enter scheduling."""
import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def span_end(tokenizer, tokens, completion, eos):
    if not completion:return 0
    limit=next((i for i,t in enumerate(tokens) if t==eos),len(tokens))
    for end in range(1,limit+1):
        text=tokenizer.decode(tokens[:end],skip_special_tokens=True)
        if len(text)>=len(completion) and text[:len(completion)]==completion:return end
    raise ValueError('saved completion does not match saved token sequence')


def audit_sample(tokenizer, sample, positions, row, mask=126336,eos=126081):
    tokens=sample['token_ids'];end=span_end(tokenizer,tokens,sample['completion'],eos)
    times=np.zeros(len(tokens),dtype=int)
    for step,group in enumerate(positions,1):
        for i in group[row]:
            if times[i]:raise ValueError('position committed twice')
            times[i]=step
    special=set(tokenizer.all_special_ids)
    visible=[i for i in range(end) if tokens[i]!=mask and tokens[i] not in special]
    body_set=set(visible)
    body_counts=[sum(i in body_set for i in group[row]) for group in positions]
    eos_index=next((i for i,t in enumerate(tokens) if t==eos),None)
    safe_step=None
    if eos_index is not None and all(tokens[i]!=mask for i in range(eos_index+1)):
        if not bool((times[:eos_index+1]>0).all()):raise ValueError('missing reveal time')
        safe_step=int(times[:eos_index+1].max())
    return dict(task_id=sample['task_id'],passed=sample['passed'],body_span_positions=end,
        rendered_tokens=len(visible),body_mask_holes=sum(t==mask for t in tokens[:end]),
        body_multi_steps=sum(k>1 for k in body_counts),body_counts=body_counts,
        rendered_body_ready_step=max((int(times[i]) for i in visible),default=0),
        eos_index=eos_index,safe_eos_completion_step=safe_step,
        body_reveal_fraction=len(visible)/max(1,int((times>0).sum())))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots',nargs='+',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained('models/llada',trust_remote_code=True,local_files_only=True)
    studies=[]
    for root in args.roots:
        source=(Path(root)/'sources'/'schedulers__latent_context_flow.py').read_text()
        for path in sorted(Path(root).glob('seed*/*/results.json')):
            p=json.loads(path.read_text());work=p['scheduler_metrics']
            if 'context_flow' in work['ranking'] and 'leaf=inputs[0].detach()' in source:continue
            if not work.get('reveal_positions') or 'token_ids' not in p['samples'][0]:continue
            records=[audit_sample(tokenizer,s,work['reveal_positions'],row) for row,s in enumerate(p['samples'])]
            safe=[s['safe_eos_completion_step'] for s in records if s['safe_eos_completion_step'] is not None]
            studies.append(dict(method=p['metrics']['variant'],passed=p['metrics']['passed'],
                forwards=p['metrics']['forward_evaluations_per_sample'],
                partial_backwards=p['metrics'].get('partial_backward_evaluations_per_sample',0.),
                mean_rendered_body_ready_step=float(np.mean([s['rendered_body_ready_step'] for s in records])),
                body_multi_steps=sum(s['body_multi_steps'] for s in records),
                mean_body_mask_holes=float(np.mean([s['body_mask_holes'] for s in records])),
                safe_eos_examples=len(safe),mean_safe_eos_step=float(np.mean(safe)) if safe else None,
                samples=records,artifact=str(path.resolve())))
    (args.out/'analysis.json').write_text(json.dumps(dict(studies=studies,
        note='Final decoded body boundaries are retrospective audit labels only. Rendered-body timing excludes skipped mask holes; safe-EOS completion additionally requires a fully committed prefix.'),indent=2)+'\n')
    lines=['# Decoded-body progress audit','',
        'Final output boundaries are used only for retrospective analysis. They never enter the scheduler.',
        'Rendered-body readiness measures when the visible output tokens were committed; mask holes can',
        'still exist inside that span. Safe EOS readiness requires every position up to the first',
        'committed EOS to be known and is the stronger runtime-compatible condition.','',
        '| Method | Passed | Full forwards | Partial MLP backwards | Body multi-reveal events | Mean visible-body ready step | Mean body mask holes | Safe EOS examples |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in studies:
        lines.append(f'| {s["method"]} | {s["passed"]}/20 | {s["forwards"]:g} | {s["partial_backwards"]:g} | {s["body_multi_steps"]} | {s["mean_rendered_body_ready_step"]:.2f} | {s["mean_body_mask_holes"]:.2f} | {s["safe_eos_examples"]}/20 |')
    (args.out/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps([{k:v for k,v in s.items() if k!='samples'} for s in studies],indent=2))


if __name__=='__main__':main()
