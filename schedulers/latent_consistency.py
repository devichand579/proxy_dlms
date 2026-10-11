"""Measured split-batch commitment consistency with EB's proposed prefix.

Checks use the frozen model and original greedy proposals. A detected conflict
can reduce count; it does not certify correctness or implement a TC bound.
"""
import torch
from schedulers.latent import _eb_prefix,_target_temporal_risk
from schedulers.utils import get_mask_by_confidence,token_entropy

last_metrics={}


def stable_subset(positions,conflict,mode='filter'):
    if mode not in ('filter','singleton'): raise ValueError('unknown consistency mode')
    if not len(positions): return positions
    if not bool(conflict.any()): return positions
    if mode=='singleton': return positions[:1]
    stable=positions[~conflict]
    return stable if len(stable) else positions[:1]


@torch.no_grad()
def generate_consistency(model,prompt,attention_mask,*,steps=128,gen_length=512,
                         mask_id=126336,eos_id=126081,enabled=True,mode='filter',
                         probe_every=1,branch_batch_size=160,stability_beta=.15,
                         progress=None,initial_canvas=None,full_attention=None):
    global last_metrics
    batch,prefix=prompt.shape
    def row(value,index): return value[index] if isinstance(value,(list,tuple)) else value
    for setting in (enabled,mode):
        if isinstance(setting,(list,tuple)) and len(setting)!=batch: raise ValueError('one setting per sequence required')
    if min(steps,probe_every)<1 or branch_batch_size<4 or branch_batch_size%4:
        raise ValueError('positive budgets and branch size divisible by four required')
    if any(row(mode,i) not in ('filter','singleton') for i in range(batch)): raise ValueError('unknown mode')
    x=torch.full((batch,prefix+gen_length),mask_id,dtype=torch.long,device=prompt.device) if initial_canvas is None else initial_canvas.clone()
    if x.shape!=(batch,prefix+gen_length): raise ValueError('canvas shape mismatch')
    x[:,:prefix]=prompt
    attention=torch.cat([attention_mask,torch.ones(batch,gen_length,device=x.device,dtype=attention_mask.dtype)],-1) if full_attention is None else full_attention
    if attention.shape!=x.shape: raise ValueError('attention shape mismatch')
    head=model.get_output_embeddings(); previous=None
    work=dict(steps_taken=0,forward_calls=0,sequence_evaluations=0,forward_work_by_row=[0]*batch,
        branch_work_by_row=[0]*batch,probe_control_work_by_row=[0]*batch,reveal_counts=[],
        enabled=enabled,mode=mode,probe_every=probe_every,branch_batch_size=branch_batch_size,
        auxiliary_training=False,matched_probe_control=True,token_values_changed=False,
        probed_batches_by_row=[0]*batch,conflicts_by_row=[0]*batch,
        numerical_mismatches_by_row=[0]*batch,count_reductions_by_row=[0]*batch,
        deferred_tokens_by_row=[0]*batch,singleton_fallbacks_by_row=[0]*batch)
    for step in range(steps):
        masked=x==mask_id
        if not masked[:,prefix:].any(): break
        states=[]; hook=head.register_forward_pre_hook(lambda _m,inputs:states.append(inputs[0].detach()))
        try: out=model(x,attention_mask=attention)
        finally: hook.remove()
        logits,hidden=out.logits,states[0]; logits[...,mask_id]=-torch.inf
        proposed=logits.argmax(-1); confidence=get_mask_by_confidence(logits,proposed)
        entropy=token_entropy(logits,eps=1e-9)
        work['forward_calls']+=1; work['sequence_evaluations']+=batch
        for i in range(batch): work['forward_work_by_row'][i]+=1
        selected=[]
        for i in range(batch):
            pos=masked[i,prefix:].nonzero().flatten()+prefix
            score=confidence[i,pos].clamp_min(1e-30).log()
            if previous is not None and len(pos):
                risk=_target_temporal_risk(logits[i,pos],hidden[i,pos],previous[i,pos],head.weight,128)
                score-=stability_beta*(risk/risk.mean().clamp_min(1e-8)).clamp(max=4)
            selected.append(_eb_prefix(pos,score,entropy[i,pos]))
        if step%probe_every==0:
            active=[i for i,pos in enumerate(selected) if row(enabled,i) and len(pos)>1]
            for offset in range(0,len(active),branch_batch_size//4):
                group=active[offset:offset+branch_batch_size//4]; n=len(group)
                rows=torch.tensor(group,device=x.device)
                copies=x[rows].repeat(4,1)
                # First 2n unchanged controls, then A-revealed and B-revealed.
                for j,i in enumerate(group):
                    a,b=selected[i][::2],selected[i][1::2]
                    copies[2*n+j,a]=proposed[i,a]; copies[3*n+j,b]=proposed[i,b]
                changed=model(copies,attention_mask=attention[rows].repeat(4,1)).logits
                changed[...,mask_id]=-torch.inf; future=changed.argmax(-1)
                work['forward_calls']+=1; work['sequence_evaluations']+=4*n
                for j,i in enumerate(group):
                    pos=selected[i]; conflict=torch.zeros(len(pos),device=x.device,dtype=torch.bool)
                    for parity,control_row,branch_row in ((1,j,2*n+j),(0,n+j,3*n+j)):
                        indices=torch.arange(parity,len(pos),2,device=x.device); target=pos[indices]
                        # Do not attribute a batch-shape mismatch to a reveal.
                        comparable=future[control_row,target]==proposed[i,target]
                        conflict[indices]=comparable&(future[branch_row,target]!=proposed[i,target])
                        work['numerical_mismatches_by_row'][i]+=int((~comparable).sum())
                    chosen=stable_subset(pos,conflict,row(mode,i))
                    work['probed_batches_by_row'][i]+=1; work['conflicts_by_row'][i]+=int(conflict.sum())
                    work['count_reductions_by_row'][i]+=int(len(chosen)<len(pos))
                    work['deferred_tokens_by_row'][i]+=len(pos)-len(chosen)
                    work['singleton_fallbacks_by_row'][i]+=int(bool(conflict.all()))
                    work['forward_work_by_row'][i]+=4; work['branch_work_by_row'][i]+=2
                    work['probe_control_work_by_row'][i]+=2; selected[i]=chosen
                del copies,changed,future
        reveal=torch.zeros_like(masked)
        for i,pos in enumerate(selected): reveal[i,pos]=True
        work['reveal_counts'].append(reveal[:,prefix:].sum(-1).tolist())
        x[reveal]=proposed[reveal]; previous=hidden.detach(); work['steps_taken']+=1
        del out,logits,hidden,states,proposed,confidence,entropy
        if progress and (step%16==15 or step==steps-1): progress(dict(work))
    work['unfinished_masks_by_row']=(x[:,prefix:]==mask_id).sum(-1).tolist()
    if sum(work['forward_work_by_row'])!=work['sequence_evaluations']: raise ValueError('incorrect work accounting')
    last_metrics=work
    return x
