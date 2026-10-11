"""Online attention to unresolved positions, with frozen model and EB counts.

Attention is an internal read pattern, not mutual information or correctness.
The model's original attention computation remains unchanged.
"""
from contextlib import contextmanager
import torch
from schedulers.latent import _eb_prefix,_target_temporal_risk
from schedulers.utils import get_mask_by_confidence,token_entropy

last_metrics = {}


def unresolved_read_risk(q,k,key_uncertainty,bias=None,query_chunk=32):
    """Mean-head attention-weighted uncertainty, excluding each query itself."""
    if q.shape[-2]!=k.shape[-2] or key_uncertainty.shape!=(q.shape[0],k.shape[-2]):
        raise ValueError('uncached self-attention and matching key weights required')
    if q.shape[1]!=k.shape[1]:
        if q.shape[1]%k.shape[1]: raise ValueError('incompatible query/key heads')
        k=k.repeat_interleave(q.shape[1]//k.shape[1],dim=1)
    keys=k.float().transpose(-1,-2); result=[]
    for start in range(0,q.shape[-2],query_chunk):
        stop=min(start+query_chunk,q.shape[-2])
        scores=(q[:,:,start:stop].float()@keys)*q.shape[-1]**-.5
        if bias is not None:
            local=bias[...,start:stop,:] if bias.shape[-2]!=1 else bias
            scores=scores.masked_fill(~local,float('-inf')) if local.dtype==torch.bool else scores+local.float()
        probabilities=scores.softmax(-1)
        incoming=(probabilities*key_uncertainty[:,None,None,:]).sum(-1)
        indices=torch.arange(start,stop,device=q.device)
        diagonal=probabilities.gather(-1,indices[None,None,:,None].expand(q.shape[0],q.shape[1],-1,1)).squeeze(-1)
        incoming-=diagonal*key_uncertainty[:,None,start:stop]
        result.append(incoming.mean(1).clamp_min(0))
    return torch.cat(result,-1)


@contextmanager
def capture_attention_reads(blocks):
    captured=[]; restore=[]
    try:
        for block in blocks:
            name='_scaled_dot_product_attention'
            existed=name in block.__dict__; previous=block.__dict__.get(name)
            original=getattr(block,name); restore.append((block,name,existed,previous))
            def wrapper(q,k,v,*args,_original=original,**kwargs):
                mask=kwargs.get('attn_mask',args[0] if args else None)
                captured.append((q.detach(),k.detach(),mask.detach() if mask is not None else None))
                return _original(q,k,v,*args,**kwargs)
            setattr(block,name,wrapper)
        yield captured
    finally:
        for block,name,existed,previous in reversed(restore):
            if existed: setattr(block,name,previous)
            else: delattr(block,name)


@torch.no_grad()
def generate_attention(model,prompt,attention_mask,*,steps=128,gen_length=512,
                       mask_id=126336,eos_id=126081,read_beta=.15,
                       uncertainty_weighted=True,attention_layers=1,
                       stability_beta=.15,progress=None,initial_canvas=None,
                       full_attention=None):
    global last_metrics
    batch,prefix=prompt.shape
    def row(value,index): return value[index] if isinstance(value,(list,tuple)) else value
    for setting in (read_beta,uncertainty_weighted):
        if isinstance(setting,(list,tuple)) and len(setting)!=batch: raise ValueError('one setting per sequence required')
    if steps<1 or attention_layers<1 or any(row(read_beta,i)<0 for i in range(batch)): raise ValueError('invalid budget or weight')
    active=any(row(read_beta,i) for i in range(batch))
    blocks=list(model.model.transformer.blocks)[-attention_layers:] if active else []
    if active and len(blocks)!=attention_layers: raise ValueError('not enough attention layers')
    x=torch.full((batch,prefix+gen_length),mask_id,dtype=torch.long,device=prompt.device) if initial_canvas is None else initial_canvas.clone()
    if x.shape!=(batch,prefix+gen_length): raise ValueError('canvas shape mismatch')
    x[:,:prefix]=prompt
    attention=torch.cat([attention_mask,torch.ones(batch,gen_length,device=x.device,dtype=attention_mask.dtype)],-1) if full_attention is None else full_attention
    if attention.shape!=x.shape: raise ValueError('attention shape mismatch')
    head=model.get_output_embeddings(); previous=None
    work=dict(steps_taken=0,forward_calls=0,sequence_evaluations=0,forward_work_by_row=[0]*batch,
        branch_work_by_row=[0]*batch,probe_control_work_by_row=[0]*batch,reveal_counts=[],
        read_beta=read_beta,uncertainty_weighted=uncertainty_weighted,attention_layers=attention_layers,
        auxiliary_training=False,self_read_counted=False,risk_sum=0.,risk_measurements=0)
    for step in range(steps):
        masked=x==mask_id
        if not masked[:,prefix:].any(): break
        states=[]; hook=head.register_forward_pre_hook(lambda _m,inputs:states.append(inputs[0].detach()))
        try:
            with capture_attention_reads(blocks) as captured:
                out=model(x,attention_mask=attention)
        finally: hook.remove()
        logits,hidden=out.logits,states[0]; logits[...,mask_id]=-torch.inf
        proposed=logits.argmax(-1); confidence=get_mask_by_confidence(logits,proposed)
        entropy=token_entropy(logits,eps=1e-9)
        incoming=None
        if active:
            if len(captured)!=len(blocks): raise ValueError('attention capture count differs from selected blocks')
            uncertainty=masked.float()*attention.bool()
            weighted=torch.tensor([row(uncertainty_weighted,i) for i in range(batch)],device=x.device)
            uncertainty*=torch.where(weighted[:,None],1-confidence,torch.ones_like(confidence))
            incoming=sum(unresolved_read_risk(q,k,uncertainty,bias) for q,k,bias in captured)/len(captured)
        work['forward_calls']+=1; work['sequence_evaluations']+=batch
        for i in range(batch): work['forward_work_by_row'][i]+=1
        reveal=torch.zeros_like(masked)
        for i in range(batch):
            pos=masked[i,prefix:].nonzero().flatten()+prefix
            if not len(pos): continue
            score=confidence[i,pos].clamp_min(1e-30).log()
            if previous is not None:
                risk=_target_temporal_risk(logits[i,pos],hidden[i,pos],previous[i,pos],head.weight,128)
                score-=stability_beta*(risk/risk.mean().clamp_min(1e-8)).clamp(max=4)
            if row(read_beta,i):
                risk=incoming[i,pos]
                score-=row(read_beta,i)*(risk/risk.mean().clamp_min(1e-8)).clamp(max=4)
                work['risk_sum']+=float(risk.sum()); work['risk_measurements']+=len(pos)
            reveal[i,_eb_prefix(pos,score,entropy[i,pos])]=True
        work['reveal_counts'].append(reveal[:,prefix:].sum(-1).tolist())
        x[reveal]=proposed[reveal]; previous=hidden.detach(); work['steps_taken']+=1
        del out,logits,hidden,states,captured,proposed,confidence,entropy,incoming
        if progress and (step%16==15 or step==steps-1): progress(dict(work))
    work['unfinished_masks_by_row']=(x[:,prefix:]==mask_id).sum(-1).tolist()
    if sum(work['forward_work_by_row'])!=work['sequence_evaluations']: raise ValueError('incorrect work accounting')
    last_metrics=work
    return x
