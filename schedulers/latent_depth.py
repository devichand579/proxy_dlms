"""Online final-layer activation changes, with EB count and frozen weights.

Intermediate states are normalized with the checkpoint's final norm. Their
head projection is a computational diagnostic, not calibrated correctness.
"""
import torch
from schedulers.latent import _eb_prefix,_target_temporal_risk
from schedulers.utils import get_mask_by_confidence,token_entropy

last_metrics = {}


def depth_fields(logits,current,earlier,weight,vocab=128):
    risks,supports = [],[]
    for start in range(0,len(current),32):
        stop = min(start+32,len(current))
        local = logits[start:stop].float()
        values,tokens = local.topk(min(vocab,local.shape[-1]),-1)
        p = values.softmax(-1); delta = current[start:stop].float()-earlier[start:stop].float()
        movement = torch.einsum('nd,nvd->nv',delta,weight[tokens].float())
        centered = movement-movement[:,:1]
        expected = (p*centered).sum(-1)
        risks.append(.5*(p*(centered-expected[:,None]).square()).sum(-1))
        # Use actual argmax, including its deterministic behavior on ties.
        proposal = local.argmax(-1)
        chosen = (delta*weight[proposal].float()).sum(-1)-movement[:,0]
        supports.append(chosen-expected)
    return torch.cat(risks),torch.cat(supports)


@torch.no_grad()
def generate_depth(model,prompt,attention_mask,*,steps=128,gen_length=512,
                   mask_id=126336,eos_id=126081,depth_back=4,
                   depth_risk_beta=0.,depth_support_beta=0.,stability_beta=.15,
                   progress=None,initial_canvas=None,full_attention=None):
    global last_metrics
    batch,prefix = prompt.shape
    def rows(value): return list(value) if isinstance(value,(tuple,list)) else [value]*batch
    risk_beta,support_beta = rows(depth_risk_beta),rows(depth_support_beta)
    if steps<1 or depth_back<1 or any(len(v)!=batch for v in (risk_beta,support_beta)):
        raise ValueError('invalid step/depth budget or per-row configuration')
    needs_depth = any(risk_beta) or any(support_beta)
    if needs_depth and getattr(model.config,'block_group_size',1)!=1:
        raise ValueError('depth capture currently requires individual block states')
    x = torch.full((batch,prefix+gen_length),mask_id,dtype=torch.long,device=prompt.device) if initial_canvas is None else initial_canvas.clone()
    if x.shape!=(batch,prefix+gen_length): raise ValueError('canvas shape mismatch')
    x[:,:prefix] = prompt
    attention = torch.cat([attention_mask,torch.ones(batch,gen_length,device=x.device,dtype=attention_mask.dtype)],-1) if full_attention is None else full_attention
    if attention.shape!=x.shape: raise ValueError('attention shape mismatch')
    head = model.get_output_embeddings(); previous = None
    work = dict(steps_taken=0,forward_calls=0,sequence_evaluations=0,
        forward_work_by_row=[0]*batch,branch_work_by_row=[0]*batch,probe_control_work_by_row=[0]*batch,
        depth_back=depth_back,depth_risk_beta=depth_risk_beta,depth_support_beta=depth_support_beta,
        depth_risk_sum=0.,depth_support_sum=0.,depth_measurements=0,
        auxiliary_training=False,reveal_counts=[],hidden_states_retained_for_depth=needs_depth)
    for step in range(steps):
        eligible = [(x[row,prefix:]==mask_id).nonzero().flatten()+prefix for row in range(batch)]
        if not any(len(pos) for pos in eligible): break
        captured = []
        hook = head.register_forward_pre_hook(lambda _m,inputs:captured.append(inputs[0].detach()))
        try: out = model(x,attention_mask=attention,output_hidden_states=needs_depth)
        finally: hook.remove()
        logits,hidden = out.logits,captured[0]; logits[...,mask_id] = -torch.inf
        earlier = None
        if needs_depth:
            if out.hidden_states is None or len(out.hidden_states)<=depth_back:
                raise ValueError('not enough intermediate states for requested depth')
            earlier = model.model.transformer.ln_f(out.hidden_states[-1-depth_back])
        predicted = logits.argmax(-1); confidence = get_mask_by_confidence(logits,predicted)
        entropy = token_entropy(logits,eps=1e-9)
        work['forward_calls'] += 1; work['sequence_evaluations'] += batch
        for row in range(batch): work['forward_work_by_row'][row] += 1
        reveal = torch.zeros_like(x,dtype=torch.bool)
        for row,pos in enumerate(eligible):
            if not len(pos): continue
            score = confidence[row,pos].clamp_min(1e-30).log()
            if previous is not None:
                risk = _target_temporal_risk(logits[row,pos],hidden[row,pos],previous[row,pos],head.weight,128)
                score -= stability_beta*(risk/risk.mean().clamp_min(1e-8)).clamp(max=4)
            if risk_beta[row] or support_beta[row]:
                risk,support = depth_fields(logits[row,pos],hidden[row,pos],earlier[row,pos],head.weight)
                score -= risk_beta[row]*(risk/risk.mean().clamp_min(1e-8)).clamp(max=4)
                score += support_beta[row]*(support/support.abs().mean().clamp_min(1e-8)).clamp(-4,4)
                work['depth_risk_sum'] += float(risk.sum()); work['depth_support_sum'] += float(support.sum())
                work['depth_measurements'] += len(pos)
            reveal[row,_eb_prefix(pos,score,entropy[row,pos])] = True
        work['reveal_counts'].append(reveal[:,prefix:].sum(-1).tolist())
        x[reveal] = predicted[reveal]; previous = hidden.detach(); work['steps_taken'] += 1
        del out,logits,hidden,captured,predicted,confidence,entropy,earlier
        if progress and (step%16==15 or step==steps-1): progress(dict(work))
    work['unfinished_masks_by_row'] = (x[:,prefix:]==mask_id).sum(-1).tolist()
    if sum(work['forward_work_by_row'])!=work['sequence_evaluations']: raise ValueError('incorrect work accounting')
    last_metrics = work
    return x
