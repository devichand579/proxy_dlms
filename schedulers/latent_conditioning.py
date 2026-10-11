"""Rank by frozen-model evidence from the original conditioning context.

The comparison is a pointwise model likelihood ratio, not calibrated correctness
or true-data mutual information. Generated anchors remain visible in both probes.
"""
import torch
from schedulers.latent import _eb_prefix,_target_temporal_risk
from schedulers.utils import get_mask_by_confidence,token_entropy

last_metrics = {}


@torch.no_grad()
def generate_conditioning(model,prompt,attention_mask,*,steps=128,gen_length=512,
                          mask_id=126336,eos_id=126081,context_beta=.5,
                          stability_beta=.15,branch_batch_size=320,progress=None,
                          initial_canvas=None,full_attention=None):
    global last_metrics
    batch,prefix = prompt.shape
    if steps<1 or branch_batch_size<2 or branch_batch_size%2:
        raise ValueError('positive steps and an even branch budget required')
    beta = context_beta if isinstance(context_beta,(list,tuple)) else [context_beta]*batch
    if len(beta)!=batch or any(v<0 for v in beta): raise ValueError('one nonnegative weight per row required')
    x = torch.full((batch,prefix+gen_length),mask_id,dtype=torch.long,device=prompt.device) if initial_canvas is None else initial_canvas.clone()
    if x.shape!=(batch,prefix+gen_length): raise ValueError('canvas shape mismatch')
    x[:,:prefix] = prompt
    attention = torch.cat([attention_mask,torch.ones(batch,gen_length,dtype=attention_mask.dtype,device=x.device)],-1) if full_attention is None else full_attention
    if attention.shape!=x.shape: raise ValueError('attention shape mismatch')
    # Only initially visible, attended positions are the conditioning context.
    # Future generated anchors are preserved in the context-removal probe.
    conditioning = (x!=mask_id)&attention.bool()
    has_conditioning = conditioning.any(-1).tolist()
    head = model.get_output_embeddings(); previous = None
    work = dict(steps_taken=0,forward_calls=0,sequence_evaluations=0,
        forward_work_by_row=[0]*batch,branch_work_by_row=[0]*batch,
        probe_control_work_by_row=[0]*batch,reveal_counts=[],
        evidence_sum=0.,evidence_abs_sum=0.,evidence_measurements=0,
        context_beta=context_beta,branch_batch_size=branch_batch_size,
        conditioning_positions_by_row=conditioning.sum(-1).tolist(),
        auxiliary_training=False,matched_probe_control=True,
        generated_anchors_masked_in_probe=False)
    def account(rows,kind=None):
        work['forward_calls'] += 1; work['sequence_evaluations'] += len(rows)
        for row in rows:
            work['forward_work_by_row'][row] += 1
            if kind: work[kind][row] += 1
    for step in range(steps):
        eligible = [(x[row,prefix:]==mask_id).nonzero().flatten()+prefix for row in range(batch)]
        if not any(len(pos) for pos in eligible): break
        captured = []
        hook = head.register_forward_pre_hook(lambda _m,inputs:captured.append(inputs[0].detach()))
        try: out = model(x,attention_mask=attention)
        finally: hook.remove()
        logits,hidden = out.logits,captured[0]; logits[...,mask_id] = -torch.inf
        predicted = logits.argmax(-1); confidence = get_mask_by_confidence(logits,predicted)
        entropy = token_entropy(logits,eps=1e-9); account(list(range(batch)))
        scores = []
        for row,pos in enumerate(eligible):
            score = confidence[row,pos].clamp_min(1e-30).log()
            if previous is not None and len(pos):
                risk = _target_temporal_risk(logits[row,pos],hidden[row,pos],previous[row,pos],head.weight,128)
                score -= stability_beta*(risk/risk.mean().clamp_min(1e-8)).clamp(max=4)
            scores.append(score)
        active = [row for row,pos in enumerate(eligible) if len(pos) and beta[row] and has_conditioning[row]]
        for offset in range(0,len(active),branch_batch_size//2):
            group = active[offset:offset+branch_batch_size//2]
            rows = torch.tensor(group,device=x.device); n = len(group)
            unconditioned = x[rows].clone(); unconditioned[conditioning[rows]] = mask_id
            paired = torch.cat([x[rows],unconditioned])
            changed = model(paired,attention_mask=attention[rows].repeat(2,1)).logits
            changed[...,mask_id] = -torch.inf
            cf = get_mask_by_confidence(changed,predicted[rows].repeat(2,1))
            for j,row in enumerate(group):
                pos = eligible[row]
                evidence = cf[j,pos].clamp_min(1e-30).log()-cf[n+j,pos].clamp_min(1e-30).log()
                normalized = (evidence/evidence.abs().mean().clamp_min(1e-8)).clamp(-4,4)
                scores[row] += beta[row]*normalized
                work['evidence_sum'] += float(evidence.sum())
                work['evidence_abs_sum'] += float(evidence.abs().sum())
                work['evidence_measurements'] += len(pos)
            account(group+group)
            for row in group:
                work['branch_work_by_row'][row] += 1
                work['probe_control_work_by_row'][row] += 1
            del paired,unconditioned,changed,cf
        reveal = torch.zeros_like(x,dtype=torch.bool)
        for row,pos in enumerate(eligible):
            if len(pos): reveal[row,_eb_prefix(pos,scores[row],entropy[row,pos])] = True
        work['reveal_counts'].append(reveal[:,prefix:].sum(-1).tolist())
        x[reveal] = predicted[reveal]; previous = hidden.detach(); work['steps_taken'] += 1
        del out,logits,hidden,captured,predicted,confidence,entropy
        if progress and (step%16==15 or step==steps-1): progress(dict(work))
    work['unfinished_masks_by_row'] = (x[:,prefix:]==mask_id).sum(-1).tolist()
    if sum(work['forward_work_by_row'])!=work['sequence_evaluations']: raise ValueError('incorrect sequence accounting')
    last_metrics = work
    return x
