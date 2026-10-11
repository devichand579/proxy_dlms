"""Measure singleton reveal effects, then combine outgoing gain and readiness.

Entropy change at surviving targets is a model utility proxy, not mutual
information or correctness. Target weights come from frozen-model geometry;
no trained readout, answers or tests enter selection.
"""
import torch

from schedulers.latent import _eb_prefix,_target_temporal_risk,_fisher_basis,_fisher_project,_unit
from schedulers.utils import get_mask_by_confidence,token_entropy

last_metrics = {}


def latent_density_weights(hidden,logits,head_weight):
    """Balance repeated output-Fisher representations without fitting a policy.

    Squared cosine density is a redundancy heuristic, not MI or correctness.
    Identical states share weight; zero-variation states fall back to uniform.
    """
    if len(hidden)<2: return torch.ones(len(hidden),device=hidden.device)
    probability,weights,_ = _fisher_basis(logits,head_weight,128)
    features = _fisher_project(hidden,probability,weights)
    features -= features.mean(0,keepdim=True)
    if float(features.square().sum())<1e-12: return torch.ones(len(hidden),device=hidden.device)
    vectors = _unit(features)
    density = (vectors@vectors.T).square().sum(-1).clamp_min(1.)
    weight = density.reciprocal()
    return weight/weight.mean().clamp_min(1e-8)


@torch.no_grad()
def generate_finite(model,prompt,attention_mask,*,steps=128,gen_length=512,
                    mask_id=126336,eos_id=126081,probe_every=16,candidates=8,
                    branch_batch_size=160,gain_beta=.25,risk_eta=1.,stability_beta=.15,
                    target_balance=False,pool_balance=False,matched_probe_control=False,progress=None,
                    initial_canvas=None,full_attention=None):
    global last_metrics
    batch,prefix = prompt.shape
    if min(steps,candidates,probe_every,branch_batch_size)<1:
        raise ValueError('positive generation and probe budgets required')
    def row_value(value,row): return value[row] if isinstance(value,(list,tuple)) else value
    x = torch.full((batch,prefix+gen_length),mask_id,dtype=torch.long,device=prompt.device) if initial_canvas is None else initial_canvas.clone()
    x[:,:prefix] = prompt
    attention = torch.cat([attention_mask,torch.ones(batch,gen_length,dtype=attention_mask.dtype,device=x.device)],-1) if full_attention is None else full_attention
    head = model.get_output_embeddings(); previous = None
    work = {'forward_calls':0,'sequence_evaluations':0,'steps_taken':0,
            'forward_work_by_row':[0]*batch,'branch_work_by_row':[0]*batch,
            'probe_control_work_by_row':[0]*batch,
            'candidates':candidates,'probe_every':probe_every,'gain_beta':gain_beta,
            'risk_eta':risk_eta,'target_balance':target_balance,'pool_balance':pool_balance,
            'matched_probe_control':matched_probe_control,
            'auxiliary_training':False,
            'source_entropy_counted':False,'branch_batch_size':branch_batch_size}
    for step in range(steps):
        masked = x==mask_id
        if not masked[:,prefix:].any(): break
        captured = []
        hook = head.register_forward_pre_hook(lambda _m,inputs:captured.append(inputs[0].detach()))
        try: out = model(x,attention_mask=attention)
        finally: hook.remove()
        logits,hidden = out.logits,captured[0]
        logits[...,mask_id] = -torch.inf
        predicted = logits.argmax(-1); conf = get_mask_by_confidence(logits,predicted)
        entropy = token_entropy(logits,eps=1e-9)
        work['forward_calls'] += 1; work['sequence_evaluations'] += batch
        for row in range(batch): work['forward_work_by_row'][row] += 1
        eligible = [masked[row,prefix:].nonzero().flatten()+prefix for row in range(batch)]
        relevance = []
        for row,pos in enumerate(eligible):
            need_balance = row_value(target_balance,row) or row_value(pool_balance,row)
            weight = latent_density_weights(hidden[row,pos],logits[row,pos],head.weight) if need_balance and len(pos) else torch.ones(len(pos),device=x.device)
            relevance.append(weight)
        gains = [torch.zeros(len(pos),device=x.device) for pos in eligible]
        measured_masks = [torch.zeros(len(pos),device=x.device,dtype=torch.bool) for pos in eligible]
        if step%probe_every==0:
            actions = []
            for row,pos in enumerate(eligible):
                if not len(pos) or not row_value(gain_beta,row): continue
                ranking = conf[row,pos].clamp_min(1e-30).log()
                if row_value(pool_balance,row): ranking += .5*relevance[row].clamp_min(1e-6).log()
                for index in ranking.argsort(descending=True,stable=True)[:candidates].tolist():
                    actions.append((row,index,int(pos[index])))
            for offset in range(0,len(actions),branch_batch_size):
                group = actions[offset:offset+branch_batch_size]
                rows = torch.tensor([row for row,_,_ in group],device=x.device)
                # ROCm output numerics can depend on batch shape. Match the
                # unchanged and revealed inputs when isolating a reveal effect.
                if matched_probe_control:
                    control_logits = model(x[rows],attention_mask=attention[rows]).logits
                    control_logits[...,mask_id] = -torch.inf
                    control_entropy = token_entropy(control_logits,eps=1e-9)
                    del control_logits
                    work['forward_calls'] += 1; work['sequence_evaluations'] += len(group)
                    for row,_,_ in group:
                        work['forward_work_by_row'][row] += 1
                        work['probe_control_work_by_row'][row] += 1
                branch = x[rows].clone()
                for j,(row,_,source) in enumerate(group): branch[j,source] = predicted[row,source]
                changed = model(branch,attention_mask=attention[rows]).logits
                changed[...,mask_id] = -torch.inf
                changed_entropy = token_entropy(changed,eps=1e-9)
                work['forward_calls'] += 1; work['sequence_evaluations'] += len(group)
                for j,(row,index,source) in enumerate(group):
                    pos = eligible[row]; surviving = pos!=source
                    weights = relevance[row] if row_value(target_balance,row) else torch.ones_like(relevance[row])
                    # Candidate-specific targets: only this source is excluded.
                    unchanged_entropy = control_entropy[j,pos] if matched_probe_control else entropy[row,pos]
                    gains[row][index] = ((unchanged_entropy-changed_entropy[j,pos])*weights)[surviving].sum()
                    measured_masks[row][index] = True
                    work['forward_work_by_row'][row] += 1; work['branch_work_by_row'][row] += 1
                del changed,changed_entropy,branch
                if matched_probe_control: del control_entropy
        reveal = torch.zeros_like(masked)
        for row,pos in enumerate(eligible):
            if not len(pos): continue
            ranking = conf[row,pos].clamp_min(1e-30).log()
            if previous is not None:
                risk = _target_temporal_risk(logits[row,pos],hidden[row,pos],previous[row,pos],head.weight,128)
                ranking -= stability_beta*(risk/risk.mean().clamp_min(1e-8)).clamp(max=4)
            measured = measured_masks[row]
            if measured.any():
                relative = gains[row][measured]/gains[row][measured].abs().mean().clamp_min(1e-8)
                relative = relative.clamp(-4,4)
                damage = (1-conf[row,pos[measured]])*relative.abs()
                ranking[measured] += row_value(gain_beta,row)*(relative-row_value(risk_eta,row)*damage)
            selected = _eb_prefix(pos,ranking,entropy[row,pos])
            reveal[row,selected] = True
        x[reveal] = predicted[reveal]; previous = hidden.detach()
        work['steps_taken'] += 1
        del out,logits,hidden,captured,predicted,conf,entropy
        if progress and (step%16==15 or step==steps-1): progress(dict(work))
    work['unfinished_masks_by_row'] = (x[:,prefix:]==mask_id).sum(-1).tolist()
    if sum(work['forward_work_by_row'])!=work['sequence_evaluations']:
        raise ValueError('per-row and total forward counts differ')
    last_metrics = work
    return x
