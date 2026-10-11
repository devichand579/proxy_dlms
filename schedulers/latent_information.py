"""Intrinsic multi-value response information with an explicit consistency term.

The joint is constructed from two source alternatives and model branch responses.
It is a model diagnostic, not data-distribution MI or a correctness certificate.
Targets retain a fixed top-vocabulary partition plus an OTHER bucket.
"""
import torch

from schedulers.latent import _eb_prefix,_target_temporal_risk
from schedulers.utils import get_mask_by_confidence,token_entropy
from schedulers.latent_depth import depth_fields

last_metrics = {}


def partition_probabilities(logits,tokens):
    """Preserve full mass in a fixed vocabulary partition, including OTHER."""
    logz = logits.float().logsumexp(-1)
    selected = (logits.gather(-1,tokens).float()-logz[...,None]).exp()
    other = (1-selected.sum(-1,keepdim=True)).clamp_min(0)
    probability = torch.cat([selected,other],-1)
    return probability/probability.sum(-1,keepdim=True).clamp_min(1e-30)


def response_information(conditional,rho,baseline):
    """Exact local MI and mixture defect on a common categorical partition.

    conditional: [sources, alternatives, targets, categories]
    rho: [sources, alternatives]; baseline: [sources, targets, categories].
    """
    mixture = (conditional*rho[...,None,None]).sum(1)
    def kl(p,q):
        return (p*(p.clamp_min(1e-30).log()-q.clamp_min(1e-30).log())).sum(-1)
    information = (kl(conditional,mixture[:,None])*rho[...,None]).sum(1).clamp_min(0)
    defect = kl(mixture,baseline).clamp_min(0)
    return information,defect


def decoded_prefix_weights(canvas,positions,prefix,eos_id):
    """Target relevance from the model's already committed end marker.

    Include the marker position; it is not a fitted code/body classifier.
    No marker means the whole canvas remains relevant.
    """
    ends = (canvas[prefix:]==eos_id).nonzero().flatten()
    if not len(ends): return torch.ones(len(positions),device=canvas.device)
    return (positions<=prefix+ends[0]).float()


def probe_conflict_prefix(positions,ranking,entropy,confidence,pool,conflict,budget=.1):
    """Heuristic count: measured proposal surprise plus joint conflict.

    Unprobed positions keep EB entropy costs. This is not a TC upper bound.
    Pair conflicts measure drops in support for other proposed argmax values.
    """
    if not len(positions): return positions
    cost = entropy.double().clone()
    cost[pool] = -confidence[pool].double().clamp_min(1e-30).log()
    source_lookup = {int(index):i for i,index in enumerate(pool.tolist())}
    order = ranking.argsort(descending=True,stable=True).tolist()
    picked = []; total = 0.; maximum = 0.; joint_conflict = 0.
    for index in order:
        charge = float(cost[index]); total += charge; maximum = max(maximum,charge)
        if index in source_lookup:
            local = source_lookup[index]
            for previous_index in picked:
                if previous_index in source_lookup:
                    previous_local = source_lookup[previous_index]
                    joint_conflict += .5*float(conflict[local,previous_local]+conflict[previous_local,local])
        if total-maximum+joint_conflict>budget+1e-12: break
        picked.append(index)
    return positions[torch.tensor(picked,device=positions.device,dtype=torch.long)]


@torch.no_grad()
def generate_information(model,prompt,attention_mask,*,steps=128,gen_length=512,
                         mask_id=126336,eos_id=126081,candidates=4,probe_every=16,
                         branch_batch_size=160,target_vocab=128,gain_beta=.5,
                         risk_eta=1.,incoming_beta=0.,stability_beta=.15,
                         prefix_targets=False,prefix_pool=False,
                         source_mass_weighting=False,count_mode='eb',count_budget=.1,
                         depth_risk_beta=0.,depth_back=4,
                         progress=None,initial_canvas=None,full_attention=None):
    global last_metrics
    batch,prefix = prompt.shape
    if min(steps,candidates,probe_every,target_vocab)<1 or branch_batch_size<2 or branch_batch_size%2:
        raise ValueError('positive budgets and an even branch batch size required')
    def value(v,row): return v[row] if isinstance(v,(list,tuple)) else v
    for setting in (gain_beta,risk_eta,incoming_beta,prefix_targets,prefix_pool,source_mass_weighting,count_mode,depth_risk_beta):
        if isinstance(setting,(list,tuple)) and len(setting)!=batch:
            raise ValueError('one setting per sequence required')
    if count_budget<0 or any(value(count_mode,row) not in ('eb','probe_surprise') for row in range(batch)):
        raise ValueError('unknown count mode or negative budget')
    needs_depth=any(value(depth_risk_beta,row) for row in range(batch))
    if depth_back<1 or (needs_depth and getattr(model.config,'block_group_size',1)!=1):
        raise ValueError('invalid depth budget or grouped block states')
    x = torch.full((batch,prefix+gen_length),mask_id,dtype=torch.long,device=prompt.device) if initial_canvas is None else initial_canvas.clone()
    if x.shape!=(batch,prefix+gen_length): raise ValueError('canvas shape mismatch')
    x[:,:prefix] = prompt
    attention = torch.cat([attention_mask,torch.ones(batch,gen_length,dtype=attention_mask.dtype,device=x.device)],-1) if full_attention is None else full_attention
    head = model.get_output_embeddings(); previous = None
    work = dict(steps_taken=0,forward_calls=0,sequence_evaluations=0,
        forward_work_by_row=[0]*batch,branch_work_by_row=[0]*batch,
        probe_control_work_by_row=[0]*batch,reveal_counts=[],
        source_retained_mass_sum=0.,source_measurements=0,
        target_retained_mass_sum=0.,target_measurements=0,
        information_sum=0.,consistency_defect_sum=0.,response_measurements=0,
        auxiliary_training=False,source_entropy_counted=False,
        matched_probe_control=True,target_partition_other_bucket=True,
        candidates=candidates,probe_every=probe_every,target_vocab=target_vocab,branch_batch_size=branch_batch_size,
        gain_beta=gain_beta,risk_eta=risk_eta,incoming_beta=incoming_beta)
    work.update(prefix_targets=prefix_targets,prefix_pool=prefix_pool,
                target_weights_sum=0.,target_weight_measurements=0,
                source_mass_weighting=source_mass_weighting,count_mode=count_mode,
                count_budget=count_budget,count_probe_steps=0,count_changes=0,
                count_extra_reveals=0,count_fewer_reveals=0,
                count_changes_by_row=[0]*batch,count_extra_by_row=[0]*batch,count_fewer_by_row=[0]*batch)
    work.update(depth_risk_beta=depth_risk_beta,depth_back=depth_back)
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
        try: out = model(x,attention_mask=attention,**({'output_hidden_states':True} if needs_depth else {}))
        finally: hook.remove()
        logits,hidden = out.logits,captured[0]
        earlier=None
        if needs_depth:
            if out.hidden_states is None or len(out.hidden_states)<=depth_back: raise ValueError('insufficient layer states')
            earlier=model.model.transformer.ln_f(out.hidden_states[-1-depth_back])
        logits[...,mask_id] = -torch.inf
        predicted = logits.argmax(-1); conf = get_mask_by_confidence(logits,predicted)
        entropy = token_entropy(logits,eps=1e-9); account(list(range(batch)))
        scores = []
        for row,pos in enumerate(eligible):
            score = conf[row,pos].clamp_min(1e-30).log()
            if previous is not None and len(pos):
                risk = _target_temporal_risk(logits[row,pos],hidden[row,pos],previous[row,pos],head.weight,128)
                score -= stability_beta*(risk/risk.mean().clamp_min(1e-8)).clamp(max=4)
            if len(pos) and value(depth_risk_beta,row):
                risk,_=depth_fields(logits[row,pos],hidden[row,pos],earlier[row,pos],head.weight)
                score -= value(depth_risk_beta,row)*(risk/risk.mean().clamp_min(1e-8)).clamp(max=4)
            scores.append(score)
        count_measurements = {}
        if step%probe_every==0:
            pools,tokens,rhos,conditionals,baselines = {},{},{},{},{}
            target_weights,source_support = {},{}
            actions = []
            for row,pos in enumerate(eligible):
                if not len(pos) or not (value(gain_beta,row) or value(incoming_beta,row) or value(count_mode,row)!='eb'): continue
                relevance = decoded_prefix_weights(x[row],pos,prefix,eos_id)
                possible = (relevance>0).nonzero().flatten() if value(prefix_pool,row) else torch.arange(len(pos),device=x.device)
                if not len(possible): continue
                pool = possible[scores[row][possible].argsort(descending=True,stable=True)[:candidates]]
                pools[row] = pool
                target_weights[row] = relevance if value(prefix_targets,row) else torch.ones_like(relevance)
                work['target_weights_sum'] += float(target_weights[row].sum())
                work['target_weight_measurements'] += len(pos)
                tokens[row] = logits[row,pos].topk(min(target_vocab,logits.shape[-1]-1),-1).indices
                source_logits,alternatives = logits[row,pos[pool]].float().topk(2,-1)
                rho = source_logits.softmax(-1); rhos[row] = rho
                retained = conf[row,pos[pool]]/rho[:,0].clamp_min(1e-30)
                source_support[row] = retained.clamp(0,1)
                work['source_retained_mass_sum'] += float(retained.sum())
                work['source_measurements'] += len(pool)
                shape = (len(pool),2,len(pos),tokens[row].shape[-1]+1)
                conditionals[row] = torch.empty(shape,device=x.device)
                baselines[row] = torch.empty((shape[0],shape[2],shape[3]),device=x.device)
                for source,index in enumerate(pool.tolist()):
                    for alternative in range(2):
                        actions.append((row,source,alternative,int(pos[index]),int(alternatives[source,alternative])))
            for offset in range(0,len(actions),branch_batch_size):
                group = actions[offset:offset+branch_batch_size]
                row_ids = [a[0] for a in group]; rows = torch.tensor(row_ids,device=x.device)
                control_logits = model(x[rows],attention_mask=attention[rows]).logits
                control_logits[...,mask_id] = -torch.inf; control_p = {}
                for j,(row,source,alternative,_,_) in enumerate(group):
                    if row not in control_p:
                        control_p[row] = partition_probabilities(control_logits[j,eligible[row]],tokens[row])
                    if alternative==0: baselines[row][source] = control_p[row]
                del control_logits
                account(row_ids,'probe_control_work_by_row')
                branch = x[rows].clone()
                for j,(_,_,_,source,token) in enumerate(group): branch[j,source] = token
                branch_logits = model(branch,attention_mask=attention[rows]).logits
                branch_logits[...,mask_id] = -torch.inf
                for j,(row,source,alternative,_,_) in enumerate(group):
                    p = partition_probabilities(branch_logits[j,eligible[row]],tokens[row])
                    conditionals[row][source,alternative] = p
                    work['target_retained_mass_sum'] += float((1-p[:,-1]).sum())
                    work['target_measurements'] += len(p)
                del branch_logits,branch,control_p
                account(row_ids,'branch_work_by_row')
            for row,pool in pools.items():
                info,defect = response_information(conditionals[row],rhos[row],baselines[row])
                info[torch.arange(len(pool),device=x.device),pool] = 0.
                defect[torch.arange(len(pool),device=x.device),pool] = 0.
                if value(count_mode,row)!='eb':
                    conflict = (baselines[row][:,pool,0].clamp_min(1e-30).log()
                                -conditionals[row][:,0,pool,0].clamp_min(1e-30).log()).clamp_min(0)
                    conflict.fill_diagonal_(0)
                    count_measurements[row] = (pool,conflict)
                if value(source_mass_weighting,row):
                    # r * I(Y;Z | Y in probed support) is the measured-support
                    # contribution to a full-source joint's MI. Omitted source
                    # conditionals remain unknown; do not invent their branches.
                    info *= source_support[row][:,None]
                    defect *= source_support[row][:,None]
                outgoing = (info*target_weights[row]).sum(-1)
                damage = ((info+defect)*target_weights[row]).sum(-1)
                scale = outgoing.mean().clamp_min(1e-8)
                bonus = (outgoing/scale).clamp(max=4)
                harmful = (damage/damage.mean().clamp_min(1e-8)).clamp(max=4)
                bonus -= value(risk_eta,row)*(1-conf[row,eligible[row][pool]])*harmful
                scores[row][pool] += value(gain_beta,row)*bonus
                # Every target excludes its own source intervention when present.
                incoming = (info+defect).sum(0)
                scores[row] -= value(incoming_beta,row)*(incoming/incoming.mean().clamp_min(1e-8)).clamp(max=4)
                work['information_sum'] += float(info.sum())
                work['consistency_defect_sum'] += float(defect.sum())
                work['response_measurements'] += int(info.numel()-len(pool))
            del pools,tokens,rhos,conditionals,baselines,target_weights,source_support
        reveal = torch.zeros_like(x,dtype=torch.bool)
        for row,pos in enumerate(eligible):
            if not len(pos): continue
            selected = _eb_prefix(pos,scores[row],entropy[row,pos])
            if row in count_measurements:
                pool,conflict = count_measurements[row]
                measured = probe_conflict_prefix(pos,scores[row],entropy[row,pos],conf[row,pos],pool,conflict,count_budget)
                delta = len(measured)-len(selected)
                work['count_probe_steps'] += 1; work['count_changes'] += int(delta!=0)
                work['count_extra_reveals'] += max(delta,0); work['count_fewer_reveals'] += max(-delta,0)
                work['count_changes_by_row'][row] += int(delta!=0)
                work['count_extra_by_row'][row] += max(delta,0); work['count_fewer_by_row'][row] += max(-delta,0)
                selected = measured
            reveal[row,selected] = True
        work['reveal_counts'].append(reveal[:,prefix:].sum(-1).tolist())
        x[reveal] = predicted[reveal]; previous = hidden.detach(); work['steps_taken'] += 1
        del out,logits,hidden,captured,predicted,conf,entropy,earlier
        if progress and (step%16==15 or step==steps-1): progress(dict(work))
    work['unfinished_masks_by_row'] = (x[:,prefix:]==mask_id).sum(-1).tolist()
    if sum(work['forward_work_by_row'])!=work['sequence_evaluations']: raise ValueError('incorrect work accounting')
    last_metrics = work
    return x
