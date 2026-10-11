"""One-forward count isolation using observed responses and native attention.

The primary count controllers are probability-free. The confidence/Fisher
ranking and probability-weighted gradient dispersion are labelled diagnostics.
Estimated allowances are not certified data-KL bounds.
"""
import math
from contextlib import nullcontext

import torch

from schedulers.latent import _eb_prefix, _target_temporal_risk
from schedulers.latent_count import decoder_geometry, forward_fields, known_reads
from schedulers.utils import get_mask_by_confidence, token_entropy
from schedulers.latent_context_flow import context_flow_capture, context_support

last_metrics = {}


def fisher_reference_score(logits, hidden, previous, weight):
    """Same confidence/Fisher diagnostic ranking as the count-isolation study."""
    proposed=logits.argmax(-1)
    score=get_mask_by_confidence(logits[None],proposed[None])[0].clamp_min(1e-30).log()
    if previous is not None:
        risk=_target_temporal_risk(logits,hidden,previous,weight,128)
        score-=.15*(risk/risk.mean().clamp_min(1e-8)).clamp(max=4)
    return score


def attention_field(q, k, bias, prefix):
    """Mean native last-block attention between generation positions."""
    if q.shape[1] != k.shape[1]:
        k = k.repeat_interleave(q.shape[1]//k.shape[1],dim=1)
    keys = k.float().transpose(-1,-2)
    rows = []
    for start in range(prefix,q.shape[-2],32):
        stop = min(start+32,q.shape[-2])
        scores = q[:,:,start:stop].float()@keys*q.shape[-1]**-.5
        if bias is not None:
            local = bias[...,start:stop,:] if bias.shape[-2]!=1 else bias
            scores = scores.masked_fill(~local,-torch.inf) if local.dtype==torch.bool else scores+local.float()
        rows.append(scores.softmax(-1).mean(1)[:,:,prefix:])
    result = torch.cat(rows,1)
    result.diagonal(dim1=-2,dim2=-1).zero_()
    return result


def gradient_ratio(weight, current, previous, native, current_ids, previous_ids,
                   active, floor_fraction=1e-3):
    """Uniform union-of-proposals gradient metric; token identities only.

    Variance removes common readout directions. No current logit magnitudes,
    confidence, margin, entropy, or probability weights are used.
    """
    if previous is None:
        return torch.zeros_like(active,dtype=torch.float32)
    ids = torch.cat([current_ids,previous_ids],-1).sort(-1).values
    unique = torch.ones_like(ids,dtype=torch.bool)
    unique[...,1:] = ids[...,1:]!=ids[...,:-1]
    weights = unique.float()/unique.sum(-1,keepdim=True)
    ratios=[];intrinsic=[]
    # Chunk the position axis to limit temporary decoder-gradient storage.
    for start in range(0,current.shape[1],32):
        stop=min(start+32,current.shape[1])
        directions=weight[ids[:,start:stop]].float()
        movement=((current[:,start:stop].float()-previous[:,start:stop].float())[:,:,None]*directions).sum(-1)
        baseline=(native[:,start:stop,:,None].transpose(-1,-2)*directions).sum(-1)
        local=weights[:,start:stop]
        def variance(values):
            mean=(values*local).sum(-1,keepdim=True)
            return ((values-mean).square()*local).sum(-1)
        ratios.append(variance(movement));intrinsic.append(variance(baseline))
    movement=torch.cat(ratios,1);intrinsic=torch.cat(intrinsic,1)
    floor=floor_fraction*(intrinsic*active).sum(-1)/active.sum(-1).clamp_min(1)
    return (movement/intrinsic.clamp_min(floor[:,None].clamp_min(1e-12))).clamp(0.,100.)


def uniform_ratio(projected, previous, native_variance, active, floor_fraction=1e-3):
    if previous is None:
        return torch.zeros_like(active,dtype=torch.float32)
    motion=(projected-previous).square().mean(-1)
    floor=floor_fraction*(native_variance*active).sum(-1)/active.sum(-1).clamp_min(1)
    return (motion/native_variance.clamp_min(floor[:,None].clamp_min(1e-12))).clamp(0.,100.)


def response_density(error, active, quantile=.99, attention=None):
    if not 0 <= quantile <= 1:
        raise ValueError('quantile must lie in [0,1]')
    if not bool(active.any()):return 0.
    field=error
    if attention is not None:
        incoming=(attention*active[None]).sum(-1).clamp(0.,1.)
        field=field*incoming
    return float(torch.quantile(field[active].float(),quantile))


def feedback_count(previous_count, density, tolerance, remaining, max_count=16,
                   growth_cap=2., safety=.8):
    if min(previous_count,remaining,max_count)<1 or tolerance<=0 or density<0:
        raise ValueError('invalid count/response inputs')
    if not math.isfinite(density) or growth_cap<1 or not 0<safety<=1:
        raise ValueError('invalid feedback parameters')
    proposed=previous_count*safety*math.sqrt(tolerance/max(density,1e-12))
    limit=min(remaining,max_count,max(1,int(previous_count*growth_cap)))
    return max(1,min(limit,math.floor(proposed)))


def allowance_prefix(order, error, attention, tolerance, max_count=16,
                     risk_weight=1., pair_weight=1., uncertainty_only=False):
    """Largest ranked prefix within a nonnegative additive interaction cost."""
    if tolerance<=0 or max_count<1 or min(risk_weight,pair_weight)<0:
        raise ValueError('invalid allowance parameters')
    order=order[:max_count]
    if not len(order):return order,dict(cost=0.,forced=False,costs=[])
    local_error=error[order].double().clamp_min(0)
    reads=attention[order][:,order].double()
    factor=local_error if uncertainty_only else 1+local_error
    coupling=.5*(reads+reads.T)*(factor[:,None]*factor[None]).sqrt()
    coupling.fill_diagonal_(0)
    modular=risk_weight*local_error.cumsum(0)
    dependence=pair_weight*coupling.tril(-1).sum(-1).cumsum(0)
    costs=modular+dependence
    feasible=(costs<=tolerance).nonzero().flatten()
    count=int(feasible[-1])+1 if len(feasible) else 1
    return order[:count],dict(cost=float(costs[count-1]),risk=float(modular[count-1]),
        dependence=float(dependence[count-1]),forced=not len(feasible),costs=costs.tolist())


def gradient_dispersion(logits, dictionary):
    """Projected Fisher trace; probabilities are metric weights in this control.

    The fixed dictionary contains projected decoder gradients, centred over
    vocabulary. This diagnostic uses no entropy or maximum-probability feature.
    """
    squared=dictionary.square().sum(-1)
    scale=squared.mean().clamp_min(1e-12)
    values=[]
    for start in range(0,logits.shape[1],32):
        p=logits[:,start:start+32].float().softmax(-1)
        mean=p@dictionary
        trace=p@squared-mean.square().sum(-1)
        values.append((trace.clamp_min(0)/scale).clamp(max=100.))
    return torch.cat(values,1)


def allowance_subset(order, error, attention, tolerance, max_count=16):
    """Keep the best-ranked anchor, then skip unaffordable interactions."""
    pool=order[:max_count*2]
    selected=[];cost=0.;forced=False;skipped=[]
    for index in pool.tolist():
        coupling=0.
        for other in selected:
            coupling+=float(.5*(attention[index,other]+attention[other,index])
                *torch.sqrt((1+error[index])*(1+error[other])))
        addition=float(error[index])+coupling
        if not selected or cost+addition<=tolerance:
            selected.append(index);cost+=addition
            if cost>tolerance:
                forced=True;break
            if len(selected)==max_count:break
        else:skipped.append(index)
    return torch.tensor(selected,device=order.device,dtype=torch.long),dict(
        cost=cost,forced=forced,skipped=skipped)


def committed_end_region(tokens, attended, mask_id, eos_id):
    """Only an actually committed end token defines the candidate boundary."""
    positions=torch.arange(tokens.shape[1],device=tokens.device)[None]
    ends=torch.where((tokens==eos_id)&attended.bool(),positions,tokens.shape[1]).min(-1).values
    return (tokens==mask_id)&attended.bool()&(positions<ends[:,None]),ends


@torch.no_grad()
def generate_allowance(model, prompt, attention_mask, *, steps=128, gen_length=512,
        mask_id=126336, ranking='diagnostic_fisher', controller='feedback_grounded',
        metric='gradient_union', tolerance=.025, quantile=.99, max_count=16,
        growth_cap=2., projection=None, initial_canvas=None, full_attention=None,
        resume_state=None, step_checkpoint=None, progress=None, gradient_dictionary=None,
        context_beta=.05, head_mean=None, context_mode='fraction',
        candidate_region='all',eos_id=126081,frontier_width=32):
    global last_metrics
    if ranking not in ('diagnostic_fisher','activation','context_flow','fisher_context_flow') or metric not in ('uniform','gradient_union','fisher_dispersion'):
        raise ValueError('unknown ranking/metric')
    if controller not in ('feedback','feedback_grounded','feedback_relative','feedback_relative_grounded',
                          'allowance','allowance_grounded','allowance_subset','risk_only','pair_only','singleton','eb'):
        raise ValueError('unknown count controller')
    if min(steps,gen_length,max_count)<1 or tolerance<=0 or growth_cap<1:
        raise ValueError('invalid sizes/tolerance')
    if candidate_region not in ('all','committed_eos','frontier'):raise ValueError('unknown candidate region')
    if frontier_width<1:raise ValueError('positive frontier width required')
    if metric=='fisher_dispersion' and controller not in ('allowance','singleton','eb'):
        raise ValueError('Fisher dispersion is an allowance diagnostic, not response feedback')
    batch,prefix=prompt.shape
    x=(torch.full((batch,prefix+gen_length),mask_id,device=prompt.device,dtype=torch.long)
       if initial_canvas is None else initial_canvas.clone())
    if x.shape!=(batch,prefix+gen_length):raise ValueError('canvas shape mismatch')
    x[:,:prefix]=prompt
    attention=(torch.cat([attention_mask,torch.ones((batch,gen_length),device=x.device,
                  dtype=attention_mask.dtype)],-1) if full_attention is None else full_attention)
    if attention.shape!=x.shape:raise ValueError('attention shape mismatch')
    projection=decoder_geometry(model.get_output_embeddings().weight) if projection is None else projection
    use_flow=ranking in ('context_flow','fisher_context_flow')
    original_observed=(x!=mask_id)&attention.bool()
    if use_flow and head_mean is None:head_mean=model.get_output_embeddings().weight.float().mean(0)
    if metric=='fisher_dispersion' and gradient_dictionary is None:
        gradient_dictionary=model.get_output_embeddings().weight.float()@projection
        gradient_dictionary-=gradient_dictionary.mean(0)
    config=dict(ranking=ranking,controller=controller,metric=metric,tolerance=tolerance,
                quantile=quantile,max_count=max_count,growth_cap=growth_cap,context_beta=context_beta,
                context_mode=context_mode,candidate_region=candidate_region,eos_id=eos_id,frontier_width=frontier_width)
    work=dict(config=config,controller=controller,ranking=ranking,steps_taken=0,
        forward_calls=0,sequence_evaluations=0,backward_sequence_evaluations=0,
        reveal_counts=[],count_records=[],reveal_positions=[],confidence_scoring=ranking in ('diagnostic_fisher','fisher_context_flow'),
        entropy_scoring=controller=='eb',eb_count=controller=='eb',auxiliary_training=False,
        count_uses_confidence=False,count_uses_vocabulary_probabilities=controller=='eb' or metric=='fisher_dispersion',
        probability_weighted_geometry=metric=='fisher_dispersion',readout_gradient_evaluations=0,
        partial_backward_sequence_evaluations=0,context_flow_traces=[],
        logical_eos_completion_by_row=[None]*batch,candidate_counts=[],
        forced_allowance_by_row=[0]*batch,realized_excess_by_row=[0]*batch)
    previous=None;previous_ids=None;previous_count=[1]*batch;density_ema=[None]*batch
    reference_density=[None]*batch;start_step=0
    if resume_state is not None:
        if resume_state['metrics']['config']!=config:raise ValueError('resumed configuration differs')
        restored=resume_state['canvas'].to(x.device)
        if restored.shape!=x.shape or not torch.equal(restored[:,:prefix],prompt):raise ValueError('resumed prompt differs')
        x=restored;previous=resume_state['previous_hidden'].to(x.device)
        previous_ids=resume_state['previous_ids']
        if previous_ids is not None:previous_ids=previous_ids.to(x.device)
        work=resume_state['metrics'];previous_count=resume_state['previous_count'];density_ema=resume_state['density_ema']
        reference_density=resume_state.get('reference_density',[None]*batch)
        start_step=resume_state['next_step']
        if start_step!=work['steps_taken'] or not 0<=start_step<=steps:raise ValueError('invalid resumed step')
    measured=controller not in ('singleton','eb')
    reads_needed=ranking=='activation' or 'grounded' in controller or controller in ('allowance','allowance_subset','risk_only','pair_only')
    for step in range(start_step,steps):
        full_active=(x[:,prefix:]==mask_id)&attention[:,prefix:].bool()
        body_active,ends=committed_end_region(x[:,prefix:],attention[:,prefix:],mask_id,eos_id)
        completed=(ends<gen_length)&~body_active.any(-1)
        for row in completed.nonzero().flatten().tolist():
            if work['logical_eos_completion_by_row'][row] is None:
                work['logical_eos_completion_by_row'][row]=step
        active=body_active if candidate_region=='committed_eos' else full_active
        if candidate_region=='frontier':
            indices=torch.arange(gen_length,device=x.device)[None]
            first=torch.where(full_active,indices,gen_length).min(-1).values
            active=full_active&(indices<first[:,None]+frontier_width)
        if not bool(active.any()):break
        flow_score=None
        with context_flow_capture(model) if use_flow else nullcontext(None) as flow:
            out,hidden,projected,native_variance,reads,native=forward_fields(model,x,attention,projection,reads=reads_needed)
            out.logits[...,mask_id]=-torch.inf
            proposals=out.logits.argmax(-1)
            if use_flow:
                flow_score,flow_info=context_support(model,flow,proposals,full_active,original_observed,prefix,head_mean,context_mode)
                work['partial_backward_sequence_evaluations']+=batch
                work['context_flow_traces'].append(flow_info)
        current_ids=None
        if measured and metric=='fisher_dispersion':
            error=gradient_dispersion(out.logits[:,prefix:],gradient_dictionary)
            work['readout_gradient_evaluations']+=batch
        elif measured and metric=='gradient_union':
            current_ids=out.logits[:,prefix:].topk(min(4,out.logits.shape[-1]-1),-1).indices
            error=gradient_ratio(model.get_output_embeddings().weight,hidden[:,prefix:],
                previous[:,prefix:] if previous is not None else None,native[:,prefix:],
                current_ids,previous_ids,full_active)
        else:
            previous_projected=previous[:,prefix:].float()@projection if previous is not None else None
            error=uniform_ratio(projected[:,prefix:],previous_projected,native_variance[:,prefix:],full_active)
        graph=None;context=None
        if reads_needed:
            q,k,bias=reads[0]
            if 'grounded' in controller or controller in ('allowance','allowance_subset','risk_only','pair_only'):
                graph=attention_field(q,k,bias,prefix)
            if ranking=='activation':
                context=known_reads(q,k,((x!=mask_id)&attention.bool()).float(),bias)[:,prefix:]
        entropy=token_entropy(out.logits,eps=1e-9)[:,prefix:] if controller=='eb' else None
        reveal=torch.zeros_like(active);records=[]
        for row in range(batch):
            pos=active[row].nonzero().flatten()
            if not len(pos):records.append(dict(row=row,count=0));continue
            rank_pos=full_active[row].nonzero().flatten()
            full=rank_pos+prefix
            if ranking in ('diagnostic_fisher','fisher_context_flow'):
                score=fisher_reference_score(out.logits[row,full],hidden[row,full],
                    previous[row,full] if previous is not None else None,model.get_output_embeddings().weight)
                if use_flow:
                    support=flow_score[row,rank_pos]
                    score+=context_beta*(support/support.abs().mean().clamp_min(1e-6)).clamp(-4,4)
            elif ranking=='context_flow':
                support=flow_score[row,rank_pos]
                score=(support/support.abs().mean().clamp_min(1e-6)).clamp(-4,4)
                if previous is not None:
                    motion=((hidden[row,full].float()-previous[row,full].float())@projection).square().mean(-1)
                    score-=.15*(motion/motion.mean().clamp_min(1e-12)).clamp(max=4)
            else:
                score=context[row,rank_pos]/context[row,rank_pos].mean().clamp_min(1e-12)
                if previous is not None:
                    motion=((hidden[row,full].float()-previous[row,full].float())@projection).square().mean(-1)
                    score-=.15*(motion/motion.mean().clamp_min(1e-12)).clamp(max=4)
            score=score[active[row,rank_pos]]
            order=pos[score.argsort(descending=True,stable=True)]
            detail=dict(row=row,bootstrap=False)
            if controller=='eb':
                selected=_eb_prefix(pos,score,entropy[row,pos]);detail['count']=len(selected)
            elif controller=='singleton' or previous is None:
                selected=order[:1];detail.update(count=1,bootstrap=controller!='singleton')
            elif controller.startswith('feedback'):
                density=response_density(error[row],active[row],quantile,
                    graph[row] if 'grounded' in controller else None)
                ema=density if density_ema[row] is None else .5*density+.5*density_ema[row]
                density_ema[row]=ema;estimate=max(density,ema)
                allowance=tolerance
                if 'relative' in controller:
                    if reference_density[row] is None:reference_density[row]=max(density,1e-6)
                    allowance=tolerance*reference_density[row]
                count=feedback_count(previous_count[row],estimate,allowance,len(pos),max_count,growth_cap)
                selected=order[:count]
                detail.update(count=count,realized_density=density,filtered_density=estimate,
                              previous_count=previous_count[row],allowance=allowance,reference_density=reference_density[row],
                              estimated_density=estimate*(count/previous_count[row])**2)
                work['realized_excess_by_row'][row]+=int(density>allowance)
            else:
                local_error=error[row]
                if controller=='allowance_grounded':
                    local_error=local_error*(graph[row]*active[row][None]).sum(-1).clamp(0.,1.)
                if controller=='allowance_subset':
                    selected,diag=allowance_subset(order,local_error,graph[row],tolerance,max_count)
                else:
                    selected,diag=allowance_prefix(order,local_error,graph[row],tolerance,max_count,
                        risk_weight=0. if controller=='pair_only' else 1.,
                        pair_weight=0. if controller=='risk_only' else 1.,
                        uncertainty_only=metric=='fisher_dispersion')
                detail.update(count=len(selected),**diag)
                work['forced_allowance_by_row'][row]+=int(diag['forced'])
            reveal[row,selected]=True
            previous_count[row]=len(selected);records.append(detail)
        x[:,prefix:][reveal]=proposals[:,prefix:][reveal]
        previous=hidden.detach();previous_ids=current_ids
        work['steps_taken']+=1;work['forward_calls']+=1;work['sequence_evaluations']+=batch
        work['reveal_counts'].append(reveal.sum(-1).tolist());work['count_records'].append(records)
        work['reveal_positions'].append([reveal[row].nonzero().flatten().tolist() for row in range(batch)])
        work['candidate_counts'].append(active.sum(-1).tolist())
        del out,hidden,projected,native_variance,reads,native,proposals,error,graph,context,entropy
        if progress and (step%16==15 or step==steps-1):progress(work)
        if step_checkpoint and (step%16==15 or step==steps-1):
            step_checkpoint(dict(canvas=x.detach().cpu(),previous_hidden=previous.detach().cpu(),
                previous_ids=previous_ids.detach().cpu() if previous_ids is not None else None,
                previous_count=previous_count,density_ema=density_ema,reference_density=reference_density,
                next_step=step+1,metrics=work))
    work['unfinished_masks_by_row']=((x[:,prefix:]==mask_id)&attention[:,prefix:].bool()).sum(-1).tolist()
    remaining,ends=committed_end_region(x[:,prefix:],attention[:,prefix:],mask_id,eos_id)
    for row in ((ends<gen_length)&~remaining.any(-1)).nonzero().flatten().tolist():
        if work['logical_eos_completion_by_row'][row] is None:
            work['logical_eos_completion_by_row'][row]=work['steps_taken']
    last_metrics=work
    return x
