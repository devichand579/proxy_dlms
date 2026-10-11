"""Forced-mask batch-growth control, independent of EB and token confidence.

The new scorer uses attention to observed context and activation movement.
An external ranking callback is supported solely for labelled count diagnostics.
"""
from contextlib import contextmanager
import math
import torch

last_metrics = {}


@torch.no_grad()
def decoder_geometry(weight,width=64,seed=1729,chunk=8192):
    """Sketch a centred frozen decoder-weight metric; no runtime probabilities."""
    generator=torch.Generator(device=weight.device);generator.manual_seed(seed)
    result=torch.zeros((weight.shape[1],width),device=weight.device)
    total=torch.zeros(weight.shape[1],device=weight.device)
    sign_total=torch.zeros(width,device=weight.device)
    for start in range(0,len(weight),chunk):
        block=weight[start:start+chunk].float()
        signs=torch.randint(2,(len(block),width),device=weight.device,generator=generator).float()*2-1
        result+=block.T@signs;total+=block.sum(0);sign_total+=signs.sum(0)
    result-=(total/len(weight))[:,None]*sign_total[None]
    result/=math.sqrt(len(weight)*width)
    result*=math.sqrt(width)/result.norm().clamp_min(1e-12)
    return result


def batch_count(sizes,errors,remaining,steps_left,budget=1.,paced=False):
    """Largest measured safe size under a remaining-step latent allowance.

    Use a conservative upper envelope; realized activation errors need not be
    monotone in batch size. Optional pacing overrides are recorded, not hidden.
    """
    if remaining<1 or steps_left<1 or budget<0 or len(sizes)!=len(errors):
        raise ValueError('invalid count inputs')
    if not sizes or sizes[0]!=1 or sizes!=sorted(set(sizes)) or sizes[-1]>remaining:
        raise ValueError('ordered unique sizes including one required')
    threshold=budget/steps_left
    envelope=[];highest=0.
    for error in errors:
        if not math.isfinite(error) or error<0:raise ValueError('invalid latent error')
        highest=max(highest,error);envelope.append(highest)
    feasible=[size for size,error in zip(sizes,envelope) if error<=threshold]
    selected=max(feasible) if feasible else 1
    forced=not feasible
    pace=math.ceil(remaining/steps_left)
    if paced and selected<pace:
        if pace not in sizes:raise ValueError('pacing size must be probed')
        selected=pace;forced=True
    return selected,dict(threshold=threshold,forced_pacing=forced,
                         measured_error=errors[sizes.index(selected)],error_envelope=envelope)


def known_reads(q,k,known,bias=None,query_chunk=32):
    """Attention to observed input/context, not vocabulary confidence."""
    if q.shape[1]!=k.shape[1]:k=k.repeat_interleave(q.shape[1]//k.shape[1],dim=1)
    keys=k.float().transpose(-1,-2);result=[]
    for start in range(0,q.shape[-2],query_chunk):
        stop=min(start+query_chunk,q.shape[-2])
        scores=q[:,:,start:stop].float()@keys*q.shape[-1]**-.5
        if bias is not None:
            local=bias[...,start:stop,:] if bias.shape[-2]!=1 else bias
            scores=scores.masked_fill(~local,float('-inf')) if local.dtype==torch.bool else scores+local.float()
        result.append((scores.softmax(-1)*known[:,None,None,:]).sum(-1).mean(1))
    return torch.cat(result,-1)


@contextmanager
def read_capture(block,active):
    captured=[]
    if not active:yield captured;return
    name='_scaled_dot_product_attention';existed=name in block.__dict__;prior=block.__dict__.get(name)
    original=getattr(block,name)
    def wrapper(q,k,v,*args,**kwargs):
        bias=kwargs.get('attn_mask',args[0] if args else None)
        captured.append((q.detach(),k.detach(),bias));return original(q,k,v,*args,**kwargs)
    setattr(block,name,wrapper)
    try:yield captured
    finally:
        if existed:setattr(block,name,prior)
        else:delattr(block,name)


def forward_fields(model,ids,attention,projection,reads=False):
    transformer=model.model.transformer;states={}
    hooks=[model.get_output_embeddings().register_forward_pre_hook(
        lambda _m,i:states.__setitem__('final',i[0].detach())),
        transformer.blocks[-3].register_forward_hook(
        lambda _m,_i,o:states.__setitem__('middle',o[0].detach()))]
    try:
        with read_capture(transformer.blocks[-1],reads) as captured:
            out=model(ids,attention_mask=attention)
    finally:
        for hook in hooks:hook.remove()
    hidden=states['final'];projected=hidden.float()@projection
    middle=transformer.ln_f(states['middle']).float()
    native_delta=hidden.float()-middle
    uncertainty=(projected-middle@projection).square().mean(-1)
    return out,hidden,projected,uncertainty,captured,native_delta


@torch.no_grad()
def generate_count(model,prompt,attention_mask,*,steps=128,gen_length=512,
                   mask_id=126336,eos_id=126081,budget=1.,max_count=16,
                   controller='guard',paced=False,branch_batch_size=320,
                   projection=None,ranker=None,initial_canvas=None,full_attention=None,
                   progress=None,growth_metric='uniform',proposal_width=4,
                   resume_state=None,step_checkpoint=None):
    global last_metrics
    if min(steps,max_count,branch_batch_size)<1 or budget<0 or controller not in ('guard','fixed','singleton'):
        raise ValueError('invalid budgets/controller')
    if growth_metric not in ('uniform','candidate_gradients') or proposal_width<2:
        raise ValueError('invalid latent metric')
    batch,prefix=prompt.shape
    x=torch.full((batch,prefix+gen_length),mask_id,device=prompt.device,dtype=torch.long) if initial_canvas is None else initial_canvas.clone()
    if x.shape!=(batch,prefix+gen_length):raise ValueError('canvas shape mismatch')
    x[:,:prefix]=prompt
    attention=torch.cat([attention_mask,torch.ones((batch,gen_length),device=x.device,dtype=attention_mask.dtype)],-1) if full_attention is None else full_attention
    if attention.shape!=x.shape:raise ValueError('attention shape mismatch')
    projection=decoder_geometry(model.get_output_embeddings().weight) if projection is None else projection
    work=dict(steps_taken=0,forward_calls=0,sequence_evaluations=0,forward_work_by_row=[0]*batch,
        branch_work_by_row=[0]*batch,probe_control_work_by_row=[0]*batch,reveal_counts=[],
        count_probe_records=[],forced_pacing_by_row=[0]*batch,budget=budget,max_count=max_count,
        controller=controller,paced=paced,ranking='diagnostic_fisher' if ranker else 'activation_known_context',
        confidence_scoring=ranker is not None,entropy_scoring=False,eb_count=False,
        auxiliary_training=False,branch_batch_size=branch_batch_size,growth_metric=growth_metric,
        proposal_width=proposal_width)
    def account(rows,kind=None):
        work['forward_calls']+=1;work['sequence_evaluations']+=len(rows)
        for row in rows:
            work['forward_work_by_row'][row]+=1
            if kind:work[kind][row]+=1
    previous=None;start_step=0
    if resume_state is not None:
        restored=resume_state['canvas'].to(x.device)
        if restored.shape!=x.shape or not torch.equal(restored[:,:prefix],prompt):raise ValueError('resumed prompt differs')
        saved=resume_state['metrics']
        for key in ('budget','max_count','controller','paced','ranking','branch_batch_size','growth_metric','proposal_width'):
            if saved[key]!=work[key]:raise ValueError('resumed settings differ')
        start_step=resume_state['next_step']
        if start_step!=saved['steps_taken'] or not 0<=start_step<=steps:raise ValueError('invalid resumed step')
        x=restored.clone();previous=resume_state['previous_hidden'].to(x.device);work=saved
    for step in range(start_step,steps):
        eligible=[(x[row,prefix:]==mask_id).nonzero().flatten()+prefix for row in range(batch)]
        if not any(len(p) for p in eligible):break
        out,hidden,projected,uncertainty,captured,native_delta=forward_fields(model,x,attention,projection,reads=ranker is None)
        out.logits[...,mask_id]=-torch.inf;predicted=out.logits.argmax(-1)
        account(list(range(batch)))
        if ranker is None:
            q,k,bias=captured[0]
            context=known_reads(q,k,((x!=mask_id)&attention.bool()).float(),bias)
        orders={};sizes={};errors={};actions=[];proposal_ids={}
        for row,pos in enumerate(eligible):
            if not len(pos):continue
            if ranker is not None:score=ranker(out.logits[row,pos],hidden[row,pos],previous[row,pos] if previous is not None else None,model.get_output_embeddings().weight)
            else:
                score=context[row,pos]/context[row,pos].mean().clamp_min(1e-12)
                if previous is not None:
                    motion=((hidden[row,pos].float()-previous[row,pos].float())@projection).square().mean(-1)
                    score-=.15*(motion/motion.mean().clamp_min(1e-12)).clamp(max=4)
            orders[row]=pos[score.argsort(descending=True,stable=True)]
            if controller=='fixed':sizes[row]=[min(len(pos),math.ceil(len(pos)/(steps-step)))];continue
            if controller=='singleton':sizes[row]=[1];continue
            if step==0 or len(pos)==1:sizes[row]=[1];errors[row]=[0.];continue
            cap=min(max_count,len(pos))
            candidates={1,cap}
            candidates.update(n for n in (2,4,8,16,32,64) if n<=cap)
            if paced:candidates.add(math.ceil(len(pos)/(steps-step)))
            sizes[row]=sorted(candidates);errors[row]=[0.]+[float('nan')]*(len(candidates)-1)
            for index,count in enumerate(sizes[row][1:],1):
                sentinel=int(orders[row][count-1])
                actions.append((row,index,count,sentinel))
                if growth_metric=='candidate_gradients':
                    proposal_ids[row,sentinel]=out.logits[row,sentinel].topk(min(proposal_width,out.logits.shape[-1]-1)).indices
        del out,captured
        for offset in range(0,len(actions),branch_batch_size):
            group=actions[offset:offset+branch_batch_size];row_ids=[a[0] for a in group]
            rows=torch.tensor(row_ids,device=x.device)
            control,hc,zc,_uc,_,_dc=forward_fields(model,x[rows],attention[rows],projection);del control,_uc,_dc
            account(row_ids,'probe_control_work_by_row')
            branch_ids=x[rows].clone()
            for j,(row,_index,count,_sentinel) in enumerate(group):
                positions=orders[row][:count-1];branch_ids[j,positions]=predicted[row,positions]
            branch,hb,zb,_ub,_,_db=forward_fields(model,branch_ids,attention[rows],projection);del _ub,_db,branch_ids
            branch.logits[...,mask_id]=-torch.inf
            account(row_ids,'branch_work_by_row')
            for j,(row,index,_count,sentinel) in enumerate(group):
                if growth_metric=='candidate_gradients':
                    alternate=branch.logits[j,sentinel].topk(min(proposal_width,branch.logits.shape[-1]-1)).indices
                    values=torch.cat([proposal_ids[row,sentinel],alternate]).unique()
                    directions=model.get_output_embeddings().weight[values].float()
                    directions-=directions.mean(0,keepdim=True)
                    shift=(hb[j,sentinel].float()-hc[j,sentinel].float())@directions.T
                    intrinsic=native_delta[row,sentinel]@directions.T
                    displacement=shift.square().mean()
                    floor=native_delta[row,sentinel].square().mean()*directions.square().sum(-1).mean()*1e-3+1e-12
                    scale=intrinsic.square().mean().clamp_min(floor)
                else:
                    displacement=(zb[j,sentinel]-zc[j,sentinel]).square().mean()
                    scale=uncertainty[row,sentinel].clamp_min(uncertainty[row,eligible[row]].mean()*1e-3+1e-12)
                errors[row][index]=float(displacement/scale)
            del branch,hc,hb,zc,zb
        reveal=torch.zeros_like(x,dtype=torch.bool);records=[]
        for row,order in orders.items():
            if controller in ('fixed','singleton'):count=sizes[row][0];diagnostic=dict(forced_pacing=False,measured_error=0.)
            elif step==0:count=1;diagnostic=dict(forced_pacing=False,measured_error=0.,threshold=budget/(steps-step))
            else:
                count,diagnostic=batch_count(sizes[row],errors[row],len(order),steps-step,budget,paced)
                work['forced_pacing_by_row'][row]+=int(diagnostic['forced_pacing'])
            reveal[row,order[:count]]=True
            records.append(dict(row=row,sizes=sizes[row],errors=errors.get(row),chosen=count,**diagnostic))
        work['count_probe_records'].append(records)
        work['reveal_counts'].append(reveal[:,prefix:].sum(-1).tolist())
        x[reveal]=predicted[reveal];previous=hidden.detach();work['steps_taken']+=1
        del hidden,projected,uncertainty,native_delta,predicted,orders,sizes,errors,actions,proposal_ids
        if ranker is None:del context
        if progress and (step%16==15 or step==steps-1):progress(dict(work))
        if step_checkpoint is not None:
            step_checkpoint(dict(canvas=x.detach().cpu(),previous_hidden=previous.detach().cpu(),next_step=step+1,metrics=work))
    work['unfinished_masks_by_row']=(x[:,prefix:]==mask_id).sum(-1).tolist()
    if sum(work['forward_work_by_row'])!=work['sequence_evaluations']:raise ValueError('work mismatch')
    last_metrics=work
    return x
