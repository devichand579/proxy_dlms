"""Variable-cardinality activation-response transport, without confidence.

Two source values have equal weights. Continuous branch contrasts measure
directed latent sensitivity; they are not claimed to be true mutual information.
"""
import torch

from schedulers.latent_residual import subset_optimum,spectral_pool

last_metrics = {}


def activation_metric(states):
    """Regularized PCA whitening from this canvas's continuous states only."""
    centered = states.double()-states.double().mean(0,keepdim=True)
    gram = centered@centered.T/centered.shape[-1]
    values,vectors = torch.linalg.eigh(gram)
    values = values.clamp_min(0)
    if float(values.max())<1e-12:return None
    active = values>values.max()*1e-8
    values,vectors = values[active],vectors[:,active]
    basis = centered.T@vectors/(centered.shape[-1]*values).sqrt()[None]
    basis /= (values+values.mean()).sqrt()[None]
    basis *= (basis.shape[1]/basis.square().sum().clamp_min(1e-30)).sqrt()
    return basis.float()


def flow_objective(contrast,defect,source_targets,*,incoming_weight=.5,
                   consistency_weight=0.,progress_reward=.05):
    """Return an exactly equivalent PSD quadratic for flow minus interference."""
    outgoing = contrast.sum(-1)
    incoming = contrast[:,source_targets].sum(0)*contrast.shape[1]/len(source_targets)
    scale = outgoing.mean().clamp_min(1e-12)
    modular = (outgoing-incoming_weight*incoming-consistency_weight*defect.sum(-1))/scale+progress_reward
    directed = contrast[:,source_targets]/scale
    pair = directed+directed.T
    pair.fill_diagonal_(0)
    degree = pair.sum(-1)
    # Signless Laplacian is PSD. Compensate its diagonal in the linear term
    # because binary indicators satisfy v_i^2 = v_i.
    interaction = torch.diag(degree)+pair
    gain = modular+.5*degree
    return gain,interaction


def hidden_forward(model,ids,attention):
    captured = []
    hook = model.get_output_embeddings().register_forward_pre_hook(
        lambda _m,i:captured.append(i[0].detach()))
    try: out = model(ids,attention_mask=attention)
    finally: hook.remove()
    return out,captured[0]


@torch.no_grad()
def generate_flow(model,prompt,attention_mask,*,steps=128,gen_length=512,
                  mask_id=126336,eos_id=126081,candidates=12,branch_batch_size=160,
                  incoming_weight=.5,consistency_weight=0.,progress_reward=.05,
                  geometry='raw',
                  initial_canvas=None,full_attention=None,progress=None,
                  resume_state=None,step_checkpoint=None,evaluate_controls=True):
    global last_metrics
    batch,prefix = prompt.shape
    if not 1<=candidates<=12 or min(steps,branch_batch_size)<1 or branch_batch_size%2:
        raise ValueError('invalid budgets; paired source alternatives need an even batch')
    if min(incoming_weight,consistency_weight,progress_reward)<0:
        raise ValueError('nonnegative objective coefficients required')
    if geometry not in ('raw','pca'):raise ValueError('unknown activation metric')
    if not evaluate_controls and consistency_weight:
        raise ValueError('consistency scoring requires unchanged controls')
    x = torch.full((batch,prefix+gen_length),mask_id,device=prompt.device,dtype=torch.long) if initial_canvas is None else initial_canvas.clone()
    if x.shape!=(batch,prefix+gen_length): raise ValueError('canvas shape mismatch')
    x[:,:prefix] = prompt
    attention = torch.cat([attention_mask,torch.ones((batch,gen_length),device=x.device,dtype=attention_mask.dtype)],-1) if full_attention is None else full_attention
    if attention.shape!=x.shape: raise ValueError('attention shape mismatch')
    work = dict(steps_taken=0,forward_calls=0,sequence_evaluations=0,
        forward_work_by_row=[0]*batch,branch_work_by_row=[0]*batch,
        probe_control_work_by_row=[0]*batch,reveal_counts=[],forced_singletons_by_row=[0]*batch,
        objective_sum=0.,effective_rank_sum=0.,selection_measurements=0,
        auxiliary_training=False,confidence_scoring=False,entropy_scoring=False,eb_count=False,
        source_value_weighting='uniform',candidates=candidates,branch_batch_size=branch_batch_size,
        incoming_weight=incoming_weight,consistency_weight=consistency_weight,progress_reward=progress_reward)
    work.update(geometry=geometry,latent_metric_rank_sum=0.,latent_metric_measurements=0)
    work['evaluate_controls'] = evaluate_controls
    start_step = 0
    if resume_state is not None:
        restored = resume_state['canvas'].to(device=x.device,dtype=x.dtype)
        if restored.shape!=x.shape or not torch.equal(restored[:,:prefix],prompt):
            raise ValueError('resumed canvas or prompt differs')
        saved_work = resume_state['metrics']
        for key in ('candidates','branch_batch_size','incoming_weight','consistency_weight','progress_reward','geometry'):
            if saved_work[key]!=work[key]:raise ValueError('resumed scheduler settings differ')
        start_step = resume_state['next_step']
        if not 0<=start_step<=steps or start_step!=saved_work['steps_taken']:
            raise ValueError('invalid resumed step')
        if len(saved_work['forward_work_by_row'])!=batch:
            raise ValueError('resumed batch size differs')
        x = restored.clone();work = saved_work
        work['evaluate_controls'] = evaluate_controls
    def account(rows,kind=None):
        work['forward_calls']+=1;work['sequence_evaluations']+=len(rows)
        for row in rows:
            work['forward_work_by_row'][row]+=1
            if kind:work[kind][row]+=1
    for step in range(start_step,steps):
        eligible = [(x[row,prefix:]==mask_id).nonzero().flatten()+prefix for row in range(batch)]
        if not any(len(p) for p in eligible):break
        out,hidden = hidden_forward(model,x,attention)
        out.logits[...,mask_id] = -torch.inf
        predicted = out.logits.argmax(-1)
        account(list(range(batch)))
        pools,source_targets,contrast,defect,actions,metrics = {},{},{},{},[],{}
        for row,pos in enumerate(eligible):
            if not len(pos):continue
            local = spectral_pool(hidden[row,pos],torch.zeros(len(pos),device=x.device),candidates)
            pool = pos[local];pools[row] = pool;source_targets[row] = local
            metrics[row] = activation_metric(hidden[row,pool]) if geometry=='pca' else None
            if metrics[row] is not None:
                work['latent_metric_rank_sum']+=metrics[row].shape[1]
                work['latent_metric_measurements']+=1
            contrast[row] = torch.zeros((len(pool),len(pos)),device=x.device)
            defect[row] = torch.zeros_like(contrast[row])
            # Scores are discarded: top two values define probes only.
            alternatives = out.logits[row,pool].topk(2,-1).indices
            for source,position in enumerate(pool):
                for alternative in range(2):
                    actions.append((row,source,alternative,int(position),int(alternatives[source,alternative])))
        del out,hidden
        for offset in range(0,len(actions),branch_batch_size):
            group = actions[offset:offset+branch_batch_size]
            rows_list = [a[0] for a in group];rows = torch.tensor(rows_list,device=x.device)
            control_hidden = None
            if evaluate_controls:
                control,control_hidden = hidden_forward(model,x[rows],attention[rows]);del control
                account(rows_list,'probe_control_work_by_row')
            branch_ids = x[rows].clone()
            for j,(_r,_s,_a,position,token) in enumerate(group):branch_ids[j,position] = token
            branch,branch_hidden = hidden_forward(model,branch_ids,attention[rows]);del branch,branch_ids
            account(rows_list,'branch_work_by_row')
            for j in range(0,len(group),2):
                row,source,alternative,position,_token = group[j]
                if alternative!=0 or group[j+1][:2]!=(row,source):
                    raise ValueError('source alternatives split across a batch')
                pos = eligible[row]
                first,second = branch_hidden[j,pos].float(),branch_hidden[j+1,pos].float()
                base = .5*(control_hidden[j,pos].float()+control_hidden[j+1,pos].float()) if evaluate_controls else None
                if metrics[row] is not None:
                    first,second = first@metrics[row],second@metrics[row]
                    if base is not None:base = base@metrics[row]
                contrast[row][source] = .25*(first-second).square().mean(-1)
                if base is not None:defect[row][source] = (.5*(first+second)-base).square().mean(-1)
                own = int(source_targets[row][source])
                contrast[row][source,own] = 0.;defect[row][source,own] = 0.
            del control_hidden,branch_hidden
        reveal = torch.zeros_like(x,dtype=torch.bool)
        for row,pool in pools.items():
            gain,kernel = flow_objective(contrast[row],defect[row],source_targets[row],
                incoming_weight=incoming_weight,consistency_weight=consistency_weight,
                progress_reward=progress_reward)
            selected,diag = subset_optimum(gain,kernel)
            reveal[row,pool[selected]] = True
            work['forced_singletons_by_row'][row]+=int(diag['forced_singleton'])
            work['objective_sum']+=diag['objective'];work['effective_rank_sum']+=diag['effective_rank']
            work['selection_measurements']+=1
        work['reveal_counts'].append(reveal[:,prefix:].sum(-1).tolist())
        x[reveal] = predicted[reveal];work['steps_taken']+=1
        del predicted,pools,source_targets,contrast,defect,metrics
        if progress and (step%16==15 or step==steps-1):progress(dict(work))
        if step_checkpoint is not None:
            step_checkpoint({'canvas':x.detach().cpu(),'next_step':step+1,'metrics':work})
    work['unfinished_masks_by_row'] = (x[:,prefix:]==mask_id).sum(-1).tolist()
    if sum(work['forward_work_by_row'])!=work['sequence_evaluations']:raise ValueError('work mismatch')
    last_metrics = work
    return x
