"""Latent-residual set optimization: no entropy or confidence scheduling.

Token values are greedy model proposals. All position and count decisions use
normalized intermediate/final activations and measured reveal responses.
The objective is a latent control surrogate, not a bound on data-distribution KL.
"""
import torch

last_metrics = {}


def spectral_pool(residual,cost,count):
    """Cover distinct activation-residual modes without probabilities or labels."""
    centered = residual.double()-residual.double().mean(0,keepdim=True)
    gram = centered@centered.T/centered.shape[-1]
    values,vectors = torch.linalg.eigh(gram)
    values = values.clamp_min(0)
    ridge = values.mean().clamp_min(1e-12)
    features = vectors*(values/(values+ridge)).sqrt()[None]
    features = features/(1+cost.double().clamp_min(0)).sqrt()[:,None]
    selected = []
    available = torch.ones(len(cost),device=cost.device,dtype=torch.bool)
    for _ in range(min(count,len(cost))):
        score = features.square().sum(-1)
        score[~available] = -torch.inf
        if float(score.max())<1e-12:
            score = -cost.double().clone();score[~available] = -torch.inf
        chosen = int(score.argmax());selected.append(chosen);available[chosen] = False
        direction = features[chosen]/features[chosen].norm().clamp_min(1e-12)
        features -= (features@direction)[:,None]*direction[None]
    return torch.tensor(selected,device=cost.device,dtype=torch.long)


def subset_optimum(gain, interaction):
    """Exact variable-cardinality quadratic solve through spectral coordinates.

    Maximize g.v - .5 v.M.v over binary subsets, including the empty set.
    Override an empty optimum with the best singleton to guarantee progress.
    """
    n = len(gain)
    if not 1 <= n <= 12 or interaction.shape != (n,n):
        raise ValueError('one to twelve candidates and a square kernel required')
    symmetric = .5*(interaction.double()+interaction.double().T)
    eigenvalues,eigenvectors = torch.linalg.eigh(symmetric)
    if float(eigenvalues.min()) < -1e-6*max(1.,float(eigenvalues.abs().max())):
        raise ValueError('interaction must be positive semidefinite')
    eigenvalues = eigenvalues.clamp_min(0)
    patterns = ((torch.arange(2**n,device=gain.device)[:,None]
                 >>torch.arange(n,device=gain.device))&1).double()
    modes = patterns@eigenvectors
    values = patterns@gain.double()-.5*(modes.square()*eigenvalues).sum(-1)
    maximum = values.max()
    ties = torch.isclose(values,maximum,rtol=0.,atol=1e-10).nonzero().flatten()
    sizes = patterns[ties].sum(-1)
    best = int(ties[sizes.argmin()])
    forced = not bool(patterns[best].any())
    if forced:
        singleton = gain.double()-.5*symmetric.diag()
        index = int(singleton.argmax())
        selected = torch.tensor([index],device=gain.device)
        value = float(singleton[index])
    else:
        selected = patterns[best].nonzero().flatten()
        value = float(values[best])
    total = eigenvalues.sum()
    effective_rank = float(total.square()/eigenvalues.square().sum().clamp_min(1e-30))
    return selected,dict(forced_singleton=forced,objective=value,
                         effective_rank=effective_rank,largest_eigenvalue=float(eigenvalues.max()))


def surviving_optimum(modular_gain,response,field,source_targets):
    """Exact latent progress on R\S, using a spectrum for each target.

    The surviving-target mask depends on the subset. This is not one global
    quadratic; treating it as one would credit a source for its own removal.
    """
    n,targets,_ = response.shape
    if not 1<=n<=12 or field.shape!=response.shape[1:] or len(source_targets)!=n:
        raise ValueError('invalid source/target geometry')
    scale = field.double().square().sum().clamp_min(1e-12)
    interaction = 2*torch.einsum('ijd,kjd->jik',response.double(),response.double())/scale
    eigenvalues,eigenvectors = torch.linalg.eigh(interaction)
    eigenvalues = eigenvalues.clamp_min(0)
    linear = 2*(response.double()*field.double()[None]).sum(-1)/scale
    patterns = ((torch.arange(2**n,device=response.device)[:,None]
                 >>torch.arange(n,device=response.device))&1).double()
    modes = torch.einsum('pk,jkm->pjm',patterns,eigenvectors)
    target_gain = patterns@linear-.5*(modes.square()*eigenvalues[None]).sum(-1)
    survive = torch.ones_like(target_gain)
    survive[:,source_targets] = 1-patterns
    values = (target_gain*survive).sum(-1)+patterns@modular_gain.double()
    maximum = values.max()
    ties = torch.isclose(values,maximum,rtol=0.,atol=1e-10).nonzero().flatten()
    best = int(ties[patterns[ties].sum(-1).argmin()])
    forced = not bool(patterns[best].any())
    if forced:
        singleton_patterns = (patterns.sum(-1)==1).nonzero().flatten()
        best = int(singleton_patterns[values[singleton_patterns].argmax()])
    selected = patterns[best].nonzero().flatten()
    aggregate = interaction.sum(0)
    aggregate_eigenvalues = torch.linalg.eigvalsh(aggregate).clamp_min(0)
    total = aggregate_eigenvalues.sum()
    rank = float(total.square()/aggregate_eigenvalues.square().sum().clamp_min(1e-30))
    return selected,dict(forced_singleton=forced,objective=float(values[best]),
                         effective_rank=rank,largest_eigenvalue=float(aggregate_eigenvalues.max()))


def latent_forward(model,ids,attention,depth_back):
    """Capture only three states, expressed using the same frozen final norm."""
    transformer = model.model.transformer
    blocks = transformer.blocks
    if depth_back<2 or depth_back%2 or len(blocks)<=depth_back:
        raise ValueError('an even depth gap smaller than the block count is required')
    saved = {}
    hooks = [blocks[-1-depth_back].register_forward_hook(
        lambda _m,_i,o:saved.__setitem__('early',o[0].detach())),
        blocks[-1-depth_back//2].register_forward_hook(
        lambda _m,_i,o:saved.__setitem__('middle',o[0].detach())),
        model.get_output_embeddings().register_forward_pre_hook(
        lambda _m,i:saved.__setitem__('final',i[0].detach()))]
    try:
        out = model(ids,attention_mask=attention)
    finally:
        for hook in hooks: hook.remove()
    final = saved['final']
    early = transformer.ln_f(saved['early']).float()
    middle = transformer.ln_f(saved['middle']).float()
    residual = final.float()-2*middle+early
    return out,final,residual


@torch.no_grad()
def generate_residual(model,prompt,attention_mask,*,steps=128,gen_length=512,
                      mask_id=126336,eos_id=126081,candidates=4,depth_back=4,
                      branch_batch_size=160,progress_reward=.5,
                      source_weight=.25,incoming_weight=.25,temporal_weight=.15,
                      target_mode='external',
                      pool_mode='ready',
                      initial_canvas=None,full_attention=None,progress=None):
    global last_metrics
    batch,prefix = prompt.shape
    if not 1<=candidates<=12 or min(steps,branch_batch_size)<1:
        raise ValueError('invalid step/candidate/batch budget')
    if min(progress_reward,source_weight,incoming_weight,temporal_weight)<0:
        raise ValueError('objective coefficients must be nonnegative')
    if target_mode not in ('external','survivors'): raise ValueError('unknown target mode')
    if pool_mode not in ('ready','spectral'): raise ValueError('unknown pool mode')
    x = torch.full((batch,prefix+gen_length),mask_id,device=prompt.device,dtype=torch.long) if initial_canvas is None else initial_canvas.clone()
    if x.shape!=(batch,prefix+gen_length): raise ValueError('canvas shape mismatch')
    x[:,:prefix] = prompt
    attention = torch.cat([attention_mask,torch.ones((batch,gen_length),device=x.device,dtype=attention_mask.dtype)],-1) if full_attention is None else full_attention
    if attention.shape!=x.shape: raise ValueError('attention shape mismatch')
    work = dict(steps_taken=0,forward_calls=0,sequence_evaluations=0,
        forward_work_by_row=[0]*batch,branch_work_by_row=[0]*batch,
        probe_control_work_by_row=[0]*batch,reveal_counts=[],forced_singletons_by_row=[0]*batch,
        objective_sum=0.,effective_rank_sum=0.,selection_measurements=0,
        control_residual_mismatch_sum=0.,control_measurements=0,
        auxiliary_training=False,confidence_scoring=False,entropy_scoring=False,eb_count=False,
        candidates=candidates,depth_back=depth_back,progress_reward=progress_reward,
        source_weight=source_weight,incoming_weight=incoming_weight,temporal_weight=temporal_weight,
        target_mode=target_mode,
        pool_mode=pool_mode,
        branch_batch_size=branch_batch_size)
    def account(rows,kind=None):
        work['forward_calls']+=1; work['sequence_evaluations']+=len(rows)
        for row in rows:
            work['forward_work_by_row'][row]+=1
            if kind: work[kind][row]+=1
    previous = None
    for step in range(steps):
        eligible = [(x[row,prefix:]==mask_id).nonzero().flatten()+prefix for row in range(batch)]
        if not any(len(p) for p in eligible): break
        out,hidden,residual = latent_forward(model,x,attention,depth_back)
        out.logits[...,mask_id] = -torch.inf
        proposed = out.logits.argmax(-1)
        del out
        account(list(range(batch)))
        pools,targets,source_cost,fields,actions,responses,incoming = {},{},{},{},[],{},{}
        for row,pos in enumerate(eligible):
            if not len(pos): continue
            energy = residual[row,pos].square().mean(-1)
            cost = energy/energy.mean().clamp_min(1e-12)
            if previous is not None:
                motion = (hidden[row,pos].float()-previous[row,pos].float()).square().mean(-1)
                cost = cost+temporal_weight*motion/motion.mean().clamp_min(1e-12)
            local = spectral_pool(residual[row,pos],cost,candidates) if pool_mode=='spectral' else cost.argsort(stable=True)[:candidates]
            pool = pos[local]; pools[row] = pool; source_cost[row] = cost[local]
            external = torch.ones(len(pos),device=x.device,dtype=torch.bool)
            external[local] = False
            targets[row] = pos if target_mode=='survivors' else pos[external]
            fields[row] = residual[row,targets[row]]
            responses[row] = torch.zeros((len(pool),len(targets[row]),hidden.shape[-1]),device=x.device)
            incoming[row] = torch.zeros((len(pool),len(pool)),device=x.device)
            actions += [(row,source,int(position),int(proposed[row,position])) for source,position in enumerate(pool)]
        for offset in range(0,len(actions),branch_batch_size):
            group = actions[offset:offset+branch_batch_size]
            row_ids = [a[0] for a in group]
            rows = torch.tensor(row_ids,device=x.device)
            control,control_hidden,control_residual = latent_forward(model,x[rows],attention[rows],depth_back)
            del control
            account(row_ids,'probe_control_work_by_row')
            branch_ids = x[rows].clone()
            for j,(_row,_source,position,token) in enumerate(group): branch_ids[j,position] = token
            branch,branch_hidden,branch_residual = latent_forward(model,branch_ids,attention[rows],depth_back)
            del branch,branch_ids
            account(row_ids,'branch_work_by_row')
            for j,(row,source,_position,_token) in enumerate(group):
                target = targets[row]; pool = pools[row]
                responses[row][source] = control_residual[j,target]-branch_residual[j,target]
                change = (branch_hidden[j,pool].float()-control_hidden[j,pool].float()).square().mean(-1)
                change[source] = 0.
                incoming[row][source] = change
                mismatch = (control_residual[j,eligible[row]]-residual[row,eligible[row]]).square().mean()
                scale = residual[row,eligible[row]].square().mean().clamp_min(1e-12)
                work['control_residual_mismatch_sum'] += float(mismatch/scale)
                work['control_measurements'] += 1
            del control_hidden,control_residual,branch_hidden,branch_residual
        reveal = torch.zeros_like(x,dtype=torch.bool)
        for row,pool in pools.items():
            response = responses[row].flatten(1)
            field = fields[row].flatten()
            normalizer = field.square().sum().clamp_min(1e-12)
            if not len(field):
                interaction = torch.zeros((len(pool),len(pool)),device=x.device)
                gain = torch.zeros(len(pool),device=x.device)
            else:
                interaction = 2*(response@response.T)/normalizer
                gain = 2*(response@field)/normalizer
            sensitivity = incoming[row].sum(0)
            sensitivity = sensitivity/sensitivity.mean().clamp_min(1e-12)
            gain = gain+progress_reward-source_weight*source_cost[row]-incoming_weight*sensitivity
            if target_mode=='survivors':
                source_targets = torch.searchsorted(eligible[row],pool)
                modular = progress_reward-source_weight*source_cost[row]-incoming_weight*sensitivity
                chosen,diagnostic = surviving_optimum(modular,responses[row],fields[row],source_targets)
            else:
                chosen,diagnostic = subset_optimum(gain,interaction)
            reveal[row,pool[chosen]] = True
            work['forced_singletons_by_row'][row] += int(diagnostic['forced_singleton'])
            work['objective_sum'] += diagnostic['objective']
            work['effective_rank_sum'] += diagnostic['effective_rank']
            work['selection_measurements'] += 1
        work['reveal_counts'].append(reveal[:,prefix:].sum(-1).tolist())
        x[reveal] = proposed[reveal]
        previous = hidden.detach()
        work['steps_taken'] += 1
        del hidden,residual,proposed,responses,fields,incoming,pools,targets,source_cost
        if progress and (step%16==15 or step==steps-1): progress(dict(work))
    work['unfinished_masks_by_row'] = (x[:,prefix:]==mask_id).sum(-1).tolist()
    if sum(work['forward_work_by_row'])!=work['sequence_evaluations']:
        raise ValueError('forward work mismatch')
    last_metrics = work
    return x
