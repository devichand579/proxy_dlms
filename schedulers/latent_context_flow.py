"""Value-aware observed-key support through the final MLP and normalization.

One normal forward, one partial backward through the final tokenwise MLP/norm.
This is a local mechanistic contribution, not a whole-model causal attribution.
"""
from contextlib import contextmanager

import torch
from torch.nn import functional as F


@contextmanager
def context_flow_capture(model):
    transformer=model.model.transformer
    block=transformer.blocks[-1];norm=transformer.ln_f
    if not hasattr(block,'ff_norm') or not hasattr(block,'attn_out'):
        raise ValueError('context-flow scoring requires a tokenwise final MLP and attention output projection')
    saved={};restores=[];hooks=[]
    parameters=list(block.parameters())+list(norm.parameters())
    flags=[p.requires_grad for p in parameters]
    for p in parameters:p.requires_grad_(False)
    def patch(obj,name,wrapper):
        existed=name in obj.__dict__;prior=obj.__dict__.get(name)
        original=getattr(obj,name);restores.append((obj,name,existed,prior))
        setattr(obj,name,wrapper(original))
    def block_wrapper(original):
        def call(*args,**kwargs):
            with torch.enable_grad():return original(*args,**kwargs)
        return call
    def norm_wrapper(original):
        def call(x,*args,**kwargs):
            if x.requires_grad:
                with torch.enable_grad():return original(x,*args,**kwargs)
            return original(x,*args,**kwargs)
        return call
    def attention_wrapper(original):
        def call(q,k,v,*args,**kwargs):
            bias=kwargs.get('attn_mask',args[0] if args else None)
            saved['reads']=(q.detach(),k.detach(),v.detach(),bias)
            return original(q,k,v,*args,**kwargs)
        return call
    def ff_input(_module,inputs):
        # Preserve tensor identity: the block retains this same tensor as its
        # residual skip. A detached replacement drops the identity derivative.
        leaf=inputs[0]
        if leaf.requires_grad or leaf.grad_fn is not None:
            raise ValueError('context-flow boundary must have a detached attention prefix')
        leaf.requires_grad_(True)
        saved['ff_input']=leaf
        return (leaf,)+tuple(inputs[1:])
    try:
        patch(block,'forward',block_wrapper)
        patch(norm,'forward',norm_wrapper)
        patch(block,'_scaled_dot_product_attention',attention_wrapper)
        hooks.append(block.ff_norm.register_forward_pre_hook(ff_input))
        hooks.append(model.get_output_embeddings().register_forward_pre_hook(
            lambda _m,i:saved.__setitem__('head',i[0])))
        yield saved
    finally:
        for hook in hooks:hook.remove()
        for obj,name,existed,prior in reversed(restores):
            if existed:setattr(obj,name,prior)
            else:delattr(obj,name)
        for p,flag in zip(parameters,flags):p.requires_grad_(flag)


def observed_value_update(block, reads, observed, prefix, chunk=32):
    q,k,v,bias=reads
    if q.shape[1]!=k.shape[1]:
        multiplier=q.shape[1]//k.shape[1]
        k=k.repeat_interleave(multiplier,1);v=v.repeat_interleave(multiplier,1)
    keys=k.float().transpose(-1,-2)
    values=v.float()*observed[:,None,:,None]
    result=[];total=[]
    with torch.no_grad():
        for start in range(prefix,q.shape[-2],chunk):
            stop=min(start+chunk,q.shape[-2])
            scores=q[:,:,start:stop].float()@keys*q.shape[-1]**-.5
            if bias is not None:
                local=bias[...,start:stop,:] if bias.shape[-2]!=1 else bias
                scores=scores.masked_fill(~local,-torch.inf) if local.dtype==torch.bool else scores+local.float()
            probabilities=scores.softmax(-1)
            attended=probabilities@values
            attended=attended.transpose(1,2).reshape(len(q),stop-start,-1)
            result.append(F.linear(attended,block.attn_out.weight.float()))
            all_values=(probabilities@v.float()).transpose(1,2).reshape(len(q),stop-start,-1)
            total.append(F.linear(all_values,block.attn_out.weight.float()))
    return torch.cat(result,1),torch.cat(total,1)


def context_support(model, captured, proposals, active, observed, prefix, head_mean, mode='fraction'):
    """Cosine between known-key value contribution and proposed-value gradient."""
    head=model.get_output_embeddings()
    direction=head.weight[proposals[:,prefix:]].detach().float()-head_mean
    direction/=direction.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    with torch.enable_grad():
        final=captured['head'][:,prefix:].float()
        loss=(final*direction*active[:,:,None]).sum()
        gradient=torch.autograd.grad(loss,captured['ff_input'],retain_graph=False)[0][:,prefix:].float()
    contribution,total=observed_value_update(model.model.transformer.blocks[-1],captured['reads'],observed,prefix)
    dot=(gradient*contribution).sum(-1)
    if mode not in ('fraction','cosine'):raise ValueError('unknown context support mode')
    scale=total.norm(dim=-1) if mode=='fraction' else contribution.norm(dim=-1)
    floor=(scale*active).sum(-1)/active.sum(-1).clamp_min(1)*1e-3
    denominator=gradient.norm(dim=-1)*scale.clamp_min(floor[:,None].clamp_min(1e-12))
    score=(dot/denominator.clamp_min(1e-12)).clamp(-4.,4.)
    return score,dict(gradient_norm=gradient.norm(dim=-1).mean(-1).tolist(),
                      contribution_norm=contribution.norm(dim=-1).mean(-1).tolist())
