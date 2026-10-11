"""Exact small distributions for the error-budget objective; no neural claim."""
import itertools
import json
from pathlib import Path

import numpy as np


def entropy(p):
    p=p[p>0]
    return float(-(p*np.log(p)).sum())


def marginal(q, states, positions):
    if not positions:return np.array([1.])
    codes=states[:,positions]@(2**np.arange(len(positions)))
    return np.bincount(codes,weights=q,minlength=2**len(positions))


def conditional(q, states, known, values):
    keep=np.ones(len(q),dtype=bool)
    for i,value in zip(known,values):keep&=states[:,i]==value
    result=q*keep
    if result.sum()==0:raise ValueError('reference history outside data support')
    return result/result.sum()


def schedule(q, states, mode, allowance=.1):
    known=[];batches=[];remaining=list(range(states.shape[1]))
    while remaining:
        current=conditional(q,states,known,[0]*len(known))
        hs={i:entropy(marginal(current,states,[i])) for i in remaining}
        feasible=[]
        for k in range(1,len(remaining)+1):
            s=remaining[:k]
            if mode=='entropy_bound':cost=sum(hs[i] for i in s)-max(hs[i] for i in s)
            elif mode=='exact_tc':cost=sum(hs[i] for i in s)-entropy(marginal(current,states,s))
            elif mode=='pairwise':
                cost=sum(hs[i]+hs[j]-entropy(marginal(current,states,[i,j]))
                         for i,j in itertools.combinations(s,2))
            elif mode=='draft_natural_weights':cost=0. # Its monotone objective selects all.
            else:raise ValueError('unknown policy')
            if cost<=allowance+1e-10:feasible.append(k)
        k=max(feasible);s=remaining[:k]
        batches.append(s);known+=s;remaining=remaining[k:]
    # These symmetric fixtures have value-independent schedule sizes. Evaluate
    # the induced product-of-true-marginals likelihood on every data outcome.
    kl=0.
    for row,prob in enumerate(q):
        if prob==0:continue
        x=states[row];known=[];logp=0.
        for s in batches:
            current=conditional(q,states,known,x[known])
            for i in s:logp+=np.log(marginal(current,states,[i])[x[i]])
            known+=s
        kl+=prob*(np.log(prob)-logp)
    return dict(counts=list(map(len,batches)),steps=len(batches),sampling_KL_nats=max(0.,float(kl)))


def main():
    n=8;states=np.array(list(itertools.product((0,1),repeat=n)))
    uniform=np.ones(len(states))/len(states)
    copied=((states.sum(1)==0)|(states.sum(1)==n)).astype(float);copied/=copied.sum()
    parity=(states.sum(1)%2==0).astype(float);parity/=parity.sum()
    result={}
    for name,q in [('independent',uniform),('copied',copied),('parity',parity)]:
        result[name]={mode:schedule(q,states,mode) for mode in
            ('entropy_bound','exact_tc','pairwise','draft_natural_weights')}
    assert result['independent']['exact_tc']['counts']==[8]
    assert result['independent']['entropy_bound']['steps']==8
    assert result['copied']['exact_tc']['counts']==[1,7]
    assert abs(result['copied']['draft_natural_weights']['sampling_KL_nats']-7*np.log(2))<1e-10
    assert result['parity']['exact_tc']['counts']==[7,1]
    assert abs(result['parity']['pairwise']['sampling_KL_nats']-np.log(2))<1e-10
    out=Path('outputs/latent_allowance_toys');out.mkdir(exist_ok=True)
    payload=dict(scope='Oracle discrete Bayes model: sampling-KL diagnostic, not LLaDA or greedy-code performance',
                 bits=n,allowance=.1,results=result)
    (out/'diagnostic.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2))


if __name__=='__main__':main()
