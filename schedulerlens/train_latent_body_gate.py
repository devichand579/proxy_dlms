"""Calibrate a latent content/filler gate while retaining confidence risk on code body."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from load_model import load_tokenizer
from schedulerlens.latent_bench import load_documents, set_seed
from schedulerlens.train_latent_risk import examples, metrics
from schedulers.latent_risk import RiskHead


def labels_for(data, seed):
    rng=np.random.default_rng(seed);labels=[]
    for ex in data:
        for fraction in (.2,.5,.8,1.):
            selected=rng.random(512)<fraction
            selected[rng.integers(0,len(ex['body']))]=True
            labels.extend((np.flatnonzero(selected)<len(ex['body'])).tolist())
    return torch.tensor(labels,dtype=torch.float32)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path('outputs/latent_risk_mbpp_v2'))
    parser.add_argument('--features',choices=('all','token','scalar'),default='all')
    args=parser.parse_args();set_seed(82);torch.set_num_threads(4)
    dataset=torch.load(args.directory/'features.pt',weights_only=True)
    artifact=torch.load(args.directory/'risk_head.pt',weights_only=True)
    tokenizer=load_tokenizer('models/llada');human,_=load_documents(None,164)
    excluded={d['entry_point'] for d in human}
    train,_=examples(tokenizer,'train',artifact['metadata']['train_problems'],excluded)
    validation,_=examples(tokenizer,'validation',artifact['metadata']['validation_problems'],excluded)
    y,vy=labels_for(train,81),labels_for(validation,82)
    x,vx=dataset['train_features'],dataset['validation_features']
    h_projection,e_projection=artifact['h_projection'],artifact['e_projection']
    if args.features in ('token','scalar'):
        width=artifact['e_projection'].shape[1]+4 if args.features=='token' else 4
        x,vx=x[:,-width:],vx[:,-width:]
        h_projection=torch.empty(artifact['h_projection'].shape[0],0)
        if args.features=='scalar':
            e_projection=torch.empty(artifact['e_projection'].shape[0],0)
    assert len(x)==len(y) and len(vx)==len(vy)
    mean,scale=x.mean(0),x.std(0).clamp_min(.01)
    xx,vxx=(x-mean)/scale,(vx-mean)/scale
    head=RiskHead(x.shape[1]);optimizer=torch.optim.AdamW(head.parameters(),lr=.002,weight_decay=.01)
    best=None;best_brier=float('inf')
    for epoch in range(12):
        for index in torch.randperm(len(x)).split(4096):
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(head(xx[index]),y[index])
            loss.backward();optimizer.step()
        with torch.no_grad():pred=torch.cat([head(p).sigmoid() for p in vxx.split(4096)])
        score=metrics(pred,vy)
        if score['brier']<best_brier:
            best_brier=score['brier'];best={k:v.detach().clone() for k,v in head.state_dict().items()}
        print(json.dumps({'epoch':epoch,**score}),flush=True)
    head.load_state_dict(best)
    with torch.no_grad():
        pred=torch.cat([head(p).sigmoid() for p in vxx.split(4096)])
    harmless=torch.zeros(126464,dtype=torch.bool)
    for token in set(tokenizer.get_vocab().values()):
        if not tokenizer.decode([token],skip_special_tokens=True).strip():harmless[token]=True
    summary={**artifact['metadata'],'feature_mode':'body_gate','feature_inputs':args.features,
             'label':'reference code-body membership versus harmless tail',
             'gate_validation':metrics(pred,vy),'hybrid':'confidence error on inferred body; nonwhitespace penalty on inferred tail'}
    name='body_gate' if args.features=='all' else 'body_gate_'+args.features
    torch.save({'head':best,'mean':mean,'scale':scale,'h_projection':h_projection,
                'e_projection':e_projection,'harmless':harmless,'metadata':summary},args.directory/(name+'.pt'))
    (args.directory/(name+'_validation.json')).write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
