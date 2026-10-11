"""Compare the latent readout with a trained scalar-only calibration control."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from load_model import load_tokenizer
from schedulerlens.latent_bench import load_documents, set_seed
from schedulerlens.train_latent_risk import examples, metrics
from schedulers.latent_risk import RiskHead


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("outputs/latent_risk_mbpp_v2"))
    args = parser.parse_args()
    torch.set_num_threads(4); set_seed(81)
    data = torch.load(args.directory/'features.pt',weights_only=True)
    artifact = torch.load(args.directory/'risk_head.pt',weights_only=True)
    x,y = data['train_features'][:,-4:], data['train_labels']
    vx,vy = data['validation_features'][:,-4:], data['validation_labels']
    mean,scale=x.mean(0),x.std(0).clamp_min(.01)
    xx=(x-mean)/scale; vxx=(vx-mean)/scale
    head=RiskHead(4); optimizer=torch.optim.AdamW(head.parameters(),lr=.002,weight_decay=.01)
    best=None; best_brier=float('inf')
    for epoch in range(12):
        for index in torch.randperm(len(x)).split(4096):
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(head(xx[index]),y[index])
            loss.backward();optimizer.step()
        with torch.no_grad():prediction=torch.cat([head(p).sigmoid() for p in vxx.split(4096)])
        score=metrics(prediction,vy)
        if score['brier']<best_brier:
            best_brier=score['brier'];best={k:v.detach().clone() for k,v in head.state_dict().items()}
    head.load_state_dict(best)
    with torch.no_grad():scalar=torch.cat([head(p).sigmoid() for p in vxx.split(4096)])
    full_head=RiskHead(len(artifact['mean']));full_head.load_state_dict(artifact['head'])
    with torch.no_grad():
        full=torch.cat([full_head((p-artifact['mean'])/artifact['scale']).sigmoid() for p in data['validation_features'].split(4096)])
    # Reconstruct the same deterministic masking indices, without model forwards.
    tok=load_tokenizer('models/llada');human,_=load_documents(None,164)
    validation,_=examples(tok,'validation',artifact['metadata']['validation_problems'],{d['entry_point'] for d in human})
    rng=np.random.default_rng(82);body=[]
    for ex in validation:
        for fraction in (.2,.5,.8,1.):
            selected=rng.random(512)<fraction
            selected[rng.integers(0,len(ex['body']))]=True
            positions=np.flatnonzero(selected)
            body.extend((positions<len(ex['body'])).tolist())
    body=torch.tensor(body)
    assert len(body)==len(vy)
    result={}
    for name,pred in [('raw_confidence',data['validation_confidence_risk']),('scalar_readout',scalar),('latent_readout',full)]:
        result[name]={'all':metrics(pred,vy),'body_only':metrics(pred[body],vy[body])}
    torch.save({'head':best,'mean':mean,'scale':scale,
        'h_projection':torch.empty(artifact['h_projection'].shape[0],0),
        'e_projection':torch.empty(artifact['e_projection'].shape[0],0),
        'metadata':{**artifact['metadata'],'feature_mode':'scalar_only'}},args.directory/'scalar_head.pt')
    (args.directory/'calibration_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
