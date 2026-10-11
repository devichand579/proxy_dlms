"""Offline reveal-value feature ablations using the same cached measured effects."""
import argparse
import json
from pathlib import Path

import torch

from schedulerlens.latent_bench import set_seed
from schedulerlens.train_latent_value import audit
from schedulers.latent_risk import RiskHead


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('outputs/latent_value_mbpp'))
    args = parser.parse_args(); torch.set_num_threads(4)
    d = torch.load(args.directory/'features.pt',weights_only=True)
    a = torch.load(args.directory/'value_head.pt',weights_only=True)
    h, e = a['h_projection'].shape[1], a['e_projection'].shape[1]
    choices = {'scalar':torch.arange(h+e,h+e+4),
               'token':torch.arange(h,h+e+4), 'local':torch.arange(h+e+4)}
    result = {'full':a['metadata']['validation'], 'full_body_only':a['metadata']['body_only']}
    for name, indices in choices.items():
        set_seed(91)
        x, vx = d['x'][:,indices], d['vx'][:,indices]
        y, vy, gain_scale = d['y'], d['vy'], a['metadata']['gain_scale']
        mean, scale = x.mean(0), x.std(0).clamp_min(.01)
        xx, vxx = (x-mean)/scale, (vx-mean)/scale
        head = RiskHead(len(indices)); optimizer = torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.03)
        best, best_mse = None, float('inf')
        for epoch in range(60):
            for rows in torch.randperm(len(x)).split(256):
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(head(xx[rows]),y[rows]/gain_scale)
                loss.backward(); optimizer.step()
            with torch.no_grad(): pred = head(vxx)*gain_scale
            mse = float(((pred-vy)**2).mean())
            if mse < best_mse:
                best_mse = mse; best = {k:v.detach().clone() for k,v in head.state_dict().items()}
        head.load_state_dict(best)
        with torch.no_grad(): pred = head(vxx)*gain_scale
        result[name] = {'all':audit(pred,vy), 'body_only':audit(pred[d['vbody']],vy[d['vbody']])}
        artifact = {**a, 'head':best, 'mean':mean, 'scale':scale, 'feature_indices':indices,
                    'metadata':{**a['metadata'], 'feature_inputs':name,
                                'validation':result[name]['all'], 'body_only':result[name]['body_only']}}
        torch.save(artifact,args.directory/f'value_head_{name}.pt')
    (args.directory/'feature_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__': main()
