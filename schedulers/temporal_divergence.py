"""Full-vocabulary posterior movement controls for the latent Fisher approximation."""
import math
import torch


def temporal_divergence(current, previous, metric='kl', chunk_size=128):
    if metric not in ('kl','js'):
        raise ValueError('unknown temporal divergence')
    pieces = []
    for cur, old in zip(current.split(chunk_size),previous.split(chunk_size)):
        p, q = cur.float().log_softmax(-1), old.float().log_softmax(-1)
        if metric == 'kl':
            delta = (p-q).masked_fill(~p.isfinite(),0.)
            value = (p.exp()*delta).sum(-1)
        else:
            middle = torch.logaddexp(p,q)-math.log(2)
            dp = (p-middle).masked_fill(~p.isfinite(),0.)
            dq = (q-middle).masked_fill(~q.isfinite(),0.)
            value = .5*((p.exp()*dp).sum(-1)+(q.exp()*dq).sum(-1))
        pieces.append(value.clamp_min(0.))
    return torch.cat(pieces)
