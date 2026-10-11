"""Exact oracle-state fixtures for the forced-mask count mechanism.

This validates count logic on known distributions, not a LLaDA activation proxy.
"""
import json
import math
from pathlib import Path

from schedulers.latent_count import batch_count


def schedule(kind,dimension=64,steps=16,cap=16):
    remaining=dimension;counts=[]
    for step in range(steps):
        if not remaining:break
        if step==0:chosen=1
        else:
            sizes=sorted({1,min(cap,remaining)}|{n for n in [2,4,8,16] if n<=min(cap,remaining)})
            # Exact posterior-mean latent fixture: for parity, a kept target
            # changes only when every other unresolved bit is supplied. For
            # repeated bits after the first reveal, every target is resolved.
            errors=[1. if kind=='parity' and n==remaining and n>1 else 0. for n in sizes]
            chosen,_=batch_count(sizes,errors,remaining,steps-step,budget=1.)
        counts.append(chosen);remaining-=chosen
    return counts


def main():
    out=Path('outputs/latent_count_toys');out.mkdir(exist_ok=True)
    repeated=schedule('repeated');parity=schedule('parity')
    assert sum(repeated)==sum(parity)==64
    assert repeated[0]==parity[-1]==1
    result=dict(dimension=64,step_budget=16,
        oracle_activation_count_schedule=dict(repeated=repeated,parity=parity),
        exact_sampling_kl_nats=dict(uniform_count4_repeated=3*math.log(2),uniform_count4_parity=math.log(2),
            guarded_repeated=0.,guarded_parity=0.),
        assumptions='Coherent exact binary joints; hypothetical target activations encode exact posterior means and intrinsic uncertainty. This is an oracle fixture, not a neural-network measurement.',
        limitations='Tests count logic only. A fixed schedule using both initial and final singletons also solves these two boundary examples. No claim that all fixed schedules fail; no LLaDA or greedy-task guarantee.')
    (out/'diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
