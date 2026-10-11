"""Paired win/loss + exact McNemar p between runs.  python3 das/pair.py runA_dir:runB_dir ..."""
import glob
import json
import os
import sys
from math import comb


def load(d):
    if os.path.exists(d + '/results.json'):
        return {int(k): v in (True, 'True') for k, v in json.load(open(d + '/results.json'))['passed'].items()}
    r = {}
    for f in glob.glob(d + '/batch_*.json'):
        for s in json.load(open(f))['samples']:
            r[s['task']] = s['passed'] in (True, 'True')
    return r


def p(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


for a, b in [x.split(':') for x in sys.argv[1:]]:
    A, B = load(a), load(b)
    ks = set(A) & set(B)
    w = sum(A[k] and not B[k] for k in ks)
    l = sum(B[k] and not A[k] for k in ks)
    print(f"{a} ({sum(A[k] for k in ks)}) vs {b} ({sum(B[k] for k in ks)}) n={len(ks)}: {w}/{l} p={p(w, l):.3g}")
