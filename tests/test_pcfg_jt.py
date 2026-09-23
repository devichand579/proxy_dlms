"""Pairwise PCFG Jt must match reconstructed mutual information."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / "llada-pcfg"), str(REPO / "llada-pcfg" / "scripts")]

from pcfg_data import build_stack  # noqa: E402
from pcfg_engine import entropy  # noqa: E402


def _reconstruct_mi(eng, L: int, a: int, b: int, committed: dict) -> float:
    qa = eng.marginal(L, committed, a)
    qb = eng.marginal(L, committed, b)
    cond = 0.0
    for v, pv in enumerate(qa):
        if pv <= 0.0:
            continue
        child = dict(committed)
        child[a] = int(v)
        cond += float(pv) * entropy(eng.marginal(L, child, b))
    return float(entropy(qb) - cond)


def test_pairwise_jt_matches_reconstructed_mi() -> None:
    _g, _cnf, eng, _vocab = build_stack()
    L = 32
    committed: dict = {}
    a, b = 8, 9
    recon = _reconstruct_mi(eng, L, a, b, committed)
    jt = float(eng.Jt(L, committed, [a, b]))
    assert recon > 0.5, recon
    assert jt == np.clip(jt, 0.0, None)
    assert abs(jt - recon) < 0.05, (jt, recon)
