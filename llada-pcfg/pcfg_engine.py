"""
Exact arbitrary-order conditionals for a CNF PCFG (ground-truth q).

CKY inside/outside runs in the Rust extension (pcfg_engine_rs). This module
exposes a thin Python API plus numeric helpers used by training/validation.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12

try:
    from pcfg_engine_rs import PyPCFGEngine as _RustPCFGEngine
except ImportError as exc:
    raise ImportError(
        "pcfg_engine_rs is required. Build from repo root:\n"
        "  pip install maturin && maturin develop --release"
    ) from exc


def kl(p, q):
    return float(np.sum(np.where(p > 0, p * (np.log(p + EPS) - np.log(q + EPS)), 0.0)))


def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p + EPS)))


def corrupt(p, temp=1.0, sigma=0.0, rng=None):
    lg = np.log(p + EPS)
    if sigma:
        lg = lg + sigma * (rng or np.random).standard_normal(p.shape)
    lg = lg / temp
    lg -= lg.max()
    e = np.exp(lg)
    return e / e.sum()


class PCFGState:
    """Incremental inside + outside for scheduler-driven unmasking."""

    def __init__(self, inner, engine: "PCFGEngine"):
        self._inner = inner
        self.engine = engine
        self.L = inner.l

    def commit(self, pos, token):
        self._inner.commit(int(pos), int(token))

    def all_marginals(self):
        return self.engine._marginals_from_rust(self._inner.all_marginals())

    def marginals_batch(self, query_positions):
        return self.engine._marginals_from_rust(
            self._inner.marginals_batch(list(query_positions))
        )

    def Mt(self, S, model_fn):
        all_m = self.marginals_batch(S)
        return sum(kl(all_m[l], model_fn(all_m[l])) for l in S if l in all_m)

    def Jt(self, S):
        return self.engine.Jt(self.L, self._inner.committed_dict(), S)


class PCFGEngine:
    """Rust-backed PCFG engine with a stable Python API."""

    def __init__(self, cnf):
        self._inner = _RustPCFGEngine(cnf)
        self.cnf = cnf
        self.V = self._inner.V
        self.n = self._inner.n
        self.t_idx = self._inner.t_idx
        self.nt_idx = {A: i for i, A in enumerate(cnf.NT)}
        self.start = self.nt_idx[cnf.start]
        self.NT = cnf.NT

    @staticmethod
    def _marginals_from_rust(obj):
        if isinstance(obj, dict):
            src = obj.items()
        else:
            src = obj.items() if hasattr(obj, "items") else []
        return {int(k): np.asarray(v, dtype=float) for k, v in src}

    def _commit(self, committed):
        return {int(k): int(v) for k, v in dict(committed).items()}

    @property
    def backend(self):
        return "rust"

    def clear_cache(self):
        self._inner.clear_cache()

    def Z(self, L, committed, use_log=None):
        _ = use_log
        return self._inner.Z(L, self._commit(committed))

    def all_marginals(self, L, committed):
        return self._marginals_from_rust(
            self._inner.all_marginals(L, self._commit(committed))
        )

    def marginals_batch(self, L, committed, query_positions):
        return self._marginals_from_rust(
            self._inner.marginals_batch(L, self._commit(committed), list(query_positions))
        )

    def marginals_suffix(self, L, prefix_committed, query_positions=None):
        clamp = dict(prefix_committed)
        if query_positions is None:
            used = set(clamp)
            query_positions = [p for p in range(L) if p not in used]
        return self.marginals_batch(L, clamp, query_positions)

    def marginal(self, L, committed, pos, use_log=None):
        _ = use_log
        return self.marginals_batch(L, committed, [pos]).get(pos, np.zeros(self.V))

    def marginals(self, L, committed, positions, use_log=None):
        _ = use_log
        batch = self.marginals_batch(L, committed, positions)
        return {p: batch.get(p, np.zeros(self.V)) for p in positions}

    def marginals_rightmost_suffix(self, L, prefix_indices, query_positions, use_log=None):
        _ = use_log
        return self.marginals_batch(L, prefix_indices, query_positions)

    def Jt(self, L, committed, S):
        return self._inner.Jt_entropy(L, self._commit(committed), sorted(S))

    def Jt_entropy(self, L, committed, S):
        return self.Jt(L, committed, S)

    def Mt(self, L, committed, S, model_fn):
        all_m = self.marginals_batch(L, committed, S)
        return sum(kl(all_m[l], model_fn(all_m[l])) for l in S if l in all_m)

    def make_state(self, L, committed=None):
        st = self._inner.make_state(L, self._commit(committed or {}))
        return PCFGState(st, self)


def engine_backend(engine=None):
    if engine is not None and hasattr(engine, "backend"):
        return engine.backend
    return "rust"
