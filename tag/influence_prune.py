"""Prune TAG by fixed-point influence: |A| column-stochastic, (I-A^T)u=r, then node/edge mass."""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

import numpy as np

from .graph import TAGEdge, TAGNode


def _logit_source_vector(
    nodes: List[TAGNode], n: int, id_to_i: Dict[str, int]
) -> np.ndarray:
    r = np.zeros(n, dtype=np.float64)
    for node in nodes:
        if node.kind != "out" or node.id not in id_to_i:
            continue
        ps = node.meta.get("p_sink", 0.0)
        try:
            ps = float(ps)
        except (TypeError, ValueError):
            ps = 0.0
        if ps > 0:
            r[id_to_i[node.id]] = ps
    s = r.sum()
    if s > 0:
        r /= s
    return r


def _aggregate_abs_weights(
    nodes: List[TAGNode], edges: List[TAGEdge]
) -> Tuple[Dict[str, int], np.ndarray]:
    n = len(nodes)
    id_to_i = {nodes[i].id: i for i in range(n)}
    w = np.zeros((n, n), dtype=np.float64)
    for e in edges:
        if e.src not in id_to_i or e.dst not in id_to_i:
            continue
        j = id_to_i[e.src]
        i = id_to_i[e.dst]
        w[i, j] += float(abs(e.weight))
    return id_to_i, w


def _column_stochastic(a_raw: np.ndarray) -> np.ndarray:
    a = a_raw.copy()
    n = a.shape[0]
    for j in range(n):
        col_sum = a[:, j].sum()
        if col_sum > 0:
            a[:, j] /= col_sum
        else:
            a[j, j] = 1.0
    return a


def _damped_stochastic(a: np.ndarray, mix: float) -> np.ndarray:
    n = a.shape[0]
    if n == 0 or mix <= 0:
        return a
    m = min(float(mix), 0.5)
    uu = np.full((n, n), 1.0 / n, dtype=np.float64)
    return (1.0 - m) * a + m * uu


def _solve_u_minus_r(a: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Return u - r for (I - A^T) u = r; r is normalized."""
    n = a.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    rs = r.sum()
    if rs <= 0:
        return np.zeros(n, dtype=np.float64)
    rc = (r / rs).astype(np.float64)
    i = np.eye(n, dtype=np.float64)
    m = i - a.T
    if n < 5000 and n**2 * 8 < 1_200_000_000:
        try:
            u = np.linalg.solve(m, rc)
        except np.linalg.LinAlgError:
            u = np.linalg.lstsq(m, rc, rcond=None)[0]
        return u - rc
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import spsolve

    m_sp = csr_matrix(m)
    u = spsolve(m_sp, rc)
    u = np.asarray(u, dtype=np.float64).ravel()
    return u - rc


def prune_tag_influence(
    nodes: List[TAGNode],
    edges: List[TAGEdge],
    *,
    node_mass: float = 0.8,
    edge_mass: float = 0.98,
    mix_uniform: float = 1e-4,
) -> Tuple[List[TAGNode], List[TAGEdge], Dict[str, Any]]:
    """
    Influence scores, then keep feat by cumulative node_mass, in/out/err always, then
    edges by mass on the induced subgraph up to edge_mass.
    """
    n = len(nodes)
    if n == 0 or not edges:
        return nodes, edges, {"method": "influence", "skipped": "empty"}
    if not any(x.kind == "out" for x in nodes):
        return nodes, edges, {"method": "influence", "skipped": "no_logit_sinks"}

    id_to_i, a_raw = _aggregate_abs_weights(nodes, edges)
    r = _logit_source_vector(nodes, n, id_to_i)
    if r.sum() <= 0:
        return nodes, edges, {"method": "influence", "skipped": "no_logit_r"}

    a = _damped_stochastic(_column_stochastic(a_raw), mix_uniform)
    scores = _solve_u_minus_r(a, r)

    # --- Phase 2: feat by cumulative mass on scores among feat only ---
    feat_i = [i for i, nx in enumerate(nodes) if nx.kind == "feat"]
    tot_feat = float(sum(scores[i] for i in feat_i))
    keep_feat: Set[str] = set()
    if feat_i:
        if tot_feat > 0:
            order = sorted(feat_i, key=lambda i: scores[i], reverse=True)
            cum = 0.0
            for i in order:
                keep_feat.add(nodes[i].id)
                cum += scores[i] / tot_feat
                if cum + 1e-15 >= node_mass:
                    break
        else:
            # uniform: keep minimal prefix to satisfy mass (interpret as equal)
            m_keep = max(1, int(np.ceil(node_mass * len(feat_i))))
            order = feat_i
            for i in order[:m_keep]:
                keep_feat.add(nodes[i].id)

    keep = set(keep_feat)
    for nx in nodes:
        if nx.kind in ("in", "out", "err"):
            keep.add(nx.id)
    if feat_i and not keep_feat and tot_feat <= 0:
        n_keep = max(1, int(np.ceil(node_mass * len(feat_i))))
        for i in feat_i[:n_keep]:
            keep.add(nodes[i].id)
    nodes2 = [n for n in nodes if n.id in keep]
    e2 = [e for e in edges if e.src in keep and e.dst in keep]
    if not e2:
        return nodes2, e2, {
            "method": "influence",
            "node_mass": node_mass,
            "edge_mass": edge_mass,
            "n_nodes_before": n,
            "n_nodes_after": len(nodes2),
            "n_edges_after": 0,
        }

    # --- Phase 3: subgraph ---
    m2 = len(nodes2)
    id2 = {nodes2[i].id: i for i in range(m2)}
    a2 = np.zeros((m2, m2), dtype=np.float64)
    for e in e2:
        j = id2[e.src]
        i = id2[e.dst]
        a2[i, j] += float(abs(e.weight))
    a2 = _damped_stochastic(_column_stochastic(a2), mix_uniform)
    r2 = _logit_source_vector(nodes2, m2, id2)
    scores2 = _solve_u_minus_r(a2, r2) if r2.sum() > 0 else np.zeros(m2, dtype=np.float64)

    es: List[Tuple[str, str, float]] = []
    for e in e2:
        si, di = id2[e.src], id2[e.dst]
        a_ts = a2[di, si]  # s→t: column si, row di
        es.append((e.src, e.dst, float(scores2[di] * a_ts)))

    es_aug = [(s, d, max(0.0, wsc)) for s, d, wsc in es]
    es_aug.sort(key=lambda t: -t[2])
    wsum = float(sum(x[2] for x in es_aug))
    keep_pairs: Set[Tuple[str, str]] = set()
    if wsum > 0:
        cum2 = 0.0
        for src, dst, wsc in es_aug:
            if wsc <= 0:
                continue
            keep_pairs.add((src, dst))
            cum2 += wsc / wsum
            if cum2 + 1e-12 >= edge_mass:
                break
    if not keep_pairs and es:
        keep_pairs = {(es[0][0], es[0][1])}
    if not keep_pairs:
        keep_e = e2
    else:
        keep_e = [e for e in e2 if (e.src, e.dst) in keep_pairs]
    if not keep_e and e2:
        keep_e = e2[:1]

    meta: Dict[str, Any] = {
        "method": "influence",
        "node_mass": node_mass,
        "edge_mass": edge_mass,
        "n_nodes_before": n,
        "n_nodes_after": len(nodes2),
        "n_edges_before": len(edges),
        "n_edges_after": len(keep_e),
    }
    return nodes2, keep_e, meta
