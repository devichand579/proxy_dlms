"""BFS-style pruning: per-destination top fraction of in-edges by |weight|. For mass-based cuts use ``influence_prune``."""

from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, List, Set, Tuple

from .graph import TAGEdge, TAGNode


def prune_tag_edges_bfs(
    nodes: List[TAGNode],
    edges: List[TAGEdge],
    *,
    sink_kinds: Tuple[str, ...] = ("out",),
    top_p: float = 0.1,
) -> List[TAGEdge]:
    """Keep all edges into sink nodes; else keep top_p fraction of in-edges by |w| (at least one)."""
    sink_ids = {n.id for n in nodes if n.kind in sink_kinds}
    n_by_dst: DefaultDict[str, List[Tuple[int, float]]] = defaultdict(list)
    for i, e in enumerate(edges):
        n_by_dst[e.dst].append((i, float(abs(e.weight))))

    keep: Set[int] = set()
    for dst, li in n_by_dst.items():
        if dst in sink_ids:
            for j, _ in li:
                keep.add(j)
            continue
        if not li:
            continue
        n_keep = max(1, int(len(li) * top_p)) if 0.0 < top_p < 1.0 else len(li)
        n_keep = min(n_keep, len(li))
        li = sorted(li, key=lambda t: t[1], reverse=True)
        for j, _w in li[:n_keep]:
            keep.add(j)

    return [e for idx, e in enumerate(edges) if idx in keep]
