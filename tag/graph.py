"""Build TAG node/edge lists from activations and precomputed virtual weights."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import torch

from cst.model.cst_bank import CSTBank, effective_window_size

NodeKind = Literal["feat", "in", "out", "err"]


@dataclass
class TAGNode:
    id: str
    kind: NodeKind
    pos: int
    t: int
    f: int = -1
    label: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TAGEdge:
    src: str
    dst: str
    weight: float
    meta: Dict[str, Any] = field(default_factory=dict)


def _nid_feat(i: int, t: int, f: int) -> str:
    return f"f_{i}_{t}_{f}"


def _nid_in(i: int) -> str:
    return f"in_{i}"


def _nid_out(i: int) -> str:
    return f"out_{i}"


def _nid_err(i: int, t: int) -> str:
    return f"err_{i}_{t}"




def _active_mask(a: torch.Tensor, thr: float = 0.0) -> torch.Tensor:
    """a [B,L,k] → bool [B,L,k] for a > thr (strictly positive after JumpReLU)."""
    return a > thr


def _virtual_submatrix(
    bank: CSTBank,
    t: int,
    delta: int,
    fi: torch.Tensor,
    gj: torch.Tensor,
) -> torch.Tensor:
    """Block of V[fi, gj] without building the full k×k matrix."""
    cst_t = bank.get(t)
    t2 = t + delta
    cst_t2 = bank.get(t2)
    w_dec = cst_t.W_dec[delta - 1].weight
    w_enc = cst_t2.W_enc.weight
    dev = w_dec.device
    fi_d = fi.to(device=dev, dtype=torch.long)
    gj_d = gj.to(device=dev, dtype=torch.long)
    a = w_dec[:, fi_d]  # [d, len(fi)]
    b = w_enc[gj_d, :]  # [len(gj), d]
    return (a.t() @ b.t()).cpu()  # [len(fi), len(gj)] on CPU for downstream float ops


def build_tag_graph(
    bank: CSTBank,
    activations: Sequence[torch.Tensor],
    *,
    prompt_len: int,
    seq_len: int,
    w_cst: int,
    max_steps: int,
    top_output_mass: float = 0.95,
    max_output_tokens: int = 10,
    final_logits: Optional[torch.Tensor] = None,
    unembed_weight: Optional[torch.Tensor] = None,
    residual_l2: Optional[torch.Tensor] = None,
    residual_d: Optional[torch.Tensor] = None,
    active_threshold: float = 0.0,
) -> Tuple[List[TAGNode], List[TAGEdge]]:
    """
    Build input, feature, cross-step, optional output, and optional error nodes/edges.
    List index s maps to CST step t = s+1. ``V`` blocks are materialized in small submatrices only.
    With ``final_logits`` and ``unembed_weight``: out nodes and feat→out (else last-step a only).
    With ``residual_d``: err nodes and err→feat; with ``residual_l2`` only: err nodes, no err edges.
    """
    nodes: List[TAGNode] = []
    edges: List[TAGEdge] = []
    node_ids: set[str] = set()

    def add_node(n: TAGNode) -> None:
        if n.id not in node_ids:
            node_ids.add(n.id)
            nodes.append(n)

    def add_edge(e: TAGEdge) -> None:
        edges.append(e)

    # --- Input nodes (response positions): sources for first CST step ---
    resp = range(int(prompt_len), int(seq_len))
    for i in resp:
        nid = _nid_in(i)
        add_node(TAGNode(nid, "in", i, 0, -1, label=f"in@{i}"))

    # --- Feature nodes: (i, t, f) for a > 0 ---
    # activations[s] is for CST step t = s+1
    n_t = len(activations)
    for s in range(n_t):
        t = s + 1
        a = activations[s]  # [B,L,k] CPU
        if a.shape[0] < 1:
            continue
        am = _active_mask(a[0:1], thr=active_threshold)  # [1,L,k]
        b, L, k = a.shape
        for i in range(L):
            if i < prompt_len:
                continue
            for f in range(k):
                if not am[0, i, f].item():
                    continue
                nid = _nid_feat(i, t, f)
                val = float(a[0, i, f].item())
                add_node(TAGNode(nid, "feat", i, t, f, label=f"a({i},{t},{f})={val:.3g}", meta={"a": val}))

    # --- Edges: input → (i, 1, f) ---
    if n_t >= 1:
        a1 = activations[0]
        for i in resp:
            for f in range(a1.shape[-1]):
                if not (a1[0, i, f] > active_threshold):
                    continue
                dst = _nid_feat(i, 1, f)
                if dst not in node_ids:
                    continue
                w = float(a1[0, i, f].item())
                add_edge(TAGEdge(_nid_in(i), dst, w, {"type": "input_to_feat", "t": 1}))

    # --- Cross-step feature → feature: (i,t,f) → (j, t+δ, g) ---
    w_eff = int(w_cst)
    for s in range(n_t):
        t = s + 1
        w_t = effective_window_size(t, w_eff, max_steps)
        for delta in range(1, w_t + 1):
            t2 = t + delta
            if t2 > n_t:
                continue
            a_from = activations[s]  # step t
            a_to = activations[t2 - 1]  # step t+delta
            # Bilinear: w = a^t_{i,f} V_{f,g} a^{t2}_{j,g}; only over active f,g
            for i in range(int(prompt_len), int(seq_len)):
                af_row = a_from[0, i]
                fi = torch.nonzero(af_row > active_threshold, as_tuple=False).squeeze(-1)
                if fi.numel() == 0:
                    continue
                for j in range(int(prompt_len), int(seq_len)):
                    ag_row = a_to[0, j]
                    gj = torch.nonzero(ag_row > active_threshold, as_tuple=False).squeeze(-1)
                    if gj.numel() == 0:
                        continue
                    v_sub = _virtual_submatrix(bank, t, int(delta), fi, gj)
                    dtype = v_sub.dtype
                    af_sub = af_row[fi].to(dtype=dtype)  # [Fi]
                    ag_sub = ag_row[gj].to(dtype=dtype)  # [Gj]
                    w_block = (af_sub[:, None] * v_sub) * ag_sub[None, :]
                    nfi, ngj = int(fi.numel()), int(gj.numel())
                    for p in range(nfi):
                        f_ = int(fi[p].item())
                        src = _nid_feat(i, t, f_)
                        if src not in node_ids:
                            continue
                        for q in range(ngj):
                            g_ = int(gj[q].item())
                            dstn = _nid_feat(j, t2, g_)
                            if dstn not in node_ids:
                                continue
                            w_ijk = float(w_block[p, q].item())
                            if w_ijk == 0.0:
                                continue
                            add_edge(
                                TAGEdge(
                                    src,
                                    dstn,
                                    w_ijk,
                                    {
                                        "type": f"cross_{t}_{t2}",
                                        "delta": int(delta),
                                        "f": f_,
                                        "g": g_,
                                    },
                                )
                            )

    # --- Optional error nodes; err → feature if ``residual_d`` [A, L, d] (ε replaces a^t in V) ---
    if residual_d is not None and residual_d.numel() > 0:
        a_rows, a_pos = int(residual_d.shape[0]), int(residual_d.shape[1])
    elif residual_l2 is not None and residual_l2.numel() > 0:
        a_rows, a_pos = int(residual_l2.shape[0]), int(residual_l2.shape[1])
    else:
        a_rows, a_pos = 0, 0
    if a_rows > 0:
        for row in range(a_rows):
            t_err = int(min(row + w_eff, n_t))
            if t_err < 1:
                continue
            for i in resp:
                if i >= a_pos:
                    continue
                if residual_d is not None and residual_d.numel() > 0:
                    eps = residual_d[row, i].float()  # [d]
                    er = float(eps.norm().item())
                else:
                    eps = None
                    er = float(residual_l2[row, i].item())  # type: ignore[union-attr]
                if er <= 0:
                    continue
                nid = _nid_err(i, t_err)
                add_node(
                    TAGNode(
                        nid,
                        "err",
                        i,
                        t_err,
                        -1,
                        label=f"err@({i})",
                        meta={"l2": er, "row": int(row), "t_err": t_err},
                    )
                )
                if eps is None or n_t < 1 or t_err > n_t:
                    continue
                t_from = t_err
                w_t = effective_window_size(t_from, w_eff, max_steps)
                a_from = activations[t_from - 1]
                cst_from = bank.get(t_from)
                for delta in range(1, w_t + 1):
                    t2 = t_from + delta
                    if t2 > n_t:
                        break
                    a_to = activations[t2 - 1]
                    w_m = cst_from.W_dec[delta - 1].weight
                    dev = w_m.device
                    dty = w_m.dtype
                    eps_d = eps.to(device=dev, dtype=dty)
                    # b_f = ε^T W^dec_{:,f}  →  [k]
                    b = torch.matmul(eps_d, w_m)  # [d] @ [d, k] -> [k]
                    b = b.float().cpu()
                    fi = torch.nonzero(
                        a_from[0, i] > active_threshold, as_tuple=False
                    ).squeeze(-1)
                    if fi.numel() == 0:
                        continue
                    b_sub = b[fi]
                    for j in range(int(prompt_len), int(seq_len)):
                        ag_row = a_to[0, j]
                        gj = torch.nonzero(
                            ag_row > active_threshold, as_tuple=False
                        ).squeeze(-1)
                        if gj.numel() == 0:
                            continue
                        v_sub = _virtual_submatrix(
                            bank, t_from, int(delta), fi, gj
                        )
                        dtype_ = v_sub.dtype
                        b_ = b_sub.to(dtype=dtype_)
                        ag_ = ag_row[gj].to(dtype=dtype_)
                        w_block = (b_[:, None] * v_sub) * ag_[None, :]
                        ngj = int(gj.numel())
                        for q in range(ngj):
                            w_sum = float(w_block[:, q].sum().item())
                            if w_sum == 0.0:
                                continue
                            g_ = int(gj[q].item())
                            ddst = _nid_feat(j, t2, g_)
                            if ddst not in node_ids:
                                continue
                            add_edge(
                                TAGEdge(
                                    nid,
                                    ddst,
                                    w_sum,
                                    {
                                        "type": "err_to_feat",
                                        "row": int(row),
                                        "delta": int(delta),
                                        "t_from": t_from,
                                        "t_to": t2,
                                        "g": g_,
                                    },
                                )
                            )

    # --- Output sinks from final_logits ---
    if final_logits is not None and final_logits.numel() > 0:
        # [1, L, V]
        plp = int(prompt_len)
        lg = final_logits[0, plp:].float()  # [Lr, V]
        Lr = lg.shape[0]
        for local_i, pos in enumerate(range(plp, plp + Lr)):
            probs = torch.softmax(lg[local_i], dim=-1)
            sorted_p, sorted_idx = torch.sort(probs, descending=True)
            c = 0.0
            out_ids: List[int] = []
            out_probs: List[float] = []
            for j in range(min(probs.shape[0], max_output_tokens * 5)):
                pj = float(sorted_p[j].item())
                c += pj
                out_ids.append(int(sorted_idx[j].item()))
                out_probs.append(pj)
                if c >= top_output_mass and len(out_ids) >= 1:
                    break
            out_ids = out_ids[: max_output_tokens]
            out_probs = out_probs[: len(out_ids)]
            # For influence pruning: up to 10 tokens, cumulative prob ≥ 0.95 (per out position)
            k_logit = 0
            cum95 = 0.0
            logit_p_tokens: List[Dict[str, object]] = []
            for j in range(len(out_ids)):
                if k_logit >= 10:
                    break
                logit_p_tokens.append(
                    {"id": int(out_ids[j]), "p": float(out_probs[j])}
                )
                cum95 += out_probs[j]
                k_logit += 1
                if cum95 >= 0.95:
                    break
            if not logit_p_tokens and out_ids:
                logit_p_tokens = [{"id": int(out_ids[0]), "p": float(out_probs[0])}]
            p_sink = float(sum(x["p"] for x in logit_p_tokens))
            onid = _nid_out(pos)
            add_node(
                TAGNode(
                    onid,
                    "out",
                    pos,
                    n_t + 1,
                    -1,
                    label=f"out@{pos}",
                    meta={
                        "top_ids": out_ids,
                        "p_mass": c,
                        "logit_p_tokens": logit_p_tokens,
                        "p_sink": p_sink,
                        "role": "logits_readout",
                    },
                )
            )
            if n_t < 1:
                continue
            t_eff = n_t
            tok_ids = [int(x["id"]) for x in logit_p_tokens]
            w_u = unembed_weight
            d_model = int(bank.get(1).W_dec[0].weight.shape[0])
            if (
                w_u is not None
                and w_u.dim() == 2
                and int(w_u.shape[1]) == d_model
            ):
                vocab = int(w_u.shape[0])
                for t in range(max(1, t_eff - w_cst + 1), t_eff + 1):
                    s_idx = t - 1
                    a_t = activations[s_idx]
                    delta_path = t_eff - t
                    d_dec = 1 if delta_path == 0 else int(delta_path)
                    w_tlim = effective_window_size(t, w_cst, max_steps)
                    if d_dec < 1 or d_dec > w_tlim:
                        continue
                    cst_t = bank.get(t)
                    w_dec = cst_t.W_dec[d_dec - 1].weight  # [d, k]
                    dev = w_dec.device
                    for f in range(a_t.shape[-1]):
                        if not (a_t[0, pos, f] > active_threshold):
                            continue
                        af = float(a_t[0, pos, f].item())
                        u_col = w_dec[:, f]  # [d]
                        fn = _nid_feat(pos, t, f)
                        if fn not in node_ids:
                            continue
                        for vid in tok_ids:
                            if vid < 0 or vid >= vocab:
                                continue
                            w_row = w_u[vid, :].to(device=dev, dtype=w_dec.dtype)
                            coef = float(torch.dot(u_col, w_row).item())
                            wgt = af * coef
                            if wgt == 0.0:
                                continue
                            add_edge(
                                TAGEdge(
                                    fn,
                                    onid,
                                    wgt,
                                    {
                                        "type": "feat_to_out_unembed",
                                        "f": int(f),
                                        "t": int(t),
                                        "v": int(vid),
                                        "delta": int(d_dec),
                                    },
                                )
                            )
            else:
                # Fallback: last step only, weight = activation
                a_last = activations[-1]
                t_last = n_t
                for f in range(a_last.shape[-1]):
                    if not (a_last[0, pos, f] > active_threshold):
                        continue
                    fn = _nid_feat(pos, t_last, f)
                    if fn not in node_ids:
                        continue
                    add_edge(
                        TAGEdge(
                            fn,
                            onid,
                            float(a_last[0, pos, f].item()),
                            {"type": "feat_to_out", "f": f},
                        )
                    )

    return nodes, edges


def graph_to_dict(nodes: List[TAGNode], edges: List[TAGEdge]) -> Dict[str, Any]:
    return {
        "nodes": [
            {
                "id": n.id,
                "kind": n.kind,
                "pos": n.pos,
                "t": n.t,
                "f": n.f,
                "label": n.label,
                "meta": n.meta,
            }
            for n in nodes
        ],
        "edges": [
            {
                "source": e.src,
                "target": e.dst,
                "weight": e.weight,
                "meta": e.meta,
            }
            for e in edges
        ],
    }
