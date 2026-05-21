"""Run TAG end to end: hiddens, graph, prune, export."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from load_model import infer_diffusion_token_ids, load_model
from cst.eval.logits_head import logits_from_prelogit_hidden, unembedding_weight_matrix
from cst.model.cst_bank import CSTBank
from cst.train.hidden_chunk import load_chunk_H

from .activations import (
    compute_activation_list,
    per_step_residual_vectors,
)
from .capture import capture_prompt_to_pt
from .cst_load import load_cst
from .graph import TAGEdge, TAGNode, build_tag_graph, graph_to_dict
from .influence_prune import prune_tag_influence
from .viz import write_graph_json, write_tag_html


@torch.no_grad()
def _run_tag_graph_only(
    h: torch.Tensor,
    meta: dict,
    cst: CSTBank,
    ccfg: dict,
    *,
    model_path: str | None,
    device: str,
    top_output_mass: float,
    max_output_tokens: int,
    influence_node_mass: float,
    influence_edge_mass: float,
    mix_uniform: float,
    with_residual: bool,
    active_threshold: float,
    output_exclude_eos: bool = False,
    eos_id: Optional[int] = None,
) -> Dict[str, Any]:
    t_steps, b, l, d = h.shape
    w = int(cst.w)
    msteps = int(ccfg.get("max_steps", t_steps))
    activations = compute_activation_list(h, cst, max_step=msteps)
    final_logits: Optional[torch.Tensor] = None
    unembed_w: Optional[torch.Tensor] = None
    eos_for_readout: Optional[int] = eos_id
    if model_path and Path(model_path).is_dir():
        m = load_model(model_path, device=device, capture_pre_logits=False, compile_model=False)
        h_last = h[-1, :1].to(device=device)
        lg = logits_from_prelogit_hidden(m, h_last)
        final_logits = lg.cpu()
        if output_exclude_eos and eos_for_readout is None:
            _mid, eos_inf, _eot = infer_diffusion_token_ids(str(model_path))
            eos_for_readout = int(eos_inf) if eos_inf is not None else None
        try:
            unembed_w = unembedding_weight_matrix(m).float().cpu()
        except TypeError:
            unembed_w = None
    res: Optional[torch.Tensor] = None
    if with_residual:
        dtyp = next(cst.csts["1"].parameters()).dtype
        h_dev = h.to(device=device, dtype=dtyp)
        res = per_step_residual_vectors(h_dev, cst, w, max_anchor=t_steps)
        if res.numel():
            res = res.cpu()
    nodes, edges = build_tag_graph(
        cst,
        activations,
        prompt_len=int(meta.get("prompt_len", 0) or 0),
        seq_len=l,
        w_cst=w,
        max_steps=msteps,
        top_output_mass=top_output_mass,
        max_output_tokens=max_output_tokens,
        final_logits=final_logits,
        output_exclude_eos=output_exclude_eos,
        eos_id=eos_for_readout,
        unembed_weight=unembed_w,
        residual_l2=None,
        residual_d=res,
        active_threshold=active_threshold,
    )
    if not edges:
        graph = graph_to_dict(nodes, edges)
        pr_meta: Dict[str, Any] = {}
    else:
        n2, e2, pr_meta = prune_tag_influence(
            nodes,
            edges,
            node_mass=influence_node_mass,
            edge_mass=influence_edge_mass,
            mix_uniform=mix_uniform,
        )
        graph = graph_to_dict(n2, e2)
    return {
        "graph": graph,
        "meta": dict(meta),
        "cst_config": ccfg,
        "h_shape": [t_steps, b, l, d],
        "prune": pr_meta,
    }


@torch.no_grad()
def run_tag_from_chunk(
    hidden_pt: str,
    cst_checkpoint: str,
    *,
    cst_config_path: str | None = None,
    model_path: str | None = None,
    device: str = "cuda",
    top_output_mass: float = 0.95,
    max_output_tokens: int = 10,
    influence_node_mass: float = 0.8,
    influence_edge_mass: float = 0.98,
    mix_uniform: float = 1e-4,
    with_residual: bool = False,
    active_threshold: float = 0.0,
    output_exclude_eos: bool = False,
    eos_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Build TAG from an existing hidden .pt file."""
    h_cpu, meta, _ = load_chunk_H(hidden_pt)
    cst, ccfg = load_cst(cst_checkpoint, cst_config_path, device=device)
    return _run_tag_graph_only(
        h_cpu,
        meta,
        cst,
        ccfg,
        model_path=model_path,
        device=device,
        top_output_mass=top_output_mass,
        max_output_tokens=max_output_tokens,
        influence_node_mass=influence_node_mass,
        influence_edge_mass=influence_edge_mass,
        mix_uniform=mix_uniform,
        with_residual=with_residual,
        active_threshold=active_threshold,
        output_exclude_eos=output_exclude_eos,
        eos_id=eos_id,
    )


@torch.no_grad()
def run_tag_full(
    model_path: str,
    cst_checkpoint: str,
    prompt: str,
    out_dir: str,
    *,
    cst_config_path: str | None = None,
    mode_key: str = "pd",
    gen_length: int = 256,
    steps: int = 256,
    device: str = "cuda",
    influence_node_mass: float = 0.8,
    influence_edge_mass: float = 0.98,
    mix_uniform: float = 1e-4,
    with_residual: bool = False,
    top_output_mass: float = 0.95,
    max_output_tokens: int = 10,
    write_html: bool = True,
    use_eos_harness: bool = True,
    output_exclude_eos: bool = False,
    eos_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Capture a prompt, build TAG, write JSON and HTML under out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pt = str(out / "tag_trace.pt")
    meta = capture_prompt_to_pt(
        model_path,
        prompt,
        pt,
        device=device,
        mode_key=mode_key,
        gen_length=gen_length,
        steps=steps,
        use_eos_harness=use_eos_harness,
    )
    gout = run_tag_from_chunk(
        pt,
        cst_checkpoint,
        cst_config_path=cst_config_path,
        model_path=model_path,
        device=device,
        influence_node_mass=influence_node_mass,
        influence_edge_mass=influence_edge_mass,
        mix_uniform=mix_uniform,
        with_residual=with_residual,
        top_output_mass=top_output_mass,
        max_output_tokens=max_output_tokens,
        output_exclude_eos=output_exclude_eos,
        eos_id=eos_id,
    )
    g = gout["graph"]
    write_graph_json(g, out / "tag_graph.json")
    if write_html:
        write_tag_html(g, out / "tag_view.html", title="TAG")
    return {
        "capture_meta": meta,
        "tag": gout,
        "paths": {
            "pt": pt,
            "json": str(out / "tag_graph.json"),
            "html": str(out / "tag_view.html") if write_html else "",
        },
    }
