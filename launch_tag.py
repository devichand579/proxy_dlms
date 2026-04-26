"""Hydra entry: TAG (Temporal Attribution Graph) for a single prompt or existing hidden ``.pt``."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

# Project root
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from repo_env_defaults import apply_repo_env_defaults


@hydra.main(version_base="1.3", config_path="config", config_name="tag")
def main(cfg: DictConfig) -> int:
    apply_repo_env_defaults()
    from tag.pipeline import run_tag_from_chunk, run_tag_full
    from tag.viz import write_tag_html, write_graph_json

    c: Any = cfg
    ckpt = c.get("cst_checkpoint")
    if ckpt in (None, "null", ""):
        raise ValueError("Set cst_checkpoint=/path/to/cst_..._train_best.pt")
    cst_path = str(ckpt)
    out_dir = Path(str(c.get("out_dir", "outputs/tag")))
    out_dir.mkdir(parents=True, exist_ok=True)
    cconf = c.get("cst_config")
    cst_config_path = str(cconf) if cconf not in (None, "null", "") else None
    mpath = str(c.get("model_path", ""))
    device = str(c.get("device", "cuda"))
    oee = c.get("output_exclude_eos", False)
    if isinstance(oee, str):
        oee = oee.strip().lower() in ("true", "1", "yes", "on")
    else:
        oee = bool(oee)
    eid = c.get("eos_id", None)
    if eid in (None, "null", "none", ""):
        eid = None
    else:
        eid = int(eid)
    hpt = c.get("hidden_pt")
    if hpt and str(hpt) not in ("null", "none", ""):
        g = run_tag_from_chunk(
            str(hpt),
            cst_path,
            cst_config_path=cst_config_path,
            model_path=mpath or None,
            device=device,
            prune_p=float(c.get("prune_p", 0.1)),
            prune_mode=str(c.get("prune_mode", "influence")),
            influence_node_mass=float(c.get("influence_node_mass", 0.8)),
            influence_edge_mass=float(c.get("influence_edge_mass", 0.98)),
            mix_uniform=float(c.get("mix_uniform", 1e-4)),
            with_residual=bool(c.get("with_residual", False)),
            output_exclude_eos=oee,
            eos_id=eid,
        )
        write_graph_json(g["graph"], out_dir / "tag_graph.json")
        if bool(c.get("write_html", True)):
            write_tag_html(g["graph"], out_dir / "tag_view.html")
        print(json.dumps({"graph_nodes": len(g["graph"].get("nodes", [])), "wrote": str(out_dir)}))
        return 0

    prompt = c.get("prompt")
    if not prompt or str(prompt).lower() in ("null", "none", ""):
        raise ValueError("Set prompt=... for full capture+TAG, or set hidden_pt=... to an existing .pt")
    ueh = c.get("use_eos_harness", True)
    if isinstance(ueh, str):
        use_eos_harness = ueh.strip().lower() in ("true", "1", "yes", "on")
    else:
        use_eos_harness = True if ueh is None else bool(ueh)
    r = run_tag_full(
        mpath,
        cst_path,
        str(prompt),
        str(out_dir),
        cst_config_path=cst_config_path,
        mode_key=str(c.get("mode_key", "pd")),
        gen_length=int(c.get("gen_length", 256)),
        steps=int(c.get("steps", 256)),
        device=device,
        prune_p=float(c.get("prune_p", 0.1)),
        prune_mode=str(c.get("prune_mode", "influence")),
        influence_node_mass=float(c.get("influence_node_mass", 0.8)),
        influence_edge_mass=float(c.get("influence_edge_mass", 0.98)),
        mix_uniform=float(c.get("mix_uniform", 1e-4)),
        with_residual=bool(c.get("with_residual", False)),
        write_html=bool(c.get("write_html", True)),
        use_eos_harness=use_eos_harness,
        output_exclude_eos=oee,
        eos_id=eid,
    )
    print(json.dumps(r.get("paths", {}), indent=0))
    return 0


if __name__ == "__main__":
    main()
