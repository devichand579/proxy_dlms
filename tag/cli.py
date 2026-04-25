"""CLI: ``python -m tag ...`` (capture+TAG or chunk-only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import run_tag_from_chunk, run_tag_full


def main() -> int:
    p = argparse.ArgumentParser(description="Build TAG: Temporal Attribution Graph for CST on one prompt or .pt")
    p.add_argument("--model_path", type=str, help="Local LM (must match CST training)")
    p.add_argument("--cst_checkpoint", type=str, required=True, help="Path to cst_*_train_best.pt or _latest.pt")
    p.add_argument("--cst_config", type=str, default=None, help="YAML with cst: block (default: config/cst.yaml)")
    p.add_argument("--prompt", type=str, default=None, help="If set, run full capture+TAG; else use --hidden_pt only")
    p.add_argument("--hidden_pt", type=str, default=None, help="Existing save_hidden_records .pt (skip generation)")
    p.add_argument(
        "--out_dir",
        type=str,
        default="outputs/tag_run",
        help="Output directory (tag_trace.pt, tag_graph.json, tag_view.html for full run)",
    )
    p.add_argument("--mode_key", type=str, default="pd", help="Short scheduler: pd, bd, top_k_m, fhs, …")
    p.add_argument("--gen_length", type=int, default=256)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument(
        "--no_eos_harness",
        action="store_true",
        help="Reference-style: full generation steps and full-tail decode (no EOS early stop / truncate)",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--prune_p", type=float, default=0.1, help="With --prune_mode bfs: keep this fraction of in-edges per node")
    p.add_argument(
        "--prune_mode",
        type=str,
        default="influence",
        choices=("influence", "bfs", "none"),
        help="influence: indirect-influence + mass (default); bfs: per-destination top-p of in-edges; none: no prune",
    )
    p.add_argument(
        "--influence_node_mass",
        type=float,
        default=0.8,
        help="With influence: cumulative mass of feat nodes to keep (0-1)",
    )
    p.add_argument(
        "--influence_edge_mass",
        type=float,
        default=0.98,
        help="With influence: cumulative mass of edge scores on pruned subgraph (0-1)",
    )
    p.add_argument("--with_residual", action="store_true", help="Add error-typed nodes (residual L2 per step)")
    p.add_argument("--no_html", action="store_true")
    p.add_argument("--no_logits", action="store_true", help="Do not load model for last-step logits (no output sinks)")
    args = p.parse_args()
    mpath = args.model_path
    if args.hidden_pt and Path(args.hidden_pt).is_file():
        g = run_tag_from_chunk(
            args.hidden_pt,
            args.cst_checkpoint,
            cst_config_path=args.cst_config,
            model_path=None if args.no_logits else mpath,
            device=args.device,
            prune_p=args.prune_p,
            prune_mode=args.prune_mode,
            influence_node_mass=args.influence_node_mass,
            influence_edge_mass=args.influence_edge_mass,
            with_residual=args.with_residual,
        )
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "tag_graph.json").open("w", encoding="utf-8") as f:
            json.dump(g["graph"], f, indent=2, ensure_ascii=False)
        if not args.no_html:
            from .viz import write_tag_html
            write_tag_html(g["graph"], out / "tag_view.html")
        print(
            json.dumps(
                {
                    **g["meta"],
                    "h_shape": g["h_shape"],
                    "prune": g.get("prune", {}),
                    "wrote": str(out / "tag_graph.json"),
                },
                indent=0,
            )
        )
        return 0
    if not args.prompt or not mpath:
        p.error("Either provide --hidden_pt to an existing .pt, or both --prompt and --model_path for full run")
    r = run_tag_full(
        mpath,
        args.cst_checkpoint,
        str(args.prompt),
        args.out_dir,
        cst_config_path=args.cst_config,
        mode_key=args.mode_key,
        gen_length=args.gen_length,
        steps=args.steps,
        device=args.device,
        prune_p=args.prune_p,
        prune_mode=args.prune_mode,
        influence_node_mass=args.influence_node_mass,
        influence_edge_mass=args.influence_edge_mass,
        with_residual=args.with_residual,
        write_html=not args.no_html,
        use_eos_harness=not args.no_eos_harness,
    )
    print(json.dumps(r.get("paths", {}), indent=0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
