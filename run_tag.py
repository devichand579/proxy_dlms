#!/usr/bin/env python3
"""CLI for TAG: full capture or an existing hidden .pt (config: config/tag.yaml)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from repo_env_defaults import apply_repo_env_defaults

import hydra
from omegaconf import DictConfig

_CONFIG_DIR = _REPO_ROOT / "config"
logger = logging.getLogger("run_tag")


def _cfg_bool(val: Any, default: bool = False) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ("", "null", "none"):
        return default
    return s in ("true", "1", "yes", "on")


def _cfg_optional_str(val: Any) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip()
    if s.lower() in ("", "null", "none"):
        return None
    return s


def _cfg_optional_int(val: Any) -> Optional[int]:
    s = _cfg_optional_str(val)
    if s is None:
        return None
    return int(s)


def run_from_cfg(cfg: DictConfig) -> int:
    from tag.pipeline import run_tag_from_chunk, run_tag_full
    from tag.viz import write_graph_json, write_tag_html

    ckpt = _cfg_optional_str(cfg.get("cst_checkpoint"))
    if not ckpt:
        raise ValueError("Set cst_checkpoint=/path/to/cst_..._train_best.pt")

    out_dir = Path(str(cfg.get("out_dir", "outputs/tag")))
    out_dir.mkdir(parents=True, exist_ok=True)
    cst_config_path = _cfg_optional_str(cfg.get("cst_config"))
    model_path = str(cfg.get("model_path", ""))
    device = str(cfg.get("device", "cuda"))
    output_exclude_eos = _cfg_bool(cfg.get("output_exclude_eos"), False)
    eos_id = _cfg_optional_int(cfg.get("eos_id"))
    no_logits = _cfg_bool(cfg.get("no_logits"), False)
    write_html = _cfg_bool(cfg.get("write_html"), True)

    influence_node_mass = float(cfg.get("influence_node_mass", 0.8))
    influence_edge_mass = float(cfg.get("influence_edge_mass", 0.98))
    mix_uniform = float(cfg.get("mix_uniform", 1e-4))
    with_residual = _cfg_bool(cfg.get("with_residual"), False)
    top_output_mass = float(cfg.get("top_output_mass", 0.95))
    max_output_tokens = int(cfg.get("max_output_tokens", 10))

    hidden_pt = _cfg_optional_str(cfg.get("hidden_pt"))
    if hidden_pt:
        logits_model = None if no_logits else (model_path or None)
        g = run_tag_from_chunk(
            hidden_pt,
            ckpt,
            cst_config_path=cst_config_path,
            model_path=logits_model,
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
        write_graph_json(g["graph"], out_dir / "tag_graph.json")
        if write_html:
            write_tag_html(g["graph"], out_dir / "tag_view.html")
        print(
            json.dumps(
                {
                    **g.get("meta", {}),
                    "h_shape": g.get("h_shape"),
                    "prune": g.get("prune", {}),
                    "graph_nodes": len(g["graph"].get("nodes", [])),
                    "wrote": str(out_dir),
                },
                indent=0,
            )
        )
        return 0

    prompt = _cfg_optional_str(cfg.get("prompt"))
    if not prompt:
        raise ValueError(
            "Set prompt=... for full capture+TAG, or hidden_pt=... for an existing .pt chunk"
        )
    if not model_path or not Path(model_path).exists():
        raise ValueError("Full TAG run requires model_path=... (local LM directory)")

    use_eos_harness = _cfg_bool(cfg.get("use_eos_harness"), True)
    r = run_tag_full(
        model_path,
        ckpt,
        prompt,
        str(out_dir),
        cst_config_path=cst_config_path,
        mode_key=str(cfg.get("mode_key", "pd")),
        gen_length=int(cfg.get("gen_length", 256)),
        steps=int(cfg.get("steps", 256)),
        device=device,
        influence_node_mass=influence_node_mass,
        influence_edge_mass=influence_edge_mass,
        mix_uniform=mix_uniform,
        with_residual=with_residual,
        top_output_mass=top_output_mass,
        max_output_tokens=max_output_tokens,
        write_html=write_html,
        use_eos_harness=use_eos_harness,
        output_exclude_eos=output_exclude_eos,
        eos_id=eos_id,
    )
    print(json.dumps(r.get("paths", {}), indent=0))
    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build TAG: Temporal Attribution Graph for CST on one prompt or .pt chunk"
    )
    p.add_argument("--model_path", type=str, help="Local LM (must match CST training)")
    p.add_argument("--cst_checkpoint", type=str, required=True)
    p.add_argument("--cst_config", type=str, default=None)
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--hidden_pt", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="outputs/tag")
    p.add_argument("--mode_key", type=str, default="pd")
    p.add_argument("--gen_length", type=int, default=256)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument(
        "--no_eos_harness",
        action="store_true",
        help="Reference-style: full generation steps and full-tail decode",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--influence_node_mass", type=float, default=0.8)
    p.add_argument("--influence_edge_mass", type=float, default=0.98)
    p.add_argument("--mix_uniform", type=float, default=1e-4)
    p.add_argument("--with_residual", action="store_true")
    p.add_argument("--no_html", action="store_true")
    p.add_argument("--no_logits", action="store_true")
    p.add_argument("--output_exclude_eos", action="store_true")
    p.add_argument("--eos_id", type=int, default=None)
    p.add_argument("--top_output_mass", type=float, default=0.95)
    p.add_argument("--max_output_tokens", type=int, default=10)
    return p


def run_from_argv(argv: list[str]) -> int:
    """CLI entry for python -m tag."""
    from omegaconf import OmegaConf

    args = build_argparser().parse_args(argv)
    cfg = OmegaConf.create(
        {
            "model_path": args.model_path or "./models/llada",
            "cst_checkpoint": args.cst_checkpoint,
            "cst_config": args.cst_config,
            "prompt": args.prompt,
            "hidden_pt": args.hidden_pt,
            "out_dir": args.out_dir,
            "mode_key": args.mode_key,
            "gen_length": args.gen_length,
            "steps": args.steps,
            "use_eos_harness": not args.no_eos_harness,
            "device": args.device,
            "influence_node_mass": args.influence_node_mass,
            "influence_edge_mass": args.influence_edge_mass,
            "mix_uniform": args.mix_uniform,
            "with_residual": args.with_residual,
            "write_html": not args.no_html,
            "no_logits": args.no_logits,
            "output_exclude_eos": args.output_exclude_eos,
            "eos_id": args.eos_id,
            "top_output_mass": args.top_output_mass,
            "max_output_tokens": args.max_output_tokens,
        }
    )
    return run_from_cfg(cfg)


@hydra.main(version_base="1.3", config_path=str(_CONFIG_DIR), config_name="tag")
def main(cfg: DictConfig) -> None:
    apply_repo_env_defaults()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(int(run_from_cfg(cfg)))


if __name__ == "__main__":
    main()
