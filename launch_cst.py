"""Entry for CST train (collect + fit) or eval (offline hiddens). Config: `config/cst.yaml`."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

_CONFIG_DIR = Path(__file__).resolve().parent / "config"

# Defaults when ``mode=eval`` and user omits an ``eval:`` block in config (overrides: ``eval.hidden_dir=...`` etc.)
_OFFLINE_EVAL_DEFAULTS = {
    "hidden_dir": "outputs/eval_hidden",
    # Prefer explicit path. If missing, run_eval falls back: cst_best.pt, then newest cst_*_train_best.pt
    "checkpoint": "outputs/cst_ckpt/cst_best.pt",
    "model_path": None,
    "ntf_conf_low": 0.2,
    "ntf_conf_high": 0.8,
    "device": "cuda",
    "w": None,
    "wandb_name": "cst-eval",
}


def _merge_offline_eval_cfg(cfg: DictConfig):
    """Merge `eval` overrides with `launch_cst` defaults when `eval:` is absent in yaml."""
    base = OmegaConf.create(_OFFLINE_EVAL_DEFAULTS)
    o = cfg.get("eval")
    if o is not None:
        return OmegaConf.merge(base, o)
    return base


def _run_offline_eval(cfg: DictConfig) -> int:
    """Run `cst.eval.run_eval_cst.run_eval_on_dir`; optional W&B logging."""
    try:
        import wandb
    except ImportError:
        wandb = None

    from cst.eval.run_eval_cst import run_eval_on_dir
    from cst.naming import resolve_train_best_in_dir

    ev = _merge_offline_eval_cfg(cfg)
    mp = ev.get("model_path")
    ck = str(ev.get("checkpoint", _OFFLINE_EVAL_DEFAULTS["checkpoint"]))
    ckp = Path(ck)
    if not ckp.is_file() and ckp.parent.is_dir():
        alt = resolve_train_best_in_dir(ckp.parent)
        if alt is not None and alt.is_file():
            ck = str(alt)
    out = run_eval_on_dir(
        hidden_dir=str(ev.get("hidden_dir", _OFFLINE_EVAL_DEFAULTS["hidden_dir"])),
        ckpt_path=ck,
        cfg_bank=OmegaConf.to_container(cfg.cst, resolve=True),
        device=str(ev.get("device", "cuda")),
        w=int(ev.get("w")) if ev.get("w") is not None else None,
        model_path=str(mp) if mp not in (None, "", "null") else None,
        ntf_conf_low=float(ev.get("ntf_conf_low", 0.2)),
        ntf_conf_high=float(ev.get("ntf_conf_high", 0.8)),
    )
    print(OmegaConf.to_yaml(out))
    use_wb = ev.get("wandb")
    if use_wb is None:
        use_wb = cfg.get("wandb", True)
    if bool(use_wb) and wandb is not None:
        wb = wandb.init(
            project=str(cfg.get("wandb_project", "strace-cst")),
            name=str(ev.get("wandb_name", "cst-eval")),
            reinit=True,
        )
        if "summary" in out:
            wb.log({f"eval/{k}": v for k, v in out["summary"].items()})
        wb.finish()
    return 0


@hydra.main(version_base="1.2", config_path=str(_CONFIG_DIR), config_name="cst")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    mode = str(cfg.get("mode", "train")).lower()
    if mode == "eval":
        sys.exit(int(_run_offline_eval(cfg)))
    if mode == "train":
        from cst.train.run import run_train

        sys.exit(int(run_train(cfg)))
    raise ValueError(f"unknown mode={mode!r} (use train or eval)")


if __name__ == "__main__":
    main()
