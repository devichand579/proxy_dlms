"""Load :class:`CSTBank` and weights the same way as eval/cst training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import torch
from omegaconf import OmegaConf

from cst.model.cst_bank import CSTBank
from cst.train.trainer import load_cst_bank_weights

ConfPath = Union[str, Path, None]


def load_cst_config_dict(cst_config_path: ConfPath) -> dict:
    root = Path(__file__).resolve().parent.parent
    if cst_config_path is None or str(cst_config_path).strip() == "" or str(cst_config_path).strip().lower() in (
        "null",
        "none",
    ):
        ypath = root / "config" / "cst.yaml"
    else:
        ypath = Path(str(cst_config_path).strip())
        if not ypath.is_file():
            ypath = root / ypath
    if not ypath.is_file():
        raise FileNotFoundError(f"CST config yaml not found: {ypath}")
    o = OmegaConf.load(str(ypath))
    c = o.get("cst")
    if c is None:
        raise KeyError(f"{ypath} must contain a 'cst:' block")
    return OmegaConf.to_container(c, resolve=True)  # type: ignore[return-value]


def build_cst_bank(
    ccfg: Dict[str, Any],
    *,
    device: torch.device | str = "cuda",
) -> CSTBank:
    bank = CSTBank(
        max_steps=int(ccfg["max_steps"]),
        d=int(ccfg["d_model"]),
        k=int(ccfg["k"]),
        w=int(ccfg["w"]),
        max_on_gpu=int(ccfg.get("max_on_gpu", ccfg["max_steps"])),
        initial_threshold=float(ccfg.get("jumprelu_initial_threshold", 0.03)),
        bandwidth=float(ccfg.get("jumprelu_bandwidth", 1.0)),
    )
    dev = device if isinstance(device, torch.device) else torch.device(device)
    bank.set_device(dev)
    return bank


def load_cst(
    checkpoint_path: str | Path,
    cst_config_path: ConfPath = None,
    *,
    device: str = "cuda",
) -> tuple[CSTBank, dict]:
    ccfg = load_cst_config_dict(cst_config_path)
    bank = build_cst_bank(ccfg, device=device)
    load_cst_bank_weights(str(checkpoint_path), bank)
    bank.to_device_all(torch.device(device))
    return bank, ccfg
