"""Predictor–controller swap and adaptive generation."""

import os
from pathlib import Path

import torch

from load_model import infer_b44_eos_unmask_safety
from schedulerlens.causal.generate import generate_pc
from schedulerlens.causal.models import load_role

if os.environ.get("TRACING_ENABLED", "false").lower() == "true":
    from traces.trace_wrapper import trace_scheduler
else:
    def trace_scheduler(*args, **kwargs):
        return lambda fn: fn


_PC_CACHE: dict = {}
_ADAPTIVE_OTHER: dict = {}


def _adaptive_roles(harness, model_path):
    """Harness model plus the other of LLaDA / LLaDA-1.5.

    The harness is forwarded first so a trace step records that forward.
    ``split`` places the other checkpoint on the other GPU.
    """
    directory = Path(model_path).resolve()
    name = directory.name.lower().replace("-", "_")
    if name not in ("llada", "llada1_5"):
        raise ValueError(
            f"adaptive selection needs the harness to be llada or llada1_5, got {directory.name}"
        )
    dev = str(next(harness.parameters()).device)
    other = "llada1_5" if name == "llada" else "llada"
    if os.environ.get("PC_DEVICE_MAP", "share") == "split" and torch.cuda.device_count() >= 2:
        other_dev = "cuda:0" if dev.endswith(":1") else "cuda:1"
    else:
        other_dev = dev
    key = (other, other_dev)
    if key not in _ADAPTIVE_OTHER:
        other_model, other_dir, _mask, _eos, _eot = load_role(
            other, other_dev, compile_model=False
        )
        _ADAPTIVE_OTHER[key] = (other_model, infer_b44_eos_unmask_safety(other_dir))
    other_model, other_drop = _ADAPTIVE_OTHER[key]
    return {
        name: (harness, infer_b44_eos_unmask_safety(directory)),
        other: (other_model, other_drop),
    }


def _pc_controller(predictor):
    name = os.environ.get("PC_CONTROLLER", "").strip()
    if not name:
        return predictor, None
    pred_dev = str(next(predictor.parameters()).device)
    if os.environ.get("PC_DEVICE_MAP", "share") == "split" and torch.cuda.device_count() >= 2:
        ctrl_dev = "cuda:1" if pred_dev in ("cuda:0", "cuda") else pred_dev
    else:
        ctrl_dev = pred_dev
    key = (name, ctrl_dev)
    if key not in _PC_CACHE:
        model, directory, _mask, _eos, _eot = load_role(name, ctrl_dev, compile_model=False)
        _PC_CACHE[key] = (model, infer_b44_eos_unmask_safety(directory))
    return _PC_CACHE[key]


@trace_scheduler()
def generate_proxy(
    model,
    input_ids,
    attention_mask,
    *,
    steps,
    gen_length,
    block_length,
    mode,
    top_k,
    mask_id,
    eos_id,
    eot_id,
    logits_eos_inf,
    confidence_eos_eot_inf,
    use_eos_harness,
    trace_name=None,
    upload_to_wandb=True,
    intervention_mode=None,
    intervention_steps=None,
    intervention_k=1,
    model_path=None,
):
    """Swap uses one controller. Adaptive loads LLaDA and LLaDA-1.5."""
    if mode == "pc_adaptive":
        if model_path is None:
            raise ValueError("adaptive selection needs model_path of the harness checkpoint")
        controller, controller_drop_eos = model, False
        roles = _adaptive_roles(model, model_path)
    else:
        controller, controller_drop_eos = _pc_controller(model)
        roles = None
        if controller_drop_eos is None:
            controller_drop_eos = bool(logits_eos_inf or confidence_eos_eot_inf)
    pc_mode = "adaptive" if mode == "pc_adaptive" else "swap"
    return generate_pc(
        model,
        controller,
        input_ids,
        attention_mask,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        phi=os.environ.get("PC_PHI", "eb"),
        mode=pc_mode,
        top_k=int(os.environ.get("PC_TOP_K", top_k)),
        gamma=float(os.environ.get("PC_GAMMA", "0.1")),
        mask_id=mask_id,
        eos_id=eos_id,
        eot_id=eot_id,
        logits_eos_inf=logits_eos_inf,
        confidence_eos_eot_inf=confidence_eos_eot_inf,
        controller_drop_eos=controller_drop_eos,
        use_eos_harness=use_eos_harness,
        roles=roles,
    )
