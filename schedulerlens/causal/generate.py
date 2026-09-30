"""Live predictor–controller generation (swap and adaptive)."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import torch

from schedulerlens.causal.adaptive import DEFAULT_CANDIDATES, pick_cross_j
from schedulerlens.causal.scores import ScoreBundle, bundle_for_select, bundle_from_logits
from schedulerlens.causal.select import select_gen
from schedulerlens.proxies import M_GAMMA, j_hat, m_hat
from schedulers.utils import (
    check_all_generated_unmasked,
    check_eos_stop_condition,
    mask_eos_eot_logits,
)
from traces.trace_wrapper import note_trace_commit, rewrite_open_step_row

# Filled by the most recent ``generate_pc`` call (gen-region selections per step).
last_step_log: list[dict[str, Any]] = []


def _device_of(model) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def generate_pc(
    predictor,
    controller,
    prompt: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    *,
    steps: int = 256,
    gen_length: int = 256,
    block_length: Optional[int] = None,
    phi: str = "eb",
    mode: str = "swap",
    candidates: Sequence[str] = DEFAULT_CANDIDATES,
    top_k: int = 2,
    gamma: float = 0.1,
    mask_id: int = 126336,
    eos_id: int = 126081,
    eot_id: Optional[int] = None,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    controller_drop_eos: bool = False,
    use_eos_harness: bool = True,
    roles: Optional[dict] = None,
):
    """Generate with predictor tokens and controller scores.

    M̂ and Ĵ use the raw forward. When a role is LLaDA-1.5, EOS and EoT are
    removed only from the scores that choose its sets and from the token it
    writes. ``mode='swap'`` applies ``phi`` to controller scores.
    ``mode='adaptive'`` builds six sets from LLaDA and six from LLaDA-1.5.
    Ĵ stays with the model that built the set. Each model supplies an M̂ on
    that set. The lowest Ĝ = M̂ + Ĵ wins. That M̂'s model writes the token.
    Returns ``(x, steps_taken, n_blocks)`` like the other schedulers.
    """
    global last_step_log
    last_step_log = []
    if mode not in ("swap", "adaptive"):
        raise ValueError(f"mode must be swap or adaptive, got {mode!r}")

    if mode == "adaptive" and (roles is None or set(roles) != {"llada", "llada1_5"}):
        raise ValueError("adaptive mode requires both llada and llada1_5 roles")
    dev_p = _device_of(predictor)
    dev_c = _device_of(controller)
    prompt = prompt.to(dev_p)
    batch, prompt_len = prompt.shape
    x = torch.full(
        (batch, prompt_len + gen_length),
        int(mask_id),
        dtype=torch.long,
        device=dev_p,
    )
    x[:, :prompt_len] = prompt
    if attention_mask is not None:
        attention_mask = attention_mask.to(dev_p)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (batch, gen_length),
                    dtype=attention_mask.dtype,
                    device=dev_p,
                ),
            ],
            dim=-1,
        )
    steps_taken = 0
    blocks = 1 if not block_length or block_length >= gen_length else gen_length // int(block_length)

    for step in range(int(steps)):
        mask_index = x == int(mask_id)
        if not mask_index[:, prompt_len:].any():
            break
        steps_taken += 1
        raw_roles = select_roles = commits = None
        if mode == "adaptive":
            raw_roles, select_roles, commits = _score_roles(
                roles, x, attention_mask, eos_id=eos_id, eot_id=eot_id
            )
            x0 = torch.empty_like(x)
        else:
            logits_p = predictor(x, attention_mask=attention_mask).logits
            x_c = x.to(dev_c)
            attn_c = None if attention_mask is None else attention_mask.to(dev_c)
            if predictor is controller and dev_p == dev_c:
                logits_c = logits_p
            else:
                logits_c = controller(x_c, attention_mask=attn_c).logits
            pred_bundles = bundle_from_logits(logits_p)
            ctrl_raw = bundle_from_logits(logits_c)
            ctrl_select = (
                bundle_for_select(logits_c, drop_eos=True, eos_id=eos_id, eot_id=eot_id)
                if controller_drop_eos
                else ctrl_raw
            )
            # Entropies in the trace are the controller's raw distribution.
            for b in range(batch):
                rewrite_open_step_row(
                    b, entropies=ctrl_raw[b].gen(prompt_len).entropy
                )
            if logits_eos_inf or confidence_eos_eot_inf:
                commit_logits = logits_p.clone()
                mask_eos_eot_logits(
                    commit_logits, logits_eos_inf, confidence_eos_eot_inf, eos_id, eot_id
                )
                x0 = torch.argmax(commit_logits, dim=-1)
            else:
                x0 = torch.argmax(logits_p, dim=-1)
        transfer = torch.zeros_like(x, dtype=torch.bool)
        for b in range(batch):
            gen_mask = mask_index[b, prompt_len:].detach().cpu().numpy()
            if mode == "adaptive":
                (
                    ctrl_model, pred_model, rule, chosen, j_hat, g_hat,
                    m_llada, m_llada1_5, j_llada, j_llada1_5,
                ) = pick_cross_j(
                    {name: raw_roles[name][b].gen(prompt_len) for name in raw_roles},
                    {name: select_roles[name][b].gen(prompt_len) for name in select_roles},
                    gen_mask,
                    step=step,
                    steps=steps,
                    candidates=candidates,
                    block_length=block_length,
                    top_k=top_k,
                    gamma=gamma,
                )
                x0[b] = commits[pred_model][b]
                pred_scores = raw_roles[pred_model][b].gen(prompt_len)
                ctrl_scores = raw_roles[ctrl_model][b].gen(prompt_len)
                rewrite_open_step_row(
                    b,
                    predictions=pred_scores.pred,
                    confidences=pred_scores.conf,
                    entropies=ctrl_scores.entropy,
                )
                if pred_model == "llada":
                    m_hat = m_llada
                    m_hat_other = m_llada1_5
                    j_hat_other = j_llada
                else:
                    m_hat = m_llada1_5
                    m_hat_other = m_llada
                    j_hat_other = j_llada1_5
                step_log = {
                    "batch": b,
                    "step": step,
                    "rule": rule,
                    "set_model": ctrl_model,
                    "pred_model": pred_model,
                    "ctrl_model": ctrl_model,
                    "reveal": int(chosen.sum()),
                    "M_hat_llada": m_llada,
                    "M_hat_llada1_5": m_llada1_5,
                    "J_hat": j_hat,
                    "J_hat_other": j_hat_other,
                    "G_hat": g_hat,
                    "m_hat": m_hat,
                    "m_hat_other": m_hat_other,
                    "j_hat": j_hat,
                    "j_hat_other": j_hat_other,
                    "g_hat": g_hat,
                }
            else:
                chosen, rule, m_hat, j_hat = _choose_row(
                    phi=phi,
                    select_scores=ctrl_select[b].gen(prompt_len),
                    measure_scores=ctrl_raw[b].gen(prompt_len),
                    gen_mask=gen_mask,
                    step=step,
                    steps=steps,
                    block_length=block_length,
                    top_k=top_k,
                    gamma=gamma,
                )
                step_log = {
                    "batch": b,
                    "step": step,
                    "rule": rule,
                    "reveal": int(chosen.sum()),
                    "M_hat": m_hat,
                    "J_hat": j_hat,
                }
            full = np.zeros(x.shape[1], dtype=bool)
            full[prompt_len:] = chosen
            transfer[b] = torch.as_tensor(full, device=dev_p)
            step_log["active"] = bool(gen_mask.any())
            last_step_log.append(step_log)
        if mode == "adaptive":
            note_trace_commit(x0)
        # In-place write so the tracing hook records selected positions and tokens.
        x[transfer] = x0[transfer]
        if use_eos_harness and check_eos_stop_condition(x, eos_id, int(mask_id), prompt_len):
            break
        if check_all_generated_unmasked(x, int(mask_id), prompt_len):
            break
    return x, steps_taken, blocks


def _score_roles(roles, x, attention_mask, *, eos_id: int, eot_id: Optional[int]):
    """Forward both models. LLaDA-1.5 drops EOS only for its sets and its commit."""
    raw = {}
    select = {}
    commits = {}
    for name, (model, drop_eos) in roles.items():
        device = _device_of(model)
        x_role = x if x.device == device else x.to(device)
        attn = None if attention_mask is None else attention_mask.to(device)
        logits = model(x_role, attention_mask=attn).logits
        raw[name] = bundle_from_logits(logits)
        if drop_eos:
            select[name] = bundle_for_select(
                logits, drop_eos=True, eos_id=eos_id, eot_id=eot_id
            )
            commit_logits = logits.clone()
            mask_eos_eot_logits(commit_logits, True, True, eos_id, eot_id)
            commits[name] = torch.argmax(commit_logits, dim=-1).to(x.device)
        else:
            select[name] = raw[name]
            commits[name] = torch.argmax(logits, dim=-1).to(x.device)
    return raw, select, commits


def _choose_row(
    *,
    phi: str,
    select_scores: ScoreBundle,
    measure_scores: ScoreBundle,
    gen_mask: np.ndarray,
    step: int,
    steps: int,
    block_length: Optional[int],
    top_k: int,
    gamma: float,
) -> tuple[np.ndarray, str, float, float]:
    chosen = select_gen(
        phi,
        select_scores,
        gen_mask,
        step=step,
        steps=steps,
        block_length=block_length,
        top_k=top_k,
        gamma=gamma,
    )
    if chosen.any():
        step_m = m_hat(measure_scores.conf[chosen], measure_scores.entropy[chosen], M_GAMMA)
        step_j = j_hat(measure_scores.entropy[chosen])
    else:
        step_m = 0.0
        step_j = 0.0
    return chosen, phi, step_m, step_j
