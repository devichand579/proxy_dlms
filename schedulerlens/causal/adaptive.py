"""Choose the unmask set with the lowest Ĝ = M̂ + Ĵ."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from schedulerlens.causal.scores import ScoreBundle
from schedulerlens.causal.select import SelectError, SUPPORTED_PHI, select_gen
from schedulerlens.proxies import M_GAMMA, j_hat, m_hat

DEFAULT_CANDIDATES = ("pd", "bd", "top_k_c", "top_k_m", "eb", "dus")
CROSS_MODELS = ("llada", "llada1_5")


def ensure_progress(chosen: np.ndarray, conf: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """Keep at least one reveal when any eligible mask remains."""
    out = np.asarray(chosen, dtype=bool).copy()
    if out.any() or not np.asarray(eligible, dtype=bool).any():
        return out
    idx = np.flatnonzero(eligible)
    best = idx[int(np.argmax(conf[idx]))]
    out[best] = True
    return out


def pick_cross_j(
    raw: dict[str, ScoreBundle],
    select: dict[str, ScoreBundle],
    mask_gen: np.ndarray,
    *,
    step: int,
    steps: int,
    candidates: Sequence[str] = DEFAULT_CANDIDATES,
    block_length: Optional[int] = None,
    top_k: int = 2,
    gamma: float = 0.1,
) -> tuple[str, str, str, np.ndarray, float, float, float, float, float, float]:
    """Pick the lowest Ĝ = M̂(predictor) + Ĵ(controller).

    Each model, as controller, builds one set per scheduler (12). Ĵ on a set
    is that controller's entropy. Each of LLaDA and LLaDA-1.5 then supplies an
    M̂ on the set, giving 24 values of Ĝ. The lowest wins. The predictor is
    the model whose M̂ entered that Ĝ, and it writes the token. The controller
    is the model that built the set. A set whose positions are already
    unmasked is skipped. M̂ and Ĵ use only the positions that are still masked.

    Returns ``(controller, predictor, rule, chosen, j_hat, g_hat, m_llada, m_llada1_5, j_llada, j_llada1_5)``.
    """
    mask = np.asarray(mask_gen, dtype=bool)
    best_g = float("inf")
    best_j = 0.0
    best_controller = ""
    best_predictor = "llada"
    best_rule = ""
    best_sel = np.zeros(mask.shape[0], dtype=bool)
    best_m = {"llada": 0.0, "llada1_5": 0.0}
    best_js = {"llada": 0.0, "llada1_5": 0.0}
    for controller in CROSS_MODELS:
        for phi in candidates:
            if phi not in SUPPORTED_PHI:
                continue
            try:
                sel = select_gen(
                    phi,
                    select[controller],
                    mask,
                    step=step,
                    steps=steps,
                    block_length=block_length,
                    top_k=top_k,
                    gamma=gamma,
                )
            except SelectError:
                continue
            if not sel.any():
                continue
            measured_m = {}
            measured_j = {}
            for model in CROSS_MODELS:
                conf = raw[model].conf[sel]
                ent = raw[model].entropy[sel]
                measured_m[model] = m_hat(conf, ent, M_GAMMA)
                measured_j[model] = j_hat(ent)
            # Ĵ stays with the model that built the set. Each predictor adds its own M̂.
            j_ctrl = measured_j[controller]
            for predictor in CROSS_MODELS:
                g_value = measured_m[predictor] + j_ctrl
                if g_value < best_g:
                    best_g = g_value
                    best_j = j_ctrl
                    best_controller = controller
                    best_predictor = predictor
                    best_rule = phi
                    best_sel = sel
                    best_m = dict(measured_m)
                    best_js = dict(measured_j)
    if not best_sel.any():
        best_sel = ensure_progress(best_sel, raw["llada"].conf, mask)
        best_controller = "llada"
        best_rule = "fallback"
        if best_sel.any():
            for model in CROSS_MODELS:
                conf = raw[model].conf[best_sel]
                ent = raw[model].entropy[best_sel]
                best_m[model] = m_hat(conf, ent, M_GAMMA)
                best_js[model] = j_hat(ent)
            best_j = best_js["llada"]
            best_predictor = min(CROSS_MODELS, key=lambda name: (best_m[name], name))
            best_g = best_m[best_predictor] + best_j
        else:
            best_j = 0.0
            best_g = 0.0
            best_predictor = "llada"
    return (
        best_controller,
        best_predictor,
        best_rule,
        best_sel,
        float(best_j),
        float(best_g),
        float(best_m["llada"]),
        float(best_m["llada1_5"]),
        float(best_js["llada"]),
        float(best_js["llada1_5"]),
    )
