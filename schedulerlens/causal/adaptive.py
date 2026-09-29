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
    """Pick the lowest Ĝ = M̂ + Ĵ across both models and every scheduler.

    Each model builds one set per scheduler (12). Each set is scored with both
    models, giving 24 pairs. Ĝ is that model's M̂ plus that model's Ĵ. The
    lowest Ĝ wins. A set whose positions are already unmasked is skipped.
    M̂ and Ĵ use only the positions that are still masked.

    Returns ``(set_model, score_model, rule, chosen, j_hat, g_hat, m_llada, m_llada1_5, j_llada, j_llada1_5)``.
    On the winning set, the lower M̂ is the predictor and the lower Ĵ is the controller.
    """
    mask = np.asarray(mask_gen, dtype=bool)
    best_g = float("inf")
    best_j = 0.0
    best_set_model = ""
    best_j_model = ""
    best_rule = ""
    best_sel = np.zeros(mask.shape[0], dtype=bool)
    best_m = {"llada": 0.0, "llada1_5": 0.0}
    best_js = {"llada": 0.0, "llada1_5": 0.0}
    for set_model in CROSS_MODELS:
        for phi in candidates:
            if phi not in SUPPORTED_PHI:
                continue
            try:
                sel = select_gen(
                    phi,
                    select[set_model],
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
            measured = {}
            for measure_model in CROSS_MODELS:
                conf = raw[measure_model].conf[sel]
                ent = raw[measure_model].entropy[sel]
                measured[measure_model] = (
                    m_hat(conf, ent, M_GAMMA),
                    j_hat(ent),
                )
            for score_model in CROSS_MODELS:
                m_value, j_value = measured[score_model]
                g_value = m_value + j_value
                if g_value < best_g:
                    best_g = g_value
                    best_j = j_value
                    best_set_model = set_model
                    best_j_model = score_model
                    best_rule = phi
                    best_sel = sel
                    best_m = {name: measured[name][0] for name in CROSS_MODELS}
                    best_js = {name: measured[name][1] for name in CROSS_MODELS}
    if not best_sel.any():
        best_sel = ensure_progress(best_sel, raw["llada"].conf, mask)
        best_set_model = "llada"
        best_j_model = "llada"
        best_rule = "fallback"
        if best_sel.any():
            for measure_model in CROSS_MODELS:
                conf = raw[measure_model].conf[best_sel]
                ent = raw[measure_model].entropy[best_sel]
                best_m[measure_model] = m_hat(conf, ent, M_GAMMA)
                best_js[measure_model] = j_hat(ent)
            best_j = best_js["llada"]
            best_g = best_m["llada"] + best_j
        else:
            best_j = 0.0
            best_g = 0.0
    return (
        best_set_model,
        best_j_model,
        best_rule,
        best_sel,
        float(best_j),
        float(best_g),
        float(best_m["llada"]),
        float(best_m["llada1_5"]),
        float(best_js["llada"]),
        float(best_js["llada1_5"]),
    )
