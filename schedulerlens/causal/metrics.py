"""Fixed-history deltas between the predictor set and the controller set."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from schedulerlens.proxies import M_GAMMA, j_hat, m_hat


@dataclass
class StepDelta:
    delta_pred_m: float
    delta_ctrl_m: float
    delta_ctrl_j: float
    delta_own_j: float
    jaccard: float
    step: int = -1


def _on(values: np.ndarray, sel: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)[np.asarray(sel, dtype=bool)]


def jaccard(s0: np.ndarray, s1: np.ndarray) -> float:
    """|S0 ∩ S1| / |S0 ∪ S1|. Empty sets score 1."""
    a = np.asarray(s0, dtype=bool)
    b = np.asarray(s1, dtype=bool)
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return float(inter / union) if union else 1.0


def step_deltas(
    *,
    conf_pred: np.ndarray,
    ent_pred: np.ndarray,
    conf_ctrl: np.ndarray,
    ent_ctrl: np.ndarray,
    s0: np.ndarray,
    s1: np.ndarray,
) -> StepDelta:
    """Prediction vs control gaps at one frozen state.

    S0 is the predictor's set and S1 is the controller's set. The set-change
    gaps, delta_ctrl_m and delta_ctrl_j, use the predictor's raw scores.
    delta_own_j scores S1 with the controller's raw entropy. The controller
    supplies the scores that choose S1.
    """
    m_pred_s0 = m_hat(_on(conf_pred, s0), _on(ent_pred, s0), M_GAMMA)
    m_pred_s1 = m_hat(_on(conf_pred, s1), _on(ent_pred, s1), M_GAMMA)
    m_ctrl_s0 = m_hat(_on(conf_ctrl, s0), _on(ent_ctrl, s0), M_GAMMA)
    j0 = j_hat(_on(ent_pred, s0))
    j1 = j_hat(_on(ent_pred, s1))
    j1_ctrl = j_hat(_on(ent_ctrl, s1))
    return StepDelta(
        delta_pred_m=m_ctrl_s0 - m_pred_s0,
        delta_ctrl_m=m_pred_s1 - m_pred_s0,
        delta_ctrl_j=j1 - j0,
        delta_own_j=j1_ctrl - j0,
        jaccard=jaccard(s0, s1),
    )
