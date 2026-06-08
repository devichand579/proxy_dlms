"""Per-step M̂_t and Ĵ_t proxies from a generation trace."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# Step-level series aggregated to sample/task caps and trends.
PROXY_ATTRS = (
    "M_hat_t",
    "mean_regret",
    "cascade_gain",
    "J_hat_t",
    "instability",
    "rank_volatility",
)
M_PROXY_ATTRS = ("M_hat_t", "mean_regret", "cascade_gain")
J_PROXY_ATTRS = ("J_hat_t", "instability", "rank_volatility")


@dataclass
class CapsPair:
    entropy: float = 0.0
    uniform: float = 0.0


@dataclass
class StepProxies:
    step: int
    M_hat_t: float          # ordering quality proxy
    J_hat_t: float          # parallelism error proxy
    weight: float           # entropy-drop weight w_t
    # sub-components (for diagnostics)
    mean_regret: float = 0.0
    cascade_gain: float = 0.0
    instability: float = 0.0
    rank_volatility: float = 0.0
    n_committed: int = 0
    n_selected: int = 0
    n_remaining: int = 0


@dataclass
class SampleProxies:
    """Per-sample caps (entropy + uniform) for M̂/Ĵ and sub-proxies."""
    M_hat: float
    J_hat: float
    M_hat_uniform: float
    J_hat_uniform: float
    caps: Dict[str, CapsPair] = field(default_factory=dict)
    step_proxies: List[StepProxies] = field(default_factory=list)


def _empty_caps() -> Dict[str, CapsPair]:
    return {name: CapsPair() for name in PROXY_ATTRS}


def _caps_pair(step_proxies: List[StepProxies], attr: str) -> CapsPair:
    if not step_proxies:
        return CapsPair()
    values = [float(getattr(sp, attr)) for sp in step_proxies]
    uniform = float(np.mean(values))
    total_weight = sum(sp.weight for sp in step_proxies)
    if total_weight <= 0:
        return CapsPair(entropy=uniform, uniform=uniform)
    entropy = float(
        sum(sp.weight * v for sp, v in zip(step_proxies, values)) / total_weight
    )
    return CapsPair(entropy=entropy, uniform=uniform)


def _aggregate_sample_caps(step_proxies: List[StepProxies]) -> Dict[str, CapsPair]:
    return {name: _caps_pair(step_proxies, name) for name in PROXY_ATTRS}


def _kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    """Kendall tau between two rank vectors."""
    n = len(a)
    if n < 2:
        return 0.0
    concordant = discordant = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            sign_a = np.sign(a[i] - a[j])
            sign_b = np.sign(b[i] - b[j])
            prod = sign_a * sign_b
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
    denom = n * (n - 1) / 2
    if denom == 0:
        return 0.0
    return (concordant - discordant) / denom


def _fast_kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    """Kendall tau via scipy, with a numpy fallback."""
    if len(a) < 2:
        return 0.0
    try:
        from scipy.stats import kendalltau
        tau, _ = kendalltau(a, b)
        return float(tau) if np.isfinite(tau) else 0.0
    except Exception:
        return _kendall_tau(a, b)


def compute_sample_proxies(trace_dict: dict) -> SampleProxies:
    """Compute M̂, Ĵ, and per-step proxies for one trace dict."""
    step_traces = trace_dict.get("step_traces", [])
    unmasking_order: Dict[int, int] = {
        int(k): int(v) for k, v in trace_dict.get("unmasking_order", {}).items()
    }
    T = len(step_traces)
    if T == 0:
        empty = _empty_caps()
        return SampleProxies(
            M_hat=0.0,
            J_hat=0.0,
            M_hat_uniform=0.0,
            J_hat_uniform=0.0,
            caps=empty,
        )

    # ---- Pre-load all step tensors (gen-region, shape [gen_length]) ---- #
    def _to_np(data, dtype=np.float32) -> Optional[np.ndarray]:
        if data is None:
            return None
        arr = np.array(data, dtype=dtype)
        # shape may be [[...]] (batch dim); squeeze to 1-D
        while arr.ndim > 1:
            arr = arr[0]
        return arr

    steps: List[dict] = []
    for st in step_traces:
        steps.append({
            "mask_index": _to_np(st.get("mask_index"), dtype=np.bool_),
            "predictions": _to_np(st.get("predictions"), dtype=np.int64),
            "confidences": _to_np(st.get("confidences"), dtype=np.float32),
            "entropies": _to_np(st.get("entropies"), dtype=np.float32),
            "selected_positions": _to_np(st.get("selected_positions"), dtype=np.bool_),
            "selected_tokens": _to_np(st.get("selected_tokens"), dtype=np.int64),
        })

    # Build committed_token[i] = token committed at position i
    # and commit_conf[i] = confidence at the step it was committed
    gen_length = int(trace_dict.get("gen_length") or (
        steps[0]["mask_index"].shape[0] if steps[0]["mask_index"] is not None else 0
    ))

    committed_token = np.full(gen_length, -1, dtype=np.int64)
    commit_conf = np.zeros(gen_length, dtype=np.float32)

    for t, st in enumerate(steps):
        sel = st["selected_positions"]
        sel_tok = st["selected_tokens"]
        conf = st["confidences"]
        if sel is None or sel_tok is None:
            continue
        positions = np.where(sel)[0]
        for pos in positions:
            if committed_token[pos] == -1:
                committed_token[pos] = sel_tok[pos] if sel_tok[pos] != 0 else (
                    steps[t]["predictions"][pos] if steps[t]["predictions"] is not None else -1
                )
                commit_conf[pos] = conf[pos] if conf is not None else 0.0

    # ---- Per-step proxy computation ---- #
    step_proxy_list: List[StepProxies] = []

    for t in range(T):
        st = steps[t]
        mask_t = st["mask_index"]          # bool [gen_length]: masked at START of step t
        sel_t = st["selected_positions"]   # bool [gen_length]: unmasked at step t
        conf_t = st["confidences"]
        pred_t = st["predictions"]
        ent_t = st["entropies"]

        if mask_t is None or sel_t is None:
            continue

        # S_t: positions selected at step t
        S_t = np.where(sel_t)[0]
        # C_t: committed before step t (not masked at start of t)
        C_t = np.where(~mask_t)[0]
        # S̄_t: still masked after step t
        S_bar_t = np.where(mask_t & ~sel_t)[0]

        n_selected = len(S_t)
        n_committed = len(C_t)
        n_remaining = len(S_bar_t)

        # ---- Entropy-drop weight w_t ---- #
        weight = 0.0
        if ent_t is not None and n_selected > 0:
            weight = float(np.sum(ent_t[S_t]))

        # ---- Cascade gain cg(t) ---- #
        cascade_gain_val = 0.0
        if t > 0 and n_remaining > 0:
            conf_prev = steps[t - 1]["confidences"]
            if conf_t is not None and conf_prev is not None:
                delta = conf_t[S_bar_t] - conf_prev[S_bar_t]
                cascade_gain_val = float(np.mean(delta))

        # ---- Mean positive regret over C_t ---- #
        mean_regret_val = 0.0
        if n_committed > 0 and conf_t is not None and pred_t is not None:
            regrets = []
            for pos in C_t:
                tok = committed_token[pos]
                if tok < 0:
                    continue
                # Approximate p_θ(x̂_i | context_t):
                # If model still predicts committed token, use current confidence.
                # Otherwise probability is below current argmax confidence → approx 0.
                if pred_t[pos] == tok:
                    prob_committed = float(conf_t[pos])
                else:
                    prob_committed = 0.0
                r = float(commit_conf[pos]) - prob_committed
                regrets.append(max(0.0, r))
            if regrets:
                mean_regret_val = float(np.mean(regrets))

        # ---- M̂_t ---- #
        M_hat_t = mean_regret_val / (1.0 + max(0.0, cascade_gain_val))

        # ---- Prediction instability inst(t) ---- #
        instability_val = 0.0
        if n_selected > 0 and t > 0:
            pred_prev = steps[t - 1]["predictions"]
            if pred_t is not None and pred_prev is not None:
                same = np.sum(pred_t[S_t] == pred_prev[S_t])
                instability_val = 1.0 - float(same) / n_selected

        # ---- Rank volatility vol(t) ---- #
        rank_vol_val = 0.0
        if t > 0 and gen_length >= 2:
            conf_prev = steps[t - 1]["confidences"]
            if conf_t is not None and conf_prev is not None:
                c_cur = conf_t.astype(np.float64)
                c_prv = conf_prev.astype(np.float64)
                r_cur = np.argsort(np.argsort(c_cur)).astype(np.float64)
                r_prv = np.argsort(np.argsort(c_prv)).astype(np.float64)
                tau = _fast_kendall_tau(r_cur, r_prv)
                rank_vol_val = 1.0 - (tau + 1.0) / 2.0

        # ---- Ĵ_t ---- #
        J_hat_t = instability_val * rank_vol_val

        step_proxy_list.append(StepProxies(
            step=t,
            M_hat_t=M_hat_t,
            J_hat_t=J_hat_t,
            weight=weight,
            mean_regret=mean_regret_val,
            cascade_gain=cascade_gain_val,
            instability=instability_val,
            rank_volatility=rank_vol_val,
            n_committed=n_committed,
            n_selected=n_selected,
            n_remaining=n_remaining,
        ))

    caps = _aggregate_sample_caps(step_proxy_list)
    M_hat = caps["M_hat_t"].entropy
    J_hat = caps["J_hat_t"].entropy
    M_hat_uniform = caps["M_hat_t"].uniform
    J_hat_uniform = caps["J_hat_t"].uniform

    return SampleProxies(
        M_hat=M_hat,
        J_hat=J_hat,
        M_hat_uniform=M_hat_uniform,
        J_hat_uniform=J_hat_uniform,
        caps=caps,
        step_proxies=step_proxy_list,
    )
