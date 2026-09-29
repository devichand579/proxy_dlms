"""Per-step M̂ and Ĵ from a generation trace.

M̂ sums −log c over the lowest-entropy tokens of the committed set whose entropies sum to at most γ = 1.
Ĵ is ΣH − max H on that set when it has at least two tokens. Sample caps sum the steps; Ĝ is M̂ + Ĵ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Entropy budget for M̂. Committed tokens are admitted from lowest entropy upward
# while the sum of admitted entropies stays within this value.
M_GAMMA = 1.0

# Step-level series aggregated to sample/task caps and trends.
PROXY_ATTRS = ("M_hat_t", "J_hat_t")


class TraceProxyError(ValueError):
    """Raised when required trace fields are missing or inconsistent."""


@dataclass
class StepProxies:
    step: int
    M_hat_t: float  # m_hat on S_t (γ = M_GAMMA)
    J_hat_t: float  # j_hat on S_t (|S|≥2)
    n_committed: int = 0
    n_selected: int = 0
    n_remaining: int = 0


@dataclass
class SampleProxies:
    """Per-sample caps for M̂ and Ĵ."""

    M_hat: float
    J_hat: float  # full-trajectory j_hat sum
    caps: Dict[str, float] = field(default_factory=dict)
    step_proxies: List[StepProxies] = field(default_factory=list)


def _empty_caps() -> Dict[str, float]:
    return {name: 0.0 for name in PROXY_ATTRS}


def _aggregate_sample_caps(
    step_proxies: List[StepProxies],
) -> tuple[Dict[str, float], float]:
    """Sample sums of M̂ and Ĵ."""
    m_sum = 0.0
    j_full = 0.0
    for sp in step_proxies:
        if sp.n_selected <= 0:
            continue
        m_sum += float(sp.M_hat_t)
        j_full += float(sp.J_hat_t)
    caps = {"M_hat_t": m_sum, "J_hat_t": j_full}
    return caps, j_full


def cumulative_caps(
    step_proxies: List[StepProxies],
    prefix_steps: int,
) -> tuple[float, float, float]:
    """M̂, Ĵ, and Ĝ summed over steps before ``prefix_steps``."""
    k = int(prefix_steps)
    if k <= 0:
        return 0.0, 0.0, 0.0
    m_sum = 0.0
    j_sum = 0.0
    for sp in step_proxies:
        if int(sp.step) >= k:
            continue
        if sp.n_selected <= 0:
            continue
        m_sum += float(sp.M_hat_t)
        j_sum += float(sp.J_hat_t)
    return m_sum, j_sum, m_sum + j_sum


def prefix_commit_fraction(
    step_proxies: List[StepProxies],
    prefix_steps: int,
    gen_length: int,
) -> float:
    """Share of the gen region unmasked after steps before ``prefix_steps``."""
    length = int(gen_length)
    if length <= 0 or not step_proxies:
        return 0.0
    k = int(prefix_steps)
    later = [sp for sp in step_proxies if int(sp.step) >= k]
    if later:
        committed = int(min(later, key=lambda sp: int(sp.step)).n_committed)
    else:
        last = max(step_proxies, key=lambda sp: int(sp.step))
        committed = int(last.n_committed) + int(last.n_selected)
    return float(committed) / float(length)


def _to_np(data, dtype=np.float32) -> Optional[np.ndarray]:
    if data is None:
        return None
    arr = np.array(data, dtype=dtype)
    while arr.ndim > 1:
        arr = arr[0]
    return arr


def _require_array(
    arr: Optional[np.ndarray],
    *,
    name: str,
    step: int,
    gen_length: int,
) -> np.ndarray:
    if arr is None:
        raise TraceProxyError(f"step {step}: missing required field {name!r}")
    if arr.shape[0] != gen_length:
        raise TraceProxyError(
            f"step {step}: {name} length {arr.shape[0]} != gen_length {gen_length}"
        )
    return arr


def _zero_step(step: int, n_committed: int, n_remaining: int) -> StepProxies:
    """A step that unmasks nothing."""
    return StepProxies(
        step=step,
        M_hat_t=0.0,
        J_hat_t=0.0,
        n_committed=n_committed,
        n_selected=0,
        n_remaining=n_remaining,
    )


def j_hat(entropies: np.ndarray) -> float:
    """Ĵ on one committed set."""
    if entropies.size < 2:
        return 0.0
    return float(max(np.sum(entropies) - np.max(entropies), 0.0))


def m_hat(conf: np.ndarray, ent: np.ndarray, gamma: float) -> float:
    """M̂ on one committed set: −log c inside the entropy budget γ."""
    if ent.size == 0:
        return 0.0
    order = np.argsort(ent, kind="mergesort")
    h = ent[order].astype(np.float64)
    c = np.clip(conf[order].astype(np.float64), 1e-8, 1.0)
    keep = np.cumsum(h) <= float(gamma)
    if not np.any(keep):
        return 0.0
    return float(np.sum(-np.log(c[keep])))


def _mask_diff_selection(
    steps: List[dict],
    t: int,
    gen_length: int,
) -> Optional[np.ndarray]:
    """Positions unmasked between this step and the next."""
    mask_t = steps[t]["mask_index"]
    if mask_t is None:
        return None
    mask_t = _require_array(mask_t, name="mask_index", step=t, gen_length=gen_length)
    if t + 1 >= len(steps):
        return np.zeros(gen_length, dtype=bool)
    mask_next = steps[t + 1]["mask_index"]
    if mask_next is None:
        return None
    mask_next = _require_array(
        mask_next, name="mask_index", step=t + 1, gen_length=gen_length
    )
    return (mask_t & ~mask_next).astype(bool, copy=False)


def validate_trace_for_metrics(
    trace_dict: dict,
    *,
    trace_name: Optional[Union[str, Path]] = None,
) -> None:
    """Raise if the trace is missing fields or still needs repair."""
    from traces.trace_checks import (
        count_phantom_extra_steps,
        count_selection_mask_mismatches,
        sample_tail_trimmed,
    )
    from traces.utils import (
        is_fdm_trace_name,
        is_fhs_trace_name,
        parse_trace_step_budget,
    )

    steps_raw = trace_dict.get("step_traces") or []
    if not steps_raw:
        return

    gen_length = int(trace_dict.get("gen_length") or 0)
    if gen_length <= 0:
        raise TraceProxyError(f"invalid gen_length={gen_length}")

    if not sample_tail_trimmed(trace_dict):
        raise TraceProxyError(
            "trailing step not trimmed (schedulerlens_trimmed_tail); "
            "run convert/repair-npz before SchedulerLens metrics"
        )

    required = ("mask_index", "predictions", "confidences", "entropies")
    for t, st in enumerate(steps_raw):
        for name in required:
            val = st.get(name)
            if val is None:
                raise TraceProxyError(f"step {t}: missing required field {name!r}")
            arr = _to_np(val, dtype=np.float32 if name in ("confidences", "entropies") else np.int64)
            if arr is None or arr.shape[0] != gen_length:
                raise TraceProxyError(
                    f"step {t}: {name} length "
                    f"{None if arr is None else arr.shape[0]} != gen_length {gen_length}"
                )

    is_fdm = is_fdm_trace_name(trace_name)
    is_fhs = is_fhs_trace_name(trace_name)
    budget = parse_trace_step_budget(trace_name)
    phantom = count_phantom_extra_steps(
        steps_raw,
        gen_length=gen_length,
        is_fdm=is_fdm,
        is_fhs=is_fhs,
        step_budget=budget,
    )
    if phantom > 0:
        kinds = []
        if is_fdm:
            kinds.append("FDM plateau")
        if is_fhs:
            kinds.append("FHS corrector")
        kind = "/".join(kinds) if kinds else "phantom"
        raise TraceProxyError(
            f"{phantom} {kind} extra steps; "
            "run convert/repair-npz before SchedulerLens metrics"
        )

    mismatches = count_selection_mask_mismatches(steps_raw, gen_length=gen_length)
    if mismatches:
        head = "; ".join(mismatches[:5])
        more = f" (+{len(mismatches) - 5} more)" if len(mismatches) > 5 else ""
        raise TraceProxyError(
            f"selected_positions vs mask_diff mismatch; repair the trace first: {head}{more}"
        )

    # Build normalized steps for set-equality / missing-selection checks.
    steps: List[dict] = []
    for st in steps_raw:
        steps.append({
            "mask_index": _to_np(st.get("mask_index"), dtype=np.bool_),
            "predictions": _to_np(st.get("predictions"), dtype=np.int64),
            "selected_positions": _to_np(st.get("selected_positions"), dtype=np.bool_),
            "selected_tokens": _to_np(st.get("selected_tokens"), dtype=np.int64),
        })

    last_t = len(steps) - 1
    for t in range(len(steps)):
        is_last = t == last_t
        inferred = _mask_diff_selection(steps, t, gen_length)
        if inferred is None:
            raise TraceProxyError(
                f"step {t}: cannot read mask_index pair for mask_diff validation"
            )
        n_diff = int(np.sum(inferred))
        sel = steps[t]["selected_positions"]
        sel_tok = steps[t]["selected_tokens"]
        # Last step has no next mask after repair drops the trailing step; trust selected_*.
        if is_last:
            if sel is None and sel_tok is not None:
                raise TraceProxyError(
                    f"step {t}: selected_tokens present without selected_positions"
                )
            if sel is not None and sel_tok is None:
                raise TraceProxyError(
                    f"step {t}: selected_positions present without selected_tokens"
                )
            if sel is not None:
                _require_array(sel, name="selected_positions", step=t, gen_length=gen_length)
                _require_array(sel_tok, name="selected_tokens", step=t, gen_length=gen_length)
            continue
        if n_diff > 0 and sel is None:
            raise TraceProxyError(
                f"step {t}: missing selected_positions but mask_diff has {n_diff} unmasks; "
                "repair the trace first"
            )
        if sel is None and sel_tok is not None:
            raise TraceProxyError(
                f"step {t}: selected_tokens present without selected_positions"
            )
        if sel is not None and sel_tok is None:
            raise TraceProxyError(
                f"step {t}: selected_positions present without selected_tokens"
            )
        if sel is None:
            continue
        sel_b = _require_array(sel, name="selected_positions", step=t, gen_length=gen_length)
        sel_b = sel_b.astype(bool, copy=False)
        _require_array(sel_tok, name="selected_tokens", step=t, gen_length=gen_length)
        if not np.array_equal(sel_b, inferred):
            only_sel = int(np.sum(sel_b & ~inferred))
            only_diff = int(np.sum(inferred & ~sel_b))
            raise TraceProxyError(
                f"step {t}: selected_positions set != mask_diff "
                f"(sel_count={int(sel_b.sum())} diff_count={n_diff} "
                f"only_sel={only_sel} only_diff={only_diff}); repair the trace first"
            )


def _resolve_selection(
    steps: List[dict],
    t: int,
    gen_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Selected positions and tokens for one step."""
    st = steps[t]
    sel = st["selected_positions"]
    sel_tok = st["selected_tokens"]
    inferred = _mask_diff_selection(steps, t, gen_length)
    is_last = t == len(steps) - 1

    if sel is not None and sel_tok is not None:
        sel = _require_array(sel, name="selected_positions", step=t, gen_length=gen_length)
        sel_tok = _require_array(sel_tok, name="selected_tokens", step=t, gen_length=gen_length)
        sel_b = sel.astype(bool, copy=False)
        # Non-last steps must match mask_diff; last step is trusted after repair drop.
        if (
            not is_last
            and inferred is not None
            and not np.array_equal(sel_b, inferred)
        ):
            raise TraceProxyError(
                f"step {t}: selected_positions != mask_diff "
                f"(sel={int(sel_b.sum())} diff={int(inferred.sum())}); repair the trace first"
            )
        return sel_b, np.asarray(sel_tok, dtype=np.int64)

    if not is_last and inferred is not None and inferred.any():
        raise TraceProxyError(
            f"step {t}: missing selected_positions but mask_diff has {int(inferred.sum())} "
            "unmasks; repair the trace first"
        )
    if sel is not None or sel_tok is not None:
        raise TraceProxyError(
            f"step {t}: incomplete selected_* pair (one of positions/tokens is missing)"
        )
    return np.zeros(gen_length, dtype=bool), np.zeros(gen_length, dtype=np.int64)


def compute_sample_proxies(
    trace_dict: dict,
    *,
    trace_name: Optional[Union[str, Path]] = None,
    gamma: Optional[float] = None,
) -> SampleProxies:
    """M̂, Ĵ, and the per-step values for one sample."""
    validate_trace_for_metrics(trace_dict, trace_name=trace_name)

    step_traces = trace_dict.get("step_traces", [])
    T = len(step_traces)
    if T == 0:
        empty = _empty_caps()
        return SampleProxies(
            M_hat=0.0,
            J_hat=0.0,
            caps=empty,
        )

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

    if steps[0]["mask_index"] is None:
        raise TraceProxyError("step 0: missing required field 'mask_index'")
    gen_length = int(trace_dict.get("gen_length") or steps[0]["mask_index"].shape[0])
    if gen_length <= 0:
        raise TraceProxyError(f"invalid gen_length={gen_length}")

    # Resolve selection for every step up front (fail-hard; no silent repair).
    resolved: List[Tuple[np.ndarray, np.ndarray]] = []
    for t in range(T):
        resolved.append(_resolve_selection(steps, t, gen_length))

    step_proxy_list: List[StepProxies] = []
    budget = float(M_GAMMA if gamma is None else gamma)

    for t in range(T):
        mask_t = _require_array(
            steps[t]["mask_index"], name="mask_index", step=t, gen_length=gen_length
        )
        conf_t = _require_array(
            steps[t]["confidences"], name="confidences", step=t, gen_length=gen_length
        )
        ent_t = _require_array(
            steps[t]["entropies"], name="entropies", step=t, gen_length=gen_length
        )
        sel_t, _sel_tok = resolved[t]

        S_t = np.where(sel_t)[0]
        n_selected = int(S_t.size)
        n_committed = int(np.sum(~mask_t))
        n_remaining = int(np.sum(mask_t & ~sel_t))

        if n_selected == 0:
            step_proxy_list.append(_zero_step(t, n_committed, n_remaining))
            continue

        c_sel = conf_t[S_t]
        h_sel = ent_t[S_t]
        m_hat_t = m_hat(c_sel, h_sel, budget)
        j_hat_t = j_hat(h_sel.astype(np.float64))

        step_proxy_list.append(
            StepProxies(
                step=t,
                M_hat_t=m_hat_t,
                J_hat_t=j_hat_t,
                n_committed=n_committed,
                n_selected=n_selected,
                n_remaining=n_remaining,
            )
        )

    caps, j_full = _aggregate_sample_caps(step_proxy_list)
    m_cap = float(caps["M_hat_t"])
    return SampleProxies(
        M_hat=m_cap,
        J_hat=j_full,
        caps=caps,
        step_proxies=step_proxy_list,
    )
