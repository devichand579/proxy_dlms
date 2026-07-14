"""PCFG oracle and trace-commit metrics for SchedulerLens."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LLADA_PCFG_DIR = _REPO_ROOT / "llada-pcfg"
_LLADA_PCFG_SCRIPTS = _LLADA_PCFG_DIR / "scripts"
for _path in (_REPO_ROOT, _LLADA_PCFG_DIR, _LLADA_PCFG_SCRIPTS):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from pcfg_data import ARCH_LEN, Vocab  # noqa: E402
from pcfg_engine import PCFGEngine, kl  # noqa: E402

EPS = 1e-12


@dataclass
class OracleStepMetrics:
    step: int
    Mt: float
    Jt: float
    oracle_nll: float
    model_nll: float
    n_committed: int


@dataclass
class OracleSampleMetrics:
    Mt_total: float
    Jt_total: float
    oracle_nll: float
    model_nll: float
    oracle_ppl: float
    model_ppl: float
    steps: list[OracleStepMetrics]


def _ppl(nll: float) -> float:
    if not math.isfinite(nll):
        return float("nan")
    return math.exp(min(float(nll), 20.0))


def _squeeze_gen(arr) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 2 and a.shape[0] == 1:
        a = a[0]
    return a


def extract_unmask_events(trace_dict: dict) -> list[tuple[int, int, int]]:
    """Return (step, gen_pos, token_id) for each commit in a trace."""
    events: list[tuple[int, int, int]] = []
    step_traces = trace_dict.get("step_traces", [])
    for st in step_traces:
        step = int(st.get("step", 0))
        sel_pos = st.get("selected_positions")
        sel_tok = st.get("selected_tokens")
        if sel_pos is not None and sel_tok is not None:
            pos = _squeeze_gen(sel_pos).astype(bool)
            tok = _squeeze_gen(sel_tok).astype(np.int64)
            for p in range(len(pos)):
                if pos[p]:
                    events.append((step, int(p), int(tok[p])))
            continue
        mask = _squeeze_gen(st.get("mask_index", [])).astype(bool)
        preds = _squeeze_gen(st.get("predictions", [])).astype(np.int64)
        if step + 1 < len(step_traces):
            next_mask = _squeeze_gen(step_traces[step + 1]["mask_index"]).astype(bool)
            for p in range(len(mask)):
                if mask[p] and not next_mask[p]:
                    events.append((step, p, int(preds[p])))
    return events


def reference_token_ids(trace_dict: dict, metadata: dict | None = None) -> list[int] | None:
    meta = metadata or trace_dict.get("metadata") or {}
    ref = meta.get("reference_token_ids")
    if ref is not None:
        return [int(x) for x in ref]
    final_seq = trace_dict.get("final_sequence")
    if final_seq is None:
        return None
    seq = _squeeze_gen(final_seq).astype(np.int64).tolist()
    prompt_len = int(trace_dict.get("prompt_len") or 0)
    gen_len = int(trace_dict.get("gen_length") or len(seq) - prompt_len)
    return [int(x) for x in seq[prompt_len : prompt_len + gen_len]]


def model_logprobs_from_logits(logits: torch.Tensor, positions: Iterable[int]) -> dict[int, np.ndarray]:
    """Softmax log-probs over vocab for each sequence position."""
    logp = F.log_softmax(logits.float(), dim=-1)
    out = {}
    for p in positions:
        out[int(p)] = logp[0, int(p)].detach().cpu().numpy()
    return out


def model_nll_at_commit(
    logprobs: dict[int, np.ndarray],
    pos: int,
    token_id: int,
) -> float:
    lp = logprobs.get(int(pos))
    if lp is None or token_id < 0 or token_id >= lp.shape[0]:
        return float("inf")
    return float(-lp[int(token_id)])


def oracle_nll_at_commit(
    engine: PCFGEngine,
    tokens: Sequence[str],
    committed: dict[int, int],
    pos: int,
) -> float:
    marg = engine.marginal(len(tokens), committed, int(pos))
    tid = committed[int(pos)]
    return float(-math.log(float(marg[tid]) + EPS))


def trace_model_commit_nll(trace_dict: dict, reference_ids: Sequence[int] | None = None) -> float:
    """Proxy NLL from trace confidences at commit time."""
    ref = reference_ids if reference_ids is not None else reference_token_ids(trace_dict)
    if ref is None:
        return float("nan")
    nll = 0.0
    n = 0
    for step, pos, tok in extract_unmask_events(trace_dict):
        if pos >= len(ref):
            continue
        target = int(ref[pos])
        st = trace_dict["step_traces"][step]
        preds = _squeeze_gen(st.get("predictions", [])).astype(np.int64)
        confs = _squeeze_gen(st.get("confidences", [])).astype(np.float64)
        if int(preds[pos]) == target and confs[pos] > 0:
            nll += -math.log(float(confs[pos]) + EPS)
        else:
            nll += 20.0
        n += 1
    return nll / max(n, 1)


def trace_oracle_nll(
    engine: PCFGEngine,
    trace_dict: dict,
    token_strings: Sequence[str],
) -> float:
    """Exact oracle NLL -log q(x_i | committed) along the trace unmask order."""
    tmap = engine.t_idx
    n = len(token_strings)
    committed: dict[int, int] = {}
    nll = 0.0
    count = 0
    for _step, pos, tok in extract_unmask_events(trace_dict):
        if pos >= n:
            continue
        tid = tmap[token_strings[int(pos)]]
        committed[int(pos)] = tid
        marg = engine.marginal(n, {k: v for k, v in committed.items() if k != pos}, int(pos))
        nll += -math.log(float(marg[tid]) + EPS)
        count += 1
    return nll / max(count, 1)


@torch.no_grad()
def replay_oracle_mt_jt(
    model,
    trace_dict: dict,
    engine: PCFGEngine,
    vocab: Vocab,
    device: torch.device | str,
) -> OracleSampleMetrics:
    """Re-forward model at each trace step; compute exact Mt and Jt vs PCFG oracle."""
    meta = trace_dict.get("metadata") or {}
    token_strings = meta.get("reference_tokens")
    if token_strings is None:
        ref_ids = reference_token_ids(trace_dict)
        if ref_ids is None:
            raise ValueError("trace missing reference_tokens / final_sequence")
        token_strings = vocab.decode(ref_ids[: meta.get("n_content", len(ref_ids))])

    n_content = int(meta.get("n_content", len(token_strings)))
    input_ids = torch.full((1, ARCH_LEN), vocab.pad_id, dtype=torch.long, device=device)
    attention = torch.zeros((1, ARCH_LEN), dtype=torch.long, device=device)
    input_ids[0, :n_content] = vocab.mask_id
    input_ids[0, n_content] = vocab.mask_id
    attention[0, : n_content + 1] = 1

    tmap = engine.t_idx
    committed: dict[int, int] = {}
    step_metrics: list[OracleStepMetrics] = []
    Mt_total = 0.0
    Jt_total = 0.0
    oracle_nll_sum = 0.0
    model_nll_sum = 0.0
    n_tokens = 0

    events_by_step: dict[int, list[tuple[int, int]]] = {}
    for step, pos, tok in extract_unmask_events(trace_dict):
        events_by_step.setdefault(step, []).append((pos, tok))

    for step in sorted(events_by_step):
        S = [p for p, _ in events_by_step[step]]
        out = model(input_ids=input_ids, attention_mask=attention)
        logps = model_logprobs_from_logits(out.logits, S)
        q_batch = engine.marginals_batch(n_content, committed, S)

        step_mt = 0.0
        step_onll = 0.0
        step_mnll = 0.0
        for pos, tok in events_by_step[step]:
            q = q_batch.get(int(pos), np.zeros(engine.V))
            p = np.exp(logps[int(pos)])
            step_mt += kl(q, p)
            tid = tmap[token_strings[int(pos)]]
            step_onll += -math.log(float(q[tid]) + EPS)
            step_mnll += model_nll_at_commit(logps, pos, int(tok))
            committed[int(pos)] = tid
            input_ids[0, int(pos)] = int(tok)
            n_tokens += 1

        jt = float(engine.Jt(n_content, committed, S))
        step_metrics.append(
            OracleStepMetrics(
                step=step,
                Mt=step_mt,
                Jt=jt,
                oracle_nll=step_onll / max(len(S), 1),
                model_nll=step_mnll / max(len(S), 1),
                n_committed=len(S),
            )
        )
        Mt_total += step_mt
        Jt_total += jt
        oracle_nll_sum += step_onll
        model_nll_sum += step_mnll

    denom = max(n_tokens, 1)
    oracle_mean = oracle_nll_sum / denom
    model_mean = model_nll_sum / denom
    return OracleSampleMetrics(
        Mt_total=Mt_total,
        Jt_total=Jt_total,
        oracle_nll=oracle_mean,
        model_nll=model_mean,
        oracle_ppl=_ppl(oracle_mean),
        model_ppl=_ppl(model_mean),
        steps=step_metrics,
    )
