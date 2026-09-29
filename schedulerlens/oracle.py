"""PCFG oracle and trace-commit metrics for SchedulerLens."""

from __future__ import annotations

import logging
import math
import os
import queue
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    import torch
    from pcfg_data import Vocab
    from pcfg_engine import PCFGEngine

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LLADA_PCFG_DIR = _REPO_ROOT / "llada-pcfg"
_LLADA_PCFG_SCRIPTS = _LLADA_PCFG_DIR / "scripts"

EPS = 1e-12

logger = logging.getLogger(__name__)


def _ensure_pcfg_path() -> None:
    for _path in (_REPO_ROOT, _LLADA_PCFG_DIR, _LLADA_PCFG_SCRIPTS):
        _text = str(_path)
        if _text not in sys.path:
            sys.path.insert(0, _text)


@dataclass
class OracleTokenMetrics:
    """Oracle quantities for one committed token."""

    gen_pos: int
    content_pos: int
    sel_tok: int
    ref_tok: int
    mt_kl: float  # KL(q‖p) contribution at this position
    oracle_nll: float  # -log q(ref | committed)
    model_nll: float  # -log p(sel | model logits)


@dataclass
class OracleStepMetrics:
    step: int
    Mt: float
    Jt: float
    oracle_nll: float
    model_nll: float
    n_selected: int
    tokens: list[OracleTokenMetrics] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "Mt": self.Mt,
            "Jt": self.Jt,
            "oracle_nll": self.oracle_nll,
            "model_nll": self.model_nll,
            "n_selected": self.n_selected,
            "tokens": [
                {
                    "gen_pos": t.gen_pos,
                    "content_pos": t.content_pos,
                    "sel_tok": t.sel_tok,
                    "ref_tok": t.ref_tok,
                    "mt_kl": t.mt_kl,
                    "oracle_nll": t.oracle_nll,
                    "model_nll": t.model_nll,
                }
                for t in self.tokens
            ],
        }


@dataclass
class OracleSampleMetrics:
    Mt_total: float
    Jt_total: float
    oracle_nll: float
    model_nll: float
    oracle_ppl: float
    model_ppl: float
    steps: list[OracleStepMetrics]

    def to_dict(self) -> dict:
        return {
            "Mt_total": self.Mt_total,
            "Jt_total": self.Jt_total,
            "oracle_nll": self.oracle_nll,
            "model_nll": self.model_nll,
            "oracle_ppl": self.oracle_ppl,
            "model_ppl": self.model_ppl,
            "steps": [s.to_dict() for s in self.steps],
        }


def nll_to_ppl(nll: float) -> float:
    if not math.isfinite(nll):
        return float("nan")
    return math.exp(min(float(nll), 20.0))


def _squeeze_gen(arr) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 2 and a.shape[0] == 1:
        a = a[0]
    return a


def extract_unmask_events(
    trace_dict: dict,
    *,
    trace_name: str | Path | None = None,
) -> list[tuple[int, int, int]]:
    """Each commit as step, gen position, and token id."""
    from schedulerlens.proxies import TraceProxyError, validate_trace_for_metrics

    validate_trace_for_metrics(trace_dict, trace_name=trace_name)

    events: list[tuple[int, int, int]] = []
    step_traces = trace_dict.get("step_traces", [])
    for t, st in enumerate(step_traces):
        step = int(st.get("step", t))
        sel_pos = st.get("selected_positions")
        sel_tok = st.get("selected_tokens")
        if sel_pos is None:
            # Validated: empty mask_diff only. No silent mask-diff inference here.
            continue
        if sel_tok is None:
            raise TraceProxyError(
                f"step {step}: selected_positions present without selected_tokens"
            )
        pos = _squeeze_gen(sel_pos).astype(bool)
        tok = _squeeze_gen(sel_tok).astype(np.int64)
        if pos.shape[0] != tok.shape[0]:
            raise TraceProxyError(
                f"step {step}: selected_positions length {pos.shape[0]} != "
                f"selected_tokens length {tok.shape[0]}"
            )
        for p in range(len(pos)):
            if pos[p]:
                events.append((step, int(p), int(tok[p])))
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


def _pcfg_layout(trace_dict: dict, vocab=None) -> dict:
    """Content, prompt, and gen lengths for this trace."""
    meta = trace_dict.get("metadata") or {}
    token_strings = meta.get("reference_tokens")
    if token_strings is None:
        raise ValueError("PCFG trace missing metadata.reference_tokens")
    token_strings = list(token_strings)
    n_content = int(meta.get("n_content", len(token_strings)))
    n_visible = int(meta.get("n_visible", 0))
    if not (0 <= n_visible < n_content):
        raise ValueError(f"invalid n_visible={n_visible} for n_content={n_content}")
    if len(token_strings) < n_content:
        raise ValueError(
            f"reference_tokens length {len(token_strings)} < n_content {n_content}"
        )
    prompt_len = int(trace_dict.get("prompt_len") or n_visible)
    gen_length = int(trace_dict.get("gen_length") or meta.get("gen_length") or 0)
    if gen_length <= 0:
        raise ValueError("trace missing gen_length")
    suffix_len = n_content - n_visible
    return {
        "token_strings": token_strings,
        "n_content": n_content,
        "n_visible": n_visible,
        "prompt_len": prompt_len,
        "gen_length": gen_length,
        "suffix_len": suffix_len,
        "meta": meta,
        "vocab": vocab,
    }


def gen_to_content_pos(gen_pos: int, n_visible: int, n_content: int) -> int | None:
    """Content index for a gen position, or None outside the content."""
    content_pos = int(n_visible) + int(gen_pos)
    if content_pos < 0 or content_pos >= int(n_content):
        return None
    return content_pos


def model_logprobs_from_logits(logits, positions: Iterable[int]) -> dict[int, np.ndarray]:
    """Log-probabilities at the given positions."""
    import torch.nn.functional as F

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
    engine,
    tokens: Sequence[str],
    committed: dict[int, int],
    pos: int,
) -> float:
    marg = engine.marginal(len(tokens), committed, int(pos))
    tid = committed[int(pos)]
    return float(-math.log(float(marg[tid]) + EPS))


def trace_model_commit_nll(
    trace_dict: dict,
    reference_ids: Sequence[int] | None = None,
    *,
    trace_name: Union[str, Path, None] = None,
) -> float:
    """Mean −log c of committed tokens from the trace confidences."""
    ref = reference_ids if reference_ids is not None else reference_token_ids(trace_dict)
    if ref is None:
        return float("nan")
    nll = 0.0
    n = 0
    for step, pos, tok in extract_unmask_events(trace_dict, trace_name=trace_name):
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
    engine,
    trace_dict: dict,
    token_strings: Sequence[str] | None = None,
    *,
    trace_name: Union[str, Path, None] = None,
) -> float:
    """Mean oracle −log q along the unmask order."""
    meta = trace_dict.get("metadata") or {}
    tokens = list(token_strings or meta.get("reference_tokens") or [])
    if not tokens:
        raise ValueError("trace_oracle_nll requires reference_tokens")
    n_content = int(meta.get("n_content", len(tokens)))
    n_visible = int(meta.get("n_visible", 0))
    tokens = tokens[:n_content]
    tmap = engine.t_idx
    committed: dict[int, int] = {i: tmap[tokens[i]] for i in range(n_visible)}
    nll = 0.0
    count = 0
    for _step, gen_pos, _tok in extract_unmask_events(
        trace_dict, trace_name=trace_name
    ):
        content_pos = gen_to_content_pos(gen_pos, n_visible, n_content)
        if content_pos is None:
            continue
        tid = tmap[tokens[content_pos]]
        marg = engine.marginal(n_content, committed, content_pos)
        nll += -math.log(float(marg[tid]) + EPS)
        committed[content_pos] = tid
        count += 1
    return nll / max(count, 1)


def _build_eval_style_input(
    *,
    token_strings: Sequence[str],
    n_visible: int,
    n_content: int,
    prompt_len: int,
    gen_length: int,
    vocab,
    device,
):
    """Left-padded input: visible prefix, then a masked gen region."""
    import torch

    _ensure_pcfg_path()
    from pcfg_data import ARCH_LEN

    ids = vocab.encode(list(token_strings[:n_content]))
    seq_len = max(int(prompt_len) + int(gen_length), ARCH_LEN)
    input_ids = torch.full((1, seq_len), vocab.pad_id, dtype=torch.long, device=device)
    attention = torch.zeros((1, seq_len), dtype=torch.long, device=device)
    pl = int(prompt_len)
    gl = int(gen_length)
    nv = int(n_visible)
    if nv > pl:
        raise ValueError(f"n_visible={nv} > prompt_len={pl}")
    if nv > 0:
        input_ids[0, pl - nv : pl] = torch.tensor(
            ids[:nv], dtype=torch.long, device=device
        )
        attention[0, pl - nv : pl] = 1
    input_ids[0, pl : pl + gl] = vocab.mask_id
    attention[0, pl : pl + gl] = 1
    return input_ids, attention


@dataclass
class _StepJob:
    """One unmask step and the teacher-forced state before that commit."""

    step: int
    n_content: int
    S_content: list[int]
    content_events: list[tuple[int, int, int]]  # gen_pos, cpos, sel_tok
    model_positions: list[int]
    committed: dict[int, int]
    tokens: list[str]
    tmap: dict
    input_ids: Any  # CPU torch.LongTensor [1, L], or None for Jt-only jobs
    attention: Any


class _EnginePool:
    """Pool of PCFG engines, one per thread."""

    def __init__(self, size: int):
        _ensure_pcfg_path()
        from pcfg_data import build_stack

        size = max(1, int(size))
        self._q: queue.Queue = queue.Queue()
        for _ in range(size):
            _g, _cnf, eng, _vocab = build_stack()
            self._q.put(eng)

    def borrow(self):
        return self._q.get()

    def release(self, eng) -> None:
        if hasattr(eng, "clear_cache"):
            eng.clear_cache()
        self._q.put(eng)


def _default_step_workers() -> int:
    return max(1, min(8, (os.cpu_count() or 4)))


def _precompute_step_jobs(
    trace_dict: dict,
    vocab,
    *,
    tmap: Optional[dict] = None,
    trace_name: Union[str, Path, None] = None,
    include_model_inputs: bool = True,
) -> list[_StepJob]:
    """One independent job per unmask step."""
    layout = _pcfg_layout(trace_dict, vocab)
    tokens = layout["token_strings"][: layout["n_content"]]
    n_content = layout["n_content"]
    n_visible = layout["n_visible"]
    prompt_len = layout["prompt_len"]
    gen_length = layout["gen_length"]
    if tmap is None:
        _ensure_pcfg_path()
        from pcfg_data import build_stack

        _g, _cnf, eng, _v = build_stack()
        tmap = eng.t_idx
        if hasattr(eng, "clear_cache"):
            eng.clear_cache()
        del eng

    input_ids = None
    attention = None
    content_ids = None
    if include_model_inputs:
        import torch

        content_ids = vocab.encode(tokens)
        cpu = torch.device("cpu")
        input_ids, attention = _build_eval_style_input(
            token_strings=tokens,
            n_visible=n_visible,
            n_content=n_content,
            prompt_len=prompt_len,
            gen_length=gen_length,
            vocab=vocab,
            device=cpu,
        )

    committed: dict[int, int] = {i: tmap[tokens[i]] for i in range(n_visible)}
    events_by_step: dict[int, list[tuple[int, int]]] = {}
    for step, gen_pos, tok in extract_unmask_events(
        trace_dict, trace_name=trace_name
    ):
        events_by_step.setdefault(step, []).append((gen_pos, tok))

    jobs: list[_StepJob] = []
    for step in sorted(events_by_step):
        content_events: list[tuple[int, int, int]] = []
        for gen_pos, sel_tok in events_by_step[step]:
            cpos = gen_to_content_pos(gen_pos, n_visible, n_content)
            if cpos is None:
                continue
            content_events.append((int(gen_pos), int(cpos), int(sel_tok)))
        if not content_events:
            continue

        S_content = [cpos for _, cpos, _ in content_events]
        model_positions = [prompt_len + gpos for gpos, _, _ in content_events]
        jobs.append(
            _StepJob(
                step=step,
                n_content=n_content,
                S_content=S_content,
                content_events=content_events,
                model_positions=model_positions,
                committed=dict(committed),
                tokens=tokens,
                tmap=tmap,
                input_ids=None if input_ids is None else input_ids.clone(),
                attention=None if attention is None else attention.clone(),
            )
        )
        # Advance teacher-forced state for the next step's snapshot.
        for gen_pos, cpos, _sel in content_events:
            ref_tid = tmap[tokens[cpos]]
            committed[int(cpos)] = ref_tid
            if input_ids is not None and content_ids is not None:
                input_ids[0, int(prompt_len + gen_pos)] = int(content_ids[cpos])

    return jobs


def _batched_forward_logprobs(
    model,
    jobs: Sequence[_StepJob],
    device,
    *,
    model_lock: Optional[threading.Lock] = None,
) -> list[dict[int, np.ndarray]]:
    """Log-probabilities from one batched model forward."""
    import torch

    if not jobs:
        return []
    ids = torch.cat([j.input_ids for j in jobs], dim=0).to(device)
    attn = torch.cat([j.attention for j in jobs], dim=0).to(device)

    def _run():
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=attn)
            logits = out.logits
            del out
            result = []
            for i, job in enumerate(jobs):
                result.append(
                    model_logprobs_from_logits(logits[i : i + 1], job.model_positions)
                )
            del logits
            return result

    if model_lock is not None:
        with model_lock:
            return _run()
    return _run()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    return int(raw)


def default_jt_exact_max() -> int:
    """Largest committed set scored with exact Jt."""
    return max(2, _env_int("SCHEDULERLENS_JT_EXACT_MAX", 3))


def default_jt_markov_order() -> int:
    """Markov order used once the committed set is larger than the exact limit."""
    return max(1, _env_int("SCHEDULERLENS_JT_MARKOV_ORDER", 2))


def default_jt_concurrency() -> int:
    """How many exact Jt calls may run at once."""
    return max(1, _env_int("SCHEDULERLENS_JT_CONCURRENCY", 2))


def default_jt_large_set() -> int:
    """Set size above which Jt uses pairwise joints. Zero disables the cutoff."""
    return max(0, _env_int("SCHEDULERLENS_JT_LARGE_SET", 6))


_JT_SEM: Optional[threading.Semaphore] = None
_JT_SEM_LOCK = threading.Lock()


def _jt_semaphore() -> threading.Semaphore:
    global _JT_SEM
    with _JT_SEM_LOCK:
        if _JT_SEM is None:
            _JT_SEM = threading.Semaphore(default_jt_concurrency())
        return _JT_SEM


def _jt_exact(engine, n_content: int, committed: dict, positions: Sequence[int]) -> float:
    """Exact Jt on these positions."""
    S = sorted({int(p) for p in positions})
    if len(S) < 2:
        return 0.0
    sem = _jt_semaphore()
    sem.acquire()
    try:
        return float(engine.Jt(n_content, committed, S))
    finally:
        if hasattr(engine, "clear_cache"):
            engine.clear_cache()
        sem.release()


def compute_jt(
    engine,
    n_content: int,
    committed: dict,
    S: Sequence[int],
    *,
    exact_max: Optional[int] = None,
    markov_order: Optional[int] = None,
    large_set: Optional[int] = None,
) -> float:
    """Jt on the committed set: exact when the set is small, Markov when it is large."""
    positions = sorted({int(p) for p in S})
    if len(positions) < 2:
        return 0.0

    k_exact = max(2, int(exact_max if exact_max is not None else default_jt_exact_max()))
    order = max(1, int(markov_order if markov_order is not None else default_jt_markov_order()))
    large = int(large_set if large_set is not None else default_jt_large_set())
    # Huge simultaneous sets: pairwise only (size-2 joints), not size-3 windows.
    if large > 0 and len(positions) > large:
        order = 1
        k_exact = 2
    # Never ask exact Jt for sets larger than k_exact.
    order = min(order, k_exact - 1)

    if len(positions) <= k_exact:
        return _jt_exact(engine, n_content, committed, positions)

    # Cache exact subset Jt within this call (left window often reappears).
    cache: dict[tuple[int, ...], float] = {}

    def _cached(subset: Sequence[int]) -> float:
        key = tuple(subset)
        if len(key) < 2:
            return 0.0
        if key not in cache:
            cache[key] = _jt_exact(engine, n_content, committed, key)
        return cache[key]

    total = 0.0
    for i in range(1, len(positions)):
        lo = max(0, i - order)
        left = positions[lo:i]
        window = positions[lo : i + 1]
        total += _cached(window) - _cached(left)
    return float(total)


def compute_sample_jt_steps(
    trace_dict: dict,
    engine,
    vocab,
    *,
    trace_name: Union[str, Path, None] = None,
    jt_exact_max: Optional[int] = None,
    jt_markov_order: Optional[int] = None,
    jt_large_set: Optional[int] = None,
) -> list[tuple[int, int, float]]:
    """Jt at each unmask step, from the PCFG only."""
    tmap = getattr(engine, "t_idx", None)
    jobs = _precompute_step_jobs(
        trace_dict,
        vocab,
        tmap=tmap,
        trace_name=trace_name,
        include_model_inputs=False,
    )
    out: list[tuple[int, int, float]] = []
    for job in jobs:
        n_sel = len(job.S_content)
        jt = compute_jt(
            engine,
            job.n_content,
            job.committed,
            job.S_content,
            exact_max=jt_exact_max,
            markov_order=jt_markov_order,
            large_set=jt_large_set,
        )
        if hasattr(engine, "clear_cache"):
            engine.clear_cache()
        out.append((int(job.step), int(n_sel), float(jt)))
    return out


def _pcfg_metrics_for_step(
    engine,
    job: _StepJob,
    logps: dict[int, np.ndarray],
    kl,
    *,
    jt_exact_max: Optional[int] = None,
    jt_markov_order: Optional[int] = None,
    jt_large_set: Optional[int] = None,
) -> OracleStepMetrics:
    """Mt, Jt, and the NLLs for one precomputed step."""
    q_batch = engine.marginals_batch(job.n_content, job.committed, job.S_content)
    jt = compute_jt(
        engine,
        job.n_content,
        job.committed,
        job.S_content,
        exact_max=jt_exact_max,
        markov_order=jt_markov_order,
        large_set=jt_large_set,
    )
    if hasattr(engine, "clear_cache"):
        engine.clear_cache()

    step_mt = 0.0
    step_onll = 0.0
    step_mnll = 0.0
    token_rows: list[OracleTokenMetrics] = []
    for i, (gen_pos, cpos, sel_tok) in enumerate(job.content_events):
        model_pos = int(job.model_positions[i])
        q = q_batch.get(int(cpos), np.zeros(engine.V))
        p = np.exp(logps[int(model_pos)])
        if p.shape[0] != q.shape[0]:
            p = p[: q.shape[0]]
            p = p / max(float(p.sum()), EPS)
        mt_kl = float(kl(q, p))
        ref_tid = int(job.tmap[job.tokens[cpos]])
        onll = -math.log(float(q[ref_tid]) + EPS)
        mnll = float(model_nll_at_commit(logps, model_pos, int(sel_tok)))
        step_mt += mt_kl
        step_onll += onll
        step_mnll += mnll
        token_rows.append(
            OracleTokenMetrics(
                gen_pos=int(gen_pos),
                content_pos=int(cpos),
                sel_tok=int(sel_tok),
                ref_tok=ref_tid,
                mt_kl=mt_kl,
                oracle_nll=onll,
                model_nll=mnll,
            )
        )

    n_ev = max(len(job.content_events), 1)
    return OracleStepMetrics(
        step=job.step,
        Mt=step_mt,
        Jt=jt,
        oracle_nll=step_onll / n_ev,
        model_nll=step_mnll / n_ev,
        n_selected=len(job.content_events),
        tokens=token_rows,
    )


def replay_oracle_mt_jt(
    model,
    trace_dict: dict,
    engine,
    vocab,
    device,
    *,
    forward_batch_size: int = 8,
    step_workers: Optional[int] = None,
    engine_pool: Optional[_EnginePool] = None,
    model_lock: Optional[threading.Lock] = None,
    trace_name: Union[str, Path, None] = None,
    jt_exact_max: Optional[int] = None,
    jt_markov_order: Optional[int] = None,
    jt_large_set: Optional[int] = None,
) -> OracleSampleMetrics:
    """Mt and Jt for one sample, replaying the model against the PCFG."""
    _ensure_pcfg_path()
    from pcfg_engine import kl

    tmap = getattr(engine, "t_idx", None) if engine is not None else None
    jobs = _precompute_step_jobs(
        trace_dict, vocab, tmap=tmap, trace_name=trace_name
    )
    if not jobs:
        return OracleSampleMetrics(
            Mt_total=0.0,
            Jt_total=0.0,
            oracle_nll=float("nan"),
            model_nll=float("nan"),
            oracle_ppl=float("nan"),
            model_ppl=float("nan"),
            steps=[],
        )

    k_exact = max(2, int(jt_exact_max if jt_exact_max is not None else default_jt_exact_max()))
    m_order = max(1, int(jt_markov_order if jt_markov_order is not None else default_jt_markov_order()))
    large = int(jt_large_set if jt_large_set is not None else default_jt_large_set())
    max_s = max((len(j.S_content) for j in jobs), default=0)
    if max_s > k_exact:
        logger.debug(
            "Oracle Jt: max |S|=%d exact_max=%d markov=%d large_set=%d",
            max_s,
            k_exact,
            min(m_order, k_exact - 1),
            large,
        )

    n_workers = max(1, int(step_workers if step_workers is not None else _default_step_workers()))
    batch = max(1, int(forward_batch_size))
    owns_pool = engine_pool is None
    pool = engine_pool or _EnginePool(n_workers)

    step_metrics: list[Optional[OracleStepMetrics]] = [None] * len(jobs)

    def _run_pcfg(idx: int, logps: dict[int, np.ndarray]) -> None:
        eng = pool.borrow()
        try:
            step_metrics[idx] = _pcfg_metrics_for_step(
                eng,
                jobs[idx],
                logps,
                kl,
                jt_exact_max=k_exact,
                jt_markov_order=m_order,
                jt_large_set=large,
            )
        finally:
            pool.release(eng)

    try:
        with ThreadPoolExecutor(max_workers=n_workers) as cpu_ex:
            pending = []
            for start in range(0, len(jobs), batch):
                chunk = jobs[start : start + batch]
                logps_list = _batched_forward_logprobs(
                    model, chunk, device, model_lock=model_lock
                )
                for i, logps in enumerate(logps_list):
                    pending.append(cpu_ex.submit(_run_pcfg, start + i, logps))
                # Bound in-flight CPU work so chart RAM stays limited.
                if len(pending) >= n_workers * 2:
                    for fut in as_completed(pending):
                        fut.result()
                    pending.clear()
            for fut in as_completed(pending):
                fut.result()
    finally:
        # Drop cloned step inputs promptly (can be large for long traces).
        for job in jobs:
            job.input_ids = None  # type: ignore[assignment]
            job.attention = None  # type: ignore[assignment]
        if owns_pool and engine is not None and hasattr(engine, "clear_cache"):
            engine.clear_cache()

    ordered = [m for m in step_metrics if m is not None]
    ordered.sort(key=lambda m: m.step)
    Mt_total = sum(m.Mt for m in ordered)
    Jt_total = sum(m.Jt for m in ordered)
    # Token-weighted NLL (same as sequential path).
    oracle_nll_sum = 0.0
    model_nll_sum = 0.0
    n_tokens = 0
    for m in ordered:
        # Recover sums from per-step means × n_selected.
        oracle_nll_sum += float(m.oracle_nll) * int(m.n_selected)
        model_nll_sum += float(m.model_nll) * int(m.n_selected)
        n_tokens += int(m.n_selected)
    denom = max(n_tokens, 1)
    oracle_mean = oracle_nll_sum / denom
    model_mean = model_nll_sum / denom
    return OracleSampleMetrics(
        Mt_total=Mt_total,
        Jt_total=Jt_total,
        oracle_nll=oracle_mean,
        model_nll=model_mean,
        oracle_ppl=nll_to_ppl(oracle_mean),
        model_ppl=nll_to_ppl(model_mean),
        steps=ordered,
    )

# --- perplexity aggregation ---


@dataclass
class SamplePerplexity:
    model_commit_nll: float
    model_commit_ppl: float
    oracle_nll: float = float("nan")
    oracle_ppl: float = float("nan")
    Mt_total: float = float("nan")
    Jt_total: float = float("nan")
    n_commits: int = 0


@dataclass
class TaskPerplexity:
    model_commit_nll: float
    model_commit_ppl: float
    oracle_nll: float
    oracle_ppl: float
    Mt_total: float
    Jt_total: float
    n_samples: int
    sample_model_nll: List[float] = field(default_factory=list)
    sample_oracle_nll: List[float] = field(default_factory=list)
    sample_Mt: List[float] = field(default_factory=list)
    sample_Jt: List[float] = field(default_factory=list)
    mt_trend: List[float] = field(default_factory=list)
    jt_trend: List[float] = field(default_factory=list)
    # Optional in-memory sample objects (usually empty; flat tables preferred).
    sample_oracle: List[Optional[OracleSampleMetrics]] = field(default_factory=list)
    # Flat oracle tables accumulated during replay (compact; for oracle.npz).
    oracle_tables: Dict[str, list] = field(default_factory=dict)
    jt_exact_max: int = 0
    jt_markov_order: int = 0

    def to_dict(self) -> dict:
        n_tok = len(self.oracle_tables.get("tok_sample_idx") or [])
        n_step = len(self.oracle_tables.get("step_sample_idx") or [])
        return {
            "model_commit_nll": self.model_commit_nll,
            "model_commit_ppl": self.model_commit_ppl,
            "oracle_nll": self.oracle_nll,
            "oracle_ppl": self.oracle_ppl,
            "Mt_total": self.Mt_total,
            "Jt_total": self.Jt_total,
            "n_samples": self.n_samples,
            "sample_model_nll": self.sample_model_nll,
            "sample_oracle_nll": self.sample_oracle_nll,
            "sample_Mt": self.sample_Mt,
            "sample_Jt": self.sample_Jt,
            "mt_trend": self.mt_trend,
            "jt_trend": self.jt_trend,
            "jt_exact_max": self.jt_exact_max,
            "jt_markov_order": self.jt_markov_order,
            "n_oracle_steps": n_step,
            "n_oracle_tokens": n_tok,
        }


def empty_task_perplexity(*, n_samples: int = 0) -> TaskPerplexity:
    """A perplexity row filled with NaN."""
    nan = float("nan")
    return TaskPerplexity(
        model_commit_nll=nan,
        model_commit_ppl=nan,
        oracle_nll=nan,
        oracle_ppl=nan,
        Mt_total=nan,
        Jt_total=nan,
        n_samples=int(n_samples),
    )

def _compute_sample_perplexity_inner(
    trace_dict: dict,
    *,
    engine=None,
    vocab=None,
    model=None,
    device=None,
    replay_oracle: bool = False,
    forward_batch_size: int = 8,
    step_workers: Optional[int] = None,
    engine_pool: Optional[_EnginePool] = None,
    model_lock: Optional[threading.Lock] = None,
    trace_name: Union[str, Path, None] = None,
    jt_exact_max: Optional[int] = None,
    jt_markov_order: Optional[int] = None,
    jt_large_set: Optional[int] = None,
):
    """Perplexity for one sample, with oracle metrics when replay is on."""
    meta = trace_dict.get("metadata") or {}
    ref_ids = meta.get("reference_token_ids")
    n_commits = len(extract_unmask_events(trace_dict, trace_name=trace_name))

    if replay_oracle and model is not None and vocab is not None:
        # Full Mt/Jt replay also yields oracle/model NLL — skip extra chart walk.
        oracle = replay_oracle_mt_jt(
            model,
            trace_dict,
            engine,
            vocab,
            device or "cpu",
            forward_batch_size=forward_batch_size,
            step_workers=step_workers,
            engine_pool=engine_pool,
            model_lock=model_lock,
            trace_name=trace_name,
            jt_exact_max=jt_exact_max,
            jt_markov_order=jt_markov_order,
            jt_large_set=jt_large_set,
        )
        return SamplePerplexity(
            model_commit_nll=oracle.model_nll,
            model_commit_ppl=nll_to_ppl(oracle.model_nll),
            oracle_nll=oracle.oracle_nll,
            oracle_ppl=nll_to_ppl(oracle.oracle_nll),
            Mt_total=oracle.Mt_total,
            Jt_total=oracle.Jt_total,
            n_commits=n_commits,
        ), oracle

    model_nll = trace_model_commit_nll(
        trace_dict, ref_ids, trace_name=trace_name
    )
    # Oracle NLL / Mt / Jt only come from --replay-oracle (or cached oracle.npz).
    return SamplePerplexity(
        model_commit_nll=model_nll,
        model_commit_ppl=nll_to_ppl(model_nll),
        oracle_nll=float("nan"),
        oracle_ppl=float("nan"),
        Mt_total=float("nan"),
        Jt_total=float("nan"),
        n_commits=n_commits,
    ), None


def compute_sample_perplexity(
    trace_dict: dict,
    *,
    engine=None,
    vocab=None,
    model=None,
    device=None,
    replay_oracle: bool = False,
    forward_batch_size: int = 8,
    step_workers: Optional[int] = None,
    trace_name: Union[str, Path, None] = None,
    jt_exact_max: Optional[int] = None,
    jt_markov_order: Optional[int] = None,
    jt_large_set: Optional[int] = None,
) -> SamplePerplexity:
    """Commit perplexity for one sample, plus oracle fields when replay is on."""
    sp, _oracle = _compute_sample_perplexity_inner(
        trace_dict,
        engine=engine,
        vocab=vocab,
        model=model,
        device=device,
        replay_oracle=replay_oracle,
        forward_batch_size=forward_batch_size,
        step_workers=step_workers,
        trace_name=trace_name,
        jt_exact_max=jt_exact_max,
        jt_markov_order=jt_markov_order,
        jt_large_set=jt_large_set,
    )
    return sp


def compute_task_perplexity(
    trace_path: Union[str, Path],
    *,
    engine=None,
    vocab=None,
    model=None,
    device=None,
    replay_oracle: bool = False,
    max_samples: Optional[int] = None,
    verbose: bool = True,
    oracle_workers: int = 1,
    step_workers: Optional[int] = None,
    forward_batch_size: int = 8,
    jt_exact_max: Optional[int] = None,
    jt_markov_order: Optional[int] = None,
    jt_large_set: Optional[int] = None,
) -> TaskPerplexity:
    """Mean perplexity and oracle metrics over one trace file."""
    from traces.trace_npz import iter_npz_traces

    _ensure_pcfg_path()
    # PCFG stack only needed for live Mt/Jt replay (not for model-commit PPL).
    if replay_oracle and (engine is None or vocab is None):
        try:
            from pcfg_data import build_stack

            _g, _cnf, engine, vocab = build_stack()
        except Exception:
            engine = None
            vocab = None

    # Indexed by NPZ sample order (scalars only; token detail → oracle_tables).
    by_sample: dict[int, SamplePerplexity] = {}
    oracle_tables: Dict[str, list] = {
        "step_sample_idx": [],
        "step": [],
        "step_Mt": [],
        "step_Jt": [],
        "step_oracle_nll": [],
        "step_model_nll": [],
        "step_n_selected": [],
        "tok_sample_idx": [],
        "tok_step": [],
        "tok_gen_pos": [],
        "tok_content_pos": [],
        "tok_sel": [],
        "tok_ref": [],
        "tok_mt_kl": [],
        "tok_oracle_nll": [],
        "tok_model_nll": [],
    }
    mt_sum: List[float] = []
    mt_count: List[int] = []
    jt_sum: List[float] = []
    jt_count: List[int] = []
    n_steps = 0
    trend_lock = threading.Lock()

    def _ensure_trend_len(t_max: int) -> None:
        nonlocal mt_sum, mt_count, jt_sum, jt_count, n_steps
        if t_max <= n_steps:
            return
        pad = t_max - n_steps
        mt_sum.extend([0.0] * pad)
        mt_count.extend([0] * pad)
        jt_sum.extend([0.0] * pad)
        jt_count.extend([0] * pad)
        n_steps = t_max

    def _fold(
        sample_idx: int,
        sp: SamplePerplexity,
        oracle: Optional[OracleSampleMetrics],
        n_trace_steps: int,
    ) -> None:
        with trend_lock:
            by_sample[int(sample_idx)] = sp
            if oracle is not None:
                si = int(sample_idx)
                for sm in oracle.steps:
                    t = int(sm.step)
                    _ensure_trend_len(t + 1)
                    if math.isfinite(sm.Mt):
                        mt_sum[t] += float(sm.Mt)
                        mt_count[t] += 1
                    if math.isfinite(sm.Jt):
                        jt_sum[t] += float(sm.Jt)
                        jt_count[t] += 1
                    oracle_tables["step_sample_idx"].append(si)
                    oracle_tables["step"].append(t)
                    oracle_tables["step_Mt"].append(float(sm.Mt))
                    oracle_tables["step_Jt"].append(float(sm.Jt))
                    oracle_tables["step_oracle_nll"].append(float(sm.oracle_nll))
                    oracle_tables["step_model_nll"].append(float(sm.model_nll))
                    oracle_tables["step_n_selected"].append(int(sm.n_selected))
                    for tok in sm.tokens:
                        oracle_tables["tok_sample_idx"].append(si)
                        oracle_tables["tok_step"].append(t)
                        oracle_tables["tok_gen_pos"].append(int(tok.gen_pos))
                        oracle_tables["tok_content_pos"].append(int(tok.content_pos))
                        oracle_tables["tok_sel"].append(int(tok.sel_tok))
                        oracle_tables["tok_ref"].append(int(tok.ref_tok))
                        oracle_tables["tok_mt_kl"].append(float(tok.mt_kl))
                        oracle_tables["tok_oracle_nll"].append(float(tok.oracle_nll))
                        oracle_tables["tok_model_nll"].append(float(tok.model_nll))
                # Drop heavy per-step objects; flat tables keep the detail.
                oracle.steps.clear()
            else:
                _ensure_trend_len(n_trace_steps)

    n_sample_workers = max(1, int(oracle_workers))
    n_step_workers = max(
        1, int(step_workers if step_workers is not None else _default_step_workers())
    )
    # When many samples run concurrently, keep per-sample step parallelism modest.
    if n_sample_workers > 1:
        n_step_workers = max(1, min(n_step_workers, 2))

    do_replay = bool(replay_oracle and model is not None and vocab is not None)
    pool: Optional[_EnginePool] = None
    model_lock: Optional[threading.Lock] = None
    k_exact = max(2, int(jt_exact_max if jt_exact_max is not None else default_jt_exact_max()))
    m_order = max(1, int(jt_markov_order if jt_markov_order is not None else default_jt_markov_order()))
    large = int(jt_large_set if jt_large_set is not None else default_jt_large_set())
    if do_replay:
        pool_size = n_sample_workers * n_step_workers
        pool = _EnginePool(pool_size)
        model_lock = threading.Lock()
        if verbose:
            logger.info(
                "Oracle replay: samples=%d step_workers=%d forward_batch=%d "
                "engine_pool=%d jt_exact_max=%d jt_markov=%d jt_large_set=%d",
                n_sample_workers,
                n_step_workers,
                forward_batch_size,
                pool_size,
                k_exact,
                min(m_order, k_exact - 1),
                large,
            )

    def _process_one(sample_idx: int, tr: dict):
        return sample_idx, _compute_sample_perplexity_inner(
            tr,
            engine=engine,
            vocab=vocab,
            model=model,
            device=device,
            replay_oracle=do_replay,
            forward_batch_size=forward_batch_size,
            step_workers=n_step_workers,
            engine_pool=pool,
            model_lock=model_lock,
            trace_name=trace_path,
            jt_exact_max=k_exact,
            jt_markov_order=m_order,
            jt_large_set=large,
        ), len(tr.get("step_traces") or [])

    n_done = 0
    chunk: list[tuple[int, dict]] = []

    def _flush_chunk(items: list[tuple[int, dict]]) -> None:
        nonlocal n_done
        if not items:
            return
        if n_sample_workers == 1 or not do_replay:
            for si, tr in items:
                sample_idx, (sp, oracle), n_ts = _process_one(si, tr)
                _fold(sample_idx, sp, oracle, n_ts)
                n_done += 1
                if verbose and n_done % 10 == 0:
                    label = "Oracle replay" if do_replay else "Perplexity"
                    logger.info("%s: %d samples ...", label, n_done)
            return
        with ThreadPoolExecutor(max_workers=n_sample_workers) as ex:
            futs = [ex.submit(_process_one, si, tr) for si, tr in items]
            for fut in as_completed(futs):
                sample_idx, (sp, oracle), n_ts = fut.result()
                _fold(sample_idx, sp, oracle, n_ts)
                n_done += 1
                if verbose and n_done % 10 == 0:
                    label = "Oracle replay" if do_replay else "Perplexity"
                    logger.info("%s: %d samples ...", label, n_done)

    for i, tr in enumerate(iter_npz_traces(trace_path)):
        if max_samples is not None and i >= max_samples:
            break
        chunk.append((i, tr))
        if len(chunk) >= max(n_sample_workers, 1):
            _flush_chunk(chunk)
            chunk = []
    _flush_chunk(chunk)

    if verbose and do_replay:
        logger.info("Oracle replay: %d samples total.", n_done)

    def _mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    mt_trend: List[float] = []
    jt_trend: List[float] = []
    if n_steps > 0 and (any(mt_count) or any(jt_count)):
        mt_trend = [
            (mt_sum[t] / mt_count[t]) if mt_count[t] else float("nan")
            for t in range(n_steps)
        ]
        jt_trend = [
            (jt_sum[t] / jt_count[t]) if jt_count[t] else float("nan")
            for t in range(n_steps)
        ]

    ordered_idx = sorted(by_sample)
    model_nlls = [by_sample[i].model_commit_nll for i in ordered_idx]
    oracle_nlls = [by_sample[i].oracle_nll for i in ordered_idx]
    sample_Mt = [by_sample[i].Mt_total for i in ordered_idx]
    sample_Jt = [by_sample[i].Jt_total for i in ordered_idx]

    model_mean = _mean([x for x in model_nlls if math.isfinite(x)])
    oracle_mean = _mean([x for x in oracle_nlls if math.isfinite(x)])
    return TaskPerplexity(
        model_commit_nll=model_mean,
        model_commit_ppl=nll_to_ppl(model_mean),
        oracle_nll=oracle_mean,
        oracle_ppl=nll_to_ppl(oracle_mean),
        Mt_total=_mean([x for x in sample_Mt if math.isfinite(x)]),
        Jt_total=_mean([x for x in sample_Jt if math.isfinite(x)]),
        n_samples=len(ordered_idx),
        sample_model_nll=model_nlls,
        sample_oracle_nll=oracle_nlls,
        sample_Mt=sample_Mt,
        sample_Jt=sample_Jt,
        mt_trend=mt_trend,
        jt_trend=jt_trend,
        sample_oracle=[],
        oracle_tables=oracle_tables if do_replay else {},
        jt_exact_max=k_exact if do_replay else 0,
        jt_markov_order=min(m_order, k_exact - 1) if do_replay else 0,
    )


def save_oracle_artifacts(
    path: Union[str, Path],
    ppl: TaskPerplexity,
    *,
    stem: str = "",
) -> Path:
    """Write the oracle replay to an NPZ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = int(ppl.n_samples)
    sample_oracle_nll = np.asarray(ppl.sample_oracle_nll, dtype=np.float64)
    sample_model_nll = np.asarray(ppl.sample_model_nll, dtype=np.float64)
    sample_Mt = np.asarray(ppl.sample_Mt, dtype=np.float64)
    sample_Jt = np.asarray(ppl.sample_Jt, dtype=np.float64)
    sample_oracle_ppl = np.asarray(
        [nll_to_ppl(float(x)) for x in sample_oracle_nll], dtype=np.float64
    )
    sample_model_ppl = np.asarray(
        [nll_to_ppl(float(x)) for x in sample_model_nll], dtype=np.float64
    )

    tables = ppl.oracle_tables or {}
    # Fallback: rebuild from sample_oracle objects if tables were not filled.
    if not tables.get("step_sample_idx") and ppl.sample_oracle:
        tables = {
            "step_sample_idx": [],
            "step": [],
            "step_Mt": [],
            "step_Jt": [],
            "step_oracle_nll": [],
            "step_model_nll": [],
            "step_n_selected": [],
            "tok_sample_idx": [],
            "tok_step": [],
            "tok_gen_pos": [],
            "tok_content_pos": [],
            "tok_sel": [],
            "tok_ref": [],
            "tok_mt_kl": [],
            "tok_oracle_nll": [],
            "tok_model_nll": [],
        }
        for si, oracle in enumerate(ppl.sample_oracle):
            if oracle is None:
                continue
            for sm in oracle.steps:
                tables["step_sample_idx"].append(si)
                tables["step"].append(int(sm.step))
                tables["step_Mt"].append(float(sm.Mt))
                tables["step_Jt"].append(float(sm.Jt))
                tables["step_oracle_nll"].append(float(sm.oracle_nll))
                tables["step_model_nll"].append(float(sm.model_nll))
                tables["step_n_selected"].append(int(sm.n_selected))
                for t in sm.tokens:
                    tables["tok_sample_idx"].append(si)
                    tables["tok_step"].append(int(sm.step))
                    tables["tok_gen_pos"].append(int(t.gen_pos))
                    tables["tok_content_pos"].append(int(t.content_pos))
                    tables["tok_sel"].append(int(t.sel_tok))
                    tables["tok_ref"].append(int(t.ref_tok))
                    tables["tok_mt_kl"].append(float(t.mt_kl))
                    tables["tok_oracle_nll"].append(float(t.oracle_nll))
                    tables["tok_model_nll"].append(float(t.model_nll))

    np.savez_compressed(
        path,
        stem=np.asarray(stem),
        n_samples=np.asarray(n, dtype=np.int64),
        jt_exact_max=np.asarray(int(ppl.jt_exact_max), dtype=np.int64),
        jt_markov_order=np.asarray(int(ppl.jt_markov_order), dtype=np.int64),
        oracle_nll=np.asarray(ppl.oracle_nll, dtype=np.float64),
        oracle_ppl=np.asarray(ppl.oracle_ppl, dtype=np.float64),
        model_commit_nll=np.asarray(ppl.model_commit_nll, dtype=np.float64),
        model_commit_ppl=np.asarray(ppl.model_commit_ppl, dtype=np.float64),
        Mt_total=np.asarray(ppl.Mt_total, dtype=np.float64),
        Jt_total=np.asarray(ppl.Jt_total, dtype=np.float64),
        sample_oracle_nll=sample_oracle_nll,
        sample_model_nll=sample_model_nll,
        sample_oracle_ppl=sample_oracle_ppl,
        sample_model_ppl=sample_model_ppl,
        sample_Mt=sample_Mt,
        sample_Jt=sample_Jt,
        mt_trend=np.asarray(ppl.mt_trend, dtype=np.float64),
        jt_trend=np.asarray(ppl.jt_trend, dtype=np.float64),
        step_sample_idx=np.asarray(tables.get("step_sample_idx", []), dtype=np.int32),
        step=np.asarray(tables.get("step", []), dtype=np.int32),
        step_Mt=np.asarray(tables.get("step_Mt", []), dtype=np.float64),
        step_Jt=np.asarray(tables.get("step_Jt", []), dtype=np.float64),
        step_oracle_nll=np.asarray(tables.get("step_oracle_nll", []), dtype=np.float64),
        step_model_nll=np.asarray(tables.get("step_model_nll", []), dtype=np.float64),
        step_n_selected=np.asarray(tables.get("step_n_selected", []), dtype=np.int32),
        tok_sample_idx=np.asarray(tables.get("tok_sample_idx", []), dtype=np.int32),
        tok_step=np.asarray(tables.get("tok_step", []), dtype=np.int32),
        tok_gen_pos=np.asarray(tables.get("tok_gen_pos", []), dtype=np.int32),
        tok_content_pos=np.asarray(tables.get("tok_content_pos", []), dtype=np.int32),
        tok_sel=np.asarray(tables.get("tok_sel", []), dtype=np.int32),
        tok_ref=np.asarray(tables.get("tok_ref", []), dtype=np.int32),
        tok_mt_kl=np.asarray(tables.get("tok_mt_kl", []), dtype=np.float64),
        tok_oracle_nll=np.asarray(tables.get("tok_oracle_nll", []), dtype=np.float64),
        tok_model_nll=np.asarray(tables.get("tok_model_nll", []), dtype=np.float64),
    )
    return path


def load_oracle_artifacts(path: Union[str, Path]) -> dict[str, np.ndarray]:
    """Arrays stored in an oracle NPZ."""
    path = Path(path)
    with np.load(path, allow_pickle=True) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def _oracle_scalar(arr, default: float = float("nan")) -> float:
    a = np.asarray(arr)
    if a.size == 0:
        return float(default)
    return float(np.asarray(a).reshape(-1)[0])


def _oracle_float_list(arr) -> List[float]:
    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    return [float(x) for x in a.tolist()]


def _oracle_stem_from_array(arr) -> str:
    a = np.asarray(arr)
    if a.size == 0:
        return ""
    raw = a.reshape(-1)[0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    text = str(raw).strip()
    if not text:
        return ""
    return Path(text).stem


def task_perplexity_from_oracle_npz(path: Union[str, Path]) -> TaskPerplexity:
    """Task perplexity rebuilt from a saved oracle NPZ."""
    arts = load_oracle_artifacts(path)
    n_samples = int(_oracle_scalar(arts.get("n_samples", 0), default=0))
    return TaskPerplexity(
        model_commit_nll=_oracle_scalar(arts.get("model_commit_nll")),
        model_commit_ppl=_oracle_scalar(arts.get("model_commit_ppl")),
        oracle_nll=_oracle_scalar(arts.get("oracle_nll")),
        oracle_ppl=_oracle_scalar(arts.get("oracle_ppl")),
        Mt_total=_oracle_scalar(arts.get("Mt_total")),
        Jt_total=_oracle_scalar(arts.get("Jt_total")),
        n_samples=n_samples,
        sample_model_nll=_oracle_float_list(arts.get("sample_model_nll", [])),
        sample_oracle_nll=_oracle_float_list(arts.get("sample_oracle_nll", [])),
        sample_Mt=_oracle_float_list(arts.get("sample_Mt", [])),
        sample_Jt=_oracle_float_list(arts.get("sample_Jt", [])),
        mt_trend=_oracle_float_list(arts.get("mt_trend", [])),
        jt_trend=_oracle_float_list(arts.get("jt_trend", [])),
        jt_exact_max=int(_oracle_scalar(arts.get("jt_exact_max", 0), default=0)),
        jt_markov_order=int(
            _oracle_scalar(arts.get("jt_markov_order", 0), default=0)
        ),
    )


def _oracle_keys_for_path(path: Path, *, content_stem: str = "") -> set[str]:
    """Stems this oracle file can match."""
    keys: set[str] = set()
    name = path.stem
    if name == "oracle":
        pass
    elif name.endswith("_oracle"):
        keys.add(name[: -len("_oracle")])
    elif name.startswith("oracle_"):
        keys.add(name[len("oracle_") :])
    else:
        keys.add(name)
    if path.name == "oracle.npz" and path.parent.name not in ("", "."):
        keys.add(path.parent.name)
    if content_stem:
        keys.add(Path(content_stem).stem)
    return {k for k in keys if k}


def index_oracle_artifacts(root: Union[str, Path]) -> dict[str, Path]:
    """Trace stem to oracle NPZ, for one file or a directory of them."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Oracle path not found: {root}")

    paths: List[Path]
    if root.is_file():
        if root.suffix.lower() != ".npz":
            raise ValueError(f"Oracle path must be .npz or a directory, got {root}")
        paths = [root]
    else:
        paths = sorted(p for p in root.rglob("*.npz") if p.is_file())

    index: dict[str, Path] = {}
    for path in paths:
        content_stem = ""
        try:
            arts = load_oracle_artifacts(path)
            content_stem = _oracle_stem_from_array(arts.get("stem", []))
        except Exception as e:
            logger.warning("Skipping unreadable oracle artifact %s: %s", path, e)
            continue
        for key in _oracle_keys_for_path(path, content_stem=content_stem):
            if key in index and index[key].resolve() != path.resolve():
                logger.warning(
                    "Duplicate oracle stem %r: keeping %s (ignore %s)",
                    key,
                    index[key],
                    path,
                )
                continue
            index[key] = path
    return index


def resolve_oracle_npz(
    oracle_root: Optional[Union[str, Path]],
    stem: str,
    *,
    index: Optional[dict[str, Path]] = None,
) -> Optional[Path]:
    """Oracle NPZ for this stem, or None."""
    if oracle_root is None and index is None:
        return None
    lookup = index if index is not None else index_oracle_artifacts(oracle_root)  # type: ignore[arg-type]
    stem = Path(stem).stem
    return lookup.get(stem)

