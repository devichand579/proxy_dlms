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
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Sequence, Union

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
class OracleStepMetrics:
    step: int
    Mt: float
    Jt: float
    oracle_nll: float
    model_nll: float
    n_selected: int


@dataclass
class OracleSampleMetrics:
    Mt_total: float
    Jt_total: float
    oracle_nll: float
    model_nll: float
    oracle_ppl: float
    model_ppl: float
    steps: list[OracleStepMetrics]


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
    """Return (step, gen_pos, token_id) for each commit in a trace.

    Positions are gen-relative (0 .. gen_length-1), matching trace fields.
    Fail-hard on unrepaired selection/mask/phantom issues (same as proxies).
    """
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
    """Resolve PCFG content / prompt / gen layout from trace metadata."""
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
    """Map gen-relative index → content index, or None if outside content (e.g. EOS pad)."""
    content_pos = int(n_visible) + int(gen_pos)
    if content_pos < 0 or content_pos >= int(n_content):
        return None
    return content_pos


def model_logprobs_from_logits(logits, positions: Iterable[int]) -> dict[int, np.ndarray]:
    """Softmax log-probs over vocab for each sequence position."""
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
    """Proxy NLL from trace confidences at commit time (gen-relative indices)."""
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
    """Exact oracle NLL -log q(x_i | prefix∪prior commits) along the unmask order.

    Uses teacher-forced reference tokens. Gen indices map to content via n_visible.
    """
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
    """Left-pad visible prefix + masked gen region (matches pcfg_eval batching)."""
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
    """One unmask step with teacher-forced pre-step state (known from the trace)."""

    step: int
    n_content: int
    S_content: list[int]
    content_events: list[tuple[int, int, int]]  # gen_pos, cpos, sel_tok
    model_positions: list[int]
    committed: dict[int, int]
    tokens: list[str]
    tmap: dict
    input_ids: Any  # CPU torch.LongTensor [1, L]
    attention: Any


class _EnginePool:
    """Thread-safe pool of PCFGEngine instances (mutable chart cache is not shareable)."""

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
) -> list[_StepJob]:
    """Build per-step jobs; each has independent pre-step committed + input_ids."""
    import torch

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

    content_ids = vocab.encode(tokens)
    # Build on CPU; batching moves chunks to the model device.
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
                input_ids=input_ids.clone(),
                attention=attention.clone(),
            )
        )
        # Advance teacher-forced state for the next step's snapshot.
        for gen_pos, cpos, _sel in content_events:
            ref_tid = tmap[tokens[cpos]]
            committed[int(cpos)] = ref_tid
            input_ids[0, int(prompt_len + gen_pos)] = int(content_ids[cpos])

    return jobs


def _batched_forward_logprobs(
    model,
    jobs: Sequence[_StepJob],
    device,
    *,
    model_lock: Optional[threading.Lock] = None,
) -> list[dict[int, np.ndarray]]:
    """One batched model forward for a chunk of step jobs."""
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


def _pcfg_metrics_for_step(engine, job: _StepJob, logps: dict[int, np.ndarray], kl) -> OracleStepMetrics:
    """Mt / Jt / NLLs for one precomputed step (uses a private engine)."""
    q_batch = engine.marginals_batch(job.n_content, job.committed, job.S_content)
    jt = float(engine.Jt(job.n_content, job.committed, job.S_content))
    if hasattr(engine, "clear_cache"):
        engine.clear_cache()

    step_mt = 0.0
    step_onll = 0.0
    step_mnll = 0.0
    for i, (_gen_pos, cpos, sel_tok) in enumerate(job.content_events):
        model_pos = int(job.model_positions[i])
        q = q_batch.get(int(cpos), np.zeros(engine.V))
        p = np.exp(logps[int(model_pos)])
        if p.shape[0] != q.shape[0]:
            p = p[: q.shape[0]]
            p = p / max(float(p.sum()), EPS)
        step_mt += kl(q, p)
        ref_tid = job.tmap[job.tokens[cpos]]
        step_onll += -math.log(float(q[ref_tid]) + EPS)
        step_mnll += model_nll_at_commit(logps, model_pos, int(sel_tok))

    n_ev = max(len(job.content_events), 1)
    return OracleStepMetrics(
        step=job.step,
        Mt=step_mt,
        Jt=jt,
        oracle_nll=step_onll / n_ev,
        model_nll=step_mnll / n_ev,
        n_selected=len(job.content_events),
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
) -> OracleSampleMetrics:
    """Re-forward model at each trace step; compute exact Mt and Jt vs PCFG oracle.

    Parallelism (all optional):
    - Batched GPU/CPU model forwards over precomputed step inputs.
    - Thread pool over steps for PCFG Mt/Jt (each thread borrows an engine).
    - Pipeline: submit CPU work for chunk t while GPU runs chunk t+1.
    """
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

    n_workers = max(1, int(step_workers if step_workers is not None else _default_step_workers()))
    batch = max(1, int(forward_batch_size))
    owns_pool = engine_pool is None
    pool = engine_pool or _EnginePool(n_workers)

    step_metrics: list[Optional[OracleStepMetrics]] = [None] * len(jobs)

    def _run_pcfg(idx: int, logps: dict[int, np.ndarray]) -> None:
        eng = pool.borrow()
        try:
            step_metrics[idx] = _pcfg_metrics_for_step(eng, jobs[idx], logps, kl)
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
    mt_trend: List[float] = field(default_factory=list)
    jt_trend: List[float] = field(default_factory=list)
    # Per-sample step lists from replay (OracleStepMetrics).
    sample_oracle_steps: List[list] = field(default_factory=list)

    def to_dict(self) -> dict:
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
            "mt_trend": self.mt_trend,
            "jt_trend": self.jt_trend,
        }


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
):
    """Return (SamplePerplexity, optional OracleSampleMetrics)."""
    meta = trace_dict.get("metadata") or {}
    ref_ids = meta.get("reference_token_ids")
    token_strings = meta.get("reference_tokens")
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
    oracle_nll = float("nan")
    if engine is not None and token_strings is not None:
        oracle_nll = trace_oracle_nll(
            engine, trace_dict, token_strings, trace_name=trace_name
        )

    return SamplePerplexity(
        model_commit_nll=model_nll,
        model_commit_ppl=nll_to_ppl(model_nll),
        oracle_nll=oracle_nll,
        oracle_ppl=nll_to_ppl(oracle_nll),
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
) -> SamplePerplexity:
    """Per-sample commit NLL/PPL; oracle fields when PCFG metadata + engine present."""
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
) -> TaskPerplexity:
    """Aggregate perplexity / oracle metrics over a trace file.

    Parallelism when ``replay_oracle``:
    - ``forward_batch_size``: batched model forwards within a sample
    - ``step_workers``: parallel PCFG Mt/Jt across steps (engine pool)
    - ``oracle_workers``: parallel samples (shared model + lock; shared engine pool)
    """
    from traces.trace_npz import iter_npz_traces

    _ensure_pcfg_path()
    if engine is None or vocab is None:
        try:
            from pcfg_data import build_stack

            _g, _cnf, engine, vocab = build_stack()
        except Exception:
            engine = None
            vocab = None

    model_nlls: List[float] = []
    oracle_nlls: List[float] = []
    mt_vals: List[float] = []
    jt_vals: List[float] = []
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

    def _fold(sp: SamplePerplexity, oracle, n_trace_steps: int) -> None:
        with trend_lock:
            if math.isfinite(sp.model_commit_nll):
                model_nlls.append(sp.model_commit_nll)
            if math.isfinite(sp.oracle_nll):
                oracle_nlls.append(sp.oracle_nll)
            if math.isfinite(sp.Mt_total):
                mt_vals.append(sp.Mt_total)
            if math.isfinite(sp.Jt_total):
                jt_vals.append(sp.Jt_total)
            if oracle is not None:
                for sm in oracle.steps:
                    t = int(sm.step)
                    _ensure_trend_len(t + 1)
                    if math.isfinite(sm.Mt):
                        mt_sum[t] += float(sm.Mt)
                        mt_count[t] += 1
                    if math.isfinite(sm.Jt):
                        jt_sum[t] += float(sm.Jt)
                        jt_count[t] += 1
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
    if do_replay:
        pool_size = n_sample_workers * n_step_workers
        pool = _EnginePool(pool_size)
        model_lock = threading.Lock()
        if verbose:
            logger.info(
                "Oracle replay: samples=%d step_workers=%d forward_batch=%d engine_pool=%d",
                n_sample_workers,
                n_step_workers,
                forward_batch_size,
                pool_size,
            )

    def _process_one(tr: dict):
        return _compute_sample_perplexity_inner(
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
        ), len(tr.get("step_traces") or [])

    n_done = 0
    chunk: list[dict] = []

    def _flush_chunk(items: list[dict]) -> None:
        nonlocal n_done
        if not items:
            return
        if n_sample_workers == 1 or not do_replay:
            for tr in items:
                (sp, oracle), n_ts = _process_one(tr)
                _fold(sp, oracle, n_ts)
                n_done += 1
                if verbose and n_done % 10 == 0:
                    logger.info("Oracle replay: %d samples ...", n_done)
            return
        with ThreadPoolExecutor(max_workers=n_sample_workers) as ex:
            futs = [ex.submit(_process_one, tr) for tr in items]
            for fut in as_completed(futs):
                (sp, oracle), n_ts = fut.result()
                _fold(sp, oracle, n_ts)
                n_done += 1
                if verbose and n_done % 10 == 0:
                    logger.info("Oracle replay: %d samples ...", n_done)

    for i, tr in enumerate(iter_npz_traces(trace_path)):
        if max_samples is not None and i >= max_samples:
            break
        chunk.append(tr)
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

    model_mean = _mean(model_nlls)
    oracle_mean = _mean(oracle_nlls)
    return TaskPerplexity(
        model_commit_nll=model_mean,
        model_commit_ppl=nll_to_ppl(model_mean),
        oracle_nll=oracle_mean,
        oracle_ppl=nll_to_ppl(oracle_mean),
        Mt_total=_mean(mt_vals),
        Jt_total=_mean(jt_vals),
        n_samples=len(model_nlls),
        sample_model_nll=model_nlls,
        sample_oracle_nll=oracle_nlls,
        mt_trend=mt_trend,
        jt_trend=jt_trend,
        sample_oracle_steps=[],
    )

