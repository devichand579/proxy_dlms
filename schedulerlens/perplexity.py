"""Perplexity and oracle metrics from generation traces."""

from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LLADA_PCFG_DIR = _REPO_ROOT / "llada-pcfg"
_LLADA_PCFG_SCRIPTS = _LLADA_PCFG_DIR / "scripts"
for _path in (_REPO_ROOT, _LLADA_PCFG_DIR, _LLADA_PCFG_SCRIPTS):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)


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
        }


def _ppl(nll: float) -> float:
    if not math.isfinite(nll):
        return float("nan")
    return math.exp(min(float(nll), 20.0))


def compute_sample_perplexity(
    trace_dict: dict,
    *,
    engine=None,
    vocab=None,
    model=None,
    device=None,
    replay_oracle: bool = False,
) -> SamplePerplexity:
    """Per-sample commit NLL/PPL; oracle fields when PCFG metadata + engine present."""
    from schedulerlens.pcfg_oracle import (
        extract_unmask_events,
        replay_oracle_mt_jt,
        trace_model_commit_nll,
        trace_oracle_nll,
    )

    meta = trace_dict.get("metadata") or {}
    ref_ids = meta.get("reference_token_ids")
    token_strings = meta.get("reference_tokens")

    model_nll = trace_model_commit_nll(trace_dict, ref_ids)
    n_commits = len(extract_unmask_events(trace_dict))

    oracle_nll = float("nan")
    Mt_total = float("nan")
    Jt_total = float("nan")

    if engine is not None and token_strings is not None:
        oracle_nll = trace_oracle_nll(engine, trace_dict, token_strings)

    if replay_oracle and model is not None and engine is not None and vocab is not None:
        try:
            oracle = replay_oracle_mt_jt(model, trace_dict, engine, vocab, device or "cpu")
            Mt_total = oracle.Mt_total
            Jt_total = oracle.Jt_total
            oracle_nll = oracle.oracle_nll
            model_nll = oracle.model_nll
        except Exception as e:
            logger.warning("replay_oracle_mt_jt failed: %s", e)

    return SamplePerplexity(
        model_commit_nll=model_nll,
        model_commit_ppl=_ppl(model_nll),
        oracle_nll=oracle_nll,
        oracle_ppl=_ppl(oracle_nll),
        Mt_total=Mt_total,
        Jt_total=Jt_total,
        n_commits=n_commits,
    )


def compute_task_perplexity(
    trace_path: Union[str, Path],
    *,
    engine=None,
    vocab=None,
    model=None,
    device=None,
    replay_oracle: bool = False,
    max_samples: Optional[int] = None,
) -> TaskPerplexity:
    """Aggregate perplexity / oracle metrics over a trace file."""
    from schedulerlens.metrics import _iter_traces

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

    for i, tr in enumerate(_iter_traces(trace_path)):
        if max_samples is not None and i >= max_samples:
            break
        sp = compute_sample_perplexity(
            tr,
            engine=engine,
            vocab=vocab,
            model=model,
            device=device,
            replay_oracle=replay_oracle,
        )
        if math.isfinite(sp.model_commit_nll):
            model_nlls.append(sp.model_commit_nll)
        if math.isfinite(sp.oracle_nll):
            oracle_nlls.append(sp.oracle_nll)
        if math.isfinite(sp.Mt_total):
            mt_vals.append(sp.Mt_total)
        if math.isfinite(sp.Jt_total):
            jt_vals.append(sp.Jt_total)

    def _mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    model_mean = _mean(model_nlls)
    oracle_mean = _mean(oracle_nlls)
    return TaskPerplexity(
        model_commit_nll=model_mean,
        model_commit_ppl=_ppl(model_mean),
        oracle_nll=oracle_mean,
        oracle_ppl=_ppl(oracle_mean),
        Mt_total=_mean(mt_vals),
        Jt_total=_mean(jt_vals),
        n_samples=len(model_nlls),
        sample_model_nll=model_nlls,
        sample_oracle_nll=oracle_nlls,
    )
