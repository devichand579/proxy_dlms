"""
Evaluation metrics for PCFG / LLaDA training and suffix generation eval.

Training / validation (masked diffusion):
  - masked_perplexity = exp(loss)
  - make_val_example: deterministic validation mask draw

Suffix eval (test split, lm-eval-style prefix + masked suffix):
  - suffix_exact_match, suffix_structural_accuracy
  - build_prompt_suffix_eval_plan / build_prompt_suffix_batch

Trace/oracle metrics (Mt, Jt, commit NLL) live in schedulerlens.oracle.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from pcfg_data import ARCH_LEN, Vocab, make_train_example

# Terminals that fill lexical slots; syntax metric ignores these.
STRUCTURAL_SLOTS = frozenset({"id", "fld", "num", "flt"})


@dataclass
class PromptSuffixEvalSample:
    """One test row: fixed visible prefix + suffix to generate in ``gen_length`` slots."""

    sample_idx: int
    tokens: list[str]
    n_content: int
    n_visible: int
    ref_suffix_ids: list[int]


def prompt_visible_len(sample_idx: int, n_content: int, seed: int) -> int:
    """Deterministic visible prefix length in ``[1, max(1, n_content // 2)]``."""
    rng = random.Random(seed + sample_idx)
    hi = max(1, n_content // 2)
    return rng.randint(1, hi)


def build_prompt_suffix_eval_plan(
    token_strings: Sequence[Sequence[str]],
    vocab: Vocab,
    *,
    sample_start_idx: int = 0,
    seed: int = 0,
) -> list[PromptSuffixEvalSample]:
    """Precompute per-sample prompt splits (same for all schedulers/models with same seed)."""
    plan: list[PromptSuffixEvalSample] = []
    for offset, tokens in enumerate(token_strings):
        toks = list(tokens)
        n = len(toks)
        idx = sample_start_idx + offset
        n_vis = prompt_visible_len(idx, n, seed)
        ids = vocab.encode(toks)
        ref_suffix = ids[n_vis:] + [vocab.eos_id]
        plan.append(
            PromptSuffixEvalSample(
                sample_idx=idx,
                tokens=toks,
                n_content=n,
                n_visible=n_vis,
                ref_suffix_ids=ref_suffix,
            )
        )
    return plan


def build_prompt_suffix_batch(
    plan: Sequence[PromptSuffixEvalSample],
    vocab: Vocab,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Left-pad prompts (lm-eval style): visible tokens right-aligned before ``generate`` appends masks."""
    if not plan:
        raise ValueError("empty eval plan")
    max_pl = max((s.n_visible for s in plan), default=0)
    rows = []
    attn = []
    prompt_lens = []
    for sample in plan:
        n_vis = sample.n_visible
        ids = vocab.encode(sample.tokens[:n_vis])
        row = np.full(max_pl, vocab.pad_id, dtype=np.int64)
        mask = np.zeros(max_pl, dtype=np.int64)
        row[max_pl - n_vis :] = ids
        mask[max_pl - n_vis :] = 1
        rows.append(row)
        attn.append(mask)
        prompt_lens.append(n_vis)
    return (
        torch.tensor(np.stack(rows), dtype=torch.long),
        torch.tensor(np.stack(attn), dtype=torch.long),
        prompt_lens,
    )


def suffix_exact_match(
    generated_ids: Sequence[int],
    reference_suffix_ids: Sequence[int],
) -> bool:
    """True when generated suffix (content + eos) matches reference."""
    gen = list(generated_ids[: len(reference_suffix_ids)])
    ref = list(reference_suffix_ids)
    return gen == ref


def _structural_accuracy(
    generated_ids: Sequence[int],
    reference_ids: Sequence[int],
    vocab: Vocab,
    *,
    n_content: int,
) -> float:
    """Match rate on syntax tokens (excludes id/fld/num/flt slot fillers)."""
    matches = 0
    total = 0
    for i in range(n_content):
        ref_tok = vocab.itos[int(reference_ids[i])]
        if ref_tok in STRUCTURAL_SLOTS:
            continue
        total += 1
        if i < len(generated_ids) and int(generated_ids[i]) == int(reference_ids[i]):
            matches += 1
    return matches / max(total, 1)


def suffix_structural_accuracy(
    generated_ids: Sequence[int],
    reference_suffix_ids: Sequence[int],
    vocab: Vocab,
    *,
    suffix_tokens: Sequence[str],
) -> float:
    """Syntax-token match rate on the suffix only (excludes slot fillers)."""
    return _structural_accuracy(
        generated_ids,
        reference_suffix_ids,
        vocab,
        n_content=len(suffix_tokens),
    )


def masked_perplexity(loss: float) -> float:
    return math.exp(min(float(loss), 20.0))


def make_val_example(tokens, vocab, arch_len=ARCH_LEN, seed: int = 0):
    """Deterministic validation example: fixed mask draw from ``seed``."""
    return make_train_example(tokens, vocab, arch_len=arch_len, rng=random.Random(seed))
