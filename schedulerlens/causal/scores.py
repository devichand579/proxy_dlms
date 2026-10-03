"""Score bundles from a mask-predictor forward."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from schedulers.utils import (
    get_mask_by_confidence,
    get_mask_by_entropy,
    mask_eos_eot_logits,
)

# Score a few rows and positions at a time so the float32 vocab tensor stays small.
_SCORE_ROWS = 4
_SCORE_LEN = 32


@dataclass
class ScoreBundle:
    """Per-position scores on one sequence (full length, including the prompt)."""

    conf: np.ndarray
    entropy: np.ndarray
    pred: np.ndarray
    margin: np.ndarray
    eb_h: np.ndarray
    h_rank: np.ndarray

    def gen(self, prompt_len: int) -> "ScoreBundle":
        sl = slice(prompt_len, None)
        return ScoreBundle(
            conf=self.conf[sl],
            entropy=self.entropy[sl],
            pred=self.pred[sl],
            margin=self.margin[sl],
            eb_h=self.eb_h[sl],
            h_rank=self.h_rank[sl],
        )


def bundle_for_select(logits, *, drop_eos: bool, eos_id: int, eot_id):
    """Scores used to choose positions. LLaDA-1.5 drops EOS and EoT here."""
    if not drop_eos:
        return bundle_from_logits(logits)
    return bundle_from_logits(logits, drop_eos=True, eos_id=int(eos_id), eot_id=eot_id)


def bundle_from_logits(
    logits,
    *,
    drop_eos: bool = False,
    eos_id: int = 0,
    eot_id=None,
) -> list[ScoreBundle]:
    """Build one ``ScoreBundle`` per batch row from ``[B, L, V]`` logits.

    The vocabulary axis is reduced in small tiles, so the float32 log-softmax
    is never materialized for the whole tensor.
    """
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"logits must be a torch.Tensor, got {type(logits)}")
    if logits.ndim != 3:
        raise ValueError(f"logits must be [B, L, V], got {tuple(logits.shape)}")
    batches, length, vocab = logits.shape
    conf = np.empty((batches, length), dtype=np.float32)
    entropy = np.empty((batches, length), dtype=np.float32)
    pred = np.empty((batches, length), dtype=np.int64)
    margin = np.empty((batches, length), dtype=np.float32)
    eb_h = np.empty((batches, length), dtype=np.float32)
    h_rank = np.empty((batches, length), dtype=np.float32)
    top_k = min(2, int(vocab))
    for b0 in range(0, batches, _SCORE_ROWS):
        b1 = min(batches, b0 + _SCORE_ROWS)
        for t0 in range(0, length, _SCORE_LEN):
            t1 = min(length, t0 + _SCORE_LEN)
            piece = logits[b0:b1, t0:t1]
            if drop_eos:
                piece = piece.clone()
                mask_eos_eot_logits(piece, True, True, int(eos_id), eot_id)
            # Temperature 0 uses argmax of the logits. Confidence is p(argmax).
            pred_t = torch.argmax(piece, dim=-1)
            conf_t = get_mask_by_confidence(piece, pred_t)
            # Margin and the EB constraint use softmax, as in generate_EB and generate_topk.
            softmax_p = torch.softmax(piece.float(), dim=-1)
            eb_h_t = -(softmax_p * torch.log(softmax_p + 1e-9)).sum(-1)
            h_rank_t = get_mask_by_entropy(piece)
            top2 = torch.topk(softmax_p, k=top_k, dim=-1).values
            mar_t = top2[..., 0] if top_k == 1 else top2[..., 0] - top2[..., 1]
            # Shannon entropy for M̂ and Ĵ. A -inf logit has probability 0 and
            # log-probability -inf; 0 * -inf is NaN, and that bin contributes 0.
            log_p = torch.log_softmax(piece.float(), dim=-1)
            probs = log_p.exp()
            weighted = torch.where(
                probs > 0,
                probs * log_p,
                torch.zeros((), dtype=probs.dtype, device=probs.device),
            )
            ent_t = -weighted.sum(dim=-1)
            conf[b0:b1, t0:t1] = conf_t.detach().float().cpu().numpy()
            entropy[b0:b1, t0:t1] = ent_t.detach().float().cpu().numpy()
            pred[b0:b1, t0:t1] = pred_t.detach().cpu().numpy()
            margin[b0:b1, t0:t1] = mar_t.detach().float().cpu().numpy()
            eb_h[b0:b1, t0:t1] = eb_h_t.detach().float().cpu().numpy()
            h_rank[b0:b1, t0:t1] = h_rank_t.detach().float().cpu().numpy()
            del piece, log_p, probs, pred_t, conf_t, ent_t, softmax_p, eb_h_t, h_rank_t, top2, mar_t
    return [
        ScoreBundle(
            conf=conf[b],
            entropy=entropy[b],
            pred=pred[b],
            margin=margin[b],
            eb_h=eb_h[b],
            h_rank=h_rank[b],
        )
        for b in range(batches)
    ]
