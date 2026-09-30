"""Score bundles from a mask-predictor forward."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from schedulers.utils import mask_eos_eot_logits

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

    def gen(self, prompt_len: int) -> "ScoreBundle":
        sl = slice(prompt_len, None)
        return ScoreBundle(
            conf=self.conf[sl],
            entropy=self.entropy[sl],
            pred=self.pred[sl],
            margin=self.margin[sl],
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
    top_k = min(2, int(vocab))
    for b0 in range(0, batches, _SCORE_ROWS):
        b1 = min(batches, b0 + _SCORE_ROWS)
        for t0 in range(0, length, _SCORE_LEN):
            t1 = min(length, t0 + _SCORE_LEN)
            piece = logits[b0:b1, t0:t1]
            if drop_eos:
                piece = piece.clone()
                mask_eos_eot_logits(piece, True, True, int(eos_id), eot_id)
            log_p = torch.log_softmax(piece.float(), dim=-1)
            pred_t = torch.argmax(log_p, dim=-1)
            conf_t = torch.gather(log_p, dim=-1, index=pred_t.unsqueeze(-1)).squeeze(-1).exp()
            probs = log_p.exp()
            ent_t = -(probs * log_p).sum(dim=-1)
            top2 = torch.topk(probs, k=top_k, dim=-1).values
            mar_t = top2[..., 0] if top_k == 1 else top2[..., 0] - top2[..., 1]
            conf[b0:b1, t0:t1] = conf_t.detach().float().cpu().numpy()
            entropy[b0:b1, t0:t1] = ent_t.detach().float().cpu().numpy()
            pred[b0:b1, t0:t1] = pred_t.detach().cpu().numpy()
            margin[b0:b1, t0:t1] = mar_t.detach().float().cpu().numpy()
            del piece, log_p, probs, pred_t, conf_t, ent_t, top2, mar_t
    return [
        ScoreBundle(conf=conf[b], entropy=entropy[b], pred=pred[b], margin=margin[b])
        for b in range(batches)
    ]
