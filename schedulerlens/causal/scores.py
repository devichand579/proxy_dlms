"""Score bundles from a mask-predictor forward."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from schedulers.utils import mask_eos_eot_logits


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
    if drop_eos:
        logits = logits.clone()
        mask_eos_eot_logits(logits, True, True, int(eos_id), eot_id)
    return bundle_from_logits(logits)


def bundle_from_logits(logits) -> list[ScoreBundle]:
    """Build one ``ScoreBundle`` per batch row from ``[B, L, V]`` logits."""
    import torch
    import torch.nn.functional as F

    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"logits must be a torch.Tensor, got {type(logits)}")
    log_p = torch.log_softmax(logits.float(), dim=-1)
    pred = torch.argmax(log_p, dim=-1)
    conf = torch.gather(log_p, dim=-1, index=pred.unsqueeze(-1)).squeeze(-1).exp()
    probs = log_p.exp()
    entropy = -(probs * log_p).sum(dim=-1)
    top2 = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
    if top2.shape[-1] == 1:
        margin = top2[..., 0]
    else:
        margin = top2[..., 0] - top2[..., 1]
    conf_np = conf.detach().float().cpu().numpy()
    ent_np = entropy.detach().float().cpu().numpy()
    pred_np = pred.detach().cpu().numpy().astype(np.int64)
    mar_np = margin.detach().float().cpu().numpy()
    return [
        ScoreBundle(
            conf=conf_np[b],
            entropy=ent_np[b],
            pred=pred_np[b],
            margin=mar_np[b],
        )
        for b in range(conf_np.shape[0])
    ]
