import torch
import numpy as np
import torch.nn.functional as F
from utils import add_gumbel_noise, get_num_transfer_tokens
try:
    from ..load_model import load_model_and_tokenizer
except ImportError:
    import os
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from load_model import load_model_and_tokenizer
    



@torch.no_grad()
def _global_confidence_maxlogp(model, x_hyp, attention_mask, mask_id,
                               eos_id=None, eot_id=None,
                               confidence_eos_eot_inf=False):
    """
    Compute C_global for each hypothesis in x_hyp.

    x_hyp: [B_h, L]  (stacked hypotheses)
    Returns: c_global: [B_h] (scalar per hypothesis)

    Estimator:
      mean over remaining masked positions of max log-prob over vocab at that position.
      (Equivalent to negative per-position entropy proxy, but cheaper.)
    """
    logits = model(x_hyp, attention_mask=attention_mask).logits  # [B_h, L, V]

    if confidence_eos_eot_inf and eos_id is not None and eot_id is not None:
        logits[:, :, eos_id] = -torch.inf
        logits[:, :, eot_id] = -torch.inf

    logp = F.log_softmax(logits, dim=-1)  # [B_h, L, V]
    max_logp, _ = torch.max(logp, dim=-1)  # [B_h, L]

    mask_index = (x_hyp == mask_id)  # [B_h, L]
    # Avoid division by zero if no masks left in a hypothesis
    denom = mask_index.sum(dim=-1).clamp_min(1)  # [B_h]
    c_global = (max_logp.masked_fill(~mask_index, 0.0).sum(dim=-1)) / denom  # [B_h]
    return c_global


def _select_topn_unique_positions(scores, positions, n):
    """
    scores:   [M] float
    positions:[M] int (token positions in sequence)
    return indices of up to n entries with highest scores, with unique positions.
    """
    # Sort descending
    order = torch.argsort(scores, descending=True)
    chosen = []
    used_pos = set()
    for idx in order.tolist():
        pos = int(positions[idx])
        if pos in used_pos:
            continue
        chosen.append(idx)
        used_pos.add(pos)
        if len(chosen) >= n:
            break
    return chosen


@torch.no_grad()
def generate_fdm(
    model,
    prompt,
    attention_mask=None,
    steps=128,
    gen_length=128,
    n=1,                 # number of tokens to decode per step
    K=32,                # foreseeing width
    gamma=0.2,           # pruning threshold on prob (not logprob)
    temperature=0.0,
    mask_id=126336,
    logits_eos_inf=False,
    confidence_eos_eot_inf=False,
    eos_id=126081,
    eot_id=126348,
):
    """
    FDM sampler (single-block / pure diffusion style) compatible with your code.

    prompt: [B, Lp]
    output x: [B, Lp + gen_length]
    """
    device = model.device
    B = prompt.shape[0]
    Lp = prompt.shape[1]
    L = Lp + gen_length

    x = torch.full((B, L), mask_id, dtype=torch.long, device=device)
    x[:, :Lp] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat(
            [attention_mask, torch.ones((B, gen_length), dtype=attention_mask.dtype, device=device)],
            dim=-1
        )

    for t in range(steps):
        mask_index = (x == mask_id)  # [B, L]
        if not mask_index.any():
            break

        # 1) forward on current x
        logits = model(x, attention_mask=attention_mask).logits  # [B, L, V]

        if logits_eos_inf:
            logits[:, :, eos_id] = -torch.inf

        if confidence_eos_eot_inf:
            logits[:, :, eos_id] = -torch.inf
            logits[:, :, eot_id] = -torch.inf

        # 2) pick argmax token for each position
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, L]

        # 3) local confidence = prob of chosen token
        p = F.softmax(logits, dim=-1)  # [B, L, V]
        x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # [B, L]

        # Never decode prompt tokens
        x0_p[:, :Lp] = -np.inf

        # Only masked positions are eligible
        local_conf = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=device))  # [B, L]
        # Candidate mask: masked and prob > gamma
        candidate_mask = mask_index & (x0_p > gamma)  # [B, L]

        # Update x in-place; do per-batch element selection
        for b in range(B):
            # all eligible masked positions
            eligible_pos = torch.nonzero(mask_index[b], as_tuple=False).squeeze(-1)  # [M]
            if eligible_pos.numel() == 0:
                continue

            # fallback greedy-local positions (top-n by local confidence among masked)
            # (if fewer than n positions remain, decode as many as possible)
            n_eff = min(n, int(eligible_pos.numel()))

            cand_pos = torch.nonzero(candidate_mask[b], as_tuple=False).squeeze(-1)  # [C]
            if cand_pos.numel() == 0:
                # No candidates survive pruning => pure local
                scores = local_conf[b, eligible_pos]
                top_idx = torch.topk(scores, k=n_eff).indices
                chosen_pos = eligible_pos[top_idx]
                x[b, chosen_pos] = x0[b, chosen_pos]
                continue

            # Top-K by local confidence within candidates
            cand_scores = local_conf[b, cand_pos]  # [C]
            K_eff = min(K, int(cand_pos.numel()))
            topk_idx = torch.topk(cand_scores, k=K_eff).indices
            Lambda_pos = cand_pos[topk_idx]  # [K_eff]

            if Lambda_pos.numel() == 0:
                # Shouldn't happen, but keep safe
                scores = local_conf[b, eligible_pos]
                top_idx = torch.topk(scores, k=n_eff).indices
                chosen_pos = eligible_pos[top_idx]
                x[b, chosen_pos] = x0[b, chosen_pos]
                continue

            # 4) Build hypotheses (fill ONE token at a time)
            #    x_hyp: [K_eff, L]
            x_base = x[b].unsqueeze(0).repeat(K_eff, 1)
            # fill candidate token at corresponding pos
            x_base[torch.arange(K_eff, device=device), Lambda_pos] = x0[b, Lambda_pos]

            # attention mask for hypotheses
            if attention_mask is not None:
                attn_hyp = attention_mask[b].unsqueeze(0).repeat(K_eff, 1)
            else:
                attn_hyp = None

            # 5) Compute global confidence for each hypothesis
            c_global = _global_confidence_maxlogp(
                model=model,
                x_hyp=x_base,
                attention_mask=attn_hyp,
                mask_id=mask_id,
                eos_id=eos_id,
                eot_id=eot_id,
                confidence_eos_eot_inf=confidence_eos_eot_inf
            )  # [K_eff]

            # Local confidence for those positions 
            # Using log makes it more additive with c_global (which is in log units).
            c_local = torch.log(torch.clamp(x0_p[b, Lambda_pos], min=1e-12))  # [K_eff]

            combined = c_local + c_global  # [K_eff]

            # Pick up to n unique positions among Lambda
            n_eff2 = min(n_eff, int(Lambda_pos.numel()))
            chosen_indices = _select_topn_unique_positions(combined, Lambda_pos, n_eff2)
            chosen_pos = Lambda_pos[torch.tensor(chosen_indices, device=device)]

            # Commit those tokens
            x[b, chosen_pos] = x0[b, chosen_pos]

    return x
