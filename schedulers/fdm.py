'''
    FDM (Foreseeing Decoding Method) scheduler
    generate_fdm(model, prompt, attention_mask=None, steps=128, gen_length=128,
                 K1=8, N=8, eta1=0.8, eta2=0.7, gamma=0.2, temperature=0.,
                 mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False,
                 global_scale=1.0)
'''


import math
import torch
import torch.nn.functional as F
import numpy as np
from schedulers.utils import add_gumbel_noise, check_eos_stop_condition


@ torch.no_grad()
def _global_confidence_expected_logp(model, x_hyp, attention_mask, mask_id,
                                     eos_id=None, eot_id=None,
                                     confidence_eos_eot_inf=False):
    """
    Compute a per-hypothesis C_global estimator.

    Strategy:
      For each hypothesis (shape [B_h, L]) compute logits -> log_softmax -> p
      Per-position expected log-prob:  e_lp = sum_v p_v * log p_v  (shape [B_h, L])
      Then average e_lp over *masked* positions for each hypothesis.
      This yields an average expected log-prob per masked position (log-space).
      It is an unbiased-ish per-position expected log-prob estimator and is
      closer to the expected sequence log-likelihood than max-logp.

    Returns:
      c_global: [B_h] (float)  — average expected log-prob over masked positions.
    """
    device = x_hyp.device
    logits = model(x_hyp, attention_mask=attention_mask).logits  # [B_h, L, V]

    if confidence_eos_eot_inf and eos_id is not None and eot_id is not None:
        logits[:, :, eos_id] = -torch.inf
        logits[:, :, eot_id] = -torch.inf

    # log-probs and probs
    logp = F.log_softmax(logits, dim=-1)  # [B_h, L, V]
    p = logp.exp()                         # [B_h, L, V]

    # per-position expected log-prob: sum_v p_v * log p_v (negative entropy, <= 0)
    expected_logp_per_pos = (p * logp).sum(dim=-1)  # [B_h, L]

    # mask positions to consider
    mask_index = (x_hyp == mask_id)  # [B_h, L]
    denom = mask_index.sum(dim=-1).clamp_min(1).to(expected_logp_per_pos.dtype)  # [B_h]

    # sum over masked positions and average (so units are per-position log-prob)
    summed = expected_logp_per_pos.masked_fill(~mask_index, 0.0).sum(dim=-1)  # [B_h]
    c_global = summed / denom  # [B_h]

    return c_global  # Note: typically negative numbers (because logp <= 0)


def _select_topn_unique_positions(scores, positions, n, device=None):
    """
    Vectorized selection: pick up to n items from 'scores' with unique 'positions'.
    scores:   Tensor [M]
    positions:Tensor [M] (int positions in sequence)
    returns a tensor of chosen indices (into the original arrays), shape [k] (k <= n)
    """
    if device is None:
        device = scores.device
    M = scores.numel()
    if M == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    # sort descending and iterate until we have n unique positions
    order = torch.argsort(scores, descending=True)
    chosen = []
    used = set()
    for idx in order.tolist():
        pos = int(positions[idx].item())
        if pos in used:
            continue
        chosen.append(idx)
        used.add(pos)
        if len(chosen) >= n:
            break
    if len(chosen) == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.tensor(chosen, dtype=torch.long, device=device)



def _fdm_single_step(
    model, x, b, attention_mask,
    masked_pos,
    x0, x0_logp,
    K, n,
    log_gamma,
    mask_id,
    eos_id, eot_id,
    confidence_eos_eot_inf,
    global_scale,
):
    device = x.device

    # Candidate pruning
    cand_mask = (x0_logp[b, masked_pos] > log_gamma)
    cand_pos = masked_pos[cand_mask]

    if cand_pos.numel() == 0:
        # fallback greedy local
        scores = x0_logp[b, masked_pos]
        k = min(n, masked_pos.numel())
        topk = torch.topk(scores, k=k).indices
        chosen = masked_pos[topk]
        x[b, chosen] = x0[b, chosen]
        return

    cand_scores = x0_logp[b, cand_pos]
    K_eff = min(K, cand_pos.numel())
    topk_idx = torch.topk(cand_scores, k=K_eff).indices
    Lambda_pos = cand_pos[topk_idx]

    # Build hypotheses
    K_eff = Lambda_pos.numel()
    x_base = x[b].unsqueeze(0).repeat(K_eff, 1)
    arange = torch.arange(K_eff, device=device)
    x_base[arange, Lambda_pos] = x0[b, Lambda_pos]

    attn_hyp = attention_mask[b].unsqueeze(0).repeat(K_eff, 1) if attention_mask is not None else None

    c_global = _global_confidence_expected_logp(
        model=model,
        x_hyp=x_base,
        attention_mask=attn_hyp,
        mask_id=mask_id,
        eos_id=eos_id,
        eot_id=eot_id,
        confidence_eos_eot_inf=confidence_eos_eot_inf
    )

    c_local = x0_logp[b, Lambda_pos]
    combined = c_local + global_scale * c_global

    n_eff = min(n, Lambda_pos.numel())
    chosen_idx = _select_topn_unique_positions(combined, Lambda_pos, n_eff, device=device)
    if chosen_idx.numel() == 0:
        return
    chosen_pos = Lambda_pos[chosen_idx]
    x[b, chosen_pos] = x0[b, chosen_pos]


@ torch.no_grad()
def generate_fdm(
    model,
    prompt,
    attention_mask=None,
    steps=128,
    gen_length=128,
    K1=8,
    N=8,
    eta1=0.8,
    eta2=0.7,
    gamma=0.2,
    temperature=0.,
    mask_id=126336,
    logits_eos_inf=False,
    confidence_eos_eot_inf=False,
    global_scale=1.0,
):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        attention_mask: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        K1: Deep foreseeing width for FDM1 (exploration mode).
        N: Max parallel tokens for FDM2 and acceleration mode.
        eta1: High-confidence threshold. Above this uses pure local decoding.
        eta2: Borderline threshold. Between eta2 and eta1 uses FDM2.
        gamma: Pruning threshold for candidate selection in FDM1/FDM2.
        temperature: Categorical distribution sampling temperature.
        mask_id: The token id of [MASK] is 126336.
        logits_eos_inf: Whether to set the logits of EOS token to -inf. See Appendix B.4 of LLaDA for details
        confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf. See Appendix B.4 of LLaDA for details
        global_scale: Weight for global confidence in combined score.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    prompt_len = prompt.shape[1]
    eos_id, eot_id = 126081, 126348
    steps_taken = 0  

    log_gamma = math.log(gamma) if gamma > 0 else -1e9

    for t in range(steps):
        # Cache mask_index - compute once per step
        mask_index = (x == mask_id)
        if not mask_index.any():
            break

        steps_taken += 1  
        logits = model(x, attention_mask=attention_mask).logits

        # Set EOS/EOT to -inf before sampling to prevent them from being predicted
        if logits_eos_inf:
            logits[:, :, eos_id] = -torch.inf

        if confidence_eos_eot_inf:
            logits[:, :, eos_id] = -torch.inf
            logits[:, :, eot_id] = -torch.inf

        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

        logp = F.log_softmax(logits, dim=-1)
        x0_logp = torch.gather(logp, -1, x0.unsqueeze(-1)).squeeze(-1)

        x0_logp[:, :prompt_len] = -np.inf
        local_logconf = torch.where(mask_index, x0_logp, -np.inf)

        # Convert to probs for threshold logic
        local_p = torch.exp(local_logconf)

        for j in range(prompt.shape[0]):
            masked_pos = torch.nonzero(mask_index[j], as_tuple=False).squeeze(-1)
            if masked_pos.numel() == 0:
                continue

            p_b = local_p[j, masked_pos]

            # High confidence and borderline sets
            high_mask = p_b > eta1
            border_mask = (p_b > eta2) & (p_b <= eta1)

            num_high = int(high_mask.sum().item())
            num_border = int(border_mask.sum().item())

            # Case A: Exploration (FDM1)
            if num_high == 0:
                # Use deep FDM (K=K1, n=1)
                _fdm_single_step(
                    model, x, j, attention_mask,
                    masked_pos,
                    x0, x0_logp,
                    K=K1,
                    n=1,
                    log_gamma=log_gamma,
                    mask_id=mask_id,
                    eos_id=eos_id,
                    eot_id=eot_id,
                    confidence_eos_eot_inf=confidence_eos_eot_inf,
                    global_scale=global_scale,
                )
                continue

            # Case B: Balance (FDM2)
            if num_border > 0:
                # Cheap FDM (K=1), parallel small batch
                _fdm_single_step(
                    model, x, j, attention_mask,
                    masked_pos,
                    x0, x0_logp,
                    K=1,
                    n=min(N, masked_pos.numel()),
                    log_gamma=log_gamma,
                    mask_id=mask_id,
                    eos_id=eos_id,
                    eot_id=eot_id,
                    confidence_eos_eot_inf=confidence_eos_eot_inf,
                    global_scale=global_scale,
                )
                continue

            # Case C: Acceleration (pure local)
            # Decode many tokens purely by local confidence
            scores = local_logconf[j, masked_pos]
            k = min(N, masked_pos.numel())
            topk = torch.topk(scores, k=k).indices
            chosen_pos = masked_pos[topk]
            x[j, chosen_pos] = x0[j, chosen_pos]

        # Early stopping: Check if EOS is decoded and all tokens before it are unmasked
        if check_eos_stop_condition(x, eos_id, mask_id, prompt_len):
            return x, steps_taken  # Stop generation early

        # Clear intermediate tensors to free memory
        del logits, logits_with_noise, logp, x0_logp, local_logconf, local_p
        torch.cuda.empty_cache()

    return x, steps_taken
