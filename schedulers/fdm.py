'''
    FDM (Foreseeing Decoding Method) scheduler
'''


import math
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from load_model import LLADA_DEFAULT_EOS_ID, LLADA_DEFAULT_EOT_ID, LLADA_DEFAULT_MASK_ID
from schedulers.utils import add_gumbel_noise, check_eos_stop_condition, mask_eos_eot_logits

if os.environ.get('TRACING_ENABLED', 'false').lower() == 'true':
    from traces.trace_wrapper import trace_scheduler
else:
    def trace_scheduler(*args, **kwargs):
        return lambda x: x


@torch.no_grad()
def _global_confidence_expected_logp(model, x_hyp, attention_mask, mask_id,
                                     block_start=None, block_end=None,
                                     eos_id=None, eot_id=None,
                                     confidence_eos_eot_inf=False):
    """
    Compute a per-hypothesis C_global estimator.

    Strategy:
      For each hypothesis (shape [B_h, L]) compute logits -> log_softmax -> p
      Per-position expected log-prob:  e_lp = sum_v p_v * log p_v  (shape [B_h, L])
      Then average e_lp over *masked* positions within the active block only.
      

    Returns:
      c_global: [B_h] (float)  — average expected log-prob over masked positions in block.
    """
    device = x_hyp.device
    logits = model(x_hyp, attention_mask=attention_mask).logits  # [B_h, L, V]

    if confidence_eos_eot_inf and eos_id is not None:
        mask_eos_eot_logits(logits, False, True, eos_id, eot_id)

    # log-probs and probs
    logp = F.log_softmax(logits, dim=-1)  # [B_h, L, V]
    p = logp.exp()                         # [B_h, L, V]

    # per-position expected log-prob: sum_v p_v * log p_v (negative entropy, <= 0)
    expected_logp_per_pos = (p * logp).sum(dim=-1)  # [B_h, L]

    # mask positions to consider — restrict to current block only
    L = x_hyp.shape[1]
    if block_start is not None and block_end is not None:
        assert 0 <= block_start < block_end <= L, "block indices must be within hypothesis length"
    mask_index = (x_hyp == mask_id)  # [B_h, L]
    if block_start is not None and block_end is not None:
        block_range = torch.zeros(L, dtype=torch.bool, device=device)
        block_range[block_start:block_end] = True
        mask_index = mask_index & block_range.unsqueeze(0)  # [B_h, L]

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
    block_start, block_end,
    eos_id, eot_id,
    confidence_eos_eot_inf,
    global_scale,
):
    device = x.device

    # Candidate pruning (FDM2 uses γ=1.0 -> no pruning; FDM1 uses γ_1 or η_2)
    if log_gamma > -1e-9:  # γ_1 ≈ 1.0: skip pruning, use all masked positions
        cand_pos = masked_pos
    else:
        cand_mask = (x0_logp[b, masked_pos] > log_gamma)
        cand_pos = masked_pos[cand_mask]

    if cand_pos.numel() == 0:
        # fallback greedy local
        scores = x0_logp[b, masked_pos]
        k = min(n, masked_pos.numel())
        topk = torch.topk(scores, k=k).indices
        chosen = masked_pos[topk]
        x[b, chosen] = x0[b, chosen]
        return 0

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
        block_start=block_start,
        block_end=block_end,
        eos_id=eos_id,
        eot_id=eot_id,
        confidence_eos_eot_inf=confidence_eos_eot_inf
    )

    c_local = x0_logp[b, Lambda_pos]
    combined = c_local + global_scale * c_global

    n_eff = min(n, Lambda_pos.numel())
    chosen_idx = _select_topn_unique_positions(combined, Lambda_pos, n_eff, device=device)
    if chosen_idx.numel() == 0:
        return 1
    chosen_pos = Lambda_pos[chosen_idx]
    x[b, chosen_pos] = x0[b, chosen_pos]
    return 1


@torch.no_grad()
@trace_scheduler()  # tracing decorator
def generate_fdm(
    model,
    prompt,
    attention_mask=None,
    steps=128,
    gen_length=256,
    block_length=256,
    K1=2,
    N=4,
    eta1=0.8,
    eta2=0.7,
    gamma=0.5,
    temperature=0.,
    mask_id=LLADA_DEFAULT_MASK_ID,
    eos_id=LLADA_DEFAULT_EOS_ID,
    eot_id: Optional[int] = LLADA_DEFAULT_EOT_ID,
    logits_eos_inf=False,
    confidence_eos_eot_inf=False,
    global_scale=1.0,
    trace_name=None,
    upload_to_wandb=True,
    intervention_mode=None,
    intervention_steps=None,
    intervention_k=1,
    use_eos_harness: bool = True,
):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        attention_mask: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length. Divided across blocks.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. Confidence/unmasking is computed only within each block.
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
        trace_name: Optional trace name. If provided, trace will be saved to disk.
        upload_to_wandb: If True and trace_name is provided, upload trace to wandb as artifact.
        intervention_mode: Optional intervention mode. "drop" to prevent tokens from being unmasked, "swap" to swap tokens.
        intervention_steps: Optional list/set of step indices (0-indexed) at which to apply intervention.
        intervention_k: Number of tokens to drop/swap per intervention step. Default 1.
        use_eos_harness: If False, never return early on EOS (full step budget).
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    prompt_len = prompt.shape[1]
    steps_taken = 0  

    # gamma_1: main pruning threshold (exploration branch). eta_2: borderline-stage threshold (final else).
    log_gamma1 = math.log(gamma) if gamma > 0 else -1e9

    for num_block in range(num_blocks):
        block_start = prompt_len + num_block * block_length
        block_end = prompt_len + (num_block + 1) * block_length

        # Early stopping - check if block is already complete
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        if not block_mask_index.any():
            continue  # Skip completed blocks

        for t in range(steps_per_block):
            # Cache mask_index - compute once per step
            mask_index = (x == mask_id)
            block_mask = mask_index[:, block_start:block_end]
            if not block_mask.any():
                break

            logits = model(x, attention_mask=attention_mask).logits
            steps_taken += 1

            mask_eos_eot_logits(logits, logits_eos_inf, confidence_eos_eot_inf, eos_id, eot_id)

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

            logp = F.log_softmax(logits, dim=-1)
            x0_logp = torch.gather(logp, -1, x0.unsqueeze(-1)).squeeze(-1)

            x0_logp[:, :prompt_len] = -np.inf
            local_logconf = torch.where(mask_index, x0_logp, -np.inf)

            # Convert to probs for threshold logic
            local_p = torch.exp(local_logconf)

            for j in range(prompt.shape[0]):
                # Restrict to masked positions within current block only
                masked_pos = torch.nonzero(
                    mask_index[j, block_start:block_end], as_tuple=False
                ).squeeze(-1)
                if masked_pos.numel() == 0:
                    continue
                masked_pos = masked_pos + block_start  # Convert to global indices

                p_b = local_p[j, masked_pos]

                # High confidence and borderline sets
                high_mask = p_b > eta1
                border_mask = (p_b > eta2) & (p_b <= eta1)

                num_high = int(high_mask.sum().item())
                num_border = int(border_mask.sum().item())

              
                if num_high == 0:
                    # Exploration: NUM(p > eta1) == 0 -> FDM1(..., n=1, gamma=gamma1)
                    steps_taken += _fdm_single_step(
                        model, x, j, attention_mask,
                        masked_pos,
                        x0, x0_logp,
                        K=K1,
                        n=1,
                        log_gamma=log_gamma1,
                        mask_id=mask_id,
                        block_start=block_start,
                        block_end=block_end,
                        eos_id=eos_id,
                        eot_id=eot_id,
                        confidence_eos_eot_inf=confidence_eos_eot_inf,
                        global_scale=global_scale,
                    )
                    continue

                if num_high >= N:
                    # Many qualified -> FDM2(..., n=N, gamma=1.0).
                    steps_taken += _fdm_single_step(
                        model, x, j, attention_mask,
                        masked_pos,
                        x0, x0_logp,
                        K=1,
                        n=min(num_high, N),
                        log_gamma=0.0,
                        mask_id=mask_id,
                        block_start=block_start,
                        block_end=block_end,
                        eos_id=eos_id,
                        eot_id=eot_id,
                        confidence_eos_eot_inf=confidence_eos_eot_inf,
                        global_scale=global_scale,
                    )
                    continue

                if num_border == 0:
                    # No borderline -> FDM2(..., n=NUM(p>eta1), gamma=1.0).
                    n_dec = min(num_high, N)
                    steps_taken += _fdm_single_step(
                        model, x, j, attention_mask,
                        masked_pos,
                        x0, x0_logp,
                        K=1,
                        n=n_dec,
                        log_gamma=0.0,
                        mask_id=mask_id,
                        block_start=block_start,
                        block_end=block_end,
                        eos_id=eos_id,
                        eot_id=eot_id,
                        confidence_eos_eot_inf=confidence_eos_eot_inf,
                        global_scale=global_scale,
                    )
                    continue

                # Else: borderline exists -> FDM1(..., n=NUM(p>eta1), gamma=eta_2). eta_2 != gamma_1
                n_dec = min(num_high, masked_pos.numel())
                log_eta2 = math.log(eta2) if eta2 > 0 else -1e9
                steps_taken += _fdm_single_step(
                    model, x, j, attention_mask,
                    masked_pos,
                    x0, x0_logp,
                    K=K1,
                    n=n_dec,
                    log_gamma=log_eta2,
                    mask_id=mask_id,
                    block_start=block_start,
                    block_end=block_end,
                    eos_id=eos_id,
                    eot_id=eot_id,
                    confidence_eos_eot_inf=confidence_eos_eot_inf,
                    global_scale=global_scale,
                )

            # Early stopping: Check if EOS is decoded and all tokens before it are unmasked
            if use_eos_harness and check_eos_stop_condition(x, eos_id, mask_id, prompt_len):
                return x, steps_taken, num_block + 1  # Stop generation early

            # Clear intermediate tensors to free memory
            del logits, logits_with_noise, logp, x0_logp, local_logconf, local_p
            torch.cuda.empty_cache()

    return x, steps_taken, num_blocks
