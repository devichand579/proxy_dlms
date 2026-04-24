'''
    FHS (First-Hitting Sampling) schedulers
    1. Extrapolation Method (N <= 128)
    generate_fhs_extrapolation(model, prompt, attention_mask=None, gen_length=128, steps=128, temperature=1.0,
                               mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False, pass_time_to_model=True, trace_name=None, intervention_mode=None, intervention_steps=None, intervention_k=1)
    
    2. Predictor-Corrector Method (N > 128)
    generate_fhs_predictor_corrector(model, prompt, attention_mask=None, gen_length=256, steps=256, temperature=1.0,
                                     mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False, pass_time_to_model=True, trace_name=None, intervention_mode=None, intervention_steps=None, intervention_k=1)
    
    3. Auto-selector (chooses method based on steps)
    generate_fhs(model, prompt, attention_mask=None, gen_length=128, steps=128, temperature=1.0,
                 mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False, pass_time_to_model=True, trace_name=None, intervention_mode=None, intervention_steps=None, intervention_k=1)
'''


import os
from typing import Optional

import torch
import torch.nn.functional as F
from load_model import LLADA_DEFAULT_EOS_ID, LLADA_DEFAULT_EOT_ID, LLADA_DEFAULT_MASK_ID
from schedulers.utils import check_eos_stop_condition, mask_eos_eot_logits

if os.environ.get('TRACING_ENABLED', 'false').lower() == 'true':
    from traces.trace_wrapper import trace_scheduler
else:
    def trace_scheduler(*args, **kwargs):
        return lambda x: x


def _model_logits(model, x, attention_mask=None, t_cond=None):
    """
    Helper to support both:
      - time-agnostic model: model(x, attention_mask=...).logits
      - time-conditioned model: model(x, attention_mask=..., timesteps=t).logits  (or similar)

    """
    if t_cond is None:
        return model(x, attention_mask=attention_mask).logits

    try:
        return model(x, attention_mask=attention_mask, timesteps=t_cond).logits
    except TypeError:
        pass
    try:
        return model(x, attention_mask=attention_mask, t=t_cond).logits
    except TypeError:
        pass

    # Fall back to time-agnostic
    return model(x, attention_mask=attention_mask).logits



def _sample_next_event_time_linear(tau, n_masked):
    """
    Paper: sample next event time analytically.
    Linear schedule α(t) = 1 - t  =>  τ_{l-1} = τ_l * u^{1/l}
    """
    n_safe = n_masked.clamp(min=1.0)
    u = torch.rand_like(tau).clamp(min=1e-12)
    return tau * torch.pow(u, 1.0 / n_safe)


def _pick_token_and_tau(tau_per_token, masked, batch_idx):
    """
    FHS: pick the masked token with earliest (minimum) hitting time.
    Returns (k, tau_new) where tau_new = actual next unmasking time.
    """
    idx = torch.nonzero(masked[batch_idx]).squeeze(-1)
    if idx.numel() == 0:
        return None, None
    tau_b = tau_per_token[batch_idx]
    tau_at_masked = tau_b[idx]
    best_local = torch.argmin(tau_at_masked).item()
    k = idx[best_local].item()
    tau_new = tau_b[k].item()
    return k, tau_new


def _rescale_hitting_times_survival(tau_per_token, masked, k, tau_new, tau_upper, batch_idx):
    """
    After token k hits at τ_new, condition remaining on survival.
    Rescale τ_j ∈ (τ_new, τ_upper] → (0, 1] for next round.
    (masked already has k=False when called)
    """
    tau_b = tau_per_token[batch_idx]
    mask_b = masked[batch_idx]
    denom = tau_upper - tau_new
    denom = max(float(denom), 1e-8)
    tau_b[mask_b] = (tau_b[mask_b] - tau_new) / denom


def sample_categorical(logits, temperature=1.0, use_fp64=True):
    """
    Sample from a categorical distribution.
    """
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1)

    if use_fp64:
        logits = logits.double()

    probs = F.softmax(logits / temperature, dim=-1)
    probs = probs.clamp(min=1e-12)

    # multinomial will use dtype of probs
    return torch.multinomial(probs, num_samples=1).squeeze(-1)



def make_equal_schedule(L, N):
    """
    Requires L divisible by N.
    """
    assert L % N == 0, f"L={L} must be divisible by N={N}"
    return [L // N] * N




@ torch.no_grad()
@trace_scheduler()  # tracing decorator
def generate_fhs_extrapolation(
    model,
    prompt,
    attention_mask=None,
    gen_length=128,
    steps=128,
    temperature=1.0,
    mask_id=LLADA_DEFAULT_MASK_ID,
    eos_id=LLADA_DEFAULT_EOS_ID,
    eot_id: Optional[int] = LLADA_DEFAULT_EOT_ID,
    logits_eos_inf=False,
    confidence_eos_eot_inf=False,
    pass_time_to_model=True,
    trace_name=None,
    upload_to_wandb=True,
    intervention_mode=None,
    intervention_steps=None,
    intervention_k=1,
):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        attention_mask: A tensor of shape (1, L).
        gen_length: Generated answer length.
        steps: Number of diffusion steps (chunks). gen_length must be divisible by steps.
        temperature: Categorical distribution sampling temperature.
        mask_id: The token id of [MASK] is 126336.
        logits_eos_inf: Whether to set the logits of EOS token to -inf. See Appendix B.4 of LLaDA for details
        confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf. See Appendix B.4 of LLaDA for details
        pass_time_to_model: Whether to pass time conditioning to the model.
        trace_name: Optional trace name. If provided, trace will be saved to disk.
        upload_to_wandb: If True and trace_name is provided, upload trace to wandb as artifact.
        intervention_mode: Optional intervention mode. "drop" to prevent tokens from being unmasked, "swap" to swap tokens.
        intervention_steps: Optional list/set of step indices (0-indexed) at which to apply intervention.
        intervention_k: Number of tokens to drop/swap per intervention step. Default 1.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    assert gen_length % steps == 0
    schedule = make_equal_schedule(gen_length, steps)

    prompt_len = prompt.shape[1]
    steps_taken = 0  

    # Mask bookkeeping
    masked = torch.zeros_like(x, dtype=torch.bool)
    masked[:, prompt_len:] = True

    # Per-token hitting times: each masked token has τ_i ~ Uniform(0,1).
    # Decoding order = token with minimum τ becomes visible first.
    tau_per_token = torch.empty_like(x, dtype=torch.float32, device=model.device)
    tau_per_token[:, :prompt_len] = 0.0  # unused for prompt
    tau_per_token[:, prompt_len:] = torch.rand(
        prompt.shape[0], gen_length, device=model.device, dtype=torch.float32
    )

    tau_upper = torch.ones((prompt.shape[0],), device=model.device, dtype=torch.float32)
    l = torch.full((prompt.shape[0],), gen_length, device=model.device, dtype=torch.float32)

    mu_prev = None
    tau_prev = None
    EXTRAPOL_EPS = 1e-6  # guard against tau_curr ≈ tau_prev

    for step, Ln in enumerate(schedule):
        # Early stopping - check if all tokens are decoded
        if not masked.any():
            break

        for i in range(Ln):
            # Paper: τ = actual next unmasking time = min(τ_i) among masked
            picks = [_pick_token_and_tau(tau_per_token, masked, b) for b in range(prompt.shape[0])]
            tau_new = torch.tensor(
                [picks[b][1] if picks[b][1] is not None else 1.0 for b in range(prompt.shape[0])],
                device=model.device, dtype=torch.float32
            )

            # Network eval ONCE per chunk
            if i == 0:
                steps_taken += 1  
                t_cond = tau_new if pass_time_to_model else None
                logits = _model_logits(model, x, attention_mask, t_cond)

                mask_eos_eot_logits(logits, logits_eos_inf, confidence_eos_eot_inf, eos_id, eot_id)

                mu = logits
                tau_curr = tau_new.clone()

            # ----- Extrapolation (2nd order) -----
            if mu_prev is None:
                mu_hat = mu
            else:
                # Lagrange interpolation in time; reshape for (batch, seq, vocab) broadcast
                # Guard against tau_curr ≈ tau_prev which makes denominator ~0 and coefficients explode
                denom = tau_curr - tau_prev
                use_extrapol = denom.abs() >= EXTRAPOL_EPS
                denom_safe = torch.where(use_extrapol, denom, EXTRAPOL_EPS * torch.ones_like(denom))
                coeff1 = (tau_new - tau_prev) / denom_safe
                coeff2 = (tau_new - tau_curr) / (-denom_safe)
                coeff1 = coeff1.reshape(-1, 1, 1)
                coeff2 = coeff2.reshape(-1, 1, 1)
                mu_hat = torch.where(
                    use_extrapol.reshape(-1, 1, 1),
                    coeff1 * mu + coeff2 * mu_prev,
                    mu,
                )

            # Decode ONE token per batch; τ_new = actual hitting time; rescale on survival
            for b in range(prompt.shape[0]):
                k, tau_new_b = picks[b][0], picks[b][1]
                if k is None:
                    continue
                token = sample_categorical(mu_hat[b, k], temperature)
                x[b, k] = token
                masked[b, k] = False
                l[b] -= 1
                _rescale_hitting_times_survival(
                    tau_per_token, masked, k, tau_new_b, tau_upper[b].item(), b
                )

            # Early stopping: Check if EOS is decoded and all tokens before it are unmasked
            if check_eos_stop_condition(x, eos_id, mask_id, prompt_len):
                return x, steps_taken  # Stop generation early

        # Save previous network output for next extrapolation
        mu_prev = mu
        tau_prev = tau_curr

        # Clear intermediate tensors to free memory
        if i == Ln - 1:
            del mu, logits
            torch.cuda.empty_cache()

    return x, steps_taken


@ torch.no_grad()
@trace_scheduler()  # tracing decorator
def generate_fhs_predictor_corrector(
    model,
    prompt,
    attention_mask=None,
    gen_length=256,
    steps=256,
    temperature=1.0,
    mask_id=LLADA_DEFAULT_MASK_ID,
    eos_id=LLADA_DEFAULT_EOS_ID,
    eot_id: Optional[int] = LLADA_DEFAULT_EOT_ID,
    logits_eos_inf=False,
    confidence_eos_eot_inf=False,
    pass_time_to_model=True,
    trace_name=None,
    upload_to_wandb=True,
    intervention_mode=None,
    intervention_steps=None,
    intervention_k=1,
):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        attention_mask: A tensor of shape (1, L).
        gen_length: Generated answer length.
        steps: Number of diffusion steps (chunks). gen_length must be divisible by steps.
        temperature: Categorical distribution sampling temperature.
        mask_id: The token id of [MASK] is 126336.
        logits_eos_inf: Whether to set the logits of EOS token to -inf. See Appendix B.4 of LLaDA for details
        confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf. See Appendix B.4 of LLaDA for details
        pass_time_to_model: Whether to pass time conditioning to the model.
        trace_name: Optional trace name. If provided, trace will be saved to disk.
        upload_to_wandb: If True and trace_name is provided, upload trace to wandb as artifact.
        intervention_mode: Optional intervention mode. "drop" to prevent tokens from being unmasked, "swap" to swap tokens.
        intervention_steps: Optional list/set of step indices (0-indexed) at which to apply intervention.
        intervention_k: Number of tokens to drop/swap per intervention step. Default 1.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    assert gen_length % steps == 0
    schedule = make_equal_schedule(gen_length, steps)

    prompt_len = prompt.shape[1]
    steps_taken = 0 

    # Mask bookkeeping
    masked = torch.zeros_like(x, dtype=torch.bool)
    masked[:, prompt_len:] = True

    # Per-token hitting times: each masked token has τ_i ~ Uniform(0,1)
    tau_per_token = torch.empty_like(x, dtype=torch.float32, device=model.device)
    tau_per_token[:, :prompt_len] = 0.0
    tau_per_token[:, prompt_len:] = torch.rand(
        prompt.shape[0], gen_length, device=model.device, dtype=torch.float32
    )

    tau_upper = torch.ones((prompt.shape[0],), device=model.device, dtype=torch.float32)
    l = torch.full((prompt.shape[0],), gen_length, device=model.device, dtype=torch.float32)

    for step, Ln in enumerate(schedule):
        # Early stopping - check if all tokens are decoded
        if not masked.any():
            break

        # Track positions decoded in this chunk (for corrector: refine same tokens only)
        decoded_this_chunk = [[] for _ in range(prompt.shape[0])]

        for i in range(Ln):
            # τ = actual next unmasking time = min(τ_i) among masked
            picks = [_pick_token_and_tau(tau_per_token, masked, b) for b in range(prompt.shape[0])]
            tau_new = torch.tensor(
                [picks[b][1] if picks[b][1] is not None else 1.0 for b in range(prompt.shape[0])],
                device=model.device, dtype=torch.float32
            )
            t_cond = tau_new if pass_time_to_model else None

            # ----- Predictor -----
            steps_taken += 1
            logits = _model_logits(model, x, attention_mask, t_cond)

            mask_eos_eot_logits(logits, logits_eos_inf, confidence_eos_eot_inf, eos_id, eot_id)

            mu_pred = logits

            # Decode ONE token per batch; τ_new = actual hitting time; rescale on survival
            for b in range(prompt.shape[0]):
                k, tau_new_b = picks[b][0], picks[b][1]
                if k is None:
                    continue
                token = sample_categorical(mu_pred[b, k], temperature)
                x[b, k] = token
                masked[b, k] = False
                l[b] -= 1
                decoded_this_chunk[b].append(k)
                _rescale_hitting_times_survival(
                    tau_per_token, masked, k, tau_new_b, tau_upper[b].item(), b
                )

            # ----- Corrector (only at end of chunk): refine SAME tokens, do not unmask new ones
            if i == Ln - 1:
                steps_taken += 1
                logits_corr = _model_logits(model, x, attention_mask, t_cond)

                mask_eos_eot_logits(logits_corr, logits_eos_inf, confidence_eos_eot_inf, eos_id, eot_id)

                mu_corr = logits_corr

                # Refine tokens at positions decoded in predictor (re-sample, no new unmasking)
                for b in range(prompt.shape[0]):
                    for k in decoded_this_chunk[b]:
                        token = sample_categorical(mu_corr[b, k], temperature)
                        x[b, k] = token

                del logits_corr, mu_corr

            # Early stopping: Check if EOS is decoded and all tokens before it are unmasked
            if check_eos_stop_condition(x, eos_id, mask_id, prompt_len):
                return x, steps_taken  # Stop generation early

        # Clear intermediate tensors to free memory
        del mu_pred, logits
        torch.cuda.empty_cache()

    return x, steps_taken


@ torch.no_grad()
@trace_scheduler()  # tracing decorator
def generate_fhs(
    model,
    prompt,
    attention_mask=None,
    gen_length=128,
    steps=128,
    temperature=1.0,
    mask_id=LLADA_DEFAULT_MASK_ID,
    eos_id=LLADA_DEFAULT_EOS_ID,
    eot_id: Optional[int] = LLADA_DEFAULT_EOT_ID,
    logits_eos_inf=False,
    confidence_eos_eot_inf=False,
    pass_time_to_model=True,
    trace_name=None,
    upload_to_wandb=True,
    intervention_mode=None,
    intervention_steps=None,
    intervention_k=1,
):
    '''
    Matches paper recommendation:
      - Extrapolation for steps <= 128
      - Predictor-Corrector for steps > 128
    
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        attention_mask: A tensor of shape (1, L).
        gen_length: Generated answer length.
        steps: Number of diffusion steps (chunks). gen_length must be divisible by steps.
        temperature: Categorical distribution sampling temperature.
        mask_id: The token id of [MASK] is 126336.
        logits_eos_inf: Whether to set the logits of EOS token to -inf. See Appendix B.4 of LLaDA for details
        confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf. See Appendix B.4 of LLaDA for details
        pass_time_to_model: Whether to pass time conditioning to the model.
        trace_name: Optional trace name. If provided, trace will be saved to disk.
        upload_to_wandb: If True and trace_name is provided, upload trace to wandb as artifact.
        intervention_mode: Optional intervention mode. "drop" to prevent tokens from being unmasked, "swap" to swap tokens.
        intervention_steps: Optional list/set of step indices (0-indexed) at which to apply intervention.
        intervention_k: Number of tokens to drop/swap per intervention step. Default 1.
    '''
    trace_kw = dict(
        trace_name=trace_name,
        upload_to_wandb=upload_to_wandb,
        intervention_mode=intervention_mode,
        intervention_steps=intervention_steps,
        intervention_k=intervention_k,
    )
    if steps <= 128:
        return generate_fhs_extrapolation(
            model,
            prompt,
            attention_mask,
            gen_length=gen_length,
            steps=steps,
            temperature=temperature,
            mask_id=mask_id,
            eos_id=eos_id,
            eot_id=eot_id,
            logits_eos_inf=logits_eos_inf,
            confidence_eos_eot_inf=confidence_eos_eot_inf,
            pass_time_to_model=pass_time_to_model,
            **trace_kw,
        )
    return generate_fhs_predictor_corrector(
        model,
        prompt,
        attention_mask,
        gen_length=gen_length,
        steps=steps,
        temperature=temperature,
        mask_id=mask_id,
        eos_id=eos_id,
        eot_id=eot_id,
        logits_eos_inf=logits_eos_inf,
        confidence_eos_eot_inf=confidence_eos_eot_inf,
        pass_time_to_model=pass_time_to_model,
        **trace_kw,
    )
