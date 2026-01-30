'''
    FHS (First-Hitting Sampling) schedulers
    1. Extrapolation Method (N <= 128)
    generate_fhs_extrapolation(model, prompt, attention_mask=None, gen_length=128, steps=128, temperature=1.0,
                               mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False, pass_time_to_model=True)
    
    2. Predictor-Corrector Method (N > 128)
    generate_fhs_predictor_corrector(model, prompt, attention_mask=None, gen_length=256, steps=256, temperature=1.0,
                                     mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False, pass_time_to_model=True)
    
    3. Auto-selector (chooses method based on steps)
    generate_fhs(model, prompt, attention_mask=None, gen_length=128, steps=128, temperature=1.0,
                 mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False, pass_time_to_model=True)
'''


import torch
import torch.nn.functional as F
from schedulers.utils import  check_eos_stop_condition


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



def update_tau_linear(tau, n_masked):
    """
    Linear schedule:
      alpha(t) = 1 - t
      => tau_{l-1} = tau_l * u^{1/l}
    """
    u = torch.rand_like(tau).clamp(min=1e-12)
    return tau * torch.pow(u, 1.0 / n_masked)


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
def generate_fhs_extrapolation(
    model,
    prompt,
    attention_mask=None,
    gen_length=128,
    steps=128,
    temperature=1.0,
    mask_id=126336,
    logits_eos_inf=False,
    confidence_eos_eot_inf=False,
    pass_time_to_model=True,
):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        attention_mask: A tensor of shape (1, L).
        gen_length: Generated answer length.
        steps: Number of diffusion steps (chunks). Must be divisible by gen_length.
        temperature: Categorical distribution sampling temperature.
        mask_id: The token id of [MASK] is 126336.
        logits_eos_inf: Whether to set the logits of EOS token to -inf. See Appendix B.4 of LLaDA for details
        confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf. See Appendix B.4 of LLaDA for details
        pass_time_to_model: Whether to pass time conditioning to the model.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    assert gen_length % steps == 0
    schedule = make_equal_schedule(gen_length, steps)

    prompt_len = prompt.shape[1]
    eos_id, eot_id = 126081, 126348
    steps_taken = 0  

    # Mask bookkeeping
    masked = torch.zeros_like(x, dtype=torch.bool)
    masked[:, prompt_len:] = True

    tau = torch.ones((prompt.shape[0],), device=model.device)
    l = torch.full((prompt.shape[0],), gen_length, device=model.device, dtype=torch.float32)

    mu_prev = None
    tau_prev = None

    for step, Ln in enumerate(schedule):
        # Early stopping - check if all tokens are decoded
        if not masked.any():
            break

        for i in range(Ln):
            # First-hitting time update
            tau_new = update_tau_linear(tau, l)

            # Network eval ONCE per chunk
            if i == 0:
                steps_taken += 1  
                t_cond = tau_new if pass_time_to_model else None
                logits = _model_logits(model, x, attention_mask, t_cond)

                # Set EOS/EOT to -inf before sampling to prevent them from being predicted
                if logits_eos_inf:
                    logits[:, :, eos_id] = -torch.inf

                if confidence_eos_eot_inf:
                    logits[:, :, eos_id] = -torch.inf
                    logits[:, :, eot_id] = -torch.inf

                mu = logits
                tau_curr = tau_new.clone()

            # ----- Extrapolation (2nd order) -----
            if mu_prev is None:
                mu_hat = mu
            else:
                # Lagrange interpolation in time
                mu_hat = (
                    (tau_new - tau_prev) / (tau_curr - tau_prev) * mu
                    + (tau_new - tau_curr) / (tau_prev - tau_curr) * mu_prev
                )

            # Sample ONE token per batch
            for b in range(prompt.shape[0]):
                if not masked[b].any():
                    continue
                idx = torch.nonzero(masked[b]).squeeze(1)
                if idx.numel() == 0:
                    continue
                k = idx[torch.randint(0, idx.numel(), (1,), device=model.device).item()]
                token = sample_categorical(mu_hat[b, k], temperature)
                x[b, k] = token
                masked[b, k] = False
                l[b] -= 1

            tau = tau_new

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
def generate_fhs_predictor_corrector(
    model,
    prompt,
    attention_mask=None,
    gen_length=256,
    steps=256,
    temperature=1.0,
    mask_id=126336,
    logits_eos_inf=False,
    confidence_eos_eot_inf=False,
    pass_time_to_model=True,
):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        attention_mask: A tensor of shape (1, L).
        gen_length: Generated answer length.
        steps: Number of diffusion steps (chunks). Must be divisible by gen_length.
        temperature: Categorical distribution sampling temperature.
        mask_id: The token id of [MASK] is 126336.
        logits_eos_inf: Whether to set the logits of EOS token to -inf. See Appendix B.4 of LLaDA for details
        confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf. See Appendix B.4 of LLaDA for details
        pass_time_to_model: Whether to pass time conditioning to the model.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    assert gen_length % steps == 0
    schedule = make_equal_schedule(gen_length, steps)

    prompt_len = prompt.shape[1]
    eos_id, eot_id = 126081, 126348
    steps_taken = 0 

    # Mask bookkeeping
    masked = torch.zeros_like(x, dtype=torch.bool)
    masked[:, prompt_len:] = True

    tau = torch.ones((prompt.shape[0],), device=model.device)
    l = torch.full((prompt.shape[0],), gen_length, device=model.device, dtype=torch.float32)

    for step, Ln in enumerate(schedule):
        # Early stopping - check if all tokens are decoded
        if not masked.any():
            break

        for i in range(Ln):
            # First-hitting time update
            tau_new = update_tau_linear(tau, l)
            t_cond = tau_new if pass_time_to_model else None

            # ----- Predictor -----
            steps_taken += 1  
            logits = _model_logits(model, x, attention_mask, t_cond)

            # Set EOS/EOT to -inf before sampling to prevent them from being predicted
            if logits_eos_inf:
                logits[:, :, eos_id] = -torch.inf

            if confidence_eos_eot_inf:
                logits[:, :, eos_id] = -torch.inf
                logits[:, :, eot_id] = -torch.inf

            mu_pred = logits

            # Sample ONE token per batch
            for b in range(prompt.shape[0]):
                if not masked[b].any():
                    continue
                idx = torch.nonzero(masked[b]).squeeze(1)
                if idx.numel() == 0:
                    continue
                k = idx[torch.randint(0, idx.numel(), (1,), device=model.device).item()]
                token = sample_categorical(mu_pred[b, k], temperature)
                x[b, k] = token
                masked[b, k] = False
                l[b] -= 1

            # ----- Corrector (only at end of chunk) -----
            if i == Ln - 1:
                # Refine using updated x - corrector step
                steps_taken += 1
                logits_corr = _model_logits(model, x, attention_mask, t_cond)
                
                # Set EOS/EOT to -inf before sampling
                if logits_eos_inf:
                    logits_corr[:, :, eos_id] = -torch.inf
                
                if confidence_eos_eot_inf:
                    logits_corr[:, :, eos_id] = -torch.inf
                    logits_corr[:, :, eot_id] = -torch.inf
                
                mu_corr = logits_corr
                
                # Refine masked tokens using corrector output
                for b in range(prompt.shape[0]):
                    if not masked[b].any():
                        continue
                    idx = torch.nonzero(masked[b]).squeeze(1)
                    if idx.numel() == 0:
                        continue
                    k = idx[torch.randint(0, idx.numel(), (1,), device=model.device).item()]
                    token = sample_categorical(mu_corr[b, k], temperature)
                    x[b, k] = token
                    masked[b, k] = False
                    l[b] -= 1
                
                del logits_corr, mu_corr

            tau = tau_new

            # Early stopping: Check if EOS is decoded and all tokens before it are unmasked
            if check_eos_stop_condition(x, eos_id, mask_id, prompt_len):
                return x, steps_taken  # Stop generation early

        # Clear intermediate tensors to free memory
        del mu_pred, logits
        torch.cuda.empty_cache()

    return x, steps_taken


@ torch.no_grad()
def generate_fhs(
    model,
    prompt,
    attention_mask=None,
    gen_length=128,
    steps=128,
    temperature=1.0,
    mask_id=126336,
    logits_eos_inf=False,
    confidence_eos_eot_inf=False,
    pass_time_to_model=True,
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
        steps: Number of diffusion steps (chunks). Must be divisible by gen_length.
        temperature: Categorical distribution sampling temperature.
        mask_id: The token id of [MASK] is 126336.
        logits_eos_inf: Whether to set the logits of EOS token to -inf. See Appendix B.4 of LLaDA for details
        confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf. See Appendix B.4 of LLaDA for details
        pass_time_to_model: Whether to pass time conditioning to the model.
    '''
    if steps <= 128:
        return generate_fhs_extrapolation(
            model, prompt, attention_mask,
            gen_length, steps, temperature,
            mask_id, logits_eos_inf, confidence_eos_eot_inf, pass_time_to_model
        )
    else:
        return generate_fhs_predictor_corrector(
            model, prompt, attention_mask,
            gen_length, steps, temperature,
            mask_id, logits_eos_inf, confidence_eos_eot_inf, pass_time_to_model
        )
