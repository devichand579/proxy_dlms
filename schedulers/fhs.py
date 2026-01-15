import torch
import numpy as np
import torch.nn.functional as F

from utils import add_gumbel_noise


def _model_logits(model, x, attention_mask=None, t_cond=None):
    """
    Helper to support both:
      - time-agnostic model: model(x, attention_mask=...).logits
      - time-conditioned model: model(x, attention_mask=..., timesteps=t).logits  (or similar)

    If your model doesn't accept t, it will just ignore it.
    """
    if t_cond is None:
        return model(x, attention_mask=attention_mask).logits

    # Try common keyword names
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


@torch.no_grad()
def generate_fhs(
    model,
    prompt,
    attention_mask=None,
    gen_length=128,
    temperature=0.0,
    mask_id=126336,
    schedule="linear",
    # EOS/EOT handling:
    logits_eos_inf=False,
    confidence_eos_eot_inf=False,
    eos_id=126081,
    eot_id=126348,
    # optionally pass time to model (if supported)
    pass_time_to_model=False,
):
    device = model.device
    B = prompt.shape[0]
    Lp = prompt.shape[1]
    L = Lp + gen_length

    # init x: prompt + masks
    x = torch.full((B, L), mask_id, dtype=torch.long, device=device)
    x[:, :Lp] = prompt.clone()

    # attention mask: extend with ones for generated tokens
    if attention_mask is not None:
        attention_mask = torch.cat(
            [attention_mask, torch.ones((B, gen_length), dtype=attention_mask.dtype, device=device)],
            dim=-1
        )

    # positions we will generate
    gen_pos = torch.arange(Lp, L, device=device)  # [gen_length]

    # u ~ Uniform(0,1), tau = alpha^{-1}(u)
    u = torch.rand((B, gen_length), device=device)
    tau = u  # [B, gen_length], in [0,1]

    # Sort positions by tau (ascending): decode earliest hits first
    order_idx = torch.argsort(tau, dim=-1)  # [B, gen_length]
    # Map order indices to actual sequence positions
    order_pos = gen_pos.unsqueeze(0).expand(B, -1).gather(dim=1, index=order_idx)  # [B, gen_length]
    order_tau = tau.gather(dim=1, index=order_idx)  # [B, gen_length]

    # Pointer for each batch
    ptr = torch.zeros((B,), dtype=torch.long, device=device)

    # Loop until all generated positions are filled
    while True:
        # Check completion
        if torch.all(ptr >= gen_length):
            break

        # Choose next positions to fill (up to chunk_size per batch)
        # We'll build a mask "to_fill" over [B, L]
        to_fill = torch.zeros((B, L), dtype=torch.bool, device=device)
        t_cond = None

        for b in range(B):
            p = int(ptr[b].item())
            if p >= gen_length:
                continue
            q = min(p, gen_length)
            pos = order_pos[b, p:q]  # positions in x
            to_fill[b, pos] = True

        if pass_time_to_model:
            # Use the tau of the LAST position filled in the chunk as conditioning time.
            # Shape [B], values in [0,1]
            t_vals = []
            for b in range(B):
                p = int(ptr[b].item())
                if p >= gen_length:
                    t_vals.append(torch.tensor(1.0, device=device))
                    continue
                q = min(p, gen_length)
                t_vals.append(order_tau[b, q - 1])
            t_cond = torch.stack(t_vals, dim=0)  # [B]

        # Forward pass once for all batches
        logits = _model_logits(model, x, attention_mask=attention_mask, t_cond=t_cond)  # [B, L, V]

        # EOS / EOT logits handling
        if logits_eos_inf:
            logits[:, :, eos_id] = -torch.inf

        if confidence_eos_eot_inf:
            logits[:, :, eos_id] = -torch.inf
            logits[:, :, eot_id] = -torch.inf

        # Sample / pick tokens
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)  # [B, L]

        # Commit only the selected positions
        x[to_fill] = x0[to_fill]

        # Advance pointers
        for b in range(B):
            if ptr[b] < gen_length:
                ptr[b] = min(ptr[b], gen_length)

    return x
