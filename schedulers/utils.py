import torch
import torch.nn.functional as F
from typing import Optional


def mask_eos_eot_logits(
    logits: torch.Tensor,
    logits_eos_inf: bool,
    confidence_eos_eot_inf: bool,
    eos_id: int,
    eot_id: Optional[int],
) -> None:
    """In-place optional EOS (and EoT if set) -inf in logits; see LLaDA Appendix B.4."""
    if logits_eos_inf or confidence_eos_eot_inf:
        logits[:, :, eos_id] = -torch.inf
    if confidence_eos_eot_inf and eot_id is not None:
        logits[:, :, eot_id] = -torch.inf


def add_gumbel_noise(logits, temperature):
    """Gumbel–max noise for sampling; `temperature==0` passthrough. Uses float64 (MDM note)."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """Per-step token transfer counts for a linear unmasking schedule (LLaDA)."""
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    step_idx = torch.arange(steps, device=mask_index.device).unsqueeze(0)
    num_transfer_tokens += (step_idx < remainder).long()

    return num_transfer_tokens


def get_mask_by_confidence(logits: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    """Per-position p(argmax) from softmax(logits) at `x0`."""
    log_p = F.log_softmax(logits.float(), dim=-1)
    x0_log_p = torch.gather(log_p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    return x0_log_p.exp()


def get_mask_by_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Per-position entropy of softmax(logits)."""
    p = F.softmax(logits.float(), dim=-1)
    entropy = -torch.sum(p * torch.log(p + 1e-10), dim=-1)
    return entropy


def check_eos_stop_condition(
    x: torch.Tensor,
    eos_id: int,
    mask_id: int,
    prompt_len: int,
    require_all_done: bool = True,
) -> bool:
    """True if generated region has a valid EOS stop (all non-mask before first EOS, batch logic per `require_all_done`)."""
    generated_part = x[:, prompt_len:]  # Shape: (batch_size, gen_length)
    batch_size = x.shape[0]

    for batch_idx in range(batch_size):
        eos_positions = (generated_part[batch_idx] == eos_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) == 0:
            if require_all_done:
                return False  # This sample not done
            continue
        first_eos_pos = eos_positions[0].item()
        tokens_before_eos = generated_part[batch_idx, :first_eos_pos]
        if (tokens_before_eos == mask_id).any():
            if require_all_done:
                return False  # This sample has EOS but still has masks before it
            continue
        # This sample is done (EOS with all before unmasked)
        if not require_all_done:
            return True  # Any done -> stop
    # All samples checked: if require_all_done, every sample was done
    return require_all_done


def check_all_generated_unmasked(
    x: torch.Tensor,
    mask_id: int,
    prompt_len: int,
) -> bool:
    """True when every batch row has no [MASK] left in the generated tail."""
    if prompt_len >= x.shape[1]:
        return True
    generated = x[:, prompt_len:]
    return not (generated == mask_id).any().item()