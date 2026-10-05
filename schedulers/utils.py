import torch
from typing import Optional

from traces.trace_wrapper import note_sampled_tokens

_SEQ_CHUNK = 32


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


def mask_invalid_token_logits(
    logits: torch.Tensor,
    invalid_token_ids: Optional[torch.Tensor],
) -> None:
    """In-place set ``-inf`` for token ids that have no decodable string in the tokenizer.

    The model's output logits span the full (padded) embedding matrix, which includes
    reserved/undefined ids the tokenizer cannot decode. Greedy decoding never selects
    them, but stochastic samplers (e.g. FHS with ``temperature>0``) can, which later
    crashes ``convert_tokens_to_string`` ("expected str instance, NoneType found").
    Masking them here guarantees only decodable tokens are ever sampled.
    """
    if invalid_token_ids is None or invalid_token_ids.numel() == 0:
        return
    if invalid_token_ids.device != logits.device:
        invalid_token_ids = invalid_token_ids.to(logits.device)
    logits[..., invalid_token_ids] = -torch.inf


def zero_invalid_token_probs(
    probs: torch.Tensor,
    invalid_token_ids: Optional[torch.Tensor],
) -> None:
    """In-place zero probability mass on undecodable token ids, then safe to renormalize."""
    if invalid_token_ids is None or invalid_token_ids.numel() == 0:
        return
    if invalid_token_ids.device != probs.device:
        invalid_token_ids = invalid_token_ids.to(probs.device)
    probs[..., invalid_token_ids] = 0.0


def add_gumbel_noise(logits, temperature):
    """Gumbel–max noise for sampling; `temperature==0` passthrough. Uses float64 (MDM note)."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def sample_tokens(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Token ids for one step, without a second full-vocabulary tensor.

    ``temperature == 0`` is ``argmax``. Otherwise each sequence chunk uses the
    same float64 Gumbel-max scores as ``add_gumbel_noise``.
    """
    if temperature == 0:
        return torch.argmax(logits, dim=-1)
    batch, length, _vocab = logits.shape
    chosen = torch.empty(batch, length, dtype=torch.long, device=logits.device)
    # One row at a time matches the random stream of a single contiguous draw.
    for row in range(batch):
        scores = logits[row : row + 1]
        for start in range(0, length, _SEQ_CHUNK):
            end = min(length, start + _SEQ_CHUNK)
            piece = scores[:, start:end].to(torch.float64)
            noise = torch.rand_like(piece)
            gumbel_noise = (-torch.log(noise)) ** temperature
            chosen[row : row + 1, start:end] = (piece.exp() / gumbel_noise).argmax(dim=-1)
    note_sampled_tokens(chosen)
    return chosen


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
    """Per-position p(token) from softmax(logits) at ``x0``."""
    out = torch.empty(x0.shape, dtype=torch.float32, device=logits.device)
    for start in range(0, logits.shape[1], _SEQ_CHUNK):
        end = min(logits.shape[1], start + _SEQ_CHUNK)
        piece = logits[:, start:end].float()
        chosen = piece.gather(-1, x0[:, start:end].unsqueeze(-1)).squeeze(-1)
        out[:, start:end] = (chosen - torch.logsumexp(piece, dim=-1)).exp()
    return out


def get_mask_by_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Per-position entropy of softmax(logits)."""
    return token_entropy(logits, eps=1e-10)


def token_entropy(logits: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Per-position ``-sum p log(p + eps)``, one sequence chunk at a time."""
    out = torch.empty(logits.shape[:2], dtype=torch.float32, device=logits.device)
    for start in range(0, logits.shape[1], _SEQ_CHUNK):
        end = min(logits.shape[1], start + _SEQ_CHUNK)
        probs = torch.softmax(logits[:, start:end].float(), dim=-1)
        out[:, start:end] = -(probs * torch.log(probs + eps)).sum(dim=-1)
    return out


def top_two_margin(logits: torch.Tensor) -> torch.Tensor:
    """Per-position gap between the two largest softmax probabilities."""
    out = torch.empty(logits.shape[:2], dtype=torch.float32, device=logits.device)
    for start in range(0, logits.shape[1], _SEQ_CHUNK):
        end = min(logits.shape[1], start + _SEQ_CHUNK)
        probs = torch.softmax(logits[:, start:end].float(), dim=-1)
        top2 = torch.topk(probs, 2, dim=-1).values
        out[:, start:end] = top2[..., 0] - top2[..., 1]
    return out


def token_log_prob(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Per-position log softmax probability of ``tokens``."""
    out = torch.empty(tokens.shape, dtype=torch.float32, device=logits.device)
    for start in range(0, logits.shape[1], _SEQ_CHUNK):
        end = min(logits.shape[1], start + _SEQ_CHUNK)
        piece = logits[:, start:end].float()
        chosen = piece.gather(-1, tokens[:, start:end].unsqueeze(-1)).squeeze(-1)
        out[:, start:end] = chosen - torch.logsumexp(piece, dim=-1)
    return out


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