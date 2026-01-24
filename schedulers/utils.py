import torch
import torch.nn.functional as F
from typing import List


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens

def get_mask_by_confidence(logits: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    """
    Calculate token confidence based on predicted token probabilities.
    
    Confidence is defined as the probability assigned to the predicted token.
    Higher confidence indicates the model is more certain about the prediction.
    
    Args:
        logits: Model output logits of shape (batch_size, seq_len, vocab_size)
        x0: Predicted tokens of shape (batch_size, seq_len)
        
    Returns:
        Confidence scores for each position
    """
    p = F.softmax(logits.to(torch.float64), dim=-1)
    x0_p = torch.squeeze(
        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
    )
    return x0_p

def get_mask_by_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Calculate entropy-based uncertainty for each token position.
    
    Entropy measures the uncertainty in the model's prediction.
    Higher entropy indicates less certain predictions.
    
    Args:
        logits: Model output logits of shape (batch_size, seq_len, vocab_size)
        
    Returns:
        Entropy values for each position
    """
    p = F.softmax(logits.to(torch.float64), dim=-1)
    entropy = -torch.sum(p * torch.log(p + 1e-10), dim=-1)
    return entropy

def check_stop_generation(
    x: torch.Tensor, 
    stop_tokens: List[int], 
    stop_on_eos: bool, 
    mask_id: int, 
    prompt_len: int
) -> bool:
    """
    Check if generation should stop based on stop tokens or EOS conditions.
    
    Args:
        x: Current token sequence
        stop_tokens: List of token IDs that should trigger stopping
        stop_on_eos: Whether to enable early stopping
        mask_id: ID of the mask token
        prompt_len: Length of the input prompt
        
    Returns:
        True if generation should stop, False otherwise
    """
    if not stop_on_eos:
        return False
    
    if stop_tokens:
        generated_part = x[:, prompt_len:]
        for stop_token in stop_tokens:
            if stop_token in generated_part.flatten():
                return True
    
    return False