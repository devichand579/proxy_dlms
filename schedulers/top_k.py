import torch
import numpy as np
import torch.nn.functional as F
from utils import add_gumbel_noise

import torch
import torch.nn.functional as F
import numpy as np
from utils import add_gumbel_noise

@torch.no_grad()
def generate_block_topk(
    model,
    prompt,
    attention_mask=None,
    gen_length=128,
    block_length=32,
    steps_per_block=32,
    top_k=4,
    metric="top_margin",   # "top_margin", "confidence", "entropy", "random"
    temperature=0.0,
    mask_id=126336,
    eos_id=126081,
    eot_id=126348
):
    device = model.device
    B, L = prompt.shape
    num_blocks = gen_length // block_length
    assert gen_length % block_length == 0
    assert gen_length % top_k == 0
    assert top_k >= 1

    # Initialize latent sequence
    x = torch.full((B, L + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :L] = prompt

    if attention_mask is not None:
        attention_mask = torch.cat(
            [attention_mask, torch.ones((B, gen_length), device=device)],
            dim=1
        )

    # Iterate blocks (autoregressive over blocks)
    for b in range(num_blocks):
        block_start = L + b * block_length
        block_end   = L + (b + 1) * block_length

        for step in range(steps_per_block):
            mask_index = (x == mask_id)

            # Only current block is allowed to change
            block_mask = torch.zeros_like(mask_index)
            block_mask[:, block_start:block_end] = True
            active_mask = mask_index & block_mask

            if not active_mask.any():
                break

            # Model prediction
            logits = model(x, attention_mask=attention_mask).logits

            # EOS / EOT must be removed from diffusion
            logits[:, :, eos_id] = -torch.inf
            logits[:, :, eot_id] = -torch.inf

            # Sample x0
            logits = add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits, dim=-1)

            probs = F.softmax(logits, dim=-1)

            # Compute ranking score
            if metric == "confidence":
                scores = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)

            elif metric == "top_margin":
                top2 = torch.topk(probs, 2, dim=-1).values
                scores = top2[..., 0] - top2[..., 1]

            elif metric == "entropy":
                scores = -(probs * torch.log(probs + 1e-9)).sum(-1)
                scores = -scores   # low entropy = easy

            elif metric == "random":
                scores = torch.rand_like(x0, dtype=torch.float)

            else:
                raise ValueError(metric)

            # Only masked tokens in this block compete
            scores = torch.where(active_mask, scores, -np.inf)

            # Select top-k per batch
            transfer = torch.zeros_like(x, dtype=torch.bool)
            for i in range(B):
                k = min(top_k, active_mask[i].sum().item())
                if k > 0:
                    _, idx = torch.topk(scores[i], k=k)
                    transfer[i, idx] = True

            # Commit updates
            x = torch.where(transfer, x0, x)

    return x