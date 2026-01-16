import torch
import numpy as np
import torch.nn.functional as F
from utils import add_gumbel_noise


@torch.no_grad()
def generate_EB(
    model,
    prompt,
    attention_mask=None,
    steps=128,
    gen_length=128,
    block_length=128,
    gamma=0.1,
    E="confidence",
    temperature=0.,
    cfg_scale=0.,
    mask_id=126336,
    eos_id=126081,
    eot_id=126348,
    generate_until_ids=None,
):
    """
    Blockwise EB sampler with correct early stopping:
    stop if EOS/EOT appears AND prefix is fully unmasked.
    """

    device = model.device
    B, L = prompt.shape
    d = L + gen_length

    # initialize x
    x = torch.full((B, d), mask_id, dtype=torch.long, device=device)
    x[:, :L] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat(
            [attention_mask, torch.ones((B, gen_length), device=device)],
            dim=-1
        )

    total_steps = 0

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    def stopping_condition(x):
        if generate_until_ids is None:
            return False
        for xb in x:
            for stop_id in generate_until_ids:
                if stop_id in xb:
                    i = (xb == stop_id).nonzero(as_tuple=True)[0][0]
                    if (xb[:i] != mask_id).all():
                        return True
        return False

    for num_block in range(num_blocks):

        block_start = L + num_block * block_length
        block_end   = L + (num_block + 1) * block_length

        for _ in range(steps):

            # global early stop dominates
            if stopping_condition(x):
                return x

            mask_index = (x == mask_id)
            block_mask = mask_index[:, block_start:block_end]

            # block finished
            if not block_mask.any():
                break

            # ---- forward ----
            logits = model(x, attention_mask=attention_mask).logits
            total_steps += 1

            # prevent EOS / EOT diffusion
            logits[:, :, eos_id] = -torch.inf
            logits[:, :, eot_id] = -torch.inf

            # sample proposal
            logits_noisy = add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits_noisy, dim=-1)

            p = F.softmax(logits, dim=-1)

            # ---- EB score + entropy ----
            if E == "confidence":
                e = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
            elif E == "margin":
                top2 = torch.topk(p, 2, dim=-1).values
                e = top2[..., 0] - top2[..., 1]
            elif E == "entropy":
                e = -(p * torch.log(p + 1e-9)).sum(-1)
                e = -e
            else:
                raise ValueError(E)

            h = -(p * torch.log(p + 1e-9)).sum(-1)

            transfer = torch.zeros_like(x, dtype=torch.bool)

            # ---- EB selection per batch ----
            for b in range(B):

                block_positions = [
                    i for i in range(block_start, block_end)
                    if mask_index[b, i]
                ]

                if not block_positions:
                    continue

                block_positions = sorted(
                    block_positions,
                    key=lambda i: e[b, i].item(),
                    reverse=True
                )

                H = []
                for i in block_positions:
                    H.append(h[b, i].item())

                    if sum(H) - max(H) <= gamma:
                        transfer[b, i] = True
                    else:
                        break

            # entropy too strict
            if not transfer.any():
                for b in range(B):
                    block_positions = [
                        i for i in range(block_start, block_end)
                        if mask_index[b, i]
                    ]
                    if not block_positions:
                        continue

                    # choose most confident token
                    best_i = max(block_positions, key=lambda i: e[b, i].item())
                    transfer[b, best_i] = True

            # apply updates
            x = torch.where(transfer, x0, x)

    return x, total_steps