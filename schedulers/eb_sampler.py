import torch
import torch.nn.functional as F
import numpy as np
from .utils import add_gumbel_noise

@torch.no_grad()
def generate_EB(
    model,
    prompt,
    attention_mask=None,
    gen_length=128,
    gamma=0.1,                     # entropy threshold γ
    E="confidence",                # error proxy E(p)
    temperature=0.0,
    mask_id=126336,
    eos_id=126081,
    eot_id=126348,
    generate_until_id=None        # token that triggers early stopping
):
    """
    Faithful implementation of Algorithm 1 from arXiv:2505.24857
    EB-Sampler with generate-until logic
    """

    device = model.device
    B, L = prompt.shape
    d = L + gen_length

    # x ← [y0, m * n]
    x = torch.full((B, d), mask_id, dtype=torch.long, device=device)
    x[:, :L] = prompt

    num_steps_taken = 0

    if attention_mask is not None:
        attention_mask = torch.cat(
            [attention_mask, torch.ones((B, gen_length), device=device)],
            dim=1
        )

    # stopping condition C(x)
    def stopping_condition(x):
        if generate_until_id is None:
            return False
        done = []
        for xb in x:
            if generate_until_id not in xb:
                done.append(False)
            else:
                i = (xb == generate_until_id).nonzero(as_tuple=True)[0][0]
                done.append(((xb[:i] != mask_id).all().item()))
        return all(done)

    while True:
        if num_steps_taken > gen_length:
            break

        # I_m = {l | x^l = m}
        mask_index = (x == mask_id)
        if not mask_index.any():
            break
        if stopping_condition(x):
            break

        # p^l = pθ,l(.|x)
        logits = model(x, attention_mask=attention_mask).logits
        num_steps_taken += 1

        # remove EOS / EOT from diffusion
        logits[:, :, eos_id] = -torch.inf
        logits[:, :, eot_id] = -torch.inf

        # sample x0 (used only for sampling when committing)
        logits_noisy = add_gumbel_noise(logits, temperature)
        x0 = torch.argmax(logits_noisy, dim=-1)

        # p̂ = softmax(logits)
        p = F.softmax(logits, dim=-1)

        # ê = E(p̂)
        if E == "confidence":
            e = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
        elif E == "margin":
            top2 = torch.topk(p, 2, dim=-1).values
            e = top2[..., 0] - top2[..., 1]
        elif E == "entropy":
            e = -(p * torch.log(p + 1e-9)).sum(-1)
            e = -e   # low entropy = good
        else:
            raise ValueError(E)

        # ĥ = H(p̂)
        h = -(p * torch.log(p + 1e-9)).sum(-1)

        # restrict to masked tokens
        e_masked = torch.where(mask_index, e, -np.inf)
        h_masked = torch.where(mask_index, h, 0.0)

        # I_sort = argsort(ê_Im)
        I_sort = torch.argsort(e_masked, dim=-1, descending=True)

        # U ← ∅
        transfer = torch.zeros_like(x, dtype=torch.bool)

        for b in range(B):
            U = []
            entropies = []

            for a in I_sort[b]:
                if not mask_index[b, a]:
                    continue

                U.append(a.item())
                entropies.append(h_masked[b, a].item())

                # check entropy bound:
                # sum(h_U) - max(h_U) ≤ γ
                if sum(entropies) - max(entropies) <= gamma:
                    # x^a = sample(p̂_a)
                    transfer[b, a] = True
                else:
                    break

        # apply unmasking
        x = torch.where(transfer, x0, x)

    return x, num_steps_taken
