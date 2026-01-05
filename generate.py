import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from datetime import datetime
import matplotlib.pyplot as plt

 # IMPORTANT

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


@ torch.no_grad()
def generate(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        logits_eos_inf: Whether to set the logits of EOS token to -inf. See Appendix B.4 of LLaDA for details
        confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf. See Appendix B.4 of LLaDA for details
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
            
            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x

@torch.no_grad()
def generate_eb(
    model,
    prompt,
    attention_mask=None,
    steps=256,
    gen_length=256,
    block_length=64,
    temperature=0.0,
    entropy_threshold=0.1,
    error_proxy="confidence",
    mask_id=126336,
):

    """
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
    """
    device = prompt.device
    batch, prompt_len = prompt.shape

    if attention_mask is None:
        attention_mask = torch.ones_like(prompt, device=device)

    x = torch.full(
        (batch, prompt_len + gen_length),
        mask_id,
        dtype=torch.long,
        device=device,
    )
    x[:, :prompt_len] = prompt

    attention_mask = torch.cat(
        [
            attention_mask,
            torch.ones((batch, gen_length), device=device, dtype=attention_mask.dtype),
        ],
        dim=1,
    )

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    steps_taken = 0
    unmask_stats = []

    for block_id in range(num_blocks):

        block_start = prompt_len + block_id * block_length
        block_end   = prompt_len + (block_id + 1) * block_length

        for _ in range(steps_per_block):

            # Mask index: only current block
            mask_index = (x == mask_id)
            mask_index[:, :block_start] = False
            mask_index[:, block_end:] = False

            if not mask_index.any():
                break

            steps_taken += 1

            logits = model(x, attention_mask=attention_mask).logits
            probs = F.softmax(logits, dim=-1)

            # Error proxy
            if error_proxy == "confidence":
                top1_vals, _ = probs.max(dim=-1)
                err = -top1_vals
            else:
                err = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)

            # Optional sampling
            if temperature > 0:
                noise = torch.rand_like(logits)
                gumbel = -torch.log(-torch.log(noise + 1e-12) + 1e-12)
                logits = logits + temperature * gumbel

            x0 = torch.argmax(logits, dim=-1)

            transfer = torch.zeros_like(x, dtype=torch.bool)

            for b in range(batch):
                allowed = mask_index[b]
                if not allowed.any():
                    continue

                err_b = torch.where(
                    allowed,
                    err[b],
                    torch.full_like(err[b], float("inf")),
                )

                ids_sorted = torch.argsort(err_b)

                ents = -(probs[b] * torch.log(probs[b] + 1e-12)).sum(dim=-1)
                ents_sorted = ents[ids_sorted]

                acc = torch.cumsum(ents_sorted, dim=0)
                max_e = torch.cummax(ents_sorted, dim=0).values
                diff = acc - max_e

                k = (diff <= entropy_threshold).sum()
                if k.item() < 1:
                    k = torch.tensor(1, device=device)

                transfer[b, ids_sorted[:k]] = True

            x[transfer] = x0[transfer]
            
            n_tokens = transfer.sum().item()
            unmask_stats.append(n_tokens)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = f"eb_unmask_{timestamp}.png"

    plt.figure(figsize=(6, 4))
    plt.plot(range(len(unmask_stats)), unmask_stats, marker="o")
    plt.xlabel("EB Step")
    plt.ylabel("Tokens Unmasked")
    plt.title(f"EB Unmasking Curve (γ={entropy_threshold})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    plt.close()

    print(f"[PLOT SAVED] {plot_file}")

    return x, steps_taken

@torch.no_grad()
def generate_eb_expt(
    model,
    prompt,
    attention_mask=None,
    steps=256,
    gen_length=256,
    block_length=64,
    temperature=0.0,
    entropy_threshold=0.1,
    error_proxy="confidence",
    mask_id=126336,
    t_intervene=0,
    num_tokens_to_disturb=1
):
    device = prompt.device
    batch, prompt_len = prompt.shape

    if attention_mask is None:
        attention_mask = torch.ones_like(prompt, device=device)

    x = torch.full(
        (batch, prompt_len + gen_length),
        mask_id,
        dtype=torch.long,
        device=device,
    )
    x[:, :prompt_len] = prompt

    attention_mask = torch.cat(
        [
            attention_mask,
            torch.ones((batch, gen_length), device=device, dtype=attention_mask.dtype),
        ],
        dim=1,
    )

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    global_step = 0

    for block_id in range(num_blocks):
        block_start = prompt_len + block_id * block_length
        block_end   = prompt_len + (block_id + 1) * block_length

        for _ in range(steps_per_block):

            mask_index = (x == mask_id)
            mask_index[:, :block_start] = False
            mask_index[:, block_end:] = False

            if not mask_index.any():
                break

            logits = model(x, attention_mask=attention_mask).logits
            probs = F.softmax(logits, dim=-1)

            if error_proxy == "confidence":
                top1_vals, _ = probs.max(dim=-1)
                err = -top1_vals
            else:
                err = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)

            if temperature > 0:
                noise = torch.rand_like(logits)
                gumbel = -torch.log(-torch.log(noise + 1e-12) + 1e-12)
                logits = logits + temperature * gumbel

            x0 = torch.argmax(logits, dim=-1)

            transfer = torch.zeros_like(x, dtype=torch.bool)

            for b in range(batch):
                allowed = mask_index[b]
                if not allowed.any():
                    continue

                err_b = torch.where(
                    allowed,
                    err[b],
                    torch.full_like(err[b], float("inf")),
                )

                ids_sorted = torch.argsort(err_b)

                ents = -(probs[b] * torch.log(probs[b] + 1e-12)).sum(dim=-1)
                ents_sorted = ents[ids_sorted]

                acc = torch.cumsum(ents_sorted, dim=0)
                max_e = torch.cummax(ents_sorted, dim=0).values
                diff = acc - max_e

                k = (diff <= entropy_threshold).sum()
                if k.item() < 1:
                    k = torch.tensor(1, device=device)

                eb_selected = ids_sorted[:k]

                # CAUSAL INTERVENTION (ONLY AT t_intervene)
                if global_step == t_intervene:
                    blocked = eb_selected[:num_tokens_to_disturb]

                    all_masked = torch.where(allowed)[0]
                    candidate = torch.tensor(
                        list(set(all_masked.tolist()) - set(eb_selected.tolist())),
                        device=device,
                    )

                    if len(candidate) > 0:
                        rand_pick = candidate[
                            torch.randperm(len(candidate))[:num_tokens_to_disturb]
                        ]
                        transfer[b, eb_selected] = True
                        transfer[b, eb_selected[:num_tokens_to_disturb]] = False
                        transfer[b, rand_pick] = True
                else:
                    transfer[b, eb_selected] = True

            x[transfer] = x0[transfer]
            global_step += 1

    return x, global_step

def main():
    device = 'cuda'

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    # The LLaDA architecture theoretically supports both left-padding and right-padding. 
    # However, the sampling code implementation is simpler with left-padding.
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    # If the padding ID equals the mask ID, you need to modify our generate function to achieve correct inference.
    assert tokenizer.pad_token_id != 126336

    prompts = [ "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"]

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    messages = [{"role": "user", "content": prompt} for prompt in prompts]
    prompts = [tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False) for message in messages]

    encoded_outputs = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt"
    )
    input_ids = encoded_outputs['input_ids'].to(device)
    attention_mask = encoded_outputs['attention_mask'].to(device)

    out = generate(model, input_ids, attention_mask, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
    for o in output:
        print(o)
        print('-' * 50)

    print("\n\n ###### EB outputs ###### \n\n\n")
    out = generate_eb(model, input_ids, attention_mask)
    output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
    for o in output:
        print(o)
        print('-' * 50)

if __name__ == '__main__':
    main()
