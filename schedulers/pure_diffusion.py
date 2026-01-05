import torch
import numpy as np
import torch.nn.functional as F
from utils import add_gumbel_noise, get_num_transfer_tokens
try:
    from ..load_model import load_model_and_tokenizer
except ImportError:
    import os
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from load_model import load_model_and_tokenizer

@ torch.no_grad()
def generate_pure_diffusion(model, prompt, attention_mask=None, steps=128, gen_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        temperature: Categorical distribution sampling temperature.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
        logits_eos_inf: Whether to set the logits of EOS token to -inf. See Appendix B.4 of LLaDA for details
        confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf. See Appendix B.4 of LLaDA for details
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    # Pure diffusion sampling: process all masked tokens together 
    mask_index = (x == mask_id)
    num_transfer_tokens = get_num_transfer_tokens(mask_index, steps)
        
    for i in range(steps):
        mask_index = (x == mask_id)
        
        # Early termination if no more masked tokens
        if not mask_index.any():
            break
            
        logits = model(x, attention_mask=attention_mask).logits

        if logits_eos_inf:
            logits[:, :, 126081] = -torch.inf

        # For pure diffusion: set EOS/EOT logits to -inf BEFORE prediction to prevent EOS from being predicted
        if confidence_eos_eot_inf:
            logits[:, :, 126081] = -torch.inf  # EOS token
            logits[:, :, 126348] = -torch.inf  # EOT token

        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

        if remasking == 'low_confidence':
            p = F.softmax(logits, dim=-1)
            x0_p = torch.squeeze(
                torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
        elif remasking == 'random':
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
        else:
            raise NotImplementedError(remasking)
        
        # Directly set confidence to 0 for EOS/EOT tokens during pure diffusion (applies to both remasking strategies)
        # This ensures EOS tokens are not selected for unmasking, preventing short generations
        if confidence_eos_eot_inf:
            x0_p = torch.where(x0 == 126081, torch.zeros_like(x0_p), x0_p)  # EOS
            x0_p = torch.where(x0 == 126348, torch.zeros_like(x0_p), x0_p)  # EOT

        # No block restriction - all masked tokens can be selected
        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, -np.inf)

        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
        for j in range(confidence.shape[0]):
            _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
            transfer_index[j, select_index] = True
        x[transfer_index] = x0[transfer_index]


    return x

def main():
    device = 'cuda'

    # Example usage
    model_dir = "./models/llada"
    model, tokenizer = load_model_and_tokenizer(model_dir, device=device)

    # The LLaDA architecture theoretically supports both left-padding and right-padding. 
    # However, the sampling code implementation is simpler with left-padding.
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    # If the padding ID equals the mask ID, you need to modify our generate function to achieve correct inference.
    assert tokenizer.pad_token_id != 126336

    prompts = [ "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"]

    encoded_outputs = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt"
    )
    input_ids = encoded_outputs['input_ids'].to(device)
    attention_mask = encoded_outputs['attention_mask'].to(device)

    out = generate_pure_diffusion(model, input_ids, attention_mask, steps=128, gen_length=128, temperature=0., remasking='low_confidence')
    output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
    for o in output:
        print(o)
        print('-' * 50)