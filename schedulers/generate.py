import torch

from .pure_diffusion import generate_pure_diffusion
from .top_k import generate_block_topk
from .eb_sampler import generate_EB

@ torch.no_grad()
def generate(model, prompt, mode="pure_diffusion"):
    '''
        modes - pure_diffusion, top_k, eb_sampler, top_margin
    '''

    print("Generating with mode", mode)

    if mode == "pure_diffusion":
        return generate_pure_diffusion(model, prompt, attention_mask=None, steps=1024, gen_length=1024, temperature=0.,
             remasking='low_confidence', mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False)
    elif mode == "top_k":
        return generate_block_topk(model, prompt, attention_mask=None, gen_length=1024, block_length=32,steps_per_block=32, top_k=1, metric="confidence", temperature=0.0)
    elif mode == "eb_sampler":
        return generate_EB(model, prompt, attention_mask=None, gen_length=512, gamma=0.1, E="confidence",   
            temperature=0.0, generate_until_id=None)
    elif mode == "top_margin":
        return generate_block_topk(model, prompt, attention_mask=None, gen_length=1024, block_length=32,steps_per_block=32, top_k=1, metric="top_margin", temperature=0.0)
    else:
        assert False
