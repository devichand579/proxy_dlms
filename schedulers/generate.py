import torch
import concurrent.futures
import time

from schedulers.llada_schedulers import generate_llada
from schedulers.top_k import generate_topk
from schedulers.fhs import generate_fhs
from schedulers.eb_sampler import generate_EB

@ torch.no_grad()
def generate(model, input_ids, attention_mask, steps=256, gen_length=256, block_length=256, mode="pure_diffusion", strategy="confidence"):
    '''
        modes - pure_diffusion, block_diffusion, top_k, FHS, EB_sampler
        strategies - confidence, random, top_margin, entropy
    '''
    
    start_time = time.time()
    num_samples = input_ids.shape[0]
    all_generated = []
    total_steps = 0
    total_blocks = 0

    # Helper function for single generation call
    def run_single_generation(prompt, mask, steps=256, gen_length=256, block_length=256, mode="pure_diffusion", strategy="confidence"):
        if mode == "pure_diffusion":
            assert strategy == "confidence" or strategy == "random", "strategy must be confidence or random"
            return generate_llada(model, prompt, mask, steps=steps, gen_length=gen_length, block_length=block_length, temperature=0., remasking=strategy)
        elif mode == "block_diffusion":
            assert strategy == "confidence" or strategy == "random", "strategy must be confidence or random"
            return generate_llada(model, prompt, mask, steps=steps, gen_length=gen_length, block_length=block_length, temperature=0., remasking=strategy)
        elif mode == "top_k":
            assert strategy == "confidence" or strategy == "top_margin" or strategy == "entropy" or strategy == "random", "strategy must be confidence, top_margin, entropy, or random"
            return generate_topk(model, prompt, mask, steps=steps, gen_length=gen_length, block_length=block_length, temperature=0., remasking=strategy)
        elif mode == "FHS":
            return generate_fhs(model, prompt, mask, steps=steps, gen_length=gen_length, temperature=1.0, pass_time_to_model=True)
        elif mode == "EB_sampler":
            assert strategy == "confidence" or strategy == "margin" or strategy == "entropy" , "strategy must be confidence, top_margin, entropy, or random"
            return generate_EB(model, prompt, mask, steps=steps, gen_length=gen_length, block_length=block_length, gamma=0.1, E=strategy, temperature=0.)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    # Process all samples concurrently
    futures = []
    assert steps <= gen_length, "Steps must be less than or equal to gen_length"
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_samples) as executor:
        for i in range(num_samples):
            # Extract single sample (maintaining 2D shape [1, L])
            prompt = input_ids[i:i+1]
            mask = attention_mask[i:i+1] if attention_mask is not None else None
            
            future = executor.submit(run_single_generation, prompt, mask, steps=steps, gen_length=gen_length, block_length=block_length, mode=mode, strategy=strategy)
            futures.append(future)

    # Collect results for all samples
    for f in futures:
        if mode == "FHS":
            gen, steps_taken = f.result()
            all_generated.append(gen)
            total_steps += steps_taken
        else:
            gen, steps_taken, blocks = f.result()
            all_generated.append(gen)
            total_steps += steps_taken
            total_blocks += blocks

    # Combine all results
    final_generated = torch.cat(all_generated, dim=0)
    
    end_time = time.time()
    total_time = end_time - start_time
    if mode == "FHS":
        return final_generated, total_steps, total_time
    else:
        return final_generated, total_steps, total_blocks, total_time
