import torch
import concurrent.futures
import time

from schedulers.llada_schedulers import generate_llada
# from .top_k import generate_block_topk
# from .eb_sampler import generate_EB

@ torch.no_grad()
def generate(model, input_ids, attention_mask, steps=256, gen_length=256, block_length=256, mode="pure_diffusion"):
    '''
        modes - pure_diffusion, top_k, eb_sampler, top_margin
    '''
    
    start_time = time.time()
    num_samples = input_ids.shape[0]
    all_generated = []
    total_steps = 0
    total_blocks = 0

    # Helper function for single generation call
    def run_single_generation(prompt, mask, steps=256, gen_length=256, block_length=256, mode="pure_diffusion"):
        if mode == "pure_diffusion":
            return generate_llada(model, prompt, mask, steps=steps, gen_length=gen_length, block_length=block_length, temperature=0., remasking='low_confidence')
        elif mode == "block_diffusion":
            return generate_llada(model, prompt, mask, steps=steps, gen_length=gen_length, block_length=block_length, temperature=0., remasking='low_confidence')
        else:
            raise ValueError(f"Unknown mode: {mode}")

    # Process all samples concurrently
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_samples) as executor:
        for i in range(num_samples):
            # Extract single sample (maintaining 2D shape [1, L])
            prompt = input_ids[i:i+1]
            mask = attention_mask[i:i+1] if attention_mask is not None else None
            
            future = executor.submit(run_single_generation, prompt, mask, steps=steps, gen_length=gen_length, block_length=block_length, mode=mode)
            futures.append(future)

    # Collect results for all samples
    for f in futures:
        gen, steps_taken, blocks = f.result()
        all_generated.append(gen)
        total_steps += steps_taken
        total_blocks += blocks

    # Combine all results
    final_generated = torch.cat(all_generated, dim=0)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return final_generated, total_steps, total_blocks, total_time
