import torch
import concurrent.futures
import contextvars
import time

from schedulers.llada_schedulers import generate_llada
from schedulers.top_k import generate_topk
from schedulers.fhs import generate_fhs
from schedulers.eb_sampler import generate_EB
from schedulers.dus import generate_dus
from schedulers.fdm import generate_fdm



@ torch.no_grad()
def generate(model, input_ids, attention_mask, steps=256, gen_length=256, block_length=256, mode="pure_diffusion", strategy="confidence", trace_name=None, upload_to_wandb=True, intervention_mode=None, intervention_steps=None, intervention_k=1):
    '''
        modes - pure_diffusion, block_diffusion, top_k_confidence, top_k_margin, FHS, EB_sampler, Dilated Scheduling, FDM
        strategies - confidence, random, top_margin, entropy
    '''
    
    start_time = time.time()
    num_samples = input_ids.shape[0]
    all_generated = []
    total_steps = 0
    total_blocks = 0

    # Helper function for single generation call
    def run_single_generation(prompt, mask, steps=256, gen_length=256, block_length=256, mode="pure_diffusion", strategy="confidence", trace_name=None, upload_to_wandb=True, intervention_mode=None, intervention_steps=None, intervention_k=1):
        common_trace_kw = dict(trace_name=trace_name, upload_to_wandb=upload_to_wandb, intervention_mode=intervention_mode, intervention_steps=intervention_steps, intervention_k=intervention_k)
        if mode == "pure_diffusion":
            assert strategy == "confidence" or strategy == "random", "strategy must be confidence or random"
            assert block_length == gen_length, "Block length must be equal to gen_length"
            return generate_llada(model, prompt, mask, steps=steps, gen_length=gen_length, block_length=block_length, temperature=0., remasking=strategy, **common_trace_kw)
        elif mode == "block_diffusion":
            assert block_length < gen_length, "Block length must be less than gen_length"
            assert strategy == "confidence" or strategy == "random", "strategy must be confidence or random"
            return generate_llada(model, prompt, mask, steps=steps, gen_length=gen_length, block_length=block_length, temperature=0., remasking=strategy, **common_trace_kw)
        elif mode == "top_k":
            assert block_length == gen_length, "Block length must be equal to gen_length"
            assert strategy == "confidence" or strategy == "top_margin" or strategy == "entropy" or strategy == "random", "strategy must be confidence, top_margin, entropy, or random"
            return generate_topk(model, prompt, mask, steps=steps, gen_length=gen_length, block_length=block_length, temperature=0., metric=strategy, **common_trace_kw)
        elif mode == "fhs":
            assert block_length == None, "Block length must be None"
            return generate_fhs(model, prompt, mask, steps=steps, gen_length=gen_length, temperature=1.0, pass_time_to_model=True, **common_trace_kw)
        elif mode == "eb_sampler":
            assert block_length == gen_length, "Block length must be equal to gen_length"
            assert strategy == "confidence" or strategy == "margin" or strategy == "entropy" , "strategy must be confidence, top_margin, entropy, or random"
            return generate_EB(model, prompt, mask, steps=steps, gen_length=gen_length, block_length=block_length, gamma=0.1, E=strategy, temperature=0., **common_trace_kw)
        elif mode == "dus":
            assert block_length <= gen_length, "Block length must be less than or equal to gen_length"
            assert strategy == "confidence" or strategy == "entropy" ,"strategy must be confidence, entropy"
            return generate_dus(model, prompt, mask, gen_length=gen_length, block_length=block_length, temperature=0., remasking=strategy, scheduler="dilated", base=2, base_skip=1, confidence_threshold=0.1, **common_trace_kw)
        elif mode == "fdm":
            assert block_length == None, "Block length must be None"
            return generate_fdm(model, prompt, mask, steps=steps, gen_length=gen_length, K1=2, N=4, eta1=0.8, eta2=0.7, gamma=0.5, temperature=0., mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False, global_scale=1.0, **common_trace_kw)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    # Process all samples concurrently. Copy context per task so worker threads see
    # the same context vars (e.g. eval's trace writer). Each task needs its own
    # Context instance because a Context cannot be entered by two threads at once.
    futures = []
    assert steps <= gen_length, "Steps must be less than or equal to gen_length"
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_samples) as executor:
        for i in range(num_samples):
            # Extract single sample (maintaining 2D shape [1, L])
            prompt = input_ids[i:i+1]
            mask = attention_mask[i:i+1] if attention_mask is not None else None

            def run_one(p=prompt, m=mask):
                return run_single_generation(
                    p, m,
                    steps=steps, gen_length=gen_length, block_length=block_length,
                    mode=mode, strategy=strategy,
                    trace_name=trace_name,
                    upload_to_wandb=upload_to_wandb,
                    intervention_mode=intervention_mode,
                    intervention_steps=intervention_steps,
                    intervention_k=intervention_k,
                )

            ctx = contextvars.copy_context()
            future = executor.submit(ctx.run, run_one)
            futures.append(future)

    # Collect results for all samples
    for f in futures:
        if mode == "fhs" or mode == "fdm":
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
    # Always return (generated, steps_taken, blocks, gen_time) for consistent unpacking in eval.py
    if mode == "fhs" or mode == "fdm":
        return final_generated, total_steps, 0, total_time
    else:
        return final_generated, total_steps, total_blocks, total_time
