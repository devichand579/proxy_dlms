import time
import argparse
from load_model import load_model_and_tokenizer
from schedulers.llada_schedulers import generate_llada
from schedulers.top_k import generate_topk
from schedulers.fhs import generate_fhs
from schedulers.eb_sampler import generate_EB
from schedulers.dus import generate_dus
from schedulers.fdm import generate_fdm


def test_pure_diffusion(model, input_ids, attention_mask, tokenizer):
    """Test pure diffusion scheduler"""
    print("Generating with pure diffusion (block_length=256)...")
    start_time = time.time()
    out_pure_diffusion, steps_pure, blocks_pure = generate_llada(
        model, input_ids, attention_mask, steps=256, gen_length=256, 
        block_length=256, temperature=0., remasking='confidence'
    )
    pure_diffusion_time = time.time() - start_time
    print(f"Pure diffusion generation time: {pure_diffusion_time:.2f} seconds")
    print(f"Steps taken: {steps_pure}")
    print(f"Blocks generated: {blocks_pure}")
    print(f"output Shape: {out_pure_diffusion.shape}")
    print()
    
    output = tokenizer.batch_decode(out_pure_diffusion[:, input_ids.shape[1]:], skip_special_tokens=True)
    return output, "Pure diffusion"


def test_block_diffusion(model, input_ids, attention_mask, tokenizer):
    """Test block diffusion scheduler"""
    print("Generating with block diffusion (block_length=32)...")
    start_time = time.time()
    out_block_diffusion, steps_block, blocks_block = generate_llada(
        model, input_ids, attention_mask, steps=256, gen_length=256, 
        block_length=32, temperature=0., remasking='confidence'
    )
    block_diffusion_time = time.time() - start_time
    print(f"Block diffusion generation time: {block_diffusion_time:.2f} seconds")
    print(f"Steps taken: {steps_block}")
    print(f"Blocks generated: {blocks_block}")
    print(f"output Shape: {out_block_diffusion.shape}")
    print()
    
    output = tokenizer.batch_decode(out_block_diffusion[:, input_ids.shape[1]:], skip_special_tokens=True)
    return output, "Block diffusion"


def test_top_k_confidence(model, input_ids, attention_mask, tokenizer):
    """Test top-k scheduler with confidence metric"""
    print("Generating with top-k (confidence metric, block_length=256)...")
    start_time = time.time()
    out_top_k_confidence, steps_top_k_confidence, blocks_top_k_confidence = generate_topk(
        model, input_ids, attention_mask, steps=256, gen_length=256, 
        block_length=256, temperature=0., top_k=1, metric='confidence'
    )
    top_k_time = time.time() - start_time
    print(f"Top-k generation time: {top_k_time:.2f} seconds")
    print(f"Steps taken: {steps_top_k_confidence}")
    print(f"Blocks generated: {blocks_top_k_confidence}")
    print(f"output Shape: {out_top_k_confidence.shape}")
    print()
    
    output = tokenizer.batch_decode(out_top_k_confidence[:, input_ids.shape[1]:], skip_special_tokens=True)
    return output, "Top-k (confidence)"


def test_top_k_margin(model, input_ids, attention_mask, tokenizer):
    """Test top-k scheduler with margin metric"""
    print("Generating with top-k (margin metric, block_length=256)...")
    start_time = time.time()
    out_top_k_margin, steps_top_k_margin, blocks_top_k_margin = generate_topk(
        model, input_ids, attention_mask, steps=256, gen_length=256, 
        block_length=256, temperature=0., top_k=1, metric='top_margin'
    )
    top_k_time = time.time() - start_time
    print(f"Top-k generation time: {top_k_time:.2f} seconds")
    print(f"Steps taken: {steps_top_k_margin}")
    print(f"Blocks generated: {blocks_top_k_margin}")
    print(f"output Shape: {out_top_k_margin.shape}")
    print()
    
    output = tokenizer.batch_decode(out_top_k_margin[:, input_ids.shape[1]:], skip_special_tokens=True)
    return output, "Top-k (margin)"


def test_fhs(model, input_ids, attention_mask, tokenizer):
    """Test FHS scheduler"""
    print("Generating with FHS (steps=256)...")
    start_time = time.time()
    out_fhs, steps_fhs = generate_fhs(
        model, input_ids, attention_mask=attention_mask, steps=256, 
        gen_length=256, temperature=1.0, pass_time_to_model=True
    )
    fhs_time = time.time() - start_time
    print(f"FHS generation time: {fhs_time:.2f} seconds")
    print(f"Steps taken: {steps_fhs}")
    print(f"output Shape: {out_fhs.shape}")
    print()
    
    output = tokenizer.batch_decode(out_fhs[:, input_ids.shape[1]:], skip_special_tokens=True)
    return output, "FHS"


def test_eb_sampler(model, input_ids, attention_mask, tokenizer):
    """Test EB sampler scheduler"""
    print("Generating with EB_sampler (confidence metric, block_length=256)...")
    start_time = time.time()
    out_eb_sampler, steps_eb_sampler, blocks_eb_sampler = generate_EB(
        model, input_ids, attention_mask, steps=256, gen_length=256, 
        block_length=256, gamma=0.1, E='confidence', temperature=0.
    )
    eb_sampler_time = time.time() - start_time
    print(f"EB_sampler generation time: {eb_sampler_time:.2f} seconds")
    print(f"Steps taken: {steps_eb_sampler}")
    print(f"Blocks generated: {blocks_eb_sampler}")
    print(f"output Shape: {out_eb_sampler.shape}")
    print()
    
    output = tokenizer.batch_decode(out_eb_sampler[:, input_ids.shape[1]:], skip_special_tokens=True)
    return output, "EB_sampler (confidence)"


def test_dus(model, input_ids, attention_mask, tokenizer):
    """Test DUS scheduler"""
    print("Generating with DUS (binary search schedule, block_length=32)...")
    start_time = time.time()
    out_dus, steps_dus, blocks_dus = generate_dus(
        model, input_ids, attention_mask, gen_length=256, 
        block_length=32, temperature=0., remasking='confidence', scheduler='dilated'
    )
    dus_time = time.time() - start_time
    print(f"DUS generation time: {dus_time:.2f} seconds")
    print(f"Steps taken: {steps_dus}")
    print(f"Blocks generated: {blocks_dus}")
    print(f"output Shape: {out_dus.shape}")
    print()
    
    output = tokenizer.batch_decode(out_dus[:, input_ids.shape[1]:], skip_special_tokens=True)
    return output, "DUS (dilated)"


def test_fdm(model, input_ids, attention_mask, tokenizer):
    """Test FDM scheduler"""
    print("Generating with FDM (steps=256)...")
    start_time = time.time()
    out_fdm, steps_fdm = generate_fdm(
        model, input_ids, attention_mask, steps=256, gen_length=256,
    )
    fdm_time = time.time() - start_time
    print(f"FDM generation time: {fdm_time:.2f} seconds")
    print(f"Steps taken: {steps_fdm}")
    print(f"output Shape: {out_fdm.shape}")
    print()
    
    output = tokenizer.batch_decode(out_fdm[:, input_ids.shape[1]:], skip_special_tokens=True)
    return output, "FDM"

def main(scheduler_name=None):
    device = 'cuda'

    # Example usage
    model_dir = "./models/llada"
    model, tokenizer = load_model_and_tokenizer(model_dir, device=device, compile_model=False)

    # The LLaDA architecture theoretically supports both left-padding and right-padding. 
    # However, the sampling code implementation is simpler with left-padding.
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    # If the padding ID equals the mask ID, you need to modify our generate function to achieve correct inference.
    assert tokenizer.pad_token_id != 126336

    prompt = "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?"

    encoded_outputs = tokenizer(
        prompt,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt"
    )
    input_ids = encoded_outputs['input_ids'].to(device)
    attention_mask = encoded_outputs['attention_mask'].to(device)

    # Define available schedulers
    schedulers = {
        'pure_diffusion': test_pure_diffusion,
        'block_diffusion': test_block_diffusion,
        'top_k_confidence': test_top_k_confidence,
        'top_k_margin': test_top_k_margin,
        'fhs': test_fhs,
        'eb_sampler': test_eb_sampler,
        'dus': test_dus,
        'fdm': test_fdm,
    }
    
    # Determine which schedulers to run
    if scheduler_name:
        if scheduler_name not in schedulers:
            print(f"Error: Unknown scheduler '{scheduler_name}'")
            print(f"Available schedulers: {', '.join(schedulers.keys())}")
            return
        schedulers_to_run = {scheduler_name: schedulers[scheduler_name]}
    else:
        schedulers_to_run = schedulers
    
    # Run selected schedulers
    results = []
    for name, test_func in schedulers_to_run.items():
        try:
            output, label = test_func(model, input_ids, attention_mask, tokenizer)
            results.append((label, output))
        except Exception as e:
            print(f"Error running {name}: {e}")
            print()
    
    # Print outputs
    if results:
        print("Output:")
        for label, output in results:
            for o in output:
                print(f"{label}:")
                print(o)
                print('-' * 50)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test different schedulers')
    parser.add_argument(
        '--scheduler', 
        type=str, 
        default=None,
        choices=['pure_diffusion', 'block_diffusion', 'top_k_confidence', 'top_k_margin', 'fhs', 'eb_sampler', 'dus', 'fdm'],
        help='Scheduler name to test. If not provided, all schedulers will be tested.'
    )
    args = parser.parse_args()
    
    print("... Testing scheduler ...")
    if args.scheduler:
        print(f"Running scheduler: {args.scheduler}")
    else:
        print("Running all schedulers")
    print()
    
    main(args.scheduler)