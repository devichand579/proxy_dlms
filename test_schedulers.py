import time
from load_model import load_model_and_tokenizer
from schedulers.llada_schedulers import generate_llada
from schedulers.top_k import generate_topk
from schedulers.fhs import generate_fhs

def main():
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

    prompt =  "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?"
                

    encoded_outputs = tokenizer(
        prompt,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt"
    )
    input_ids = encoded_outputs['input_ids'].to(device)
    attention_mask = encoded_outputs['attention_mask'].to(device)

    # Pure diffusion generation with timing
    print("Generating with pure diffusion (block_length=256)...")
    start_time = time.time()
    out_pure_diffusion, steps_pure, blocks_pure = generate_llada(model, input_ids, attention_mask, steps=256, gen_length=256, block_length=256, temperature=0., remasking='confidence')
    pure_diffusion_time = time.time() - start_time
    print(f"Pure diffusion generation time: {pure_diffusion_time:.2f} seconds")
    print(f"Steps taken: {steps_pure}")
    print(f"Blocks generated: {blocks_pure}")
    print(f"output Shape: {out_pure_diffusion.shape}")

    print()

    # Block diffusion generation with timing
    print("Generating with block diffusion (block_length=32)...")
    start_time = time.time()
    out_block_diffusion, steps_block, blocks_block = generate_llada(model, input_ids, attention_mask, steps=256, gen_length=256, block_length=256, temperature=0., remasking='confidence')
    block_diffusion_time = time.time() - start_time
    print(f"Block diffusion generation time: {block_diffusion_time:.2f} seconds")
    print(f"Steps taken: {steps_block}")
    print(f"Blocks generated: {blocks_block}")
    print(f"output Shape: {out_block_diffusion.shape}")

    print("Generating with top-k (confidence metric, block_length=256)...")
    start_time = time.time()
    out_top_k_confidence, steps_top_k_confidence, blocks_top_k_confidence = generate_topk(model, input_ids, attention_mask, steps=256, gen_length=256, block_length=256, temperature=0., top_k=1, metric='confidence')
    top_k_time = time.time() - start_time
    print(f"Top-k generation time: {top_k_time:.2f} seconds")
    print(f"Steps taken: {steps_top_k_confidence}")
    print(f"Blocks generated: {blocks_top_k_confidence}")
    print(f"output Shape: {out_top_k_confidence.shape}")

    print("Generating with top-k (margin metric, block_length=256)...")
    start_time = time.time()
    out_top_k_margin, steps_top_k_margin, blocks_top_k_margin = generate_topk(model, input_ids, attention_mask, steps=256, gen_length=256, block_length=256, temperature=0., top_k=1, metric='margin')
    top_k_time = time.time() - start_time
    print(f"Top-k generation time: {top_k_time:.2f} seconds")
    print(f"Steps taken: {steps_top_k_margin}")
    print(f"Blocks generated: {blocks_top_k_margin}")
    print(f"output Shape: {out_top_k_margin.shape}")

    print("Generating with FHS (steps=256)...")
    start_time = time.time()
    out_fhs, steps_fhs = generate_fhs(model, prompt, attention_mask=attention_mask, steps=256, gen_length=256, temperature=1.0, pass_time_to_model=True)
    fhs_time = time.time() - start_time
    print(f"FHS generation time: {fhs_time:.2f} seconds")
    print(f"Steps taken: {steps_fhs}")
    print(f"output Shape: {out_fhs.shape}")

    output_pure_diffusion = tokenizer.batch_decode(out_pure_diffusion[:, input_ids.shape[1]:], skip_special_tokens=True)
    output_block_diffusion = tokenizer.batch_decode(out_block_diffusion[:, input_ids.shape[1]:], skip_special_tokens=True)
    output_top_k_confidence = tokenizer.batch_decode(out_top_k_confidence[:, input_ids.shape[1]:], skip_special_tokens=True)
    output_top_k_margin = tokenizer.batch_decode(out_top_k_margin[:, input_ids.shape[1]:], skip_special_tokens=True)
    output_fhs = tokenizer.batch_decode(out_fhs[:, input_ids.shape[1]:], skip_special_tokens=True)

    print("Output:")
    for o in output_pure_diffusion:
        print("Pure diffusion:")
        print(o)
        print('-' * 50)
    for o in output_block_diffusion:
        print("Block diffusion:")
        print(o)
        print('-' * 50)
    for o in output_top_k_confidence:
        print("Top-k (confidence):")
        print(o)
        print('-' * 50)
    for o in output_top_k_margin:
        print("Top-k (margin):")
        print(o)
        print('-' * 50)
    for o in output_fhs:
        print("FHS:")
        print(o)
        print('-' * 50)
if __name__ == "__main__":
    print("... Testing scheduler ...")
    main()