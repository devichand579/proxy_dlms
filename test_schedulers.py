import time
from load_model import load_model_and_tokenizer
from schedulers.llada_schedulers import generate_llada


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
    out_pure_diffusion, steps_pure, blocks_pure = generate_llada(model, input_ids, attention_mask, steps=256, gen_length=256, block_length=256, temperature=0., remasking='low_confidence')
    pure_diffusion_time = time.time() - start_time
    print(f"Pure diffusion generation time: {pure_diffusion_time:.2f} seconds")
    print(f"Steps taken: {steps_pure}")
    print(f"Blocks generated: {blocks_pure}")
    print(f"output Shape: {out_pure_diffusion.shape}")

    print()

    # Block diffusion generation with timing
    print("Generating with block diffusion (block_length=32)...")
    start_time = time.time()
    out_block_diffusion, steps_block, blocks_block = generate_llada(model, input_ids, attention_mask, steps=256, gen_length=256, block_length=32, temperature=0., remasking='low_confidence')
    block_diffusion_time = time.time() - start_time
    print(f"Block diffusion generation time: {block_diffusion_time:.2f} seconds")
    print(f"Steps taken: {steps_block}")
    print(f"Blocks generated: {blocks_block}")
    print(f"output Shape: {out_block_diffusion.shape}")


    output_pure_diffusion = tokenizer.batch_decode(out_pure_diffusion[:, input_ids.shape[1]:], skip_special_tokens=True)
    output_block_diffusion = tokenizer.batch_decode(out_block_diffusion[:, input_ids.shape[1]:], skip_special_tokens=True)
    for o in output_pure_diffusion:
        print(o)
        print('-' * 50)
    for o in output_block_diffusion:
        print(o)
        print('-' * 50)

if __name__ == "__main__":
    print("... Testing scheduler ...")
    main()