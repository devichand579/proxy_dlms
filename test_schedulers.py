from load_model import load_model_and_tokenizer
from schedulers.llada_schedulers import generate_llada


def main():
    device = 'cuda'

    # Example usage
    model_dir = "./models/llada"
    model, tokenizer = load_model_and_tokenizer(model_dir, device=device, compile_model=True)

    # The LLaDA architecture theoretically supports both left-padding and right-padding. 
    # However, the sampling code implementation is simpler with left-padding.
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    # If the padding ID equals the mask ID, you need to modify our generate function to achieve correct inference.
    assert tokenizer.pad_token_id != 126336

    prompts = [ "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?"]

    encoded_outputs = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt"
    )
    input_ids = encoded_outputs['input_ids'].to(device)
    attention_mask = encoded_outputs['attention_mask'].to(device)

    out_pure_diffusion = generate_llada(model, input_ids, attention_mask, steps=1024, gen_length=1024, block_length=1024, temperature=0., remasking='low_confidence')
    out_block_diffusion = generate_llada(model, input_ids, attention_mask, steps=1024, gen_length=1024, block_length=32, temperature=0., remasking='low_confidence')
    # out = generate_block_topk(model, input_ids, attention_mask=None, gen_length=256, block_length=32, steps_per_block=32, top_k=1, metric="confidence")
    # out = generate_og(model, input_ids)
    # out = generate_EB(model, input_ids)
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