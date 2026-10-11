import torch
from das import model_path
from load_model import load_model_and_tokenizer, infer_diffusion_token_ids
from das.das_sampler import generate_das
model, tok = load_model_and_tokenizer(model_path('models/llada'), device='cuda', compile_model=False)
mask, eos, eot = infer_diffusion_token_ids('models/llada')
pad = [i for i in range(len(tok)) if i != mask and tok.decode([i]).strip() == '' and tok.decode([i]) != '']
ids = torch.tensor([tok('def add(a, b):\n    """Return a+b."""\n', add_special_tokens=False)['input_ids']], device='cuda')
for cfg in (dict(block_len=32), dict(rank='margin'), dict(block_len=32, rank='margin')):
    for steps in (8, 32):
        out, m = generate_das(model, ids, None, steps=steps, gen_length=128, mask_id=mask, eos_id=eos, eot_id=eot,
                              count='pace', pad_ids=pad, **cfg)
        print(cfg, steps, m['steps_taken'], m.get('pace_extra'), m['unfinished_masks'],
              repr(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)[:80]))
