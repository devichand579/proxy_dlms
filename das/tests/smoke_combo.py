"""Smoke + equivalence: our base='allowance' / score='info' vs the mentor's own schedulers."""
import torch
from das import model_path
from load_model import load_model_and_tokenizer, infer_diffusion_token_ids
from das.das_sampler import generate_das
from schedulers.latent_allowance import generate_allowance
from schedulers.latent_information import generate_information
model, tok = load_model_and_tokenizer(model_path('models/llada'), device='cuda', compile_model=False)
mask, eos, eot = infer_diffusion_token_ids('models/llada')
pad = [i for i in range(len(tok)) if i != mask and tok.decode([i]).strip() == '' and tok.decode([i]) != '']
texts = ['def add(a, b):\n    """Return a+b."""\n', 'def is_even(n):\n    """Return True if n is even."""\n']
enc = [tok(t, add_special_tokens=False)['input_ids'] for t in texts]
w = max(map(len, enc))
ids = torch.full((2, w), tok.pad_token_id, device='cuda'); att = torch.zeros_like(ids)
for r, e in enumerate(enc):
    ids[r, -len(e):] = torch.tensor(e); att[r, -len(e):] = 1
G, S = 64, 16
dec = lambda o: [repr(tok.decode(o[r, w:], skip_special_tokens=True)[:60]) for r in range(2)]

ours, m = generate_das(model, ids, att, steps=S, gen_length=G, mask_id=mask, eos_id=eos, eot_id=eot, count='eb', base='allowance')
ref = generate_allowance(model, ids, att, steps=S, gen_length=G, mask_id=mask, ranking='diagnostic_fisher',
                         controller='allowance', metric='fisher_dispersion', tolerance=.5)
print('allowance  same canvas:', torch.equal(ours, ref), m['commits_by_row'], dec(ours))

ours, m = generate_das(model, ids, att, steps=S, gen_length=G, mask_id=mask, eos_id=eos, eot_id=eot, count='eb',
                       score='info', info_chunk=320)
ref = generate_information(model, ids, att, steps=S, gen_length=G, mask_id=mask, eos_id=eos, branch_batch_size=320,
                           incoming_beta=.15)
print('info       same canvas:', torch.equal(ours, ref), m['forward_work_by_row'], m.get('info_probes'), dec(ours))

for cfg in (dict(base='allowance'), dict(score='info'), dict(base='allowance', use_fisher=False)):
    out, m = generate_das(model, ids, att, steps=8, gen_length=G, mask_id=mask, eos_id=eos, eot_id=eot,
                          count='pace', pad_ids=pad, **cfg)
    print('pace', cfg, m['forward_work_by_row'], m.get('pace_extra'), m['unfinished_masks'], dec(out))
