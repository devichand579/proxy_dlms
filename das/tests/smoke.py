"""Smoke test: every DAS mode runs, commits >=1 token/step, counts passes."""
import json, torch
torch.set_grad_enabled(False)
from das import model_path
from load_model import load_model_and_tokenizer, infer_diffusion_token_ids
from das import das_sampler
from das.das_sampler import _lookahead

model, tok = load_model_and_tokenizer(model_path('models/llada'), device='cuda', compile_model=False)
mask, eos, eot = infer_diffusion_token_ids('models/llada')
prompts = ['def add(a, b):\n    """Return a + b."""\n', 'def is_even(n):\n    """True if n is even."""\n']
enc = [tok(p, add_special_tokens=False)['input_ids'] for p in prompts]
w = max(map(len, enc))
ids = torch.full((2, w), tok.pad_token_id, device='cuda'); att = torch.zeros_like(ids)
for r, e in enumerate(enc):
    ids[r, -len(e):] = torch.tensor(e, device='cuda'); att[r, -len(e):] = 1
common = dict(steps=16, gen_length=64, mask_id=mask, eos_id=eos, eot_id=eot)
for cfg in [dict(count='eb'), dict(count='singleton'), dict(count='fixed', dep='none'),
            dict(count='fixed', dep='lookahead', m=6), dict(count='das', alpha=0.3, gamma=0.3),
            dict(count='das', dep='none', alpha=0.3)]:
    x, m = das_sampler.generate_das(model, ids, att, **common, **cfg)
    assert all(c >= min(16, 64) or c >= m['steps_taken'] for c in m['commits_by_row']), m['commits_by_row']
    assert all(f >= m['steps_taken'] for f in m['forward_work_by_row'])
    print(json.dumps(cfg), 'commits', m['commits_by_row'], 'nfe', m['forward_work_by_row'],
          'fired', m['gate_fired'], 'ext', m['extended_beyond_eb'], 'dep_med', m['dep_max_median'])
    print('   ', repr(tok.decode(x[0, w:], skip_special_tokens=True)[:80]))

# Lookahead sanity: clamping to the current argmax at an already-revealed-like position is a no-op
x = torch.full((1, w + 32), mask, device='cuda'); x[:, :w] = ids[:1]; a1 = torch.cat([att[:1], torch.ones(1, 32, device='cuda', dtype=att.dtype)], -1)
captured = []
def fwd(i, a, count=True):
    lg = model(i, attention_mask=a).logits; lg[..., mask] = -torch.inf; return lg
pred = fwd(x, a1).argmax(-1)
cands = [torch.arange(w, w + 4, device='cuda')]
met = dict(forward_work_by_row=[0], lookahead_sequences=0, lookahead_steps_by_row=[0])
D = _lookahead(model, fwd, x, a1, pred, [0], cands, 40, met)[0]
print('D (4 adjacent candidates):\n', D)
# same input twice -> control vs identical copy must give ~0
x2 = x.clone(); x2[0, w] = pred[0, w]
D2 = _lookahead(model, fwd, x2, a1, x2, [0], [torch.tensor([w, w + 1], device='cuda')], 40, met)[0]
print('D with candidate already committed (should be ~0 in row 0):', D2)
assert float(D2[0].abs().max()) < 1e-3
print('OK')
