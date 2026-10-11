"""Learn greedy reveal effects using disjoint MBPP code and actual branch forwards."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from load_model import load_model_and_tokenizer, infer_diffusion_token_ids
from schedulerlens.latent_bench import load_documents, set_seed
from schedulerlens.train_latent_risk import examples
from schedulers.latent_risk import RiskHead
from schedulers.latent_value import value_features

TRAINING_SOURCE_SNAPSHOT = Path(__file__).read_bytes()


def reference_log_probs(logits, reference):
    return logits.gather(-1, reference[:, None]).squeeze(-1).float() - logits.float().logsumexp(-1)


def surviving_gain(post, baseline, positions, source):
    surviving = positions != source
    return (post[surviving] - baseline[surviving]).mean() if surviving.any() else post.new_zeros(())


@torch.no_grad()
def collect(model, data, mask, eos, hp, ep, seed, base_batch=16, sources=8):
    rng = np.random.default_rng(seed)
    states = [(ex, fraction) for ex in data for fraction in (.5, 1.)]
    features, gains, body_flags = [], [], []
    counts = {'baseline_sequence_evaluations': 0, 'branch_sequence_evaluations': 0}
    for offset in range(0, len(states), base_batch):
        chunk = states[offset:offset + base_batch]
        width = max(len(ex['prefix']) for ex, _ in chunk)
        ids = torch.full((len(chunk), width + 512), eos, dtype=torch.long, device='cuda')
        attention = torch.zeros_like(ids)
        eligible, target_positions, references = [], [], []
        for row, (ex, fraction) in enumerate(chunk):
            ids[row, width-len(ex['prefix']):width] = torch.tensor(ex['prefix'], device='cuda')
            attention[row, width-len(ex['prefix']):] = 1
            ids[row, width:width+len(ex['body'])] = torch.tensor(ex['body'], device='cuda')
            selected = rng.random(512) < fraction
            selected[rng.integers(0, len(ex['body']))] = True
            pos = torch.tensor(np.flatnonzero(selected), device='cuda')
            ids[row, width+pos] = mask
            targets = pos[pos < len(ex['body'])]
            eligible.append(pos); target_positions.append(targets)
            references.append(torch.tensor(ex['body'], device='cuda')[targets])
        captured = []
        hook = model.get_output_embeddings().register_forward_pre_hook(lambda _m, inputs: captured.append(inputs[0].detach()))
        try:
            # Match baseline and branch batch shapes. ROCm can change numerical
            # results with GEMM batch shape; those changes must not become labels.
            output = model(ids.repeat_interleave(sources, 0),
                           attention_mask=attention.repeat_interleave(sources, 0))
        finally:
            hook.remove()
        counts['baseline_sequence_evaluations'] += len(chunk) * sources
        logits, hidden = output.logits[::sources], captured[0][::sources]
        logits[..., mask] = -torch.inf
        actions, baselines = [], []
        for row, (ex, _) in enumerate(chunk):
            pos, targets = eligible[row], target_positions[row]
            baselines.append(reference_log_probs(logits[row, width+targets], references[row]))
            lg = logits[row, width+pos]
            confidence = (lg.float().amax(-1) - lg.float().logsumexp(-1)).exp()
            candidates = pos[confidence.argsort(descending=True, stable=True)[:3]].tolist()
            # Oversample substantive actions for supervision; this eligibility
            # information never enters deployment features or the scheduler.
            candidates += rng.permutation(targets.cpu().numpy())[:3].tolist()
            candidates += rng.permutation(pos.cpu().numpy()).tolist()
            chosen = list(dict.fromkeys(candidates))[:sources]
            if len(chosen) != sources:
                raise ValueError('not enough eligible sources for matched branch batch')
            chosen_tensor = torch.tensor(chosen, device='cuda')
            features.append(value_features(hidden[row, width+chosen_tensor], logits[row, width+chosen_tensor],
                model.get_input_embeddings().weight, hp, ep, eos, hidden[row, width+pos]).cpu())
            for source in chosen:
                actions.append((row, source, int(logits[row, width+source].argmax())))
                body_flags.append(source < len(ex['body']))
        branch_rows = torch.tensor([row for row, _, _ in actions], device='cuda')
        branch = ids[branch_rows].clone()
        for index, (_, source, token) in enumerate(actions):
            branch[index, width+source] = token
        del output, logits, hidden, captured, lg, confidence
        changed = model(branch, attention_mask=attention[branch_rows]).logits
        changed[..., mask] = -torch.inf
        counts['branch_sequence_evaluations'] += len(actions)
        for index, (row, source, _) in enumerate(actions):
            targets = target_positions[row]
            post = reference_log_probs(changed[index, width+targets], references[row])
            gains.append(float(surviving_gain(post, baselines[row], targets, source)))
        del changed, branch, ids, attention, baselines
        print(json.dumps({'histories': min(offset+base_batch, len(states)), 'total': len(states), **counts}), flush=True)
    return torch.cat(features), torch.tensor(gains), torch.tensor(body_flags), counts


def audit(prediction, target):
    from scipy.stats import spearmanr
    p, y = prediction.numpy(), target.numpy()
    return {'mse': float(((p-y)**2).mean()), 'pearson': float(np.corrcoef(p,y)[0,1]),
            'spearman': float(spearmanr(p,y).statistic),
            'sign_accuracy': float(((p>0)==(y>0)).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-samples', type=int, default=128)
    parser.add_argument('--validation-samples', type=int, default=32)
    parser.add_argument('--out', type=Path, default=Path('outputs/latent_value_mbpp'))
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    set_seed(91); torch.set_num_threads(4); start = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer('models/llada', device='cuda', compile_model=False)
    torch.cuda.reset_peak_memory_stats()
    mask, eos, _ = infer_diffusion_token_ids('models/llada')
    human, _ = load_documents(None, 164); excluded = {d['entry_point'] for d in human}
    train, train_path = examples(tokenizer, 'train', args.train_samples, excluded)
    validation, val_path = examples(tokenizer, 'validation', args.validation_samples, excluded)
    dimension = model.get_output_embeddings().weight.shape[1]
    hp, ep = torch.randn(dimension,128).cuda()/dimension**.5, torch.randn(dimension,32).cuda()/dimension**.5
    x,y,body,counts = collect(model, train, mask, eos, hp, ep, 91)
    vx,vy,vbody,vcounts = collect(model, validation, mask, eos, hp, ep, 92)
    torch.save({'x':x, 'y':y, 'body':body, 'vx':vx, 'vy':vy, 'vbody':vbody}, args.out/'features.pt')
    del model; torch.cuda.empty_cache()
    mean, scale = x.mean(0), x.std(0).clamp_min(.01)
    gain_scale = y.std().clamp_min(.01)
    xx, vxx = (x-mean)/scale, (vx-mean)/scale
    head = RiskHead(x.shape[1]); optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.03)
    best, best_mse = None, float('inf')
    for epoch in range(60):
        for indices in torch.randperm(len(x)).split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(head(xx[indices]), y[indices]/gain_scale)
            loss.backward(); optimizer.step()
        with torch.no_grad(): prediction = head(vxx)*gain_scale
        mse = float(((prediction-vy)**2).mean())
        if mse < best_mse:
            best_mse = mse; best = {k:v.detach().clone() for k,v in head.state_dict().items()}
    head.load_state_dict(best)
    with torch.no_grad(): prediction = head(vxx)*gain_scale
    summary = {'train_problems':len(train), 'validation_problems':len(validation),
        'train_dataset':train_path, 'validation_dataset':val_path, 'model_frozen':True,
        'excluded_humaneval_entry_points':len(excluded), 'canvas_length':512,
        'target':'mean reference log-likelihood change at surviving masked code-body slots after argmax reveal',
        'source_entropy_counted':False, 'gold_consistent_training_histories':True,
        'baseline_branch_batch_shapes_matched':True,
        'validation':audit(prediction,vy), 'body_only':audit(prediction[vbody],vy[vbody]),
        'constant_mean_mse':float(((vy-y.mean())**2).mean()), 'gain_scale':float(gain_scale),
        'train_compute':counts, 'validation_compute':vcounts, 'elapsed_seconds':time.perf_counter()-start}
    summary.update(torch=torch.__version__, hip=torch.version.hip,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        code_sha256=hashlib.sha256(TRAINING_SOURCE_SNAPSHOT).hexdigest())
    torch.save({'head':best, 'mean':mean, 'scale':scale, 'h_projection':hp.cpu(), 'e_projection':ep.cpu(),
                'metadata':summary}, args.out/'value_head.pt')
    (args.out/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    (args.out/'training_source.py').write_bytes(TRAINING_SOURCE_SNAPSHOT)
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__': main()
