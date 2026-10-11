"""Compare source-excluded entropy gradients with matched discrete reveals."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from scipy.stats import spearmanr

from load_model import load_model_and_tokenizer,infer_diffusion_token_ids
from schedulers.latent import _gradient_reveal_gain


def target_stats(logits,positions,refs):
    logp = logits[positions].float().log_softmax(-1)
    entropy = -(logp.exp()*logp).sum(-1)
    likelihood = logp.gather(-1,refs[:,None]).squeeze(-1)
    return entropy,likelihood


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,default=Path('outputs/infogain_lab/arithmetic_seed0.json'))
    parser.add_argument('--out',type=Path,default=Path('outputs/infogain_lab/gradient_effect_audit.json'))
    args = parser.parse_args(); cached = json.loads(args.cache.read_text())
    model,tokenizer = load_model_and_tokenizer('models/llada',device='cuda',compile_model=False)
    mask,_,_ = infer_diffusion_token_ids('models/llada')
    actions = [(s,i) for s in cached['states'] for i in range(len(s['positions']))]
    widths = {len(s['input_ids']) for s,_ in actions}
    if len(widths)!=1: raise ValueError('audit needs equal-length histories for matched batch shapes')
    ids = torch.tensor([s['input_ids'] for s,_ in actions],device='cuda')
    attention = torch.ones_like(ids); torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
    with torch.no_grad(): output = model(ids,attention_mask=attention).logits
    proposals = output.clone(); proposals[...,mask] = -torch.inf
    predicted = proposals.argmax(-1); del proposals
    source_positions = [torch.tensor([s['positions'][i]],device='cuda') for s,i in actions]
    baseline = []
    for row,(s,i) in enumerate(actions):
        positions = torch.tensor(s['positions'],device='cuda'); refs = torch.tensor(s['reference_tokens'],device='cuda')
        baseline.append(target_stats(output[row],positions,refs))
    del output
    work = {'forward_calls':0,'sequence_evaluations':0,'backward_passes':0,'backward_sequence_evaluations':0}
    gradient = _gradient_reveal_gain(model,ids,attention,ids==mask,source_positions,predicted,mask,work)
    # The helper differentiates the mean over all active target slots. Convert
    # that derivative back to the sum at each candidate's surviving positions.
    denominator = sum(len(s['positions'])-1 for s,_ in actions)
    branch = ids.clone()
    for row,pos in enumerate(source_positions): branch[row,pos] = predicted[row,pos]
    with torch.no_grad(): changed = model(branch,attention_mask=attention).logits
    records = []
    for row,(s,i) in enumerate(actions):
        positions = torch.tensor(s['positions'],device='cuda'); refs = torch.tensor(s['reference_tokens'],device='cuda')
        ent,lp = target_stats(changed[row],positions,refs)
        surviving = torch.arange(len(positions),device='cuda') != i
        pos = int(positions[i]); token = int(predicted[row,pos])
        records.append({'sample_id':s['sample_id'],'history':s['history'],'source':i,
            'predicted_token':token,'source_error':token!=int(refs[i]),
            'gradient_estimate':float(gradient[row,pos])*denominator,
            'actual_external_entropy_gain':float((baseline[row][0][surviving]-ent[surviving]).sum()),
            'actual_reference_loglikelihood_gain':float((lp[surviving]-baseline[row][1][surviving]).sum())})
    x = np.array([r['gradient_estimate'] for r in records]); y = np.array([r['actual_external_entropy_gain'] for r in records])
    summary = {'actions':len(records),'same_baseline_branch_gradient_batch_size':len(ids),
        'source_entropy_excluded':True,'candidate_specific_surviving_targets':True,
        'pearson':float(np.corrcoef(x,y)[0,1]),'spearman':float(spearmanr(x,y).statistic),
        'sign_agreement':float(np.mean((x>0)==(y>0))),
        'false_sharpening_actions':sum(r['actual_external_entropy_gain']>0 and r['actual_reference_loglikelihood_gain']<0 and r['source_error'] for r in records),
        'forward_sequence_evaluations':2*len(ids)+work['sequence_evaluations'],
        'backward_sequence_evaluations':work['backward_sequence_evaluations'],
        'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'elapsed_seconds':time.perf_counter()-start}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps({'summary':summary,'records':records},indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__': main()
