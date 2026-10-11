"""Compare incoming readiness, outgoing gradients and candidate-pool size."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from load_model import load_model_and_tokenizer,infer_diffusion_token_ids
from schedulerlens.latent_bench import load_documents,code_results,decode,set_seed
from schedulers import latent


CONFIGURATIONS = [
    {'name':'fisher_control','gain_beta':0.,'pool_min':0,'temporal':True},
    {'name':'original_eb_gradient','gain_beta':.5,'pool_min':0,'temporal':False},
    {'name':'pool16_gradient','gain_beta':.5,'pool_min':16,'temporal':False},
    {'name':'pool16_combined025','gain_beta':.25,'pool_min':16,'temporal':True},
    {'name':'pool16_combined05','gain_beta':.5,'pool_min':16,'temporal':True},
]
RISK_CONFIGURATIONS = [
    {'name':'raw_combined','gain_beta':.5,'pool_min':16,'temporal':True,'risk_eta':0.,'readiness_gate':0.},
    {'name':'damage_eta1','gain_beta':.5,'pool_min':16,'temporal':True,'risk_eta':1.,'readiness_gate':0.},
    {'name':'damage_eta3','gain_beta':.5,'pool_min':16,'temporal':True,'risk_eta':3.,'readiness_gate':0.},
    {'name':'readiness_gated','gain_beta':.5,'pool_min':16,'temporal':True,'risk_eta':0.,'readiness_gate':1.},
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=int,default=20)
    parser.add_argument('--plan',choices=('pool','risk'),default='pool')
    parser.add_argument('--probe-batch-size',type=int,default=20)
    parser.add_argument('--out',type=Path,default=Path('outputs/latent_gradient_pool100'))
    args = parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    configurations = CONFIGURATIONS if args.plan == 'pool' else RISK_CONFIGURATIONS
    documents,dataset = load_documents(None,args.samples); set_seed(1234)
    model,tokenizer = load_model_and_tokenizer('models/llada',device='cuda',compile_model=False)
    mask,eos,eot = infer_diffusion_token_ids('models/llada')
    encoded = [tokenizer(d['prompt'],add_special_tokens=False)['input_ids'] for d in documents]
    width = max(map(len,encoded)); n = len(documents)
    ids = torch.full((n,width),tokenizer.pad_token_id,dtype=torch.long,device='cuda')
    attention = torch.zeros_like(ids)
    for row,tokens in enumerate(encoded):
        ids[row,-len(tokens):] = torch.tensor(tokens,device='cuda'); attention[row,-len(tokens):] = 1
    ids,attention = ids.repeat(len(configurations),1),attention.repeat(len(configurations),1)
    pools = [c['pool_min'] for c in configurations for _ in documents]
    gains = [c['gain_beta'] for c in configurations for _ in documents]
    temporal = [c['temporal'] for c in configurations for _ in documents]
    risks = [c.get('risk_eta',0.) for c in configurations for _ in documents]
    gates = [c.get('readiness_gate',0.) for c in configurations for _ in documents]
    source = Path(latent.__file__).read_bytes()
    (args.out/'scheduler_source.py').write_bytes(source)
    (args.out/'benchmark_source.py').write_bytes(Path(__file__).read_bytes())
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    start = time.perf_counter()
    def progress(m): print(f"Batch {len(ids)}: step {m['steps_taken']}/128",flush=True)
    generated,_,_ = latent.generate_latent(model,ids,attention,variant='eb_gradient_gain',
        steps=128,gen_length=512,block_length=512,mask_id=mask,eos_id=eos,eot_id=eot,
        use_eos_harness=False,probe_every=8,gain_beta=gains,gradient_pool_min=pools,
        gradient_temporal=temporal,gradient_probe_batch_size=args.probe_batch_size,
        gradient_risk_eta=risks,gradient_readiness_gate=gates,progress=progress)
    torch.cuda.synchronize(); elapsed = time.perf_counter()-start; m = latent.last_metrics
    summaries = []
    for cohort,c in enumerate(configurations):
        lo,hi = cohort*n,(cohort+1)*n
        outputs = generated[lo:hi]; texts = decode(tokenizer,outputs,width,eos)
        outcomes = code_results(documents,texts)
        samples = [{**r,'completion':s,'unfinished_masks':int((t[width:]==mask).sum())} for r,s,t in zip(outcomes,texts,outputs)]
        summary = {**c,'passed':sum(s['passed'] for s in samples),'samples':n,
            'unfinished_masks':sum(s['unfinished_masks'] for s in samples),
            'forward_evaluations_per_sample':sum(m['forward_work_by_row'][lo:hi])/n,
            'backward_evaluations_per_sample':sum(m['backward_work_by_row'][lo:hi])/n,
            'mean_measured_candidates':sum(m['gradient_sources_by_row'][lo:hi])/max(sum(m['gradient_measurements_by_row'][lo:hi]),1)}
        summaries.append(summary)
        (args.out/(c['name']+'.json')).write_text(json.dumps({'summary':summary,'samples':samples},indent=2)+'\n')
        print(json.dumps(summary),flush=True)
    if sum(m['forward_work_by_row'])!=m['sequence_evaluations'] or sum(m['backward_work_by_row'])!=m['backward_sequence_evaluations']:
        raise ValueError('per-row compute accounting differs from total')
    (args.out/'results.json').write_text(json.dumps({'summaries':summaries,'steps':128,'gen_length':512,
        'dataset_arrow':str(dataset),'task_ids':[d['task_id'] for d in documents],'total_batch_size':len(ids),
        'selection':'first canonical HumanEval rows, reused for exploratory tuning','seed':1234,
        'plan':args.plan,
        'gradient_probe_batch_size':args.probe_batch_size,'generation_seconds_all_cohorts':elapsed,
        'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,'code_sha256':hashlib.sha256(source).hexdigest(),
        'scheduler_metrics':m},indent=2,allow_nan=False)+'\n')


if __name__ == '__main__': main()
