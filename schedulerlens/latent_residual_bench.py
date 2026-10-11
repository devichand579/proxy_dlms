"""Small completed arithmetic tests of the fully latent quadratic controller."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from load_model import load_model_and_tokenizer,infer_diffusion_token_ids
from schedulerlens.latent_bench import set_seed
from schedulerlens.latent_small_bench import canvases
from schedulerlens.infogain_collect import encode_problem
from schedulers import latent_residual,latent,latent_flow
from schedulers.eb_sampler import generate_EB


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('outputs/latent_residual_arithmetic'))
    parser.add_argument('--rewards',default='.25,.5,.75,1')
    parser.add_argument('--candidates',type=int,default=4)
    parser.add_argument('--target-mode',choices=('external','survivors'),default='external')
    parser.add_argument('--pool-mode',choices=('ready','spectral'),default='ready')
    parser.add_argument('--method',choices=('residual','flow'),default='residual')
    parser.add_argument('--geometry',choices=('raw','pca'),default='raw')
    parser.add_argument('--seed',type=int,nargs='+',default=[0,1])
    args = parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    rewards = [float(x) for x in args.rewards.split(',')]
    prefix = 'latent_survivors' if args.target_mode=='survivors' else 'latent_residual'
    if args.pool_mode=='spectral': prefix = prefix.replace('latent_','latent_spectral_',1)
    if args.method=='flow':prefix = 'latent_flow_defect'
    if args.method=='flow' and args.geometry=='pca':prefix = 'latent_pca_flow_defect'
    suffix = 'w' if args.method=='flow' else 'p'
    configurations = [('eb_sampler',None),('eb_target_temporal',None)]+[(f'{prefix}_{suffix}{r:g}',r) for r in rewards]
    module = latent_flow if args.method=='flow' else latent_residual
    source = Path(module.__file__).read_bytes()
    snapshots = {'scheduler_source.py':source,'benchmark_source.py':Path(__file__).read_bytes()}
    if args.method=='flow':snapshots['quadratic_solver_source.py'] = Path(latent_residual.__file__).read_bytes()
    for name,content in snapshots.items():
        existing = args.out/name
        if existing.exists() and existing.read_bytes()!=content:
            raise ValueError('existing source snapshot differs; use a new output directory')
    for name,content in snapshots.items():(args.out/name).write_bytes(content)
    model,tokenizer = load_model_and_tokenizer('models/llada',device='cuda',compile_model=False)
    mask,eos,eot = infer_diffusion_token_ids('models/llada')
    for seed in args.seed:
        folder = args.out/f'seed{seed}'; folder.mkdir(exist_ok=True)
        examples = canvases(32,seed)
        encoded = [encode_problem(tokenizer,e) for e in examples]
        width = max(len(ids) for ids,_ in encoded)
        canvas = torch.full((32,width),tokenizer.pad_token_id,device='cuda',dtype=torch.long)
        attention = torch.zeros_like(canvas)
        slots,refs = [],[]
        for row,(ids,positions) in enumerate(encoded):
            canvas[row,:len(ids)] = torch.tensor(ids,device='cuda'); attention[row,:len(ids)] = 1
            slots.append(positions); refs.append([ids[p] for p in positions])
            canvas[row,positions] = mask
        summary = []
        for name,reward in configurations:
            path = folder/(name+'.json')
            if path.exists():
                saved = json.loads(path.read_text())
                if saved['code_sha256']!=hashlib.sha256(source).hexdigest() or saved['candidates']!=args.candidates:
                    raise ValueError('existing experiment differs')
                summary.append(saved['metrics']);continue
            set_seed(1234);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
            started = time.perf_counter()
            prompt = canvas[:,:0]
            print(f'START {name} seed {seed}',flush=True)
            if name=='eb_sampler':
                output,steps,_ = generate_EB(model,prompt,attention[:,:0],initial_canvas=canvas,
                    canvas_attention_mask=attention,gen_length=width,block_length=width,steps=12,
                    gamma=.1,E='confidence',mask_id=mask,eos_id=eos,eot_id=eot,
                    use_eos_harness=False,upload_to_wandb=False)
                metrics = dict(steps_taken=steps,sequence_evaluations=32*steps)
            elif name=='eb_target_temporal':
                output,_,_ = latent.generate_latent(model,prompt,attention[:,:0],initial_canvas=canvas,
                    canvas_attention_mask=attention,gen_length=width,block_length=width,steps=12,
                    variant=name,stability_beta=.15,fisher_vocab=128,mask_id=mask,eos_id=eos,
                    eot_id=eot,use_eos_harness=False)
                metrics = dict(latent.last_metrics)
            elif args.method=='flow':
                output = latent_flow.generate_flow(model,prompt,attention[:,:0],initial_canvas=canvas,
                    full_attention=attention,gen_length=width,steps=12,mask_id=mask,eos_id=eos,
                    candidates=args.candidates,consistency_weight=reward,branch_batch_size=160,
                    geometry=args.geometry)
                metrics = dict(latent_flow.last_metrics)
            else:
                output = latent_residual.generate_residual(model,prompt,attention[:,:0],initial_canvas=canvas,
                    full_attention=attention,gen_length=width,steps=12,mask_id=mask,eos_id=eos,
                    candidates=args.candidates,progress_reward=reward,branch_batch_size=160,
                    target_mode=args.target_mode,pool_mode=args.pool_mode)
                metrics = dict(latent_residual.last_metrics)
            torch.cuda.synchronize();elapsed = time.perf_counter()-started
            samples = []
            for row,positions in enumerate(slots):
                tokens = output[row,positions].tolist()
                correct = [tokenizer.decode([a]).strip().casefold()==tokenizer.decode([b]).strip().casefold() for a,b in zip(tokens,refs[row])]
                samples.append(dict(id=examples[row]['id'],tokens=tokens,references=refs[row],slot_correct=correct,
                    correct_slots=sum(correct),correct_worksheets=sum(all(correct[i:i+3]) for i in range(0,12,3))))
            result = dict(variant=name,seed=seed,samples=32,steps=12,
                slot_accuracy=sum(s['correct_slots'] for s in samples)/384,
                worksheet_accuracy=sum(s['correct_worksheets'] for s in samples)/128,
                unfinished_masks=int((output==mask).sum()),generation_seconds=elapsed,
                forward_evaluations_per_canvas=metrics['sequence_evaluations']/32,
                backward_evaluations_per_canvas=0.,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
            payload = dict(metrics=result,samples=samples,scheduler_metrics=metrics,candidates=args.candidates,
                           target_mode=args.target_mode,
                           pool_mode=args.pool_mode,
                           geometry=args.geometry,
                           progress_reward=metrics.get('progress_reward',reward),
                           consistency_weight=metrics.get('consistency_weight'),
                           code_sha256=hashlib.sha256(source).hexdigest())
            path.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
            summary.append(result)
            (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            print('RESULT '+json.dumps(result),flush=True)
            del output


if __name__=='__main__':main()
