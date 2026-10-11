"""Matched count ablations with activation-only and diagnostic Fisher rankings."""
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
from schedulers import latent,latent_count
from schedulers.eb_sampler import generate_EB
from schedulers.utils import get_mask_by_confidence


def diagnostic_fisher(logits,hidden,previous,weight):
    """Historical scorer, used only to isolate the count change; not a candidate."""
    proposal=logits.argmax(-1)
    score=get_mask_by_confidence(logits[None],proposal[None])[0].clamp_min(1e-30).log()
    if previous is not None:
        risk=latent._target_temporal_risk(logits,hidden,previous,weight,128)
        score-=.15*(risk/risk.mean().clamp_min(1e-8)).clamp(max=4)
    return score


PLANS=[('eb_sampler',None,None,None),('eb_target_temporal',None,None,None),
       ('activation_count_fixed','fixed',1.,None),
       ('activation_count_guard025','guard',.25,None),
       ('activation_count_guard1','guard',1.,None),
       ('diagnostic_fisher_count_fixed','fixed',1.,diagnostic_fisher),
       ('diagnostic_fisher_count_guard025','guard',.25,diagnostic_fisher),
       ('diagnostic_fisher_count_guard1','guard',1.,diagnostic_fisher)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('outputs/latent_count_arithmetic'))
    parser.add_argument('--seed',type=int,nargs='+',default=[0,1])
    parser.add_argument('--growth-metric',choices=('uniform','candidate_gradients'),default='uniform')
    parser.add_argument('--budgets',default='.25,1')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    snapshots={'scheduler_source.py':Path(latent_count.__file__).read_bytes(),
               'benchmark_source.py':Path(__file__).read_bytes()}
    for name,source in snapshots.items():
        path=args.out/name
        if path.exists() and path.read_bytes()!=source:raise ValueError('source differs; choose a new output directory')
    for name,source in snapshots.items():(args.out/name).write_bytes(source)
    model,tokenizer=load_model_and_tokenizer('models/llada',device='cuda',compile_model=False)
    mask,eos,eot=infer_diffusion_token_ids('models/llada')
    projection=latent_count.decoder_geometry(model.get_output_embeddings().weight)
    torch.save(projection.cpu(),args.out/'decoder_geometry.pt')
    for seed in args.seed:
        folder=args.out/f'seed{seed}';folder.mkdir(exist_ok=True)
        examples=canvases(32,seed);encoded=[encode_problem(tokenizer,e) for e in examples]
        width=max(len(ids) for ids,_ in encoded)
        canvas=torch.full((32,width),tokenizer.pad_token_id,device='cuda',dtype=torch.long)
        attention=torch.zeros_like(canvas);slots=[];references=[]
        for row,(ids,positions) in enumerate(encoded):
            canvas[row,:len(ids)]=torch.tensor(ids,device='cuda');attention[row,:len(ids)]=1
            slots.append(positions);references.append([ids[p] for p in positions]);canvas[row,positions]=mask
        summaries=[]
        plans=[p for p in PLANS if p[1]!='guard']
        for budget in [float(v) for v in args.budgets.split(',')]:
            plans += [(f'activation_count_guard{budget:g}','guard',budget,None),
                      (f'diagnostic_fisher_count_guard{budget:g}','guard',budget,diagnostic_fisher)]
        for name,controller,budget,ranker in plans:
            path=folder/(name+'.json')
            if path.exists():summaries.append(json.loads(path.read_text())['metrics']);continue
            print(f'START {name} seed {seed}',flush=True)
            set_seed(1234);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
            prompt=canvas[:,:0]
            if name=='eb_sampler':
                output,steps,_=generate_EB(model,prompt,attention[:,:0],initial_canvas=canvas,canvas_attention_mask=attention,
                    steps=12,gen_length=width,block_length=width,gamma=.1,E='confidence',mask_id=mask,eos_id=eos,eot_id=eot,
                    use_eos_harness=False,upload_to_wandb=False)
                work=dict(sequence_evaluations=32*steps,steps_taken=steps)
            elif name=='eb_target_temporal':
                output,_,_=latent.generate_latent(model,prompt,attention[:,:0],initial_canvas=canvas,canvas_attention_mask=attention,
                    steps=12,gen_length=width,block_length=width,variant=name,stability_beta=.15,fisher_vocab=128,
                    mask_id=mask,eos_id=eos,eot_id=eot,use_eos_harness=False)
                work=dict(latent.last_metrics)
            else:
                output=latent_count.generate_count(model,prompt,attention[:,:0],initial_canvas=canvas,full_attention=attention,
                    steps=12,gen_length=width,mask_id=mask,eos_id=eos,controller=controller,budget=budget,
                    projection=projection,ranker=ranker,branch_batch_size=320,growth_metric=args.growth_metric)
                work=dict(latent_count.last_metrics)
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            samples=[]
            for row,positions in enumerate(slots):
                values=output[row,positions].tolist()
                correct=[tokenizer.decode([a]).strip().casefold()==tokenizer.decode([b]).strip().casefold() for a,b in zip(values,references[row])]
                samples.append(dict(id=examples[row]['id'],tokens=values,references=references[row],slot_correct=correct,
                    correct_slots=sum(correct),correct_worksheets=sum(all(correct[i:i+3]) for i in range(0,12,3))))
            summary=dict(variant=name,seed=seed,samples=32,steps=12,slot_accuracy=sum(s['correct_slots'] for s in samples)/384,
                worksheet_accuracy=sum(s['correct_worksheets'] for s in samples)/128,unfinished_masks=int((output==mask).sum()),
                generation_seconds=seconds,forward_evaluations_per_canvas=work['sequence_evaluations']/32,
                backward_evaluations_per_canvas=0.,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
            payload=dict(metrics=summary,samples=samples,scheduler_metrics=work,seed=seed,
                         eligible_activation_only=ranker is None and controller is not None,
                         diagnostic_only=ranker is not None,code_sha256=hashlib.sha256(snapshots['scheduler_source.py']).hexdigest())
            path.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n');summaries.append(summary)
            (folder/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
            print('RESULT '+json.dumps(summary),flush=True);del output


if __name__=='__main__':main()
