"""Count-isolation HumanEval study: same scorer, independent count decisions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from load_model import load_model_and_tokenizer,infer_diffusion_token_ids
from schedulerlens.latent_bench import load_documents,decode,code_results,set_seed
from schedulerlens.latent_full_bench import save
from schedulerlens.latent_count_bench import diagnostic_fisher
from schedulers import latent_count


PLANS=[('diagnostic_fisher_fixed','fixed',diagnostic_fisher),
       ('diagnostic_fisher_guard','guard',diagnostic_fisher),
       ('activation_fixed','fixed',None),('activation_guard','guard',None),
       ('diagnostic_fisher_singleton','singleton',diagnostic_fisher),('activation_singleton','singleton',None)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('outputs/latent_count_humaneval20'))
    parser.add_argument('--plans',default=','.join(p[0] for p in PLANS[:4]))
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_DATASETS_OFFLINE']='1'
    documents,dataset=load_documents(None,20)
    source_paths=[Path(__file__),Path(latent_count.__file__),Path('schedulerlens/latent_count_bench.py'),
                  Path('schedulerlens/latent_bench.py'),Path('schedulers/latent.py'),Path('schedulers/utils.py')]
    manifest=dict(task_ids=[d['task_id'] for d in documents],dataset_arrow=str(dataset),steps=128,gen_length=512,
        batch_size=20,seed=1234,budget=.01,max_count=16,growth_metric='candidate_gradients',proposal_width=4,
        branch_batch_size=160,model_config_sha256=hashlib.sha256(Path('models/llada/config.json').read_bytes()).hexdigest(),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        note='Fisher rows are confidence-using diagnostic controls only. New count never uses EB, confidence or entropy.')
    path=args.out/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('existing experiment differs')
    save(path,manifest);snapshots=args.out/'sources';snapshots.mkdir(exist_ok=True)
    for p in source_paths:(snapshots/str(p).replace('/','__')).write_bytes(p.read_bytes())
    selected=[p for p in PLANS if p[0] in args.plans.split(',')]
    if len(selected)!=len(args.plans.split(',')):raise ValueError('unknown/duplicate plan')
    model,tokenizer=load_model_and_tokenizer('models/llada',device='cuda',compile_model=False)
    mask,eos,_=infer_diffusion_token_ids('models/llada')
    geometry_path=args.out/'decoder_geometry.pt'
    if geometry_path.exists():projection=torch.load(geometry_path,weights_only=True).to('cuda')
    else:
        projection=latent_count.decoder_geometry(model.get_output_embeddings().weight)
        torch.save(projection.cpu(),geometry_path)
    encoded=[tokenizer(d['prompt'],add_special_tokens=False)['input_ids'] for d in documents]
    width=max(map(len,encoded));ids=torch.full((20,width),tokenizer.pad_token_id,device='cuda',dtype=torch.long);attention=torch.zeros_like(ids)
    for row,tokens in enumerate(encoded):
        ids[row,-len(tokens):]=torch.tensor(tokens,device='cuda');attention[row,-len(tokens):]=1
    for name,controller,ranker in selected:
        folder=args.out/name;folder.mkdir(exist_ok=True)
        result_path=folder/'results.json';generated_path=folder/'generated.json';state_path=folder/'state.pt'
        if result_path.exists():print('COMPLETED '+name,flush=True);continue
        if generated_path.exists():payload=json.loads(generated_path.read_text())
        else:
            state=torch.load(state_path,map_location='cpu',weights_only=True) if state_path.exists() else None
            prior_seconds=state.get('generation_seconds',0.) if state else 0.
            prior_peak=state.get('peak_allocated_bytes',0) if state else 0
            set_seed(1234);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
            def checkpoint(s):
                if s['next_step']%8 and s['next_step']!=128:return
                s['generation_seconds']=prior_seconds+time.perf_counter()-start
                s['peak_allocated_bytes']=max(prior_peak,torch.cuda.max_memory_allocated())
                tmp=state_path.with_suffix('.tmp');torch.save(s,tmp);tmp.replace(state_path)
                print(f'{name} step {s["next_step"]}/128, forward work {s["metrics"]["sequence_evaluations"]/20:.1f}/problem, '
                      f'{s["generation_seconds"]:.1f}s',flush=True)
            print(f'START {name}, resume {state["next_step"] if state else 0}',flush=True)
            output=latent_count.generate_count(model,ids,attention,steps=128,gen_length=512,mask_id=mask,eos_id=eos,
                controller=controller,budget=.01,max_count=16,branch_batch_size=160,projection=projection,ranker=ranker,
                growth_metric='candidate_gradients',proposal_width=4,resume_state=state,step_checkpoint=checkpoint)
            torch.cuda.synchronize();seconds=prior_seconds+time.perf_counter()-start;m=dict(latent_count.last_metrics)
            samples=[dict(task_id=d['task_id'],completion=text,unfinished_masks=int((output[row,width:]==mask).sum()))
                     for row,(d,text) in enumerate(zip(documents,decode(tokenizer,output,width,eos)))]
            payload=dict(samples=samples,scheduler_metrics=m,generation_seconds=seconds,
                         peak_allocated_bytes=max(prior_peak,torch.cuda.max_memory_allocated()))
            save(generated_path,payload);del output
        outcomes=code_results(documents,[s['completion'] for s in payload['samples']])
        for sample,outcome in zip(payload['samples'],outcomes):sample.update(outcome)
        passed=sum(s['passed'] for s in payload['samples']);m=payload['scheduler_metrics']
        config=dict(variant=name,steps=128,gen_length=512,block_length=512,batch_size=20,seed=1234,
                    budget=.01,max_count=16,controller=controller,growth_metric='candidate_gradients',proposal_width=4,
                    diagnostic_only=ranker is not None)
        metrics=dict(passed=passed,samples=20,pass_at_1=passed/20,sequence_evaluations=m['sequence_evaluations'],
            backward_sequence_evaluations=0,unfinished_masks=sum(s['unfinished_masks'] for s in payload['samples']),
            generation_seconds=payload['generation_seconds'],peak_allocated_bytes=payload['peak_allocated_bytes'])
        save(result_path,dict(run=name,config=config,metrics=metrics,samples=payload['samples'],scheduler_metrics=m))
        print('RESULT '+json.dumps(dict(method=name,passed=passed,steps_used=m['steps_taken'],
            forward_evaluations_per_sample=m['sequence_evaluations']/20,generation_seconds=metrics['generation_seconds'])),flush=True)
    summaries=[]
    for name,controller,ranker in PLANS:
        path=args.out/name/'results.json'
        if path.exists():
            p=json.loads(path.read_text());summaries.append(dict(method=name,passed=p['metrics']['passed'],
                diagnostic_only=ranker is not None,steps_used=p['scheduler_metrics']['steps_taken'],
                forward_evaluations_per_sample=p['metrics']['sequence_evaluations']/20))
    save(args.out/'summary.json',summaries)


if __name__=='__main__':main()
