"""One-pass three-term scheduler: matched arithmetic and HumanEval ablations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from load_model import load_model_and_tokenizer, infer_diffusion_token_ids
from schedulerlens.infogain_collect import encode_problem
from schedulerlens.latent_bench import code_results, decode, load_documents, set_seed
from schedulerlens.latent_full_bench import save
from schedulerlens.latent_small_bench import canvases
from schedulers import latent_objective
from schedulers.eb_sampler import generate_EB
from schedulers.latent_count import decoder_geometry


PLANS = {
    'eb_sampler': None,
    'full_r005': dict(progress_reward=.05),
    'full_r015': dict(progress_reward=.15),
    'full_r035': dict(progress_reward=.35),
    'full_r075': dict(progress_reward=.75),
    'no_incoming': dict(progress_reward=.15, incoming_weight=0.),
    'no_dependence': dict(progress_reward=.15, dependence_weight=0.),
    'no_information': dict(progress_reward=.15, gain_weight=0.),
    'no_information_r035': dict(progress_reward=.35, gain_weight=0.),
    'singleton': dict(progress_reward=.15, singleton=True),
    'temporal_only': dict(progress_reward=.15, curvature_weight=0., temporal_weight=1.),
    'depth_only': dict(progress_reward=.15, temporal_weight=0.),
    'grad_curvature_r005': dict(risk_measure='proposal_curvature',progress_reward=.05),
    'grad_curvature_r015': dict(risk_measure='proposal_curvature',progress_reward=.15),
    'grad_curvature_singleton': dict(risk_measure='proposal_curvature',singleton=True),
    'grad_support_r005': dict(risk_measure='proposal_support',progress_reward=.05),
    'grad_support_r015': dict(risk_measure='proposal_support',progress_reward=.15),
    'grad_support_singleton': dict(risk_measure='proposal_support',singleton=True),
    'grad_support_noinformation': dict(risk_measure='proposal_support',gain_weight=0.,progress_reward=.05),
    'grad_curvature_noinformation': dict(risk_measure='proposal_curvature',gain_weight=0.,progress_reward=.05),
    'grad_curvature_nogain_r015': dict(risk_measure='proposal_curvature',gain_weight=0.,progress_reward=.15),
    'grad_curvature_g0025_r005': dict(risk_measure='proposal_curvature',gain_weight=.025,progress_reward=.05),
    'grad_curvature_g010_r005': dict(risk_measure='proposal_curvature',gain_weight=.1,progress_reward=.05),
    'grad_curvature_g0025_r015': dict(risk_measure='proposal_curvature',gain_weight=.025,progress_reward=.15),
    'grad_curvature_g0025_r035': dict(risk_measure='proposal_curvature',gain_weight=.025,progress_reward=.35),
    'grad_curvature_nogain_singleton': dict(risk_measure='proposal_curvature',gain_weight=0.,singleton=True),
    'grad_curvature_noincoming': dict(risk_measure='proposal_curvature',gain_weight=0.,progress_reward=.05,incoming_weight=0.),
    'grad_centered': dict(risk_measure='proposal_curvature',covariance_mode='centered',gain_weight=.1,progress_reward=.05),
    'grad_risk_scaled': dict(risk_measure='proposal_curvature',covariance_mode='risk_scaled',gain_weight=.1,progress_reward=.05),
    'context_centered': dict(risk_measure='context_curvature',curvature_weight=.05,covariance_mode='centered',gain_weight=.1,progress_reward=.05),
    'context_centered_r015': dict(risk_measure='context_curvature',curvature_weight=.05,covariance_mode='centered',gain_weight=.1,progress_reward=.15),
    'context_raw': dict(risk_measure='context_curvature',curvature_weight=.05,gain_weight=.1,progress_reward=.15),
    'context_only_centered': dict(risk_measure='context_curvature',curvature_weight=0.,covariance_mode='centered',gain_weight=.1,progress_reward=.15),
    'context_singleton': dict(risk_measure='context_curvature',curvature_weight=.05,covariance_mode='centered',gain_weight=.1,progress_reward=.15,singleton=True),
    'temporal_risk_scaled': dict(risk_measure='proposal_curvature',curvature_weight=0.,temporal_weight=1.,covariance_mode='risk_scaled',gain_weight=.1,progress_reward=.05),
    'temporal_scaled_gain1': dict(risk_measure='proposal_curvature',curvature_weight=0.,temporal_weight=1.,covariance_mode='risk_scaled',gain_weight=1.,progress_reward=.05),
    'temporal_scaled_singleton': dict(risk_measure='proposal_curvature',curvature_weight=0.,temporal_weight=1.,covariance_mode='risk_scaled',gain_weight=.1,progress_reward=.05,singleton=True),
    'context_nogain': dict(risk_measure='context_curvature',curvature_weight=.05,covariance_mode='centered',gain_weight=0.,progress_reward=.15),
    'context_nogain_singleton': dict(risk_measure='context_curvature',curvature_weight=.05,covariance_mode='centered',gain_weight=0.,progress_reward=.15,singleton=True),
    'context_smallgain': dict(risk_measure='context_curvature',curvature_weight=.05,covariance_mode='centered',gain_weight=.01,progress_reward=.15),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('arithmetic','humaneval'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--plans', default=','.join(PLANS))
    parser.add_argument('--seeds', type=int, nargs='+', default=[0,1])
    parser.add_argument('--samples', type=int, default=20)
    args = parser.parse_args()
    selected = args.plans.split(',')
    if len(set(selected)) != len(selected) or any(name not in PLANS for name in selected):
        parser.error('unknown/duplicate plans')
    if args.task == 'humaneval' and args.samples != 20:
        parser.error('this matched standalone benchmark requires 20 samples')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    args.out.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), Path(latent_objective.__file__),Path('schedulers/latent_count.py'),
               Path('schedulers/eb_sampler.py'),Path('schedulerlens/latent_bench.py'),
               Path('schedulerlens/latent_small_bench.py')]
    manifest = dict(task=args.task,steps=12 if args.task=='arithmetic' else 128,
        samples=32 if args.task=='arithmetic' else 20,
        generation_length='arithmetic canvas' if args.task=='arithmetic' else 512,
        model='models/llada',argmax=True,learned_policy=False,extra_forward_passes=False,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        model_config_sha256=hashlib.sha256(Path('models/llada/config.json').read_bytes()).hexdigest(),
        projection_width=16,rank=16,noise=1.,risk_weight=.15,gain_weight=1.,dependence_weight=1.,
        incoming_weight=.25,temporal_weight=.15,curvature_weight=1.,max_count=16,
        plans=PLANS,note='Exploratory: reused tasks. EB is baseline only.')
    manifest_path = args.out/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('source/settings changed; choose a new output directory')
    save(manifest_path, manifest)
    snapshots = args.out/'sources';snapshots.mkdir(exist_ok=True)
    for p in sources:
        (snapshots/str(p).replace('/','__')).write_bytes(p.read_bytes())
    model, tokenizer = load_model_and_tokenizer('models/llada',device='cuda',compile_model=False)
    mask,eos,eot = infer_diffusion_token_ids('models/llada')
    projection_path = args.out/'decoder_geometry.pt'
    if projection_path.exists():
        projection = torch.load(projection_path,weights_only=True).to('cuda')
    else:
        projection = decoder_geometry(model.get_output_embeddings().weight,width=16)
        torch.save(projection.cpu(),projection_path)
    readout_mean = model.get_output_embeddings().weight.float().mean(0)
    seeds = args.seeds if args.task=='arithmetic' else [1234]
    summaries = []
    for seed in seeds:
        if args.task=='arithmetic':
            examples = canvases(32,seed)
            encoded = [encode_problem(tokenizer,e) for e in examples]
            width = max(len(ids) for ids,_ in encoded)
            canvas = torch.full((32,width),tokenizer.pad_token_id,device='cuda',dtype=torch.long)
            full_attention = torch.zeros_like(canvas)
            slots=[];references=[]
            for row,(ids,positions) in enumerate(encoded):
                canvas[row,:len(ids)] = torch.tensor(ids,device='cuda')
                full_attention[row,:len(ids)] = 1
                slots.append(positions);references.append([ids[p] for p in positions])
                canvas[row,positions] = mask
            prompt = canvas[:,:0]; attention = full_attention[:,:0]
            generation_length=width;steps=12;batch=32
        else:
            documents, dataset = load_documents(None,20)
            encoded = [tokenizer(d['prompt'],add_special_tokens=False)['input_ids'] for d in documents]
            width=max(map(len,encoded))
            prompt=torch.full((20,width),tokenizer.pad_token_id,device='cuda',dtype=torch.long)
            attention=torch.zeros_like(prompt)
            for row,ids in enumerate(encoded):
                prompt[row,-len(ids):]=torch.tensor(ids,device='cuda');attention[row,-len(ids):]=1
            canvas=None;full_attention=None;generation_length=512;steps=128;batch=20
        for name in selected:
            folder=args.out/f'seed{seed}'/name;folder.mkdir(parents=True,exist_ok=True)
            result_path=folder/'results.json';generated_path=folder/'generated.json';state_path=folder/'state.pt'
            if result_path.exists():
                summaries.append(json.loads(result_path.read_text())['metrics']);continue
            if generated_path.exists():
                payload=json.loads(generated_path.read_text())
            else:
                set_seed(1234);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
                state = (torch.load(state_path,map_location='cpu',weights_only=True)
                         if state_path.exists() and PLANS[name] is not None else None)
                prior_seconds = state.get('generation_seconds',0.) if state else 0.
                prior_peak = state.get('peak_allocated_bytes',0) if state else 0
                start=time.perf_counter()
                def checkpoint(s):
                    if s['next_step']%16 and s['next_step']!=steps:return
                    s['generation_seconds']=prior_seconds+time.perf_counter()-start
                    s['peak_allocated_bytes']=max(prior_peak,torch.cuda.max_memory_allocated())
                    temporary=state_path.with_suffix('.tmp');torch.save(s,temporary);temporary.replace(state_path)
                    print(f'{name} seed {seed}: step {s["next_step"]}/{steps}, '
                          f'{s["generation_seconds"]:.1f}s',flush=True)
                print(f'START {name} seed {seed}, resume {state["next_step"] if state else 0}',flush=True)
                if name=='eb_sampler':
                    output,steps_used,_=generate_EB(model,prompt,attention,steps=steps,
                        gen_length=generation_length,block_length=generation_length,
                        initial_canvas=canvas,canvas_attention_mask=full_attention,
                        gamma=.1,E='confidence',mask_id=mask,eos_id=eos,eot_id=eot,
                        use_eos_harness=False,upload_to_wandb=False)
                    work=dict(steps_taken=steps_used,sequence_evaluations=batch*steps_used,
                              reveal_counts=[],confidence_scoring=True,eb_count=True)
                else:
                    output=latent_objective.generate_objective(model,prompt,attention,steps=steps,
                        gen_length=generation_length,mask_id=mask,projection=projection,
                        initial_canvas=canvas,full_attention=full_attention,**PLANS[name],
                        readout_mean=readout_mean,
                        resume_state=state,step_checkpoint=checkpoint if args.task=='humaneval' else None)
                    work=dict(latent_objective.last_metrics)
                torch.cuda.synchronize();seconds=prior_seconds+time.perf_counter()-start
                if args.task=='arithmetic':
                    samples=[]
                    for row,positions in enumerate(slots):
                        values=output[row,positions].tolist()
                        correct=[tokenizer.decode([a]).strip().casefold()==tokenizer.decode([b]).strip().casefold()
                                 for a,b in zip(values,references[row])]
                        samples.append(dict(id=examples[row]['id'],tokens=values,references=references[row],
                            slot_correct=correct,correct_slots=sum(correct),
                            correct_worksheets=sum(all(correct[i:i+3]) for i in range(0,12,3))))
                    unfinished=int(((output==mask)&full_attention.bool()).sum())
                else:
                    samples=[dict(task_id=d['task_id'],completion=text,
                                  unfinished_masks=int((output[row,width:]==mask).sum()))
                        for row,(d,text) in enumerate(zip(documents,decode(tokenizer,output,width,eos)))]
                    unfinished=sum(s['unfinished_masks'] for s in samples)
                payload=dict(samples=samples,scheduler_metrics=work,generation_seconds=seconds,
                    peak_allocated_bytes=max(prior_peak,torch.cuda.max_memory_allocated()),unfinished_masks=unfinished)
                save(generated_path,payload);del output
            work=payload['scheduler_metrics']
            metrics=dict(variant=name,seed=seed,samples=batch,steps=steps,
                forward_evaluations_per_sample=work['sequence_evaluations']/batch,
                backward_evaluations_per_sample=0.,generation_seconds=payload['generation_seconds'],
                peak_allocated_GiB=payload['peak_allocated_bytes']/2**30,
                unfinished_masks=payload['unfinished_masks'])
            if args.task=='arithmetic':
                metrics.update(slot_accuracy=sum(s['correct_slots'] for s in payload['samples'])/384,
                    worksheet_accuracy=sum(s['correct_worksheets'] for s in payload['samples'])/128)
            else:
                outcomes=code_results(documents,[s['completion'] for s in payload['samples']])
                for sample,outcome in zip(payload['samples'],outcomes):sample.update(outcome)
                metrics.update(passed=sum(s['passed'] for s in payload['samples']))
                metrics['pass_at_1']=metrics['passed']/batch
            counts=work['reveal_counts']
            metrics['mean_count']=sum(map(sum,counts))/sum(map(len,counts)) if counts else None
            metrics['forced_singletons']=sum(work.get('forced_singletons_by_row',[]))
            config=dict(variant=name,steps=steps,gen_length=generation_length,
                        block_length=generation_length,batch_size=batch,seed=seed,**(PLANS[name] or {}))
            metrics['sequence_evaluations']=work['sequence_evaluations']
            metrics['backward_sequence_evaluations']=0
            save(result_path,dict(metrics=metrics,samples=payload['samples'],scheduler_metrics=work,
                config=config,activation_only_scoring=name!='eb_sampler',
                eligible_activation_only=name!='eb_sampler' and not (PLANS[name] or {}).get('singleton',False)))
            summaries.append(metrics);save(args.out/'summary.json',summaries)
            print('RESULT '+json.dumps(metrics),flush=True)
    save(args.out/'summary.json',summaries)


if __name__=='__main__':
    main()
