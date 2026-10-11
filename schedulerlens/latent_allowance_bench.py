"""One-forward count isolation: feedback growth and latent interaction allowances."""
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
from schedulers import latent_allowance
from schedulers.eb_sampler import generate_EB
from schedulers.latent_count import decoder_geometry


PLANS = {
    'eb_sampler': None,
    'fisher_eb': dict(ranking='diagnostic_fisher',controller='eb',metric='uniform'),
    'fisher_singleton': dict(ranking='diagnostic_fisher',controller='singleton',metric='uniform'),
    'activation_singleton': dict(ranking='activation',controller='singleton',metric='uniform'),
    'fisher_uniform_fb005': dict(metric='uniform',controller='feedback',tolerance=.005),
    'fisher_uniform_fb025': dict(metric='uniform',controller='feedback',tolerance=.025),
    'fisher_uniform_fb1': dict(metric='uniform',controller='feedback',tolerance=.1),
    'fisher_uniform_grounded025': dict(metric='uniform',controller='feedback_grounded',tolerance=.025),
    'fisher_gradient_fb005': dict(metric='gradient_union',controller='feedback',tolerance=.005),
    'fisher_gradient_fb025': dict(metric='gradient_union',controller='feedback',tolerance=.025),
    'fisher_gradient_fb1': dict(metric='gradient_union',controller='feedback',tolerance=.1),
    'fisher_gradient_grounded025': dict(metric='gradient_union',controller='feedback_grounded',tolerance=.025),
    'fisher_gradient_allow002': dict(metric='gradient_union',controller='allowance',tolerance=.002),
    'fisher_gradient_allow01': dict(metric='gradient_union',controller='allowance',tolerance=.01),
    'fisher_gradient_allow05': dict(metric='gradient_union',controller='allowance',tolerance=.05),
    'fisher_gradient_risk01': dict(metric='gradient_union',controller='risk_only',tolerance=.01),
    'fisher_gradient_pair01': dict(metric='gradient_union',controller='pair_only',tolerance=.01),
    'activation_grounded025': dict(ranking='activation',metric='gradient_union',controller='feedback_grounded',tolerance=.025),
    'activation_allow01': dict(ranking='activation',metric='gradient_union',controller='allowance',tolerance=.01),
    'fisher_relative1': dict(metric='gradient_union',controller='feedback_relative',tolerance=1.),
    'fisher_relative4': dict(metric='gradient_union',controller='feedback_relative',tolerance=4.),
    'fisher_relative8': dict(metric='gradient_union',controller='feedback_relative',tolerance=8.),
    'fisher_relative_grounded4': dict(metric='gradient_union',controller='feedback_relative_grounded',tolerance=4.),
    'fisher_uniform_relative4': dict(metric='uniform',controller='feedback_relative',tolerance=4.),
    'activation_relative4': dict(ranking='activation',metric='gradient_union',controller='feedback_relative',tolerance=4.),
    'fisher_grounded_allow01': dict(metric='gradient_union',controller='allowance_grounded',tolerance=.01),
    'fisher_grounded_allow05': dict(metric='gradient_union',controller='allowance_grounded',tolerance=.05),
    'fisher_subset_allow01': dict(metric='gradient_union',controller='allowance_subset',tolerance=.01),
    'activation_grounded_allow01': dict(ranking='activation',metric='gradient_union',controller='allowance_grounded',tolerance=.01),
    'fisher_dispersion002': dict(metric='fisher_dispersion',controller='allowance',tolerance=.002),
    'fisher_dispersion01': dict(metric='fisher_dispersion',controller='allowance',tolerance=.01),
    'fisher_dispersion05': dict(metric='fisher_dispersion',controller='allowance',tolerance=.05),
    'activation_dispersion01': dict(ranking='activation',metric='fisher_dispersion',controller='allowance',tolerance=.01),
    'context_flow_singleton': dict(ranking='context_flow',controller='singleton',metric='uniform'),
    'context_flow_cosine_singleton': dict(ranking='context_flow',controller='singleton',metric='uniform',context_mode='cosine'),
    'context_flow_allow01': dict(ranking='context_flow',controller='allowance',metric='gradient_union',tolerance=.01),
    'fisher_flow_singleton005': dict(ranking='fisher_context_flow',controller='singleton',metric='uniform',context_beta=.05),
    'fisher_flow_singleton02': dict(ranking='fisher_context_flow',controller='singleton',metric='uniform',context_beta=.2),
    'fisher_flow_dispersion05': dict(ranking='fisher_context_flow',controller='allowance',metric='fisher_dispersion',tolerance=.05,context_beta=.05),
    'fisher_singleton_eos': dict(ranking='diagnostic_fisher',controller='singleton',metric='uniform',candidate_region='committed_eos'),
    'fisher_dispersion05_eos': dict(metric='fisher_dispersion',controller='allowance',tolerance=.05,candidate_region='committed_eos'),
    'activation_allow01_eos': dict(ranking='activation',metric='gradient_union',controller='allowance',tolerance=.01,candidate_region='committed_eos'),
    'fisher_singleton_frontier32': dict(ranking='diagnostic_fisher',controller='singleton',metric='uniform',candidate_region='frontier',frontier_width=32),
    'fisher_dispersion05_frontier32': dict(metric='fisher_dispersion',controller='allowance',tolerance=.05,candidate_region='frontier',frontier_width=32),
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
    sources = [Path(__file__), Path(latent_allowance.__file__),Path('schedulers/latent_count.py'),
               Path('schedulers/eb_sampler.py'),Path('schedulerlens/latent_bench.py'),
               Path('schedulerlens/latent_small_bench.py')]
    sources += [Path('schedulers/latent.py'),Path('schedulers/utils.py'),Path('load_model.py')]
    sources += [Path('schedulers/latent_context_flow.py')]
    manifest = dict(task=args.task,steps=12 if args.task=='arithmetic' else 128,
        samples=32 if args.task=='arithmetic' else 20,
        generation_length='arithmetic canvas' if args.task=='arithmetic' else 512,
        model='models/llada',argmax=True,learned_policy=False,extra_forward_passes=False,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        model_config_sha256=hashlib.sha256(Path('models/llada/config.json').read_bytes()).hexdigest(),
        projection_width=64,metric_floor_fraction=.001,metric_clip=100.,max_count=16,
        response_quantile=.99,growth_cap=2.,feedback_safety=.8,feedback_ema=.5,
        plans=PLANS,note='Exploratory reused tasks. Fisher rankings use confidence and are diagnostics; count controls are labelled.')
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
        projection = decoder_geometry(model.get_output_embeddings().weight,width=64)
        torch.save(projection.cpu(),projection_path)
    gradient_dictionary=None
    if any((PLANS[name] or {}).get('metric')=='fisher_dispersion' for name in selected):
        gradient_dictionary=model.get_output_embeddings().weight.float()@projection
        gradient_dictionary-=gradient_dictionary.mean(0)
    head_mean=None
    if any('context_flow' in (PLANS[name] or {}).get('ranking','') for name in selected):
        with torch.no_grad():head_mean=model.get_output_embeddings().weight.float().mean(0)
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
                    output=latent_allowance.generate_allowance(model,prompt,attention,steps=steps,
                        gen_length=generation_length,mask_id=mask,projection=projection,
                        initial_canvas=canvas,full_attention=full_attention,**PLANS[name],
                        gradient_dictionary=gradient_dictionary,
                        head_mean=head_mean,
                        resume_state=state,step_checkpoint=checkpoint if args.task=='humaneval' else None)
                    work=dict(latent_allowance.last_metrics)
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
                                  unfinished_masks=int((output[row,width:]==mask).sum()),
                                  token_ids=output[row,width:].tolist())
                        for row,(d,text) in enumerate(zip(documents,decode(tokenizer,output,width,eos)))]
                    unfinished=sum(s['unfinished_masks'] for s in samples)
                payload=dict(samples=samples,scheduler_metrics=work,generation_seconds=seconds,
                    peak_allocated_bytes=max(prior_peak,torch.cuda.max_memory_allocated()),unfinished_masks=unfinished)
                save(generated_path,payload);del output
            work=payload['scheduler_metrics']
            metrics=dict(variant=name,seed=seed,samples=batch,steps=steps,
                forward_evaluations_per_sample=work['sequence_evaluations']/batch,
                backward_evaluations_per_sample=0.,generation_seconds=payload['generation_seconds'],
                partial_backward_evaluations_per_sample=work.get('partial_backward_sequence_evaluations',0)/batch,
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
                config=config,diagnostic_only=(PLANS[name] or {}).get('ranking','diagnostic_fisher') in ('diagnostic_fisher','fisher_context_flow') or (PLANS[name] or {}).get('candidate_region')=='frontier',
                eligible_activation_only=(PLANS[name] or {}).get('ranking') in ('activation','context_flow') and (PLANS[name] or {}).get('controller') not in ('singleton','eb') and (PLANS[name] or {}).get('metric')!='fisher_dispersion' and (PLANS[name] or {}).get('candidate_region')!='frontier'))
            summaries.append(metrics);save(args.out/'summary.json',summaries)
            print('RESULT '+json.dumps(metrics),flush=True)
    save(args.out/'summary.json',summaries)


if __name__=='__main__':
    main()
