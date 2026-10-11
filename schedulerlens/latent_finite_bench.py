"""Actual singleton-reveal effects versus the Fisher/EB control."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from load_model import load_model_and_tokenizer,infer_diffusion_token_ids
from schedulerlens.latent_bench import load_documents,code_results,decode,set_seed
from schedulers import latent_finite,latent_information,latent_conditioning,latent_depth,latent_attention,latent_consistency

CONFIGURATIONS = [
    {'name':'fisher_control','gain_beta':0.,'risk_eta':0.,'target_balance':False,'pool_balance':False},
    {'name':'finite_all_targets','gain_beta':.5,'risk_eta':1.,'target_balance':False,'pool_balance':False},
    {'name':'finite_balanced_targets','gain_beta':.5,'risk_eta':1.,'target_balance':True,'pool_balance':False},
    {'name':'finite_balanced_sources_targets','gain_beta':.5,'risk_eta':1.,'target_balance':True,'pool_balance':True},
]
INFORMATION_CONFIGURATIONS = [
    {'name':'information_fisher_control','gain_beta':0.,'risk_eta':0.,'incoming_beta':0.},
    {'name':'response_information','gain_beta':.5,'risk_eta':0.,'incoming_beta':0.},
    {'name':'response_information_risk','gain_beta':.5,'risk_eta':1.,'incoming_beta':0.},
    {'name':'response_information_readiness','gain_beta':.5,'risk_eta':1.,'incoming_beta':.15},
]
PREFIX_CONFIGURATIONS = [
    {'name':'prefix_fisher_control','gain_beta':0.,'risk_eta':0.,'incoming_beta':0.,'prefix_targets':False,'prefix_pool':False},
    {'name':'prefix_canvas_control','gain_beta':.5,'risk_eta':1.,'incoming_beta':.15,'prefix_targets':False,'prefix_pool':False},
    {'name':'information_prefix_targets','gain_beta':.5,'risk_eta':1.,'incoming_beta':.15,'prefix_targets':True,'prefix_pool':False},
    {'name':'information_prefix_sources_targets','gain_beta':.5,'risk_eta':1.,'incoming_beta':.15,'prefix_targets':True,'prefix_pool':True},
]
SUPPORT_COUNT_CONFIGURATIONS = [
    {'name':'support_fisher_control','gain_beta':0.,'risk_eta':0.,'incoming_beta':0.,'source_mass_weighting':False,'count_mode':'eb'},
    {'name':'support_response_control','gain_beta':.5,'risk_eta':1.,'incoming_beta':.15,'source_mass_weighting':False,'count_mode':'eb'},
    {'name':'response_source_weighted','gain_beta':.5,'risk_eta':1.,'incoming_beta':.15,'source_mass_weighting':True,'count_mode':'eb'},
    {'name':'response_count_only','gain_beta':0.,'risk_eta':0.,'incoming_beta':0.,'source_mass_weighting':False,'count_mode':'probe_surprise'},
    {'name':'response_weighted_count','gain_beta':.5,'risk_eta':1.,'incoming_beta':.15,'source_mass_weighting':True,'count_mode':'probe_surprise'},
]
CONDITIONING_CONFIGURATIONS = [
    {'name':'conditioning_fisher_control','context_beta':0.},
    {'name':'conditioning_support025','context_beta':.25},
    {'name':'conditioning_support05','context_beta':.5},
    {'name':'conditioning_support1','context_beta':1.},
]
DEPTH_CONFIGURATIONS = [
    {'name':'depth_fisher_control','depth_risk_beta':0.,'depth_support_beta':0.},
    {'name':'depth_motion_risk','depth_risk_beta':.15,'depth_support_beta':0.},
    {'name':'depth_signed_support','depth_risk_beta':0.,'depth_support_beta':.15},
    {'name':'depth_support_risk','depth_risk_beta':.15,'depth_support_beta':.15},
]
INFORMATION_DEPTH_CONFIGURATIONS = [
    {'name':'information_depth_control','gain_beta':.5,'risk_eta':1.,'incoming_beta':.15,'depth_risk_beta':0.},
    {'name':'information_depth005','gain_beta':.5,'risk_eta':1.,'incoming_beta':.15,'depth_risk_beta':.05},
    {'name':'information_depth015','gain_beta':.5,'risk_eta':1.,'incoming_beta':.15,'depth_risk_beta':.15},
]
ATTENTION_CONFIGURATIONS = [
    {'name':'attention_fisher_control','read_beta':0.,'uncertainty_weighted':True},
    {'name':'attention_mask_read','read_beta':.15,'uncertainty_weighted':False},
    {'name':'attention_uncertain_read015','read_beta':.15,'uncertainty_weighted':True},
    {'name':'attention_uncertain_read03','read_beta':.3,'uncertainty_weighted':True},
]
CONSISTENCY_CONFIGURATIONS = [
    {'name':'consistency_fisher_control','enabled':False,'mode':'filter'},
    {'name':'consistency_filter','enabled':True,'mode':'filter'},
    {'name':'consistency_singleton','enabled':True,'mode':'singleton'},
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=int,default=20)
    parser.add_argument('--candidates',type=int,default=8)
    parser.add_argument('--probe-every',type=int,default=16)
    parser.add_argument('--branch-batch-size',type=int,default=160)
    parser.add_argument('--matched-probe-control',action='store_true')
    parser.add_argument('--scheduler',choices=('finite','information','conditioning','depth','attention','consistency'),default='finite')
    parser.add_argument('--attention-layers',type=int,default=1)
    parser.add_argument('--depth-back',type=int,default=4)
    parser.add_argument('--target-vocab',type=int,default=128)
    parser.add_argument('--information-plan',choices=('default','prefix','support_count','depth'),default='default')
    parser.add_argument('--configuration',choices=[c['name'] for c in CONFIGURATIONS+INFORMATION_CONFIGURATIONS+PREFIX_CONFIGURATIONS+SUPPORT_COUNT_CONFIGURATIONS+CONDITIONING_CONFIGURATIONS+DEPTH_CONFIGURATIONS+INFORMATION_DEPTH_CONFIGURATIONS+ATTENTION_CONFIGURATIONS+CONSISTENCY_CONFIGURATIONS],
                        help='Run one setting in a standalone batch for verification.')
    parser.add_argument('--out',type=Path,default=Path('outputs/latent_finite80'))
    args = parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True); set_seed(1234)
    plans = {'default':INFORMATION_CONFIGURATIONS,'prefix':PREFIX_CONFIGURATIONS,'support_count':SUPPORT_COUNT_CONFIGURATIONS,'depth':INFORMATION_DEPTH_CONFIGURATIONS}
    selected = {'finite':CONFIGURATIONS,'conditioning':CONDITIONING_CONFIGURATIONS,'depth':DEPTH_CONFIGURATIONS,'information':plans[args.information_plan],'attention':ATTENTION_CONFIGURATIONS,'consistency':CONSISTENCY_CONFIGURATIONS}[args.scheduler]
    configurations = [c for c in selected if args.configuration is None or c['name']==args.configuration]
    if not configurations: parser.error('configuration does not belong to selected scheduler')
    documents,dataset = load_documents(None,args.samples)
    model,tokenizer = load_model_and_tokenizer('models/llada',device='cuda',compile_model=False)
    mask,eos,_ = infer_diffusion_token_ids('models/llada')
    encoded = [tokenizer(d['prompt'],add_special_tokens=False)['input_ids'] for d in documents]
    width = max(map(len,encoded)); n = len(documents)
    ids = torch.full((n,width),tokenizer.pad_token_id,dtype=torch.long,device='cuda'); attention = torch.zeros_like(ids)
    for row,tokens in enumerate(encoded):
        ids[row,-len(tokens):] = torch.tensor(tokens,device='cuda'); attention[row,-len(tokens):] = 1
    ids,attention = ids.repeat(len(configurations),1),attention.repeat(len(configurations),1)
    parameters = {key:[c[key] for c in configurations for _ in documents] for key in configurations[0] if key!='name'}
    module = {'finite':latent_finite,'information':latent_information,'conditioning':latent_conditioning,'depth':latent_depth,'attention':latent_attention,'consistency':latent_consistency}[args.scheduler]
    source = Path(module.__file__).read_bytes()
    (args.out/'scheduler_source.py').write_bytes(source)
    (args.out/'benchmark_source.py').write_bytes(Path(__file__).read_bytes())
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start = time.perf_counter()
    def progress(m): print(f"Batch {len(ids)}: step {m['steps_taken']}/128, total forward work {m['sequence_evaluations']}",flush=True)
    generator = getattr(module,{'finite':'generate_finite','information':'generate_information',
                               'conditioning':'generate_conditioning','depth':'generate_depth','attention':'generate_attention','consistency':'generate_consistency'}[args.scheduler])
    extras = {}
    if args.scheduler in ('finite','information'): extras.update(candidates=args.candidates,probe_every=args.probe_every)
    if args.scheduler=='finite': extras['matched_probe_control'] = args.matched_probe_control
    if args.scheduler=='information': extras.update(target_vocab=args.target_vocab,depth_back=args.depth_back)
    if args.scheduler=='consistency': extras['probe_every']=args.probe_every
    if args.scheduler=='depth': extras['depth_back'] = args.depth_back
    elif args.scheduler=='attention': extras['attention_layers'] = args.attention_layers
    else: extras['branch_batch_size'] = args.branch_batch_size
    generated = generator(model,ids,attention,mask_id=mask,eos_id=eos,
        progress=progress,**extras,**parameters)
    torch.cuda.synchronize(); elapsed = time.perf_counter()-start; m = module.last_metrics
    summaries = []
    for cohort,c in enumerate(configurations):
        lo,hi = cohort*n,(cohort+1)*n; outputs = generated[lo:hi]
        texts = decode(tokenizer,outputs,width,eos); results = code_results(documents,texts)
        samples = [{**r,'completion':s,'unfinished_masks':int((t[width:]==mask).sum())} for r,s,t in zip(results,texts,outputs)]
        summary = {**c,'passed':sum(s['passed'] for s in samples),'samples':n,
            'unfinished_masks':sum(s['unfinished_masks'] for s in samples),
            'forward_evaluations_per_sample':sum(m['forward_work_by_row'][lo:hi])/n,
            'backward_evaluations_per_sample':0.,'branch_evaluations_per_sample':sum(m['branch_work_by_row'][lo:hi])/n}
        summary['probe_control_evaluations_per_sample'] = sum(m['probe_control_work_by_row'][lo:hi])/n
        for field in ('count_changes_by_row','count_extra_by_row','count_fewer_by_row','probed_batches_by_row','conflicts_by_row','numerical_mismatches_by_row','count_reductions_by_row','deferred_tokens_by_row'):
            if field in m: summary[field.removesuffix('_by_row')+'_per_sample'] = sum(m[field][lo:hi])/n
        summaries.append(summary)
        (args.out/(c['name']+'.json')).write_text(json.dumps({'summary':summary,'samples':samples},indent=2)+'\n')
        print(json.dumps(summary),flush=True)
    (args.out/'results.json').write_text(json.dumps({'summaries':summaries,'steps':128,'gen_length':512,
        'total_batch_size':len(ids),'dataset_arrow':str(dataset),'task_ids':[d['task_id'] for d in documents],
        'candidates':args.candidates if args.scheduler in ('finite','information') else None,
        'probe_every':args.probe_every if args.scheduler in ('finite','information','consistency') else 1,'branch_batch_size':args.branch_batch_size if args.scheduler not in ('depth','attention') else None,
        'depth_back':args.depth_back if args.scheduler=='depth' else None,
        'attention_layers':args.attention_layers if args.scheduler=='attention' else None,
        'matched_probe_control':args.matched_probe_control or args.scheduler in ('information','conditioning'),
        'scheduler':args.scheduler,'target_vocab':args.target_vocab,'information_plan':args.information_plan,
        'generation_seconds_all_cohorts':elapsed,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
        'code_sha256':hashlib.sha256(source).hexdigest(),'auxiliary_training':False,
        'scheduler_metrics':m},indent=2,allow_nan=False)+'\n')


if __name__ == '__main__': main()
