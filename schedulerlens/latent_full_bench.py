"""Checkpointed, matched full HumanEval comparison of frozen schedulers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from load_model import load_model_and_tokenizer, infer_diffusion_token_ids
from schedulerlens.latent_bench import load_documents, code_results, decode, set_seed
from schedulers.eb_sampler import generate_EB
from schedulers import latent, latent_information


METHODS = ('eb_sampler', 'eb_target_temporal', 'response_information_readiness')


def save(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('outputs/latent_full_humaneval164'))
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    args.out.mkdir(parents=True, exist_ok=True)
    documents, dataset = load_documents(None, 164)
    sources = [Path(__file__), Path(latent.__file__), Path(latent_information.__file__),
               Path('schedulers/eb_sampler.py'), Path('schedulerlens/latent_bench.py'),
               Path('schedulers/utils.py'), Path('load_model.py')]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    manifest = dict(task_ids=[d['task_id'] for d in documents], dataset_arrow=str(dataset),
        dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
        model='models/llada', model_config_sha256=hashlib.sha256(Path('models/llada/config.json').read_bytes()).hexdigest(),
        methods=list(METHODS), steps=128, gen_length=512, block_length=512,
        batch_size=20, seed=1234, temperature=0., gamma=.1, stability_beta=.15,
        fisher_vocab=128, candidates=4, probe_every=16, branch_batch_size=320,
        gain_beta=.5, risk_eta=1., incoming_beta=.15, auxiliary_training=False,
        early_stop=False, prompt_format='raw HumanEval code completion, zero shot',
        decode='first EOS, then installed HumanEval stop tokens', correctness_timeout_seconds=3.,
        exploratory_tasks=list(range(20)), remaining_tasks=list(range(20,164)), source_sha256=hashes,
        torch=torch.__version__, hip=torch.version.hip)
    manifest_path = args.out/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('existing experiment manifest differs')
    save(manifest_path, manifest)
    snapshot = args.out/'sources'; snapshot.mkdir(exist_ok=True)
    for p in sources:
        (snapshot/str(p).replace('/', '__')).write_bytes(p.read_bytes())
    model, tokenizer = load_model_and_tokenizer('models/llada', device='cuda', compile_model=False)
    mask, eos, eot = infer_diffusion_token_ids('models/llada')
    encoded = [tokenizer(d['prompt'], add_special_tokens=False)['input_ids'] for d in documents]
    for method in METHODS:
        folder = args.out/method; folder.mkdir(exist_ok=True)
        completed = []
        for offset in range(0,164,20):
            checkpoint = folder/f'batch_{offset:03d}.json'
            if checkpoint.exists():
                payload = json.loads(checkpoint.read_text())
                if payload['task_ids'] != manifest['task_ids'][offset:offset+20]:
                    raise ValueError('checkpoint task mismatch')
                print(f'RESUME {method} batch {offset//20+1}/9', flush=True)
            else:
                set_seed(1234)
                indices = list(range(offset,min(offset+20,164)))
                width = max(len(encoded[i]) for i in indices)
                ids = torch.full((len(indices),width),tokenizer.pad_token_id,device='cuda',dtype=torch.long)
                attention = torch.zeros_like(ids)
                for row,i in enumerate(indices):
                    ids[row,-len(encoded[i]):] = torch.tensor(encoded[i],device='cuda')
                    attention[row,-len(encoded[i]):] = 1
                def progress(m):
                    print(f'{method} batch {offset//20+1}/9 step {m["steps_taken"]}/128 '
                          f'forward work {m["sequence_evaluations"]/len(indices):.1f}/sample',flush=True)
                print(f'START {method} batch {offset//20+1}/9, prompt width {width}',flush=True)
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
                started = time.perf_counter()
                common = dict(steps=128,gen_length=512,mask_id=mask,eos_id=eos)
                if method == 'eb_sampler':
                    generated, steps, _ = generate_EB(model,ids,attention,block_length=512,
                        temperature=0.,eot_id=eot,use_eos_harness=False,gamma=.1,
                        E='confidence',upload_to_wandb=False,**common)
                    metrics = dict(steps_taken=steps,sequence_evaluations=steps*len(indices),
                                   forward_work_by_row=[steps]*len(indices))
                elif method == 'eb_target_temporal':
                    generated, _, _ = latent.generate_latent(model,ids,attention,variant=method,
                        block_length=512,temperature=0.,eot_id=eot,use_eos_harness=False,
                        diversity_beta=.25,stability_beta=.15,probe_every=8,fisher_vocab=128,
                        progress=progress,**common)
                    metrics = dict(latent.last_metrics)
                else:
                    generated = latent_information.generate_information(model,ids,attention,
                        candidates=4,probe_every=16,branch_batch_size=320,target_vocab=128,
                        gain_beta=.5,risk_eta=1.,incoming_beta=.15,stability_beta=.15,
                        progress=progress,**common)
                    metrics = dict(latent_information.last_metrics)
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-started
                texts = decode(tokenizer,generated,width,eos)
                samples = [dict(task_id=documents[i]['task_id'],completion=text,
                    unfinished_masks=int((generated[row,width:]==mask).sum()),
                    forward_evaluations=metrics.get('forward_work_by_row',[128]*len(indices))[row])
                    for row,(i,text) in enumerate(zip(indices,texts))]
                payload = dict(task_ids=[s['task_id'] for s in samples],samples=samples,
                    generation_seconds=elapsed,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                    scheduler_metrics=metrics)
                # Preserve generations before executing any generated code.
                save(folder/f'generated_{offset:03d}.json',payload)
                del generated,ids,attention
                outcomes = code_results([documents[i] for i in indices],texts)
                for sample,outcome in zip(samples,outcomes): sample.update(outcome)
                save(checkpoint,payload)
            completed.append(payload)
            print(f'FINISHED {method} batch {offset//20+1}/9: '
                  f'{sum(s["passed"] for s in payload["samples"])}/{len(payload["samples"])} '
                  f'in {payload["generation_seconds"]:.1f}s',flush=True)
        samples = [s for batch in completed for s in batch['samples']]
        summary = dict(method=method,passed=sum(s['passed'] for s in samples),samples=164,
            pass_at_1=sum(s['passed'] for s in samples)/164,
            forward_evaluations_per_sample=sum(s['forward_evaluations'] for s in samples)/164,
            backward_evaluations_per_sample=0.,generation_seconds=sum(p['generation_seconds'] for p in completed),
            peak_allocated_GiB=max(p['peak_allocated_GiB'] for p in completed),
            unfinished_masks=sum(s['unfinished_masks'] for s in samples))
        save(folder/'results.json',dict(summary=summary,samples=samples,config=manifest))
        print('RESULT '+json.dumps(summary),flush=True)


if __name__ == '__main__': main()
