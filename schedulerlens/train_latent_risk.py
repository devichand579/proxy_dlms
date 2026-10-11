"""Fit a tiny risk readout on cached MBPP training/validation code, excluding HumanEval names.

LLaDA is frozen. Labels measure agreement with reference tokens on data-consistent
masked histories, not a certificate of functional correctness on generated histories.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import time

import numpy as np
import torch

from datasets import Dataset
from load_model import load_model_and_tokenizer, infer_diffusion_token_ids
from schedulerlens.latent_bench import load_documents, set_seed
from schedulers.latent_risk import RiskHead, risk_features


def examples(tokenizer, split, limit, excluded):
    path = next(Path("/root/.cache/huggingface/datasets/google-research-datasets___mbpp").glob(f"**/mbpp-{split}.arrow"))
    dataset = Dataset.from_file(str(path))
    result = []
    for row in dataset:
        try:
            tree = ast.parse(row["code"])
            functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
            if not functions or any(n.name in excluded for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
                continue
            function = functions[0]
            first_body = function.body[0].lineno
            if first_body <= function.lineno:
                continue
            lines = row["code"].splitlines(keepends=True)
            # Known function header and question; reference implementation is masked.
            prefix = "# " + row["text"].replace("\n", " ") + "\n" + "".join(lines[:first_body - 1])
            body = "".join(lines[first_body - 1:])
            prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
            body_ids = tokenizer(body, add_special_tokens=False)["input_ids"]
            if len(prefix_ids) > 256 or not 1 <= len(body_ids) < 480:
                continue
            result.append({"task_id": row["task_id"], "prefix": prefix_ids, "body": body_ids})
            if len(result) >= limit:
                break
        except (SyntaxError, ValueError, IndexError):
            continue
    return result, str(path)


@torch.no_grad()
def collect(model, tokenizer, data, mask_id, eos_id, h_projection, e_projection, batch_size, seed):
    rng = np.random.default_rng(seed)
    states = [(example, fraction) for example in data for fraction in (.2, .5, .8, 1.)]
    features, labels, confidence_risk, task_ids = [], [], [], []
    harmless = torch.zeros(model.get_output_embeddings().weight.shape[0], dtype=torch.bool, device="cuda")
    for token_id in set(tokenizer.get_vocab().values()):
        if not tokenizer.decode([token_id], skip_special_tokens=True).strip():
            harmless[token_id] = True
    for offset in range(0, len(states), batch_size):
        chunk = states[offset:offset + batch_size]
        width = max(len(ex["prefix"]) for ex, _ in chunk)
        ids = torch.full((len(chunk), width + 512), tokenizer.pad_token_id, dtype=torch.long, device="cuda")
        attention = torch.zeros_like(ids)
        target = torch.full((len(chunk), 512), eos_id, dtype=torch.long, device="cuda")
        masked = torch.zeros_like(target, dtype=torch.bool)
        for row, (ex, fraction) in enumerate(chunk):
            ids[row, width - len(ex["prefix"]):width] = torch.tensor(ex["prefix"], device="cuda")
            attention[row, width - len(ex["prefix"]):] = 1
            target[row, :len(ex["body"])] = torch.tensor(ex["body"], device="cuda")
            selection = rng.random(512) < fraction
            selection[rng.integers(0, len(ex["body"]))] = True
            masked[row] = torch.tensor(selection, device="cuda")
        ids[:, width:] = torch.where(masked, mask_id, target)
        captured = []
        hook = model.get_output_embeddings().register_forward_pre_hook(lambda _m, inputs: captured.append(inputs[0].detach()))
        try:
            output = model(ids, attention_mask=attention)
        finally:
            hook.remove()
        for row, (ex, _) in enumerate(chunk):
            local = masked[row].nonzero().flatten()
            for start in range(0, len(local), 32):
                positions = local[start:start + 32]
                logits = output.logits[row, width + positions].clone()
                logits[:, mask_id] = -torch.inf
                hidden = captured[0][row, width + positions]
                f = risk_features(hidden, logits, model.get_input_embeddings().weight, h_projection, e_projection, eos_id)
                pred = logits.argmax(-1)
                features.append(f.cpu())
                padding = positions >= len(ex["body"])
                acceptable_tail = padding & harmless[pred]
                labels.append(((pred != target[row, positions]) & ~acceptable_tail).float().cpu())
                confidence_risk.append((1 - f[:, -3]).cpu())
                task_ids.extend([ex["task_id"]] * len(positions))
        del output, captured, ids, attention, target, masked
        print(f"Collected {min(offset + batch_size, len(states))}/{len(states)} masked code histories", flush=True)
    return torch.cat(features), torch.cat(labels), torch.cat(confidence_risk), task_ids


def metrics(pred, labels):
    from sklearn.metrics import roc_auc_score
    p, y = pred.numpy(), labels.numpy()
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(np.mean(bins == i) * abs(p[bins == i].mean() - y[bins == i].mean()) for i in range(10) if np.any(bins == i))
    return {"brier": float(np.mean((p-y)**2)), "auc": float(roc_auc_score(y,p)), "ece_10bin": float(ece),
            "mean_predicted_risk": float(p.mean()), "observed_reference_error": float(y.mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--validation-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--out", type=Path, default=Path("outputs/latent_risk_mbpp"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    set_seed(81)
    model, tokenizer = load_model_and_tokenizer("models/llada", device="cuda", compile_model=False)
    mask_id, eos_id, _ = infer_diffusion_token_ids("models/llada")
    human, _ = load_documents(None, 164)
    excluded = {d["entry_point"] for d in human}
    train, train_path = examples(tokenizer, "train", args.train_samples, excluded)
    validation, validation_path = examples(tokenizer, "validation", args.validation_samples, excluded)
    if not train or not validation:
        raise ValueError("not enough disjoint cached MBPP code examples")
    dimension = model.get_output_embeddings().weight.shape[1]
    generator = torch.Generator().manual_seed(81)
    hp = (torch.randn(dimension,128,generator=generator) / dimension**.5).cuda()
    ep = (torch.randn(dimension,32,generator=generator) / dimension**.5).cuda()
    start = time.perf_counter()
    x,y,base,ids = collect(model, tokenizer, train, mask_id, eos_id, hp,ep,args.batch_size,81)
    vx,vy,vbase,vids = collect(model, tokenizer, validation, mask_id,eos_id,hp,ep,args.batch_size,82)
    torch.save({"train_features":x,"train_labels":y,"train_confidence_risk":base,"train_task_ids":ids,
                "validation_features":vx,"validation_labels":vy,"validation_confidence_risk":vbase,"validation_task_ids":vids},args.out/'features.pt')
    del model
    torch.cuda.empty_cache()
    mean, scale = x.mean(0), x.std(0).clamp_min(.01)
    xx, yy = ((x-mean)/scale).cuda(), y.cuda()
    head=RiskHead(x.shape[1]).cuda()
    optimizer=torch.optim.AdamW(head.parameters(),lr=.002,weight_decay=.01)
    best_brier=float('inf'); best=None
    for epoch in range(12):
        order=torch.randperm(len(x),device='cuda')
        head.train()
        for batch in order.split(4096):
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(head(xx[batch]),yy[batch])
            loss.backward();optimizer.step()
        head.eval()
        with torch.no_grad():
            prediction=torch.cat([head(((part-mean)/scale).cuda()).sigmoid().cpu() for part in vx.split(4096)])
        score=metrics(prediction,vy)
        if score['brier']<best_brier:
            best_brier=score['brier'];best={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        print(json.dumps({'epoch':epoch,**score}),flush=True)
    head.load_state_dict(best)
    with torch.no_grad():
        prediction=torch.cat([head(((part-mean)/scale).cuda()).sigmoid().cpu() for part in vx.split(4096)])
    summary={'train_problems':len(train),'validation_problems':len(validation),'train_tokens':len(y),'validation_tokens':len(vy),
             'train_dataset':train_path,'validation_dataset':validation_path,'excluded_humaneval_entry_points':len(excluded),
             'model_frozen':True,'label':'reference disagreement on code-body slots; whitespace/end markers accepted in padding',
             'confidence_baseline':metrics(vbase,vy),'risk_readout':metrics(prediction,vy),'elapsed_seconds':time.perf_counter()-start}
    torch.save({'head':best,'mean':mean,'scale':scale,'h_projection':hp.cpu(),'e_projection':ep.cpu(),'metadata':summary},args.out/'risk_head.pt')
    (args.out/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
