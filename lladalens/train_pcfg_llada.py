"""
Train a small LLaDA variant on PCFG-sampled masked strings.

Each step samples t ~ Uniform[0, 1], masks content tokens independently with
probability t, and optimizes Eq. (3) with 1/(t·L) weighting on masked CE.

Example:
  python train_pcfg_llada.py --variant 100m --n-examples 2048 --steps 200
  python train_pcfg_llada.py --variant 100m --hf-dataset pcfg-code --steps 200
"""
from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from hf_hub import apply_pcfg_hf_env, resolve_hf_token
from lladalens.pcfg_models import count_parameters, list_variants, load_pcfg_llada, load_pcfg_tokenizer
from pcfg_data import ARCH_LEN, LABEL_IGNORE, generate_training_corpus, load_corpus_from_disk, load_corpus_from_hub, make_train_example


class PCFGTrainDataset(Dataset):
    def __init__(self, token_seqs, vocab, arch_len=ARCH_LEN):
        self.token_seqs = token_seqs
        self.vocab = vocab
        self.arch_len = arch_len

    def __len__(self):
        return len(self.token_seqs)

    def __getitem__(self, idx):
        ex = make_train_example(self.token_seqs[idx], self.vocab, arch_len=self.arch_len)
        return {
            "input_ids": torch.from_numpy(ex.input_ids.copy()),
            "attention_mask": torch.from_numpy(ex.attention_mask.copy()),
            "labels": torch.from_numpy(ex.labels.copy()),
            "mask_t": torch.tensor(ex.mask_t, dtype=torch.float32),
            "n_content": torch.tensor(ex.n_content, dtype=torch.int64),
        }


def llada_mask_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask_t: torch.Tensor,
    n_content: torch.Tensor,
) -> torch.Tensor:
    """Eq. (3): -(1/tL) sum_i 1[masked] log p_theta(x0_i | x_t)."""
    losses = []
    for b in range(logits.size(0)):
        masked = labels[b] != LABEL_IGNORE
        n_masked = int(masked.sum().item())
        if n_masked == 0:
            continue
        t = float(mask_t[b].item())
        if t <= 0.0:
            continue
        ce_sum = F.cross_entropy(
            logits[b],
            labels[b],
            ignore_index=LABEL_IGNORE,
            reduction="sum",
        )
        losses.append(ce_sum / (t * float(n_content[b].item())))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def wsd_lr(
    step: int,
    *,
    warmup_steps: int,
    stable_end_step: int,
    decay_mid_step: int,
    total_steps: int,
    peak_lr: float,
    mid_lr: float,
    min_lr: float,
) -> float:
    """Warmup-Stable-Decay schedule scaled for small runs."""
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(warmup_steps, 1)
    if step < stable_end_step:
        return peak_lr
    if step < decay_mid_step:
        frac = (step - stable_end_step) / max(decay_mid_step - stable_end_step, 1)
        return peak_lr + (mid_lr - peak_lr) * frac
    if step < total_steps:
        frac = (step - decay_mid_step) / max(total_steps - decay_mid_step, 1)
        return mid_lr + (min_lr - mid_lr) * frac
    return min_lr


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    sched_cfg,
    global_step=0,
    max_steps=None,
):
    model.train()
    total_loss = 0.0
    n_steps = 0
    step = global_step
    for batch in loader:
        for pg in optimizer.param_groups:
            pg["lr"] = wsd_lr(step, **sched_cfg)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        mask_t = batch["mask_t"].to(device)
        n_content = batch["n_content"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = llada_mask_loss(out.logits, labels, mask_t, n_content)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.item())
        n_steps += 1
        step += 1
        if max_steps is not None and n_steps >= max_steps:
            break
    return total_loss / max(n_steps, 1), n_steps, step


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=list_variants(), default="100m")
    p.add_argument("--n-examples", type=int, default=None,
                   help="Legacy: exact train string count (use --corpus or --train-tokens instead).")
    p.add_argument("--train-tokens", type=int, default=None,
                   help="Sample train corpus by token budget (via pcfg_generate).")
    p.add_argument("--corpus", type=Path, default=None,
                   help="Load pre-sampled HF corpus from pcfg_generate (save_to_disk path).")
    p.add_argument("--hf-dataset", type=str, default=None,
                   help="Load corpus from Hub (short name or org/repo; default name: pcfg-code).")
    p.add_argument("--hf-namespace", type=str, default=None,
                   help="HF user/org when --hf-dataset has no slash.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--peak-lr", type=float, default=4e-4, help="LLaDA peak LR (paper: 4e-4).")
    p.add_argument("--mid-lr", type=float, default=1e-4, help="WSD mid plateau LR (paper: 1e-4).")
    p.add_argument("--min-lr", type=float, default=1e-5, help="WSD final LR (paper: 1e-5).")
    p.add_argument("--warmup-steps", type=int, default=None, help="Linear warmup steps.")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--steps", type=int, default=None, help="Cap optimizer steps (smoke tests).")
    p.add_argument("--weight-decay", type=float, default=0.1, help="AdamW decay (paper: 0.1).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    apply_pcfg_hf_env()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, config, vocab = load_pcfg_llada(args.variant, device=device)
    n_params = count_parameters(model)
    print(f"variant={args.variant} params={n_params:,} ({n_params/1e6:.1f}M) "
          f"vocab={vocab.size} seq={config.max_sequence_length} device={device}")

    if args.corpus is not None:
        token_seqs, _, _ = load_corpus_from_disk(args.corpus)
        print(f"loaded train corpus from disk: {len(token_seqs)} strings")
    elif args.hf_dataset is not None:
        token_seqs, _, _ = load_corpus_from_hub(
            args.hf_dataset,
            namespace=args.hf_namespace,
            token=resolve_hf_token(None),
        )
        print(f"loaded train corpus from hub: {len(token_seqs)} strings")
    elif args.train_tokens is not None:
        from pcfg_generate import sample_train_corpus
        from pcfg_grammar import code_grammar

        records, stats = sample_train_corpus(
            args.train_tokens, grammar=code_grammar(), rng=random.Random(args.seed)
        )
        token_seqs = [r.tokens for r in records]
        print(
            f"sampled train corpus: {stats.n_strings} strings, "
            f"{stats.total_tokens:,} tokens (target {args.train_tokens:,})"
        )
    else:
        n_examples = args.n_examples if args.n_examples is not None else 2048
        token_seqs, _ = generate_training_corpus(n_examples, rng=random.Random(args.seed))
        print(f"sampled train corpus: {len(token_seqs)} strings (legacy --n-examples)")
    loader = DataLoader(
        PCFGTrainDataset(token_seqs, vocab),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    steps_per_epoch = len(loader)
    total_steps = args.steps if args.steps is not None else steps_per_epoch * args.epochs
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else max(1, min(2000, total_steps // 10))
    sched_cfg = {
        "warmup_steps": warmup_steps,
        "stable_end_step": max(warmup_steps + 1, int(total_steps * 0.55)),
        "decay_mid_step": max(warmup_steps + 2, int(total_steps * 0.80)),
        "total_steps": total_steps,
        "peak_lr": args.peak_lr,
        "mid_lr": args.mid_lr,
        "min_lr": args.min_lr,
    }
    print(
        f"schedule=WSD warmup={warmup_steps} total_steps={total_steps} "
        f"peak_lr={args.peak_lr:g} mid_lr={args.mid_lr:g} min_lr={args.min_lr:g} "
        f"weight_decay={args.weight_decay}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.peak_lr,
        weight_decay=args.weight_decay,
    )

    t0 = time.time()
    steps_left = args.steps
    global_step = 0
    for epoch in range(args.epochs):
        loss, n, global_step = train_one_epoch(
            model,
            loader,
            optimizer,
            device,
            sched_cfg,
            global_step=global_step,
            max_steps=steps_left,
        )
        lr = wsd_lr(global_step - 1, **sched_cfg)
        print(f"epoch={epoch} steps={n} loss={loss:.4f} ppl={math.exp(min(loss, 20)):.2f} lr={lr:g}")
        if steps_left is not None:
            steps_left -= n
            if steps_left <= 0:
                break
    print(f"done in {time.time() - t0:.1f}s")

    out_dir = args.out_dir or Path("checkpoints") / f"llada-{args.variant}-pcfg"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    config.save_pretrained(out_dir)
    load_pcfg_tokenizer().save_pretrained(out_dir)
    (out_dir / "pcfg_vocab.txt").write_text("\n".join(vocab.itos) + "\n")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
