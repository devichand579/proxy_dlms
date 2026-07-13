"""
Train a small LLaDA variant on PCFG-sampled masked strings.

Each step samples t ~ Uniform[0, 1], masks content tokens independently with
probability t, and optimizes Eq. (3) with 1/(t·L) weighting on masked CE.

Example (from repo root):
  python llada-pcfg/scripts/train_pcfg_llada.py --variant 100m --hf-dataset pcfg-code --steps 2000
  python llada-pcfg/scripts/train_pcfg_llada.py --variant 100m --hf-dataset pcfg-code --offline
  torchrun --nproc_per_node=2 llada-pcfg/scripts/train_pcfg_llada.py --variant 100m --corpus data/pcfg --steps 5000
"""
from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import _bootstrap  # noqa: F401

from utils import default_checkpoint_dir, ensure_import_paths, resolve_repo_path

ensure_import_paths()

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from hf_hub import apply_pcfg_hf_env, push_pcfg_model, resolve_hf_token
from repo_env_defaults import is_offline_env
from models.pcfg_models import (
    count_parameters,
    list_variants,
    load_pcfg_llada,
    load_pcfg_llada_checkpoint,
)
from pcfg_data import (
    ARCH_LEN,
    MAX_CONTENT_LEN,
    corpus_length_stats,
    fit_corpus_for_arch,
    load_corpus_from_disk,
    load_corpus_from_hub,
)
from pcfg_utils.checkpoint import (
    CKPT_LATEST,
    TRAIN_STATE_JSON,
    TrainState,
    ckpt_path,
    maybe_resume,
    restore_rng_states,
)
from pcfg_utils.distributed import is_main, setup_distributed, unwrap_model
from pcfg_utils.schedule import resolve_warmup_steps
from pcfg_utils.splits import split_train_val
from pcfg_utils.wandb_utils import default_wandb_run_name, init_wandb
from training.datasets import PCFGTrainDataset, PCFGValDataset
from training.loop import train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=list_variants(), default="100m")
    p.add_argument("--train-tokens", type=int, default=None)
    p.add_argument("--corpus", type=Path, default=None)
    p.add_argument("--hf-dataset", type=str, default=None)
    p.add_argument("--hf-namespace", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=16, help="Per-GPU batch size.")
    p.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps.")
    p.add_argument("--num-workers", type=int, default=2, help="DataLoader workers per rank.")
    p.add_argument("--peak-lr", type=float, default=4e-4)
    p.add_argument("--mid-lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=None,
                   help="Explicit warmup steps (overrides --warmup-fraction).")
    p.add_argument("--warmup-fraction", type=float, default=0.02,
                   help="Warmup as fraction of total steps when --warmup-steps unset (default 0.02).")
    p.add_argument("--warmup-max", type=int, default=5000,
                   help="Cap auto warmup steps (default 5000).")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--resume", type=Path, default=None, help="Resume from checkpoint dir.")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=5000,
                   help="Checkpoint every N steps (default 5000).")
    p.add_argument("--val-fraction", type=float, default=0.05,
                   help="Random fraction of train held out for validation (default 0.05).")
    p.add_argument("--val-every", type=int, default=2000,
                   help="Validation every N steps (default 2000; 0=disable).")
    p.add_argument("--push-model", action="store_true", help="Push best-val checkpoint to HF Hub at end.")
    p.add_argument("--hf-model", type=str, default=None,
                   help="Short or full HF model repo id (also default checkpoint dir when --out-dir unset).")
    p.add_argument("--hf-model-namespace", type=str, default=None)
    p.add_argument("--hf-model-private", action="store_true")
    p.add_argument("--amp", action="store_true", default=True, help="Mixed precision on CUDA.")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default="bfloat16")
    p.add_argument("--wandb-id", type=str, default=None, help="Explicit W&B run id for resume.")
    p.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    p.add_argument("--no-wandb", dest="wandb", action="store_false")
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument(
        "--offline",
        action="store_true",
        help="Use cached HF datasets/models and local W&B (no Hub/network).",
    )
    args = p.parse_args()

    offline = args.offline or is_offline_env()
    apply_pcfg_hf_env(offline=offline)
    rank, local_rank, world_size = setup_distributed()

    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    out_dir = resolve_repo_path(args.out_dir) or default_checkpoint_dir(
        args.variant, hf_model=args.hf_model,
    )
    resume_dir = resolve_repo_path(args.resume) or out_dir
    latest_ckpt = resume_dir / CKPT_LATEST
    do_resume = args.resume is not None and (resume_dir / TRAIN_STATE_JSON).exists()

    if is_main(rank):
        print(f"checkpoints -> {out_dir}", flush=True)

    if do_resume and latest_ckpt.is_dir():
        model, config, vocab = load_pcfg_llada_checkpoint(latest_ckpt, device=str(device))
        if is_main(rank):
            print(f"loaded weights from {latest_ckpt}", flush=True)
    else:
        model, config, vocab = load_pcfg_llada(args.variant, device=str(device))
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    n_params = count_parameters(unwrap_model(model))
    if is_main(rank):
        print(
            f"variant={args.variant} params={n_params:,} ({n_params/1e6:.1f}M) "
            f"vocab={vocab.size} seq={config.max_sequence_length} "
            f"device={device} world_size={world_size}",
            flush=True,
        )

    if args.corpus is not None:
        token_seqs, _, _ = load_corpus_from_disk(args.corpus)
        if is_main(rank):
            print(f"loaded corpus from disk: {len(token_seqs)} strings", flush=True)
    elif args.hf_dataset is not None:
        token_seqs, _, _ = load_corpus_from_hub(
            args.hf_dataset,
            namespace=args.hf_namespace,
            token=resolve_hf_token(None),
        )
        if is_main(rank):
            print(f"loaded corpus from hub: {len(token_seqs)} strings", flush=True)
    elif args.train_tokens is not None:
        from pcfg_generate import sample_train_corpus
        from pcfg_grammar import code_grammar

        records, stats = sample_train_corpus(
            args.train_tokens,
            grammar=code_grammar(),
            rng=random.Random(args.seed),
            progress_interval=0,
            log_fn=lambda *a, **k: None,
        )
        token_seqs = [r.tokens for r in records]
        if is_main(rank):
            print(
                f"sampled train corpus: {stats.n_strings} strings, "
                f"{stats.total_tokens:,} tokens (target {args.train_tokens:,})",
                flush=True,
            )
    else:
        p.error("one of --corpus, --hf-dataset, or --train-tokens is required")

    if is_main(rank):
        lo, hi, n_over = corpus_length_stats(token_seqs)
        print(
            f"corpus lengths (raw): min={lo} max={hi} "
            f"n_over_{MAX_CONTENT_LEN}={n_over} "
            f"(training uses max {MAX_CONTENT_LEN} content + <eos> in arch {ARCH_LEN})",
            flush=True,
        )

    token_seqs, n_clipped = fit_corpus_for_arch(token_seqs, arch_len=ARCH_LEN)
    if is_main(rank) and n_clipped > 0:
        print(
            f"clipped {n_clipped} strings that exceeded max content {MAX_CONTENT_LEN}",
            flush=True,
        )

    train_seqs, val_seqs = split_train_val(
        token_seqs, val_fraction=args.val_fraction, seed=args.seed,
    )
    if is_main(rank):
        print(
            f"train/val split: {len(train_seqs)} train, {len(val_seqs)} val "
            f"({args.val_fraction:.0%} held out)",
            flush=True,
        )

    dataset = PCFGTrainDataset(train_seqs, vocab)
    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    val_loader = None
    if val_seqs and args.val_every > 0:
        val_dataset = PCFGValDataset(val_seqs, vocab, seed=args.seed)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        if is_main(rank):
            print(f"validation: {len(val_seqs)} strings", flush=True)

    steps_per_epoch = max(1, len(loader) // args.grad_accum)
    total_steps = args.steps if args.steps is not None else steps_per_epoch * args.epochs
    warmup_steps = resolve_warmup_steps(
        total_steps,
        warmup_steps=args.warmup_steps,
        warmup_fraction=args.warmup_fraction,
        warmup_max=args.warmup_max,
    )
    sched_cfg = {
        "warmup_steps": warmup_steps,
        "stable_end_step": max(warmup_steps + 1, int(total_steps * 0.55)),
        "decay_mid_step": max(warmup_steps + 2, int(total_steps * 0.80)),
        "total_steps": total_steps,
        "peak_lr": args.peak_lr,
        "mid_lr": args.mid_lr,
        "min_lr": args.min_lr,
    }

    optimizer = torch.optim.AdamW(
        unwrap_model(model).parameters(),
        lr=args.peak_lr,
        weight_decay=args.weight_decay,
    )

    train_state = TrainState(seed=args.seed)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16

    if args.resume is not None and (resume_dir / TRAIN_STATE_JSON).exists():
        train_state, rng_states = maybe_resume(resume_dir, model, optimizer, device, scaler=scaler)
        if rng_states:
            restore_rng_states(rng_states)
        if is_main(rank):
            print(f"resumed from {resume_dir} at step={train_state.global_step}", flush=True)

    if args.wandb_id:
        train_state.wandb_run_id = args.wandb_id

    wb = None
    if is_main(rank):
        run_name = args.wandb_run_name or default_wandb_run_name(
            args, total_steps=total_steps, world_size=world_size,
        )
        wb = init_wandb(
            enabled=args.wandb,
            project=args.wandb_project or os.environ.get("WANDB_PROJECT", "strace"),
            run_name=run_name,
            config={
                **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                "variant": args.variant,
                "n_params": n_params,
                "vocab_size": vocab.size,
                "world_size": world_size,
                "effective_batch": args.batch_size * world_size * args.grad_accum,
                "total_steps": total_steps,
                "warmup_steps": warmup_steps,
            },
            resume_id=train_state.wandb_run_id,
        )
        if wb is not None and getattr(wb, "id", None):
            train_state.wandb_run_id = str(wb.id)
            wb.log({"model/n_params": n_params}, step=train_state.global_step)

        print(
            f"schedule=WSD warmup={warmup_steps} total_steps={total_steps} "
            f"peak_lr={args.peak_lr:g} batch={args.batch_size} "
            f"grad_accum={args.grad_accum} effective_batch={args.batch_size * world_size * args.grad_accum}",
            flush=True,
        )

    t0 = time.time()
    train_state = train(
        model,
        loader,
        optimizer,
        device,
        sched_cfg,
        train_state=train_state,
        total_steps=total_steps,
        grad_accum=args.grad_accum,
        max_grad_norm=args.max_grad_norm,
        log_every=args.log_every,
        save_every=args.save_every,
        val_every=args.val_every,
        val_loader=val_loader,
        out_dir=out_dir,
        config=config,
        vocab=vocab,
        wb=wb,
        rank=rank,
        sampler=sampler,
        scaler=scaler if scaler.is_enabled() else None,
        amp_enabled=args.amp and device.type == "cuda",
        amp_dtype=amp_dtype,
    )

    if is_main(rank):
        print(f"done in {time.time() - t0:.1f}s", flush=True)
        best_val_dir = ckpt_path(out_dir, "best_val")
        latest_dir = ckpt_path(out_dir, "latest")
        print(f"checkpoints: {latest_dir} (latest), {best_val_dir} (best val)", flush=True)

        if args.push_model and offline:
            print("skip HF push: --offline (Hub upload disabled)", flush=True)
        elif args.push_model:
            if not best_val_dir.is_dir():
                print("skip HF push: no checkpoint-best_val (enable --val-every)", flush=True)
            else:
                try:
                    full_repo = push_pcfg_model(
                        best_val_dir,
                        model=args.hf_model,
                        variant=args.variant,
                        namespace=args.hf_model_namespace,
                        private=args.hf_model_private,
                        token=resolve_hf_token(None),
                        commit_message=(
                            f"llada-{args.variant}-pcfg "
                            f"val_loss={train_state.best_val_loss:.4f} step={train_state.global_step}"
                        ),
                    )
                    print(f"pushed best checkpoint -> https://huggingface.co/{full_repo}", flush=True)
                except Exception as e:
                    print(f"HF model push failed: {e}", flush=True)

        if wb is not None:
            wb.finish()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
