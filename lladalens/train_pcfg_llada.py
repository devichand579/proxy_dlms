"""
Train a small LLaDA variant on PCFG-sampled masked strings.

Each step samples t ~ Uniform[0, 1], masks content tokens independently with
probability t, and optimizes Eq. (3) with 1/(t·L) weighting on masked CE.

Example:
  python train_pcfg_llada.py --variant 100m --hf-dataset pcfg-code --steps 2000
  torchrun --nproc_per_node=2 train_pcfg_llada.py --variant 100m --corpus data/pcfg --steps 5000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from hf_hub import apply_pcfg_hf_env, push_pcfg_model, resolve_hf_token
from models.pcfg_models import count_parameters, list_variants, load_pcfg_llada, load_pcfg_tokenizer
from pcfg_data import ARCH_LEN, LABEL_IGNORE, load_corpus_from_disk, load_corpus_from_hub, make_train_example
from pcfg_metrics import make_val_example, masked_perplexity

TRAIN_STATE_JSON = "train_state.json"
CKPT_LATEST = "checkpoint-latest"
CKPT_BEST_VAL = "checkpoint-best_val"


def split_train_val(
    token_seqs: list,
    *,
    val_fraction: float = 0.05,
    seed: int = 0,
) -> tuple[list, list]:
    """Hold out a random fraction of strings for validation."""
    if not token_seqs:
        return [], []
    if val_fraction <= 0.0:
        return list(token_seqs), []
    n_val = max(1, int(round(len(token_seqs) * val_fraction)))
    if n_val >= len(token_seqs):
        n_val = max(1, len(token_seqs) // 20)
    rng = random.Random(seed)
    indices = list(range(len(token_seqs)))
    rng.shuffle(indices)
    val_idx = set(indices[:n_val])
    train = [token_seqs[i] for i in range(len(token_seqs)) if i not in val_idx]
    val = [token_seqs[i] for i in val_idx]
    return train, val


def _remove_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def ckpt_path(out_dir: Path, tag: str) -> Path:
    return out_dir / f"checkpoint-{tag}"


def capture_rng_states() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_states(state: dict) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


@dataclass
class TrainState:
    global_step: int = 0
    epoch: int = 0
    best_val_loss: float = float("inf")
    wandb_run_id: str | None = None
    seed: int = 0

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> TrainState:
        data = json.loads(path.read_text())
        return cls(
            global_step=int(data.get("global_step", 0)),
            epoch=int(data.get("epoch", 0)),
            best_val_loss=float(data.get("best_val_loss", float("inf"))),
            wandb_run_id=data.get("wandb_run_id"),
            seed=int(data.get("seed", 0)),
        )


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


class PCFGValDataset(Dataset):
    """Held-out eval with deterministic masking per example index."""

    def __init__(self, token_seqs, vocab, arch_len=ARCH_LEN, seed: int = 0):
        self.token_seqs = token_seqs
        self.vocab = vocab
        self.arch_len = arch_len
        self.seed = seed

    def __len__(self):
        return len(self.token_seqs)

    def __getitem__(self, idx):
        ex = make_val_example(self.token_seqs[idx], self.vocab, arch_len=self.arch_len, seed=self.seed + idx)
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
        if int(masked.sum().item()) == 0:
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


def setup_distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 0, 1
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def is_main(rank: int) -> bool:
    return rank == 0


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def _format_lr(lr: float) -> str:
    text = f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    mantissa, exponent = text.split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{exponent}"


def default_wandb_run_name(
    args,
    *,
    total_steps: int,
    world_size: int,
) -> str:
    """Compact run name encoding key hparams for ablation sweeps."""
    parts = [
        "pcfg",
        str(args.variant),
        f"bs{args.batch_size}",
    ]
    if world_size > 1:
        parts.append(f"x{world_size}gpu")
    if args.grad_accum > 1:
        parts.append(f"ga{args.grad_accum}")
    if args.steps is not None:
        parts.append(f"s{args.steps}")
    else:
        parts.append(f"ep{args.epochs}")
    parts.append(f"lr{_format_lr(args.peak_lr)}")
    if args.weight_decay != 0.1:
        parts.append(f"wd{_format_lr(args.weight_decay)}")
    if args.val_fraction != 0.05:
        parts.append(f"vf{int(round(args.val_fraction * 100))}")
    if args.amp_dtype != "bfloat16":
        parts.append(str(args.amp_dtype))
    if args.corpus is not None:
        parts.append("disk")
    elif args.hf_dataset is not None:
        slug = args.hf_dataset.rsplit("/", 1)[-1].replace("_", "-")[:16]
        parts.append(slug)
    elif args.train_tokens is not None:
        if args.train_tokens >= 1_000_000:
            parts.append(f"tok{args.train_tokens // 1_000_000}M")
        elif args.train_tokens >= 1_000:
            parts.append(f"tok{args.train_tokens // 1_000}")
        else:
            parts.append(f"tok{args.train_tokens}")
    return "-".join(parts)


def resolve_warmup_steps(total_steps: int, *, warmup_steps: int | None, warmup_fraction: float, warmup_max: int) -> int:
    if warmup_steps is not None:
        return max(1, warmup_steps)
    return max(1, min(warmup_max, int(total_steps * warmup_fraction)))


def init_wandb(
    *,
    enabled: bool,
    project: str,
    run_name: str,
    config: dict,
    resume_id: str | None,
):
    if not enabled:
        return None
    try:
        import wandb
    except ImportError:
        if is_main(0):
            print("wandb not installed; continuing without logging", flush=True)
        return None

    init_kwargs = {
        "project": project,
        "name": run_name,
        "config": config,
        "reinit": True,
    }
    if resume_id:
        init_kwargs["id"] = resume_id
        init_kwargs["resume"] = "allow"
    return wandb.init(**init_kwargs)


def save_checkpoint(
    out_dir: Path,
    model,
    optimizer,
    train_state: TrainState,
    *,
    tag: str,
    config=None,
    vocab=None,
    scaler=None,
    rng_states: dict | None = None,
) -> Path:
    """Save one checkpoint; replaces any previous checkpoint with the same tag."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = ckpt_path(out_dir, tag)
    _remove_dir(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    unwrap_model(model).save_pretrained(ckpt_dir)
    if config is not None:
        config.save_pretrained(ckpt_dir)
    if vocab is not None:
        load_pcfg_tokenizer().save_pretrained(ckpt_dir)
        (ckpt_dir / "pcfg_vocab.txt").write_text("\n".join(vocab.itos) + "\n")
    payload = {
        "optimizer": optimizer.state_dict(),
        "train_state": asdict(train_state),
        "rng_states": rng_states or capture_rng_states(),
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, ckpt_dir / "trainer.pt")
    train_state.save(out_dir / TRAIN_STATE_JSON)
    return ckpt_dir


def maybe_resume(
    resume_dir: Path,
    model,
    optimizer,
    device,
    scaler=None,
) -> tuple[TrainState, dict | None]:
    state_path = resume_dir / TRAIN_STATE_JSON
    if not state_path.exists():
        raise FileNotFoundError(f"missing {state_path}")
    train_state = TrainState.load(state_path)
    rng_states = None

    ckpt_dir = resume_dir / CKPT_LATEST
    trainer_pt = ckpt_dir / "trainer.pt"
    if trainer_pt.exists():
        payload = torch.load(trainer_pt, map_location=device)
        optimizer.load_state_dict(payload["optimizer"])
        if scaler is not None and "scaler" in payload:
            scaler.load_state_dict(payload["scaler"])
        rng_states = payload.get("rng_states")
        if "train_state" in payload:
            ts = payload["train_state"]
            train_state.global_step = int(ts.get("global_step", train_state.global_step))
            train_state.epoch = int(ts.get("epoch", train_state.epoch))
            train_state.best_val_loss = float(ts.get("best_val_loss", train_state.best_val_loss))
            train_state.wandb_run_id = ts.get("wandb_run_id", train_state.wandb_run_id)
            train_state.seed = int(ts.get("seed", train_state.seed))
    return train_state, rng_states


@torch.no_grad()
def evaluate_val(
    model,
    loader,
    device,
    *,
    amp_enabled: bool = False,
    amp_dtype=torch.float16,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    autocast = torch.cuda.amp.autocast if amp_enabled and device.type == "cuda" else nullcontext
    for batch in loader:
        with autocast(dtype=amp_dtype):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            mask_t = batch["mask_t"].to(device, non_blocking=True)
            n_content = batch["n_content"].to(device, non_blocking=True)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = llada_mask_loss(out.logits, labels, mask_t, n_content)
        total += float(loss.item())
        n += 1
    model.train()
    return total / max(n, 1)


def train(
    model,
    loader,
    optimizer,
    device,
    sched_cfg,
    *,
    train_state: TrainState,
    total_steps: int,
    grad_accum: int,
    max_grad_norm: float,
    log_every: int,
    save_every: int,
    val_every: int,
    val_loader,
    out_dir: Path,
    config,
    vocab,
    wb,
    rank: int,
    sampler=None,
    scaler=None,
    amp_enabled: bool = False,
    amp_dtype=torch.float16,
) -> TrainState:
    model.train()
    accum_loss = 0.0
    accum_count = 0
    optimizer.zero_grad(set_to_none=True)
    autocast = torch.cuda.amp.autocast if amp_enabled and device.type == "cuda" else nullcontext

    while train_state.global_step < total_steps:
        if sampler is not None:
            sampler.set_epoch(train_state.epoch)
        for batch in loader:
            if train_state.global_step >= total_steps:
                break

            lr = wsd_lr(train_state.global_step, **sched_cfg)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            mask_t = batch["mask_t"].to(device, non_blocking=True)
            n_content = batch["n_content"].to(device, non_blocking=True)

            with autocast(dtype=amp_dtype):
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = llada_mask_loss(out.logits, labels, mask_t, n_content)

            if scaler is not None:
                (scaler.scale(loss) / grad_accum).backward()
            else:
                (loss / grad_accum).backward()

            accum_loss += float(loss.item())
            accum_count += 1

            if accum_count % grad_accum != 0:
                continue

            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            step_loss = accum_loss / grad_accum
            accum_loss = 0.0
            accum_count = 0
            train_state.global_step += 1

            run_val = (
                val_loader is not None
                and val_every > 0
                and train_state.global_step % val_every == 0
            )
            if run_val and is_main(rank):
                val_loss = evaluate_val(
                    model, val_loader, device, amp_enabled=amp_enabled, amp_dtype=amp_dtype,
                )
                val_ppl = masked_perplexity(val_loss)
                if val_loss < train_state.best_val_loss:
                    train_state.best_val_loss = val_loss
                    save_checkpoint(
                        out_dir, model, optimizer, train_state, tag="best_val",
                        config=config, vocab=vocab,
                        scaler=scaler, rng_states=capture_rng_states(),
                    )
                if wb is not None:
                    wb.log(
                        {"val/loss": val_loss, "val/ppl": val_ppl, "val/best_loss": train_state.best_val_loss},
                        step=train_state.global_step,
                    )
                print(
                    f"  val step={train_state.global_step} loss={val_loss:.4f} ppl={val_ppl:.2f} "
                    f"best_val={train_state.best_val_loss:.4f}",
                    flush=True,
                )

            if is_main(rank) and (
                train_state.global_step % log_every == 0
                or train_state.global_step == 1
                or train_state.global_step >= total_steps
            ):
                ppl = masked_perplexity(step_loss)
                msg = (
                    f"step={train_state.global_step}/{total_steps} "
                    f"loss={step_loss:.4f} ppl={ppl:.2f} lr={lr:g}"
                )
                print(msg, flush=True)
                if wb is not None:
                    wb.log(
                        {
                            "train/loss": step_loss,
                            "train/ppl": ppl,
                            "train/lr": lr,
                            "train/epoch": train_state.epoch,
                        },
                        step=train_state.global_step,
                    )

            if is_main(rank) and save_every > 0 and train_state.global_step % save_every == 0:
                save_checkpoint(
                    out_dir, model, optimizer, train_state, tag="latest",
                    config=config, vocab=vocab,
                    scaler=scaler, rng_states=capture_rng_states(),
                )

        train_state.epoch += 1

    if is_main(rank):
        save_checkpoint(
            out_dir, model, optimizer, train_state, tag="latest",
            config=config, vocab=vocab,
            scaler=scaler, rng_states=capture_rng_states(),
        )
    return train_state


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
    p.add_argument("--hf-model", type=str, default=None, help="Short or full HF model repo id.")
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
    args = p.parse_args()

    apply_pcfg_hf_env()
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

    out_dir = args.out_dir or Path("checkpoints") / f"llada-{args.variant}-pcfg"
    resume_dir = args.resume or out_dir

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

        if args.push_model:
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
