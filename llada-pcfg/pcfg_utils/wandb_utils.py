"""Weights & Biases logging helpers."""

from __future__ import annotations

from pathlib import Path

from pcfg_utils.distributed import is_main


def format_lr(lr: float) -> str:
    text = f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")
    mantissa, exponent = text.split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{exponent}"


def resolve_eval_model_name(
    *,
    model_dir: Path,
    hf_model: str | None,
    variant: str,
) -> str:
    """Short model label for eval trace / W&B run names."""
    if hf_model is not None and str(hf_model).strip():
        return str(hf_model).strip().rsplit("/", 1)[-1]
    name = model_dir.name
    if name.startswith("checkpoint-"):
        name = model_dir.parent.name
    if not name or name == ".":
        return f"llada-{variant}-pcfg"
    return name


def default_eval_run_name(model_name: str, scheduler_key: str, steps: int) -> str:
    """``{model}-pcfg_{scheduler}_s{steps}`` (trace filename stem and W&B run name)."""
    safe_model = str(model_name).replace("/", "-")
    return f"{safe_model}-pcfg_{scheduler_key}_s{int(steps)}"


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
    parts.append(f"lr{format_lr(args.peak_lr)}")
    if args.weight_decay != 0.1:
        parts.append(f"wd{format_lr(args.weight_decay)}")
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
