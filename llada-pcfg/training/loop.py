"""Training and validation loops."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch

from pcfg_metrics import masked_perplexity
from training.loss import llada_mask_loss
from pcfg_utils.checkpoint import TrainState, capture_rng_states, save_checkpoint
from pcfg_utils.distributed import is_main
from pcfg_utils.schedule import wsd_lr


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
