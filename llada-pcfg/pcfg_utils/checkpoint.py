"""Checkpoint and RNG state persistence."""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from models.pcfg_models import load_pcfg_tokenizer
from pcfg_utils.distributed import unwrap_model

TRAIN_STATE_JSON = "train_state.json"
CKPT_LATEST = "checkpoint-latest"
CKPT_BEST_VAL = "checkpoint-best_val"


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


def _cpu_byte_tensor(value) -> torch.Tensor:
    """RNG state tensors must live on CPU as uint8 (ByteTensor)."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected torch.Tensor RNG state, got {type(value)!r}")
    return value.detach().cpu().to(dtype=torch.uint8)


def restore_rng_states(state: dict) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(_cpu_byte_tensor(state["torch"]))
    if "cuda" in state and torch.cuda.is_available():
        cuda_states = state["cuda"]
        if isinstance(cuda_states, torch.Tensor):
            cuda_states = [cuda_states]
        torch.cuda.set_rng_state_all([_cpu_byte_tensor(s) for s in cuda_states])


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
        # trainer.pt holds optimizer/scaler/RNG state (includes numpy); not weights-only safe.
        # Keep trainer payload on CPU: RNG states must be CPU ByteTensors and
        # optimizer/scaler state_dicts load fine from CPU tensors.
        payload = torch.load(trainer_pt, map_location="cpu", weights_only=False)
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
