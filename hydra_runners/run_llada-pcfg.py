#!/usr/bin/env python3
"""
LLaDA-PCFG training and scheduler eval via Hydra.
Config: config/llada-pcfg.yaml
Shell: ./scripts/llada-pcfg.sh --mode train|eval [hydra overrides...]
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LLADA_PCFG_DIR = _REPO_ROOT / "llada-pcfg"
_SCRIPTS_DIR = _LLADA_PCFG_DIR / "scripts"
_CONFIG_DIR = _REPO_ROOT / "config"
_LAUNCHER_DIR = _CONFIG_DIR / "launcher"

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from repo_env_defaults import apply_offline_env, apply_repo_env_defaults, resolve_eval_device

import hydra
from omegaconf import DictConfig, OmegaConf


def _to_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    return str(val).strip().lower() in ("true", "1", "yes", "on")


def _resolve_path(val: str | Path | None, *, base: Path) -> str | None:
    if val is None or str(val).strip() in ("", "null", "None"):
        return None
    path = Path(str(val)).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    return str(path)


def _section_dict(cfg: DictConfig, name: str) -> dict[str, Any]:
    block = cfg.get(name)
    if block is None:
        return {}
    container = OmegaConf.to_container(block, resolve=True)
    return container if isinstance(container, dict) else {}


def _add_scalar(argv: list[str], block: dict[str, Any], key: str, flag: str) -> None:
    val = block.get(key)
    if val is None or str(val).strip() in ("", "null", "None"):
        return
    argv.extend([flag, str(val)])


def _build_train_argv(train: dict[str, Any], *, offline: bool = False) -> list[str]:
    argv: list[str] = []

    path_keys = {
        "corpus": "--corpus",
        "out_dir": "--out-dir",
        "resume": "--resume",
    }
    for key, flag in path_keys.items():
        resolved = _resolve_path(train.get(key), base=_REPO_ROOT)
        if resolved is not None:
            argv.extend([flag, resolved])

    scalar_keys = [
        ("variant", "--variant"),
        ("train_tokens", "--train-tokens"),
        ("hf_dataset", "--hf-dataset"),
        ("hf_namespace", "--hf-namespace"),
        ("batch_size", "--batch-size"),
        ("grad_accum", "--grad-accum"),
        ("num_workers", "--num-workers"),
        ("peak_lr", "--peak-lr"),
        ("mid_lr", "--mid-lr"),
        ("min_lr", "--min-lr"),
        ("warmup_steps", "--warmup-steps"),
        ("warmup_fraction", "--warmup-fraction"),
        ("warmup_max", "--warmup-max"),
        ("epochs", "--epochs"),
        ("steps", "--steps"),
        ("weight_decay", "--weight-decay"),
        ("max_grad_norm", "--max-grad-norm"),
        ("seed", "--seed"),
        ("device", "--device"),
        ("log_every", "--log-every"),
        ("save_every", "--save-every"),
        ("val_fraction", "--val-fraction"),
        ("val_every", "--val-every"),
        ("hf_model", "--hf-model"),
        ("hf_model_namespace", "--hf-model-namespace"),
        ("amp_dtype", "--amp-dtype"),
        ("wandb_run_name", "--wandb-run-name"),
        ("wandb_id", "--wandb-id"),
        ("wandb_project", "--wandb-project"),
    ]
    for key, flag in scalar_keys:
        _add_scalar(argv, train, key, flag)

    if train.get("amp") is False:
        argv.append("--no-amp")
    if train.get("wandb") is False:
        argv.append("--no-wandb")
    if _to_bool(train.get("push_model")):
        argv.append("--push-model")
    if _to_bool(train.get("hf_model_private")):
        argv.append("--hf-model-private")
    if offline:
        argv.append("--offline")

    sources = sum(
        1
        for k in ("corpus", "hf_dataset", "train_tokens")
        if train.get(k) is not None and str(train.get(k)).strip() not in ("", "null", "None")
    )
    if sources == 0:
        raise ValueError("train: set one of corpus, hf_dataset, or train_tokens")
    if sources > 1:
        raise ValueError("train: set only one of corpus, hf_dataset, or train_tokens")

    return argv


def _build_eval_argv(ev: dict[str, Any], *, offline: bool = False) -> list[str]:
    argv: list[str] = []

    for key, flag in {
        "model_path": "--model-path",
        "corpus": "--corpus",
        "out_dir": "--out-dir",
    }.items():
        resolved = _resolve_path(ev.get(key), base=_REPO_ROOT)
        if resolved is not None:
            argv.extend([flag, resolved])

    scalar_keys = [
        ("hf_model", "--hf-model"),
        ("variant", "--variant"),
        ("hf_dataset", "--hf-dataset"),
        ("hf_namespace", "--hf-namespace"),
        ("schedulers", "--schedulers"),
        ("max_samples", "--max-samples"),
        ("steps", "--steps"),
        ("batch_size", "--batch-size"),
        ("device", "--device"),
    ]
    for key, flag in scalar_keys:
        _add_scalar(argv, ev, key, flag)

    if ev.get("upload_traces") is False:
        argv.append("--no-upload-traces")
    if ev.get("replay_oracle") is False:
        argv.append("--no-replay-oracle")
    if offline:
        argv.append("--offline")

    has_model = (
        ev.get("model_path") is not None and str(ev.get("model_path")).strip() not in ("", "null", "None")
    ) or (
        ev.get("hf_model") is not None and str(ev.get("hf_model")).strip() not in ("", "null", "None")
    )
    if not has_model:
        raise ValueError("eval: set model_path or hf_model")

    has_corpus = (
        ev.get("corpus") is not None and str(ev.get("corpus")).strip() not in ("", "null", "None")
    ) or (
        ev.get("hf_dataset") is not None and str(ev.get("hf_dataset")).strip() not in ("", "null", "None")
    )
    if not has_corpus:
        raise ValueError("eval: set corpus or hf_dataset for test split")

    return argv


def _apply_eval_trace_env(env: dict[str, str], cfg: DictConfig) -> None:
    ev = _section_dict(cfg, "eval")
    tracing = _to_bool(ev.get("tracing_enabled", True))
    env["TRACING_ENABLED"] = "TRUE" if tracing else "FALSE"

    upload = ev.get("trace_hf_upload")
    if cfg.get("offline"):
        upload = False
    elif upload is None:
        upload = tracing
    else:
        upload = _to_bool(upload)
    env["TRACE_HF_UPLOAD"] = "TRUE" if upload else "FALSE"

    repo = ev.get("trace_hf_dataset_repo")
    if repo is not None and str(repo).strip():
        env["TRACE_HF_DATASET_REPO"] = str(repo).strip()
    ns = ev.get("trace_hf_namespace")
    if ns is not None and str(ns).strip():
        env["TRACE_HF_NAMESPACE"] = str(ns).strip()


def _apply_common_env(cfg: DictConfig) -> dict[str, str]:
    env = os.environ.copy()
    apply_repo_env_defaults()
    for key in ("HF_TOKEN", "WANDB_API_KEY", "WANDB_PROJECT"):
        if key in os.environ:
            env[key] = os.environ[key]

    project = cfg.get("wandb_project")
    if project is not None and str(project).strip():
        env["WANDB_PROJECT"] = str(project).strip()

    if _to_bool(cfg.get("offline")):
        apply_offline_env(env)

    return env


def _train_launch_prefix(nproc: int) -> list[str]:
    if nproc <= 1:
        return [sys.executable]
    return [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={nproc}",
    ]


def _wrap_slurm(cmd: list[str], *, use_slurm: bool) -> list[str]:
    if not use_slurm:
        return cmd
    if os.environ.get("SLURM_JOB_ID"):
        return ["srun", *cmd]
    return cmd


def _pythonpath_env(env: dict[str, str]) -> dict[str, str]:
    prefix = os.pathsep.join([str(_LLADA_PCFG_DIR), str(_REPO_ROOT)])
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([p for p in (prefix, existing) if p])
    return env


def run_train(cfg: DictConfig, *, use_slurm: bool = False) -> int:
    train = _section_dict(cfg, "train")
    nproc = int(train.get("nproc_per_node") or 1)
    if nproc < 1:
        raise ValueError("train.nproc_per_node must be >= 1")

    argv = _build_train_argv(train, offline=_to_bool(cfg.get("offline")))
    script = str(_SCRIPTS_DIR / "train_pcfg_llada.py")
    cmd = _wrap_slurm([*_train_launch_prefix(nproc), script, *argv], use_slurm=use_slurm)

    env = _pythonpath_env(_apply_common_env(cfg))
    device = train.get("device")
    if device is not None and str(device).strip():
        model_device, cuda_vis = resolve_eval_device(str(device))
        if cuda_vis is not None:
            env["CUDA_VISIBLE_DEVICES"] = cuda_vis
        if model_device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""

    print("Running:", " ".join(cmd), flush=True)
    return int(subprocess.run(cmd, cwd=_REPO_ROOT, env=env).returncode)


def run_eval(cfg: DictConfig, *, use_slurm: bool = False) -> int:
    ev = _section_dict(cfg, "eval")
    argv = _build_eval_argv(ev, offline=_to_bool(cfg.get("offline")))
    script = str(_SCRIPTS_DIR / "pcfg_eval.py")
    cmd = _wrap_slurm([sys.executable, script, *argv], use_slurm=use_slurm)

    env = _pythonpath_env(_apply_common_env(cfg))
    _apply_eval_trace_env(env, cfg)

    device = ev.get("device")
    if device is not None and str(device).strip():
        model_device, cuda_vis = resolve_eval_device(str(device))
        if cuda_vis is not None:
            env["CUDA_VISIBLE_DEVICES"] = cuda_vis
        if model_device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""

    print("Running:", " ".join(cmd), flush=True)
    return int(subprocess.run(cmd, cwd=_REPO_ROOT, env=env).returncode)


def dispatch(cfg: DictConfig) -> int:
    mode = str(cfg.get("mode", "train")).strip().lower()
    use_slurm = _to_bool(os.environ.get("LLADA_PCFG_SLURM_WRAPPED"))
    if mode == "train":
        return run_train(cfg, use_slurm=use_slurm)
    if mode == "eval":
        return run_eval(cfg, use_slurm=use_slurm)
    raise ValueError(f"unknown mode={mode!r} (use train or eval)")


def _normalize_hydra_cli(argv: list[str]) -> list[str]:
    if len(argv) <= 1:
        return argv

    valid_launchers: set[str] = set()
    if _LAUNCHER_DIR.is_dir():
        valid_launchers = {p.stem for p in _LAUNCHER_DIR.glob("*.yaml")}

    opts_with_value = {
        "--cfg",
        "--info",
        "--config-path",
        "--config-name",
        "--config-dir",
        "--package",
        "--experimental-rerun",
    }

    flags: list[str] = []
    overrides: list[str] = []

    i = 1
    while i < len(argv):
        a = argv[i]
        if a.startswith("hydra.launcher="):
            val = a.split("=", 1)[1]
            if val in valid_launchers:
                overrides.append(f"launcher={val}")
            else:
                overrides.append(a)
            i += 1
            continue

        if a.startswith("-"):
            flags.append(a)
            if a in opts_with_value and i + 1 < len(argv):
                flags.append(argv[i + 1])
                i += 2
            else:
                i += 1
            continue

        overrides.append(a)
        i += 1

    return [argv[0], *flags, *overrides]


@hydra.main(version_base="1.2", config_path=str(_CONFIG_DIR), config_name="llada-pcfg")
def main(cfg: DictConfig) -> None:
    sys.exit(dispatch(cfg))


if __name__ == "__main__":
    sys.argv = _normalize_hydra_cli(sys.argv)
    main()
