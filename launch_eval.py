"""
Launch eval with Hydra. Supports local runs and SLURM multirun via submitit launcher.

  # Single run (local, same as run_eval.py)
  python launch_eval.py task_name=gsm8k mode=pd

  # Multirun locally (multiple configs, sequential)
  python launch_eval.py task_name=gsm8k,humaneval mode=pd,bd --multirun

  # Multirun on SLURM (one job per combination)
  python launch_eval.py task_name=gsm8k,humaneval mode=pd,bd --multirun launcher=slurm

Config: config/config.yaml. Launcher defaults: config/launcher/local.yaml; use launcher=slurm for cluster.
"""

import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig

# Mode shortcut -> (full_mode, strategy)
MODE_MAP = {
    "pd": ("pure_diffusion", "confidence"),
    "bd": ("block_diffusion", "confidence"),
    "eb": ("eb_sampler", "confidence"),
    "fdm": ("fdm", None),
    "fhs": ("fhs", None),
    "top_k_c": ("top_k", "confidence"),
    "top_k_m": ("top_k", "top_margin"),
    "dus": ("dus", "confidence"),
}


def get_num_fewshot(task_name: str) -> int:
    if "humaneval" in task_name:
        return 0
    if "gsm8k" in task_name:
        return 4
    if "math500" in task_name or "minerva_math500" in task_name:
        return 4
    if "mbpp" in task_name:
        return 4
    if "truthfulqa" in task_name:
        return 0
    return 0


def get_gen_length_steps(task_name: str) -> tuple[int, int]:
    if "truthfulqa" in task_name:
        return 128, 128
    return 512, 512


def _to_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes")
    return bool(val)


def run(cfg: DictConfig) -> int:
    """Build and run the accelerate eval command from Hydra config."""
    # Use script dir as project root so SLURM jobs (which may run from sweep dir) still find eval.py
    project_root = Path(__file__).resolve().parent
    batch_size = int(cfg.get("batch_size", 2))
    mode_input = str(cfg.get("mode", "pd")).strip().lower()
    task_name = str(cfg.get("task_name", "gsm8k"))
    model = str(cfg.get("model", "llada"))
    compile_model = _to_bool(cfg.get("compile_model", False))
    wandb = _to_bool(cfg.get("wandb", False))
    tracing_enabled = _to_bool(cfg.get("tracing_enabled", False))

    intervention = cfg.get("intervention") or {}
    if isinstance(intervention, dict):
        intervention_mode = intervention.get("mode")
        intervention_steps = intervention.get("steps")
        intervention_k = int(intervention.get("k", 1))
    else:
        intervention_mode = None
        intervention_steps = None
        intervention_k = 1

    if mode_input in MODE_MAP:
        mode, strategy = MODE_MAP[mode_input]
    else:
        mode = mode_input

    gen_length, steps = get_gen_length_steps(task_name)
    block_length = 32 if mode == "block_diffusion" else gen_length
    num_fewshot = get_num_fewshot(task_name)
    model_path = f"./models/{model}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_suffix = f"{model}_{task_name}_{mode}_{timestamp}"
    if intervention_mode:
        steps_raw = intervention_steps if intervention_steps is not None else "0"
        if isinstance(steps_raw, (list, tuple)):
            steps_safe = "_".join(str(s) for s in steps_raw)
        else:
            steps_safe = re.sub(r"[,_]", "_", str(steps_raw))
        output_suffix = f"{model}_{task_name}_{mode}_intv_{intervention_mode}_steps{steps_safe}_k{intervention_k}_{timestamp}"
    wandb_name = output_suffix

    model_args = (
        f"model_path={model_path},gen_length={gen_length},steps={steps},"
        f"block_length={block_length},mode={mode},strategy={str(strategy)},compile_model={str(compile_model).lower()},"
        f"task_name={task_name},trace_name={output_suffix}"
    )
    if intervention_mode:
        if isinstance(intervention_steps, (list, tuple)):
            steps_arg = "_".join(str(s) for s in intervention_steps)
        else:
            steps_arg = re.sub(r",", "_", str(intervention_steps or "0"))
        model_args += f",intervention_mode={intervention_mode},intervention_steps={steps_arg},intervention_k={intervention_k}"

    env = os.environ.copy()
    env.setdefault("HF_ALLOW_CODE_EVAL", "1")
    env.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "true")
    env.setdefault("LMEVAL_LOG_LEVEL", "INFO")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env["TRACING_ENABLED"] = "TRUE" if tracing_enabled else "FALSE"
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    env.setdefault("WANDB_API_KEY", "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")
    env.setdefault("WANDB_PROJECT", str(cfg.get("wandb_project", "strace")))

    print("Configuration:")
    print(f"  batch_size={batch_size} mode={mode} task_name={task_name} model={model}")
    print(f"  compile_model={compile_model} gen_length={gen_length} steps={steps} block_length={block_length}")
    print(f"  num_fewshot={num_fewshot} wandb={wandb} tracing_enabled={tracing_enabled}")
    if intervention_mode:
        print(f"  intervention: mode={intervention_mode} steps={intervention_steps} k={intervention_k}")
    print(f"  output: outputs/{output_suffix}.json")
    print()

    cmd = [
        "env", f"TRACING_ENABLED={'TRUE' if tracing_enabled else 'FALSE'}",
        "accelerate", "launch", "eval.py",
        "-L", "4",
        "--tasks", task_name,
        "--batch_size", str(batch_size),
        "--num_fewshot", str(num_fewshot),
        "--model", model,
        "--model_args", model_args,
        "--write_out",
        "--output_path", f"outputs/{output_suffix}.json",
        "--log_samples",
        "--cache_requests", "true",
    ]
    if wandb:
        cmd += ["--wandb_args", f"project={env.get('WANDB_PROJECT', 'strace')},name={wandb_name}"]

    return subprocess.run(cmd, env=env, cwd=project_root).returncode


_CONFIG_DIR = Path(__file__).resolve().parent / "config"


@hydra.main(version_base="1.2", config_path=str(_CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    sys.exit(run(cfg))


if __name__ == "__main__":
    main()
