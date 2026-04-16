"""
Launch eval with Hydra. Supports local runs and SLURM multirun via submitit launcher.

  # Single run (local, same as run_eval.py)
  python launch_eval.py task_name=gsm8k mode=pd

  # Multirun locally (multiple configs, sequential)
  python launch_eval.py task_name=gsm8k,humaneval mode=pd,bd --multirun

  # Multirun on SLURM (one job per combination)
  python launch_eval.py task_name=gsm8k,humaneval mode=pd,bd --multirun launcher=slurm_a40

Config: config/config.yaml. Launcher defaults: config/launcher/local.yaml; use launcher=slurm_* for cluster (e.g. launcher=slurm_a40).
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
    if "mbpp3" in task_name:
        return 3
    if "mbpp" in task_name:
        return 3
    if "arc_challenge_chat" in task_name:
        return 5
    return 0


def get_gen_length_steps(task_name: str) -> tuple[int, int]:
    if "arc_challenge_chat" in task_name:
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
    batch_size = int(cfg.get("batch_size", 4))
    mode_input = str(cfg.get("mode", "pd")).strip().lower()
    task_name = str(cfg.get("task_name", "gsm8k"))
    model = str(cfg.get("model", "llada"))
    compile_model = _to_bool(cfg.get("compile_model", False))
    wandb = _to_bool(cfg.get("wandb", True))
    tracing_enabled = _to_bool(cfg.get("tracing_enabled", True))

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
    block_length_cfg = cfg.get("block_length")
    if block_length_cfg is not None:
        block_length = int(block_length_cfg)
    else:
        block_length = 32 if mode == "block_diffusion" else (64 if mode == "fdm" else (16 if mode == "dus" else gen_length))

    top_k_cfg = cfg.get("top_k")
    top_k = int(top_k_cfg) if top_k_cfg is not None else 1

    num_fewshot_cfg = cfg.get("num_fewshot")
    num_fewshot = int(num_fewshot_cfg) if num_fewshot_cfg is not None else get_num_fewshot(task_name)
    model_path = f"./models/{model}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resumable = _to_bool(cfg.get("resumable", False))
    run_id = cfg.get("run_id") or None

    if resumable and run_id:
        output_suffix = str(run_id)
    elif resumable:
        output_suffix = f"{model}_{task_name}_{mode}"
        if intervention_mode:
            steps_raw = intervention_steps if intervention_steps is not None else "0"
            if isinstance(steps_raw, (list, tuple)):
                steps_safe = "_".join(str(s) for s in steps_raw)
            else:
                steps_safe = re.sub(r"[,_]", "_", str(steps_raw))
            output_suffix = f"{model}_{task_name}_{mode}_intv_{intervention_mode}_steps{steps_safe}_k{intervention_k}"
    else:
        output_suffix = f"{model}_{task_name}_{mode}_{timestamp}"
        if intervention_mode:
            steps_raw = intervention_steps if intervention_steps is not None else "0"
            if isinstance(steps_raw, (list, tuple)):
                steps_safe = "_".join(str(s) for s in steps_raw)
            else:
                steps_safe = re.sub(r"[,_]", "_", str(steps_raw))
            output_suffix = f"{model}_{task_name}_{mode}_intv_{intervention_mode}_steps{steps_safe}_k{intervention_k}_{timestamp}"
    wandb_name = output_suffix

    checkpoint_dir = f"outputs/{output_suffix}" if resumable else None

    model_args = (
        f"model_path={model_path},gen_length={gen_length},steps={steps},"
        f"block_length={block_length},mode={mode},strategy={str(strategy)},top_k={top_k},compile_model={str(compile_model).lower()},"
        f"task_name={task_name},trace_name={output_suffix}"
    )
    if resumable and checkpoint_dir:
        model_args += f",resumable=true,output_dir={checkpoint_dir}"
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
    if resumable:
        env["RESUMABLE_RUN_ID"] = output_suffix
        env["CHECKPOINT_DIR"] = checkpoint_dir
    env.setdefault("WANDB_API_KEY", "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")
    env.setdefault("WANDB_PROJECT", str(cfg.get("wandb_project", "strace")))

    wandb_resume_args = ""
    if resumable:
        os.makedirs(checkpoint_dir, exist_ok=True)
    if resumable and wandb:
        wandb_id_file = Path(checkpoint_dir) / ".wandb_id"
        if wandb_id_file.exists():
            wandb_run_id = wandb_id_file.read_text().strip()
            wandb_resume_args = f",resume=allow,id={wandb_run_id}"
            print(f"  Resuming wandb run: {wandb_run_id}")

    print("Configuration:")
    print(f"  batch_size={batch_size} mode={mode} task_name={task_name} model={model}")
    print(f"  compile_model={compile_model} gen_length={gen_length} steps={steps} block_length={block_length} top_k={top_k}")
    print(f"  num_fewshot={num_fewshot} wandb={wandb} tracing_enabled={tracing_enabled} resumable={resumable}")
    if intervention_mode:
        print(f"  intervention: mode={intervention_mode} steps={intervention_steps} k={intervention_k}")
    print(f"  output: outputs/{output_suffix}/" if resumable else f"  output: outputs/{output_suffix}.json")
    print()

    cmd = [
        "env", f"TRACING_ENABLED={'TRUE' if tracing_enabled else 'FALSE'}",
        "accelerate", "launch", "eval.py",
        "--tasks", task_name,
        "--batch_size", str(batch_size),
        "--num_fewshot", str(num_fewshot),
        "--model", model,
        "--model_args", model_args,
        "--write_out",
        "--cache_requests", "true",
    ]
    if resumable:
        cmd += ["--output_path", f"{checkpoint_dir}/results.json", "--log_samples"]
    else:
        cmd += ["--output_path", f"outputs/{output_suffix}.json", "--log_samples"]
    if wandb:
        cmd += ["--wandb_args", f"project={env.get('WANDB_PROJECT', 'strace')},name={wandb_name}{wandb_resume_args}"]

    return subprocess.run(cmd, env=env, cwd=project_root).returncode


def _normalize_hydra_cli(argv: list[str]) -> list[str]:
    """Normalize CLI args for Hydra.

    Hydra expects all flags (e.g. --multirun, --cfg, --resolve) to come before
    positional overrides (key=value). Users often interleave them, e.g.:
      python launch_eval.py task_name=humaneval mode=bd --multirun launcher=slurm_a40
    which Hydra will reject with 'unrecognized arguments'.

    This function reorders argv so flags come first and overrides come last.

    It also supports the common pattern:
      hydra.launcher=slurm_a40
    by rewriting it to our config-group override:
      launcher=slurm_a40
    when the requested launcher exists in config/launcher/*.yaml.
    """
    if len(argv) <= 1:
        return argv

    # Discover valid launcher names from config group.
    launcher_dir = Path(__file__).resolve().parent / "config" / "launcher"
    valid_launchers: set[str] = set()
    if launcher_dir.is_dir():
        valid_launchers = {p.stem for p in launcher_dir.glob("*.yaml")}

    # Hydra options that take a value as the next argv token.
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

        # Map 'hydra.launcher=slurm_a40' -> 'launcher=slurm_a40' when it matches our launcher group.
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


_CONFIG_DIR = Path(__file__).resolve().parent / "config"


@hydra.main(version_base="1.2", config_path=str(_CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    sys.exit(run(cfg))


if __name__ == "__main__":
    sys.argv = _normalize_hydra_cli(sys.argv)
    main()
