"""
Launch eval with Hydra config. Config is in config/eval.yaml; override any key from CLI.

  python run_eval.py                          # use config defaults
  python run_eval.py task_name=gsm8k batch_size=4
  python run_eval.py mode=pd task_name=humaneval wandb=true tracing_enabled=true
  python run_eval.py intervention.mode=drop intervention.steps=0,128,256

Then: accelerate launch eval.py ... is run with the composed config.
"""

import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

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
    if "bbh" in task_name:
        return 3
    if "truthfulqa" in task_name:
        return 0
    return 0


def get_gen_length_steps(task_name: str) -> tuple[int, int]:
    if "humaneval" in task_name or "mbpp" in task_name:
        return 512, 512
    return 256, 256


def main():
    # Load config: config/eval.yaml, with CLI overrides
    config_path = Path(__file__).parent / "config"
    if not (config_path / "eval.yaml").exists():
        print(f"Config not found: {config_path / 'eval.yaml'}", file=sys.stderr)
        sys.exit(1)

    cli_overrides = [a for a in sys.argv[1:] if "=" in a and not a.startswith("-")]
    cfg = OmegaConf.load(config_path / "eval.yaml")
    if cli_overrides:
        overrides = OmegaConf.from_dotlist(cli_overrides)
        cfg = OmegaConf.merge(cfg, overrides)

    batch_size = cfg.get("batch_size", 2)
    mode_input = str(cfg.get("mode", "pd")).strip().lower()
    task_name = str(cfg.get("task_name", "gsm8k"))
    model = str(cfg.get("model", "llada"))
    compile_model = cfg.get("compile_model", False)
    if isinstance(compile_model, str):
        compile_model = compile_model.lower() in ("true", "1", "yes")
    wandb = cfg.get("wandb", False)
    if isinstance(wandb, str):
        wandb = wandb.lower() in ("true", "1", "yes")
    tracing_enabled = cfg.get("tracing_enabled", False)
    if isinstance(tracing_enabled, str):
        tracing_enabled = tracing_enabled.lower() in ("true", "1", "yes")

    intervention = cfg.get("intervention") or {}
    if isinstance(intervention, dict):
        intervention_mode = intervention.get("mode")
        intervention_steps = intervention.get("steps")
        intervention_k = intervention.get("k", 1)
    else:
        intervention_mode = None
        intervention_steps = None
        intervention_k = 1

    # Resolve mode
    if mode_input in MODE_MAP:
        mode, _ = MODE_MAP[mode_input]
    else:
        mode = mode_input

    gen_length, steps = get_gen_length_steps(task_name)
    block_length = 32 if mode == "block_diffusion" else gen_length
    num_fewshot = get_num_fewshot(task_name)
    model_path = f"./models/{model}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Output / wandb naming
    output_suffix = f"{model}_{task_name}_{mode}_{timestamp}"
    if intervention_mode:
        steps_raw = intervention_steps if intervention_steps is not None else "0"
        if isinstance(steps_raw, (list, tuple)):
            steps_safe = "_".join(str(s) for s in steps_raw)
        else:
            steps_safe = re.sub(r"[,_]", "_", str(steps_raw))
        output_suffix = f"{model}_{task_name}_{mode}_intv_{intervention_mode}_steps{steps_safe}_k{intervention_k}_{timestamp}"
    wandb_name = output_suffix

    # Model args for lm_eval (comma-separated; use _ for intervention_steps to avoid breaking parsing)
    model_args = (
        f"model_path={model_path},gen_length={gen_length},steps={steps},"
        f"block_length={block_length},mode={mode},compile_model={str(compile_model).lower()},"
        f"task_name={task_name},trace_name={output_suffix}"
    )
    if intervention_mode:
        if isinstance(intervention_steps, (list, tuple)):
            steps_arg = "_".join(str(s) for s in intervention_steps)
        else:
            steps_arg = re.sub(r",", "_", str(intervention_steps or "0"))
        model_args += f",intervention_mode={intervention_mode},intervention_steps={steps_arg},intervention_k={intervention_k}"

    # Env
    env = os.environ.copy()
    env.setdefault("HF_ALLOW_CODE_EVAL", "1")
    env.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "true")
    env.setdefault("LMEVAL_LOG_LEVEL", "INFO")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    env["CUDA_VISIBLE_DEVICES"] = str(cfg.get("cuda_visible_devices", "1"))
    env["TRACING_ENABLED"] = "TRUE" if tracing_enabled else "FALSE"
    if cfg.get("wandb_project"):
        env.setdefault("WANDB_PROJECT", str(cfg.wandb_project))

    # Print config
    print("Configuration:")
    print(f"  batch_size={batch_size} mode={mode} task_name={task_name} model={model}")
    print(f"  compile_model={compile_model} gen_length={gen_length} steps={steps} block_length={block_length}")
    print(f"  num_fewshot={num_fewshot} wandb={wandb} tracing_enabled={tracing_enabled}")
    if intervention_mode:
        print(f"  intervention: mode={intervention_mode} steps={intervention_steps} k={intervention_k}")
    print(f"  output: outputs/{output_suffix}.json")
    print()

    # Build command
    cmd = [
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
        "--cache_requests",
    ]
    if wandb:
        cmd += ["--wandb_args", f"project={env.get('WANDB_PROJECT', 'strace')},name={wandb_name}"]

    sys.exit(subprocess.run(cmd, env=env, cwd=Path(__file__).parent).returncode)


if __name__ == "__main__":
    main()
