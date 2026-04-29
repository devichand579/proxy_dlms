"""Run `eval.py` via Hydra (local or SLURM). Config `config/config.yaml`; details in `docs/evaluation.md`."""

import concurrent.futures
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from repo_env_defaults import apply_repo_env_defaults

import hydra
from omegaconf import DictConfig, OmegaConf

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


def _resumable_output_basename(model: str, task_name: str, mode: str, strategy) -> str:
    """Segment for outputs/<name>/ … ; disambiguates top_k+confidence vs top_k+top_margin."""
    if mode == "top_k":
        s = ""
        if strategy is not None:
            s = str(strategy).strip()
        if not s:
            s = "confidence"
        return f"{model}_{task_name}_{mode}_{s}"
    return f"{model}_{task_name}_{mode}"


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


def _run_single_eval(cfg: DictConfig, extra_env: Optional[dict[str, str]] = None) -> int:
    """Build and run one accelerate eval; optional env overrides (e.g. CUDA_VISIBLE_DEVICES per slot)."""
    apply_repo_env_defaults()
    # Use script dir as project root so SLURM jobs (which may run from sweep dir) still find eval.py
    project_root = Path(__file__).resolve().parent
    batch_size = int(cfg.get("batch_size", 4))
    mode_input = str(cfg.get("mode", "pd")).strip().lower()
    task_name = str(cfg.get("task_name", "gsm8k"))
    model = str(cfg.get("model", "llada"))
    compile_model = _to_bool(cfg.get("compile_model", False))
    ueh = cfg.get("use_eos_harness", True)
    if isinstance(ueh, str):
        use_eos_harness = ueh.strip().lower() in ("true", "1", "yes", "on")
    else:
        use_eos_harness = True if ueh is None else bool(ueh)
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
        strategy = cfg.get("strategy")

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
        base = _resumable_output_basename(model, task_name, mode, strategy)
        output_suffix = base
        if intervention_mode:
            steps_raw = intervention_steps if intervention_steps is not None else "0"
            if isinstance(steps_raw, (list, tuple)):
                steps_safe = "_".join(str(s) for s in steps_raw)
            else:
                steps_safe = re.sub(r"[,_]", "_", str(steps_raw))
            output_suffix = f"{base}_intv_{intervention_mode}_steps{steps_safe}_k{intervention_k}"
    else:
        base_ts = _resumable_output_basename(model, task_name, mode, strategy)
        output_suffix = f"{base_ts}_{timestamp}"
        if intervention_mode:
            steps_raw = intervention_steps if intervention_steps is not None else "0"
            if isinstance(steps_raw, (list, tuple)):
                steps_safe = "_".join(str(s) for s in steps_raw)
            else:
                steps_safe = re.sub(r"[,_]", "_", str(steps_raw))
            output_suffix = f"{base_ts}_intv_{intervention_mode}_steps{steps_safe}_k{intervention_k}_{timestamp}"
    wandb_name = output_suffix

    checkpoint_dir = f"outputs/{output_suffix}" if resumable else None

    model_args = (
        f"model_path={model_path},gen_length={gen_length},steps={steps},"
        f"block_length={block_length},mode={mode},strategy={str(strategy)},top_k={top_k},compile_model={str(compile_model).lower()},"
        f"task_name={task_name},trace_name={output_suffix},use_eos_harness={str(use_eos_harness).lower()}"
    )
    if resumable and checkpoint_dir:
        model_args += f",resumable=true,output_dir={checkpoint_dir}"
    if intervention_mode:
        if isinstance(intervention_steps, (list, tuple)):
            steps_arg = "_".join(str(s) for s in intervention_steps)
        else:
            steps_arg = re.sub(r",", "_", str(intervention_steps or "0"))
        model_args += f",intervention_mode={intervention_mode},intervention_steps={steps_arg},intervention_k={intervention_k}"

    cst_ck = cfg.get("cst_checkpoint")
    cst_ck_s = str(cst_ck).strip() if cst_ck is not None else ""
    if cst_ck_s and cst_ck_s.lower() not in ("null", "none", ""):
        # CST metrics are computed on written hidden .pt files; need capture on.
        capture_pre_logits = True
    else:
        capture_pre_logits = _to_bool(cfg.get("capture_pre_logits", False))
    hidden_capture_path = cfg.get("hidden_capture_path")
    if capture_pre_logits:
        if hidden_capture_path is None or str(hidden_capture_path).strip().lower() in ("null", "none", ""):
            if resumable and checkpoint_dir:
                hidden_dir = f"{checkpoint_dir}/hidden"
            else:
                hidden_dir = f"outputs/{output_suffix}/hidden"
        else:
            hidden_dir = str(hidden_capture_path).strip()
        os.makedirs(hidden_dir, exist_ok=True)
        model_args += f",capture_pre_logits=true,hidden_capture_path={hidden_dir}"

    cst_config_path = cfg.get("cst_config_path")
    cst_conf_s = str(cst_config_path).strip() if cst_config_path is not None else ""
    cst_config_arg = cst_conf_s if cst_conf_s and cst_conf_s.lower() not in ("null", "none", "") else ""
    if cst_ck_s and cst_ck_s.lower() not in ("null", "none", ""):
        cst_nlo = float(cfg.get("cst_ntf_conf_low", 0.2))
        cst_nhi = float(cfg.get("cst_ntf_conf_high", 0.8))
        cst_tgb = float(cfg.get("cst_target_staging_gb", 1.0))
        cst_ntf_1st = str(_to_bool(cfg.get("cst_ntf_first_non_eos", False))).lower()
        model_args += f",cst_checkpoint={cst_ck_s}"
        if cst_config_arg:
            model_args += f",cst_config_path={cst_config_arg}"
        model_args += f",cst_ntf_conf_low={cst_nlo},cst_ntf_conf_high={cst_nhi},cst_ntf_first_non_eos={cst_ntf_1st},cst_target_staging_gb={cst_tgb}"

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    env.setdefault("HF_ALLOW_CODE_EVAL", "1")
    env.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "true")
    env.setdefault("LMEVAL_LOG_LEVEL", "INFO")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env["TRACING_ENABLED"] = "TRUE" if tracing_enabled else "FALSE"
    if resumable:
        env["RESUMABLE_RUN_ID"] = output_suffix
        env["CHECKPOINT_DIR"] = checkpoint_dir
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
    print(
        f"  compile_model={compile_model} use_eos_harness={use_eos_harness} "
        f"gen_length={gen_length} steps={steps} block_length={block_length} top_k={top_k}"
    )
    print(f"  num_fewshot={num_fewshot} wandb={wandb} tracing_enabled={tracing_enabled} resumable={resumable}")
    if capture_pre_logits:
        print(f"  capture_pre_logits=true hidden_capture_path={hidden_dir}")
    if cst_ck_s and cst_ck_s.lower() not in ("null", "none", ""):
        cst_tgb_p = float(cfg.get("cst_target_staging_gb", 1.0))
        cst_fne_p = _to_bool(cfg.get("cst_ntf_first_non_eos", False))
        print(
            f"  cst_checkpoint={cst_ck_s} cst_target_staging_gb={cst_tgb_p} "
            f"cst_ntf_first_non_eos={cst_fne_p} (buffer .pt, eval+delete, lm-eval flow unchanged)"
        )
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


def _normalize_parallel_slots_payload(data: Any) -> list[Any]:
    """YAML / JSON loader may return ``list`` or ``{"runs": [ ... ]}``."""
    if data is None:
        return []
    if isinstance(data, dict):
        inner = data.get("runs")
        if isinstance(inner, list):
            return inner
        raise ValueError("parallel runs file: expected a YAML list or a mapping with key 'runs' listing slot overrides.")
    if isinstance(data, list):
        return data
    raise ValueError("parallel runs: expected a list of per-slot override dicts (or {\"runs\": [...]}).")


def _slot_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    if OmegaConf.is_config(item):
        container = OmegaConf.to_container(item, resolve=True)
        if isinstance(container, dict):
            return container
    raise ValueError(f"parallel slot entries must be dicts mapping config keys for that GPU; got {type(item)!r}")


def _parallel_slot_specs_from_file(cfg: DictConfig) -> Optional[list[dict[str, Any]]]:
    """YAML list (or ``{\"runs\": [...]}``) merged per GPU; ``None`` if unset → single eval."""
    raw_file = cfg.get("parallel_runs_file")
    if raw_file is None:
        return None
    fs = str(raw_file).strip()
    if fs.lower() in ("", "none", "null"):
        return None
    path = Path(fs).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"parallel_runs_file not found: {path.resolve()}")
    with path.open(encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    slots_raw = _normalize_parallel_slots_payload(payload)
    out = [_slot_dict(item) for item in slots_raw]
    return out or None


def run(cfg: DictConfig) -> int:
    """Either one eval from config, or one eval per YAML slot (parallel_runs_file) with CUDA_VISIBLE_DEVICES per slot."""
    slot_specs = _parallel_slot_specs_from_file(cfg)
    if slot_specs:
        n = len(slot_specs)
        base_plain = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(base_plain, dict)
        base_plain.pop("parallel_runs_file", None)
        base_oc = OmegaConf.create(base_plain)

        def _run_slot(slot_idx: int, overrides: dict[str, Any]) -> int:
            merged = OmegaConf.merge(base_oc, OmegaConf.create(overrides))
            return _run_single_eval(merged, extra_env={"CUDA_VISIBLE_DEVICES": str(slot_idx)})

        if n == 1:
            return _run_slot(0, slot_specs[0])

        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
            futures = [pool.submit(_run_slot, i, spec) for i, spec in enumerate(slot_specs)]
            codes = [f.result() for f in futures]
        for c in codes:
            if c != 0:
                return c
        return 0

    return _run_single_eval(cfg)


def _normalize_hydra_cli(argv: list[str]) -> list[str]:
    """Put Hydra flags (``--multirun``, etc.) before ``key=value`` overrides; map ``hydra.launcher=X`` → ``launcher=X`` if valid."""
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
