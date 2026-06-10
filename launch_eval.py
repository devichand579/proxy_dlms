"""Run `eval.py` via Hydra (local or SLURM). Config `config/lmeval.yaml`; details in `docs/evaluation.md`."""

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

from repo_env_defaults import apply_offline_env, apply_repo_env_defaults, resolve_eval_device
from sampling_defaults import resolve_gen_length_steps

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


def _intervention_suffix(intervention_mode, intervention_steps, intervention_k: int) -> str:
    steps_raw = intervention_steps if intervention_steps is not None else "0"
    if isinstance(steps_raw, (list, tuple)):
        steps_safe = "_".join(str(s) for s in steps_raw)
    else:
        steps_safe = re.sub(r"[,_]", "_", str(steps_raw))
    return f"_intv_{intervention_mode}_steps{steps_safe}_k{intervention_k}"


def _eval_output_basename(
    model: str,
    task_name: str,
    mode: str,
    strategy,
    *,
    steps: int,
    intervention_mode=None,
    intervention_steps=None,
    intervention_k: int = 1,
    include_steps: bool = True,
) -> str:
    """Auto run/trace basename: ``{model}_{task}_{mode}[_s{steps}]`` (+ intervention tags when set)."""
    base = _resumable_output_basename(model, task_name, mode, strategy)
    if intervention_mode:
        base = f"{base}{_intervention_suffix(intervention_mode, intervention_steps, intervention_k)}"
    if include_steps:
        base = f"{base}_s{int(steps)}"
    return base


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




def _apply_trace_hf_env(env: dict[str, str], cfg: dict, *, offline: bool = False) -> None:
    """Env vars read by ``traces.hf_trace_upload`` when eval closes a trace file."""
    tracing = _to_bool(cfg.get("tracing_enabled", True))
    upload = cfg.get("trace_hf_upload")
    if offline:
        upload = False
    elif upload is None:
        upload = tracing
    else:
        upload = _to_bool(upload)
    env["TRACE_HF_UPLOAD"] = "TRUE" if upload else "FALSE"
    repo = cfg.get("trace_hf_dataset_repo")
    if repo is not None and str(repo).strip():
        env["TRACE_HF_DATASET_REPO"] = str(repo).strip()
    ns = cfg.get("trace_hf_namespace")
    if ns is not None and str(ns).strip():
        env["TRACE_HF_NAMESPACE"] = str(ns).strip()
    if cfg.get("trace_hf_dataset_private") is not None:
        env["TRACE_HF_DATASET_PRIVATE"] = "TRUE" if _to_bool(cfg.get("trace_hf_dataset_private")) else "FALSE"
    rev = cfg.get("trace_hf_revision")
    if rev is not None and str(rev).strip():
        env["TRACE_HF_REVISION"] = str(rev).strip()


def _to_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes")
    return bool(val)


def _apply_device_env(env: dict[str, str], cfg: DictConfig, extra_env: Optional[dict[str, str]]) -> str:
    """Pin GPU via CUDA_VISIBLE_DEVICES when requested; return harness model_args device."""
    device_cfg = cfg.get("device")
    model_device, cuda_visible = resolve_eval_device(
        None if device_cfg is None else str(device_cfg)
    )
    if extra_env and "CUDA_VISIBLE_DEVICES" in extra_env and cuda_visible is None:
        cuda_visible = extra_env["CUDA_VISIBLE_DEVICES"]
    if cuda_visible is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    return model_device


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
    offline = _to_bool(cfg.get("offline", False))
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

    gen_length, steps = resolve_gen_length_steps(cfg, task_name)
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

    run_id_explicit = bool(resumable and run_id)
    if run_id_explicit:
        output_suffix = str(run_id)
    elif resumable:
        output_suffix = _eval_output_basename(
            model,
            task_name,
            mode,
            strategy,
            steps=steps,
            intervention_mode=intervention_mode,
            intervention_steps=intervention_steps,
            intervention_k=intervention_k,
            include_steps=True,
        )
    else:
        base = _eval_output_basename(
            model,
            task_name,
            mode,
            strategy,
            steps=steps,
            intervention_mode=intervention_mode,
            intervention_steps=intervention_steps,
            intervention_k=intervention_k,
            include_steps=True,
        )
        output_suffix = f"{base}_{timestamp}"
    wandb_name = output_suffix

    checkpoint_dir = f"outputs/{output_suffix}" if resumable else None

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    model_device = _apply_device_env(env, cfg, extra_env)

    model_args = (
        f"model_path={model_path},gen_length={gen_length},steps={steps},"
        f"block_length={block_length},mode={mode},strategy={str(strategy)},top_k={top_k},compile_model={str(compile_model).lower()},"
        f"task_name={task_name},trace_name={output_suffix},use_eos_harness={str(use_eos_harness).lower()},"
        f"device={model_device}"
    )
    if resumable and checkpoint_dir:
        model_args += f",resumable=true,output_dir={checkpoint_dir}"
    if intervention_mode:
        if isinstance(intervention_steps, (list, tuple)):
            steps_arg = "_".join(str(s) for s in intervention_steps)
        else:
            steps_arg = re.sub(r",", "_", str(intervention_steps or "0"))
        model_args += f",intervention_mode={intervention_mode},intervention_steps={steps_arg},intervention_k={intervention_k}"

    env.setdefault("HF_ALLOW_CODE_EVAL", "1")
    env.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "true")
    env.setdefault("LMEVAL_LOG_LEVEL", "INFO")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env["TRACING_ENABLED"] = "TRUE" if tracing_enabled else "FALSE"
    if offline:
        apply_offline_env(env)
    _apply_trace_hf_env(env, cfg, offline=offline)
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
    print(
        f"  num_fewshot={num_fewshot} wandb={wandb} offline={offline} device={model_device} "
        f"cuda_visible={env.get('CUDA_VISIBLE_DEVICES', '(unchanged)')} "
        f"tracing_enabled={tracing_enabled} "
        f"trace_hf_upload={env.get('TRACE_HF_UPLOAD')} trace_hf_repo={env.get('TRACE_HF_DATASET_REPO', 'schedulerlens')} "
        f"resumable={resumable}"
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


def count_parallel_slots(path: str | Path) -> int:
    """Number of GPU slots in a parallel runs YAML file."""
    specs = load_parallel_slot_specs(path)
    return len(specs)


def load_parallel_slot_specs(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"parallel runs file not found: {path.resolve()}")
    with path.open(encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    slots_raw = _normalize_parallel_slots_payload(payload)
    return [_slot_dict(item) for item in slots_raw]


def _parallel_slot_specs_from_file(cfg: DictConfig) -> Optional[list[dict[str, Any]]]:
    """YAML list (or ``{\"runs\": [...]}``) merged per GPU; ``None`` if unset → single eval."""
    raw_file = cfg.get("parallel_runs_file")
    if raw_file is None:
        return None
    fs = str(raw_file).strip()
    if fs.lower() in ("", "none", "null"):
        return None
    return load_parallel_slot_specs(fs) or None


def _count_visible_gpus() -> int:
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _slurm_gpus_allocated() -> Optional[int]:
    for key in ("SLURM_GPUS_ON_NODE", "SLURM_GPUS"):
        raw = os.environ.get(key)
        if raw and str(raw).strip().isdigit():
            return int(str(raw).strip())
    return None


def _is_slurm_launcher(cfg: DictConfig) -> bool:
    launcher = str(OmegaConf.select(cfg, "launcher") or "local").strip().lower()
    if launcher != "local" and launcher.startswith("slurm"):
        return True
    return OmegaConf.select(cfg, "hydra.launcher.partition") is not None


def _hydra_multirun_active() -> bool:
    return "--multirun" in sys.argv or "-m" in sys.argv


def _cfg_cli_overrides(cfg: DictConfig) -> list[str]:
    """Serialize resolved eval config back to Hydra CLI overrides for child re-invocation."""
    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict):
        return ["launcher=local"]

    overrides: list[str] = []
    for key, val in container.items():
        if key in ("hydra", "launcher"):
            continue
        if val is None:
            continue
        if key == "parallel_runs_file":
            overrides.append(f"parallel_runs_file={Path(str(val)).expanduser().resolve()}")
            continue
        if isinstance(val, bool):
            overrides.append(f"{key}={'true' if val else 'false'}")
        elif isinstance(val, (int, float, str)):
            overrides.append(f"{key}={val}")
        elif key == "intervention" and isinstance(val, dict):
            for sub_key, sub_val in val.items():
                if sub_val is None:
                    continue
                if isinstance(sub_val, bool):
                    overrides.append(f"intervention.{sub_key}={'true' if sub_val else 'false'}")
                elif isinstance(sub_val, (int, float, str)):
                    overrides.append(f"intervention.{sub_key}={sub_val}")
                elif isinstance(sub_val, list):
                    overrides.append(f"intervention.{sub_key}=[{','.join(str(x) for x in sub_val)}]")
    overrides.append("launcher=local")
    return overrides


def _parallel_slurm_worker(repo_root: str, overrides: list[str], setup_cmds: list[str]) -> int:
    root = Path(repo_root)
    for cmd in setup_cmds:
        subprocess.run(cmd, shell=True, executable="/bin/bash", cwd=root, check=True)
    env = os.environ.copy()
    env["STRACE_PARALLEL_SLURM_CHILD"] = "1"
    cmd = [sys.executable, str(root / "launch_eval.py"), *overrides]
    print("Running parallel eval child:", " ".join(cmd), flush=True)
    return int(subprocess.run(cmd, env=env, cwd=root).returncode)


def _submit_parallel_slurm_job(cfg: DictConfig, slot_count: int) -> int:
    import submitit

    hlaunch = cfg.hydra.launcher
    folder = str(hlaunch.get("submitit_folder", "job_outputs/%j"))
    executor = submitit.AutoExecutor(folder=folder)

    params: dict[str, Any] = {
        "slurm_partition": hlaunch.get("partition"),
        "timeout_min": hlaunch.get("timeout_min", 4320),
        "mem_gb": hlaunch.get("mem_gb", 64),
        "cpus_per_worker": hlaunch.get("cpus_per_task", 8),
        "nodes": hlaunch.get("nodes", 1),
        "name": "strace_parallel_eval",
        "gres": f"gpu:{slot_count}",
    }
    qos = hlaunch.get("qos")
    if qos is not None and str(qos).strip().lower() not in ("", "null", "none"):
        params["slurm_qos"] = qos
    addl = OmegaConf.to_container(hlaunch.get("additional_parameters") or {}, resolve=True)
    if isinstance(addl, dict) and addl:
        params["slurm_additional_parameters"] = addl

    executor.update_parameters(**{k: v for k, v in params.items() if v is not None})

    repo_root = Path(__file__).resolve().parent
    overrides = _cfg_cli_overrides(cfg)
    setup_cmds = [str(x) for x in (OmegaConf.select(cfg, "hydra.launcher.setup") or [])]

    job = executor.submit(_parallel_slurm_worker, str(repo_root), overrides, setup_cmds)
    print(
        f"Submitted parallel SLURM job {job.job_id} "
        f"({slot_count} GPU(s), partition={params.get('slurm_partition')})"
    )
    return 0


def _should_submit_parallel_slurm(cfg: DictConfig, slot_count: int) -> bool:
    if os.environ.get("STRACE_PARALLEL_SLURM_CHILD") == "1":
        return False
    if _hydra_multirun_active():
        return False
    if slot_count < 1 or not _is_slurm_launcher(cfg):
        return False
    allocated = _slurm_gpus_allocated()
    visible = _count_visible_gpus()
    if os.environ.get("SLURM_JOB_ID"):
        have = allocated if allocated is not None else visible
        if have >= slot_count:
            return False
    return True


def _run_parallel_evals(cfg: DictConfig, slot_specs: list[dict[str, Any]]) -> int:
    n = len(slot_specs)
    if n == 0:
        raise ValueError("parallel_runs_file is empty")
    if os.environ.get("SLURM_JOB_ID"):
        allocated = _slurm_gpus_allocated()
        visible = _count_visible_gpus()
        have = allocated if allocated is not None else visible
        if have and have < n:
            raise RuntimeError(
                f"parallel_runs_file defines {n} GPU slots but this SLURM allocation has {have} GPU(s). "
                f"Request --gres=gpu:{n} (or use launcher=slurm_dgx from the login node to auto-submit)."
            )

    base_plain = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(base_plain, dict)
    base_plain.pop("parallel_runs_file", None)
    base_oc = OmegaConf.create(base_plain)

    def _run_slot(slot_idx: int, overrides: dict[str, Any]) -> int:
        merged = OmegaConf.merge(base_oc, OmegaConf.create(overrides))
        slot_env: dict[str, str] = {}
        if merged.get("device") is None:
            slot_env["CUDA_VISIBLE_DEVICES"] = str(slot_idx)
        return _run_single_eval(merged, extra_env=slot_env or None)

    if n == 1:
        return _run_slot(0, slot_specs[0])

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(_run_slot, i, spec) for i, spec in enumerate(slot_specs)]
        codes = [f.result() for f in futures]
    for c in codes:
        if c != 0:
            return c
    return 0


def run(cfg: DictConfig) -> int:
    """Single eval, parallel multi-GPU eval, or auto-submit one multi-GPU SLURM job."""
    slot_specs = _parallel_slot_specs_from_file(cfg)
    if slot_specs:
        if _should_submit_parallel_slurm(cfg, len(slot_specs)):
            return _submit_parallel_slurm_job(cfg, len(slot_specs))
        return _run_parallel_evals(cfg, slot_specs)

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


@hydra.main(version_base="1.2", config_path=str(_CONFIG_DIR), config_name="lmeval")
def main(cfg: DictConfig) -> None:
    sys.exit(run(cfg))


if __name__ == "__main__":
    sys.argv = _normalize_hydra_cli(sys.argv)
    main()
