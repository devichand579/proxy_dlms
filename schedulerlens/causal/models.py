"""Load LLaDA / LLaDA-1.5 for predictor and controller roles."""

from __future__ import annotations

from pathlib import Path

import torch

from load_model import infer_diffusion_token_ids, load_model, load_tokenizer

ALLOWED_MODELS = ("llada", "llada1_5")
_REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_model_dir(name: str) -> Path:
    path = Path(name)
    if path.is_dir():
        return path.resolve()
    key = name.strip().lower().replace("-", "_")
    if key not in ALLOWED_MODELS:
        raise ValueError(
            f"causal experiments accept only {ALLOWED_MODELS}, got {name!r}"
        )
    local = _REPO_ROOT / "models" / key
    if not local.is_dir():
        raise FileNotFoundError(
            f"model dir not found: {local} (pass a checkpoint path or download into models/{key})"
        )
    return local.resolve()


def load_role(name: str, device: str, *, compile_model: bool = False):
    directory = resolve_model_dir(name)
    model = load_model(
        str(directory),
        device=device,
        compile_model=compile_model,
    )
    mask_id, eos_id, eot_id = infer_diffusion_token_ids(directory)
    return model, directory, int(mask_id), int(eos_id), eot_id


def role_pad_id(directory: Path) -> int:
    tokenizer = load_tokenizer(str(directory))
    if tokenizer.pad_token_id is None:
        raise ValueError(f"tokenizer has no pad token id: {directory}")
    return int(tokenizer.pad_token_id)


def role_devices(device_map: str) -> tuple[list[str], list[str]]:
    """Devices for the predictor and the controller.

    ``split`` gives each role half of the visible GPUs. An odd GPU goes to the
    predictor. ``share`` puts both roles on the first GPU, or on CPU.
    """
    if device_map not in ("split", "share"):
        raise ValueError(f"device_map must be split or share, got {device_map!r}")
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpu == 0:
        return ["cpu"], ["cpu"]
    if device_map == "share" or n_gpu == 1:
        return ["cuda:0"], ["cuda:0"]
    ctrl_n = n_gpu // 2
    pred_n = n_gpu - ctrl_n
    gpus = [f"cuda:{i}" for i in range(n_gpu)]
    return gpus[:pred_n], gpus[pred_n:]


def place_fleet(
    predictor: str,
    controller: str,
    *,
    device_map: str = "split",
    compile_model: bool = False,
):
    """Load one replica of each role on every device that role owns.

    The same checkpoint is loaded once per GPU and those replicas are shared
    by both roles. Different checkpoints split the GPUs evenly.
    """
    pred_devs, ctrl_devs = role_devices(device_map)
    same = resolve_model_dir(predictor) == resolve_model_dir(controller)
    devices = list(dict.fromkeys(pred_devs + ctrl_devs)) if same else pred_devs
    predictors = []
    pred_dir = None
    mask_id = eos_id = None
    eot_id = None
    for dev in devices:
        model, directory, mask_id, eos_id, eot_id = load_role(
            predictor, dev, compile_model=compile_model
        )
        pred_dir = directory
        predictors.append(model)
    if same:
        controllers = predictors
        ctrl_dir = pred_dir
        ctrl_devs = [str(next(m.parameters()).device) for m in controllers]
    else:
        controllers = []
        ctrl_dir = None
        for dev in ctrl_devs:
            model, directory, mask_ctrl, eos_ctrl, _eot_ctrl = load_role(
                controller, dev, compile_model=compile_model
            )
            if (mask_ctrl, eos_ctrl) != (mask_id, eos_id):
                raise ValueError(
                    f"token ids differ between {pred_dir.name} and {directory.name}; "
                    "causal audit requires a shared token space"
                )
            ctrl_dir = directory
            controllers.append(model)
    pad_id = role_pad_id(pred_dir)
    controller_pad_id = role_pad_id(ctrl_dir)
    if controller_pad_id != pad_id:
        raise ValueError(
            f"pad token ids differ between {pred_dir.name} and {ctrl_dir.name}: "
            f"{pad_id} != {controller_pad_id}"
        )
    return {
        "predictors": predictors,
        "controllers": controllers,
        "predictor": predictors[0],
        "controller": controllers[0],
        "predictor_dir": pred_dir,
        "controller_dir": ctrl_dir,
        "mask_id": mask_id,
        "eos_id": eos_id,
        "eot_id": eot_id,
        "pad_id": pad_id,
        "predictor_devices": [str(next(m.parameters()).device) for m in predictors],
        "controller_devices": [str(next(m.parameters()).device) for m in controllers],
    }
