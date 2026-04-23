"""
Load diffusion LMs (LLaDA, LLaDA 1.5, Dream, …) from local checkpoints via HuggingFace Auto classes.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from transformers import AutoModel, AutoTokenizer

BASE_DIR = Path(__file__).parent.resolve()


def _read_config(model_dir: Path) -> Dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found at {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_diffusion_token_ids(model_dir: Union[str, Path]) -> Tuple[int, int, Optional[int]]:
    """
    Read mask / EOS / optional EoT ids from config. LLaDA models use a fixed EoT id (126348); Dream has no EoT.
    """
    model_dir = Path(model_dir).resolve()
    cfg = _read_config(model_dir)
    model_type = str(cfg.get("model_type", "")).lower()
    mask_id = int(cfg["mask_token_id"])
    eos_id = int(cfg["eos_token_id"])
    if model_type == "llada":
        eot_id: Optional[int] = 126348
    else:
        eot_id = None
    return mask_id, eos_id, eot_id


def load_model(
    model_dir: str,
    torch_dtype=torch.bfloat16,
    device: str = "cuda",
    compile_model: bool = True,
    capture_pre_logits: bool = False,
):
    """
    Load model weights from a local directory (config + safetensors). Uses trust_remote_code for LLaDA / Dream modeling files shipped next to config.
    """
    model_dir = Path(model_dir).resolve()
    _read_config(model_dir)

    model = AutoModel.from_pretrained(
        str(model_dir),
        torch_dtype=torch_dtype,
        local_files_only=True,
        trust_remote_code=True,
    )

    model = model.to(device)
    model.eval()

    if capture_pre_logits:
        from cst import PreLogitsCaptureWrapper

        mid, _, _ = infer_diffusion_token_ids(model_dir)
        model = PreLogitsCaptureWrapper(model, mask_id=mid)

    if compile_model:
        try:
            model = torch.compile(model, mode="max-autotune")
        except Exception as e:
            print(f"Warning: Could not compile model: {e}")
            print("Continuing without compilation...")

    return model


def load_tokenizer(model_dir: str):
    """Load tokenizer from local files (fast JSON, Dream BPE, etc.)."""
    model_dir = Path(model_dir).resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    return AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=True,
    )


def load_model_and_tokenizer(
    model_dir: str,
    torch_dtype=torch.bfloat16,
    device: str = "cuda",
    compile_model: bool = True,
    capture_pre_logits: bool = False,
):
    model = load_model(
        model_dir,
        torch_dtype=torch_dtype,
        device=device,
        compile_model=compile_model,
        capture_pre_logits=capture_pre_logits,
    )
    tokenizer = load_tokenizer(model_dir)
    return model, tokenizer


if __name__ == "__main__":
    example_dir = BASE_DIR / "models" / "dream"
    print(f"Loading model from {example_dir}...")
    model, tokenizer = load_model_and_tokenizer(str(example_dir), compile_model=True)
    print("Model and tokenizer loaded successfully!")
    print(f"Model type: {type(model)}")
    print(f"Tokenizer type: {type(tokenizer)}")
    print(f"Inferred token ids: {infer_diffusion_token_ids(example_dir)}")
