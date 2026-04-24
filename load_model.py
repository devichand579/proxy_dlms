"""
Load diffusion LMs (LLaDA, LLaDA 1.5, Dream, …) from local checkpoints via HuggingFace Auto classes.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

BASE_DIR = Path(__file__).parent.resolve()

# Default token ids when a caller has no local ``config.json`` and no per-call overrides.
# Matches standard LLaDA / LLaDA 1.5 checkpoints in this repo; schedulers use these as
# fallbacks. Prefer :func:`infer_diffusion_token_ids` from the selected ``model_path``.
LLADA_DEFAULT_MASK_ID = 126336
LLADA_DEFAULT_EOS_ID = 126081
LLADA_DEFAULT_EOT_ID: Optional[int] = 126348


def _read_config(model_dir: Path) -> Dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found at {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_diffusion_token_ids(model_dir: Union[str, Path]) -> Tuple[int, int, Optional[int]]:
    """
    Read mask / EOS / optional EoT ids from a checkpoint's ``config.json``.

    * ``model_type == "llada"`` (LLaDA and LLaDA 1.5): EoT is ``LLADA_DEFAULT_EOT_ID``.
    * Other model types (e.g. Dream): EoT is None — schedulers only suppress EoS/EoT in logits when
      ``eot_id`` is not None.
    """
    model_dir = Path(model_dir).resolve()
    cfg = _read_config(model_dir)
    model_type = str(cfg.get("model_type", "")).lower()
    mask_id = int(cfg["mask_token_id"])
    eos_id = int(cfg["eos_token_id"])
    if model_type == "llada":
        eot_id: Optional[int] = LLADA_DEFAULT_EOT_ID
    else:
        eot_id = None
    return mask_id, eos_id, eot_id


class DreamSchedulersCompatWrapper(nn.Module):
    """
    When Dream is run through this repo's LLaDA-style schedulers, ``forward`` must
    follow ``models.dream.generation_utils.DreamGenerationMixin._sample``: RoPE
    ``position_ids`` (``tok_idx``) derived from the padding mask, and a one-step logits
    alignment ``torch.cat([logits[:, :1], logits[:, :-1]], dim=1)`` for the MaskGIT
    head. Native ``model.generate`` still uses the inner :class:`~DreamModel` and is
    unchanged.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self._inner = model

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        u = dict(kwargs)
        if len(args) >= 1 and "input_ids" not in u:
            u["input_ids"] = args[0]
        if len(args) >= 2 and "attention_mask" not in u:
            u["attention_mask"] = args[1]
        if len(args) >= 3 and "position_ids" not in u:
            u["position_ids"] = args[2]

        position_ids = u.get("position_ids")
        attention_mask = u.get("attention_mask")
        if position_ids is None and torch.is_tensor(attention_mask) and attention_mask.dim() == 2:
            f = attention_mask
            f = f.float() if f.dtype == torch.bool else f.to(dtype=torch.float32, device=attention_mask.device)
            pos = f.long().cumsum(-1) - 1
            pos.masked_fill_(f == 0, 1)
            u["position_ids"] = pos

        out = self._inner(**u)
        if hasattr(out, "logits") and out.logits is not None:
            lg = out.logits
            out.logits = torch.cat([lg[:, :1], lg[:, :-1]], dim=1)
        return out

    def __getattr__(self, name: str) -> Any:
        if name == "_inner":
            return super().__getattr__(name)
        return getattr(self._inner, name)


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

    if str(getattr(model.config, "model_type", "")).lower() == "dream":
        model = DreamSchedulersCompatWrapper(model)

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
