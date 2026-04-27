"""Load diffusion LMs (LLaDA family, Dream) with AutoModel/AutoTokenizer from local dirs."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

BASE_DIR = Path(__file__).parent.resolve()

# Defaults if no per-checkpoint read; prefer infer_diffusion_token_ids(model_path).
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
    """Return (mask_token_id, eos_token_id, eot_id). LLaDA uses fixed EoT; Dream has eot_id=None."""
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


def infer_b44_eos_unmask_safety(model_dir: Union[str, Path]) -> bool:
    """B.4 EOS/EoT suppression in schedulers: on for `.../llada1_5` or `strace_b44_eos_unmask` in config."""
    model_dir = Path(model_dir).resolve()
    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        return False
    cfg = _read_config(model_dir)
    if "strace_b44_eos_unmask" in cfg:
        v = cfg["strace_b44_eos_unmask"]
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes", "on")
        return bool(v)
    return model_dir.name == "llada1_5"  # e.g. ./models/llada1_5


class DreamSchedulersCompatWrapper(nn.Module):
    """Dream forward for schedulers: pad-derived position_ids + logits align like Dream's _sample."""

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
        # Unify across PyTorch / SDPA / Flash backends: never pass 2D int/bool into Dream — behavior differs
        # by version. Pre-build the same (B,1,1,S) float additive padding mask as DreamSdpaAttention, in
        # model activations dtype, so SDPA always sees a single contract (see models/dream/modeling_dream.py).
        if torch.is_tensor(attention_mask) and attention_mask.dim() == 2:
            if attention_mask.dtype.is_floating_point:
                keep = attention_mask > 0.5
            elif attention_mask.dtype == torch.bool:
                keep = attention_mask
            else:
                keep = attention_mask != 0

            if position_ids is None:
                pos = keep.to(dtype=torch.long, device=keep.device).cumsum(-1) - 1
                pos.masked_fill_(~keep, 0)
                u["position_ids"] = pos

            w_dtype = next(self._inner.parameters()).dtype
            if w_dtype not in (torch.float16, torch.bfloat16, torch.float32):
                w_dtype = torch.float32
            min_f = torch.finfo(w_dtype).min
            u["attention_mask"] = (~keep).to(w_dtype)[:, None, None, :].contiguous() * min_f
        # 4D mask or None: pass through; DreamSdpaAttention uses 4D as attn_mask directly.

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
    """Load weights from a local dir (trust_remote_code). Dream may be wrapped for scheduler forward."""
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
    """Tokenizer from the same directory as the model."""
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
