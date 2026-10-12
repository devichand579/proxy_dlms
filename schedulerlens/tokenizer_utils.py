"""Resolve model directory and tokenizer from a trace path."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Set, Tuple, Union

from load_model import BASE_DIR, infer_diffusion_token_ids, load_tokenizer

logger = logging.getLogger(__name__)

KNOWN_MODELS: Sequence[str] = ("llada1_5", "llada", "dream", "pcfg")


def infer_model_name(trace_path: Union[str, Path]) -> Optional[str]:
    """Infer model dirname (``llada``, ``llada1_5``, ``dream``) from trace filename stem."""
    stem = Path(trace_path).stem
    for name in KNOWN_MODELS:
        if stem == name or stem.startswith(f"{name}_"):
            return name
    return None


def resolve_model_dir(
    trace_path: Union[str, Path],
    *,
    model_path: Optional[Union[str, Path]] = None,
    models_root: Optional[Union[str, Path]] = None,
) -> Path:
    """Return local model directory for tokenizer decode."""
    if model_path is not None:
        path = Path(model_path).resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Model directory not found: {path}")
        return path

    root = Path(models_root).resolve() if models_root else BASE_DIR / "models"
    name = infer_model_name(trace_path)
    if name is None:
        raise ValueError(
            f"Cannot infer model from trace filename {Path(trace_path).name!r}. "
            f"Expected stem to start with one of {list(KNOWN_MODELS)} or pass --model-path."
        )
    path = root / name
    if not path.is_dir():
        raise FileNotFoundError(f"Inferred model directory not found: {path}")
    return path


def load_trace_tokenizer(
    trace_path: Union[str, Path],
    *,
    model_path: Optional[Union[str, Path]] = None,
    models_root: Optional[Union[str, Path]] = None,
):
    """Load tokenizer matching the model implied by the trace file."""
    model_dir = resolve_model_dir(
        trace_path, model_path=model_path, models_root=models_root
    )
    logger.info("Using tokenizer from %s", model_dir)
    return load_tokenizer(str(model_dir)), model_dir


def diffusion_skip_token_ids(model_dir: Union[str, Path]) -> Set[int]:
    """Token ids to skip in semantic analysis (mask / eos / eot)."""
    mask_id, eos_id, eot_id = infer_diffusion_token_ids(model_dir)
    skip = {int(mask_id), int(eos_id)}
    if eot_id is not None:
        skip.add(int(eot_id))
    return skip
