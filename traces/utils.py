"""
Utility functions for tracing and repo-relative path helpers.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent

TRACE_OUTPUTS_DIRNAME = "trace_outputs"
OUTPUTS_DIRNAME = "outputs"
FETCH_EVALS_DIRNAME = "fetch_evals"


def repo_root() -> Path:
    return _REPO_ROOT


def trace_outputs_dir(*parts: str) -> Path:
    path = _REPO_ROOT / TRACE_OUTPUTS_DIRNAME
    for part in parts:
        path = path / part
    return path


def outputs_dir(*parts: str) -> Path:
    path = _REPO_ROOT / OUTPUTS_DIRNAME
    for part in parts:
        path = path / part
    return path


def fetch_evals_dir(*parts: str) -> Path:
    path = _REPO_ROOT / FETCH_EVALS_DIRNAME
    for part in parts:
        path = path / part
    return path


def resolve_repo_path(path: str | Path | None, *, base: Path | None = None) -> Path | None:
    if path is None:
        return None
    text = str(path).strip()
    if not text or text.lower() in ("null", "none"):
        return None
    resolved = Path(text).expanduser()
    if not resolved.is_absolute():
        resolved = (base or _REPO_ROOT) / resolved
    return resolved.resolve()


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=indent) + "\n")
    return target


def slugify(text: str, *, max_len: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(text).strip()).strip("-")
    return slug[:max_len] if max_len else slug


def trace_run_stem(model: str, task: str, scheduler: str, *, steps: int | None = None) -> str:
    parts = [slugify(model), slugify(task), slugify(scheduler)]
    if steps is not None:
        parts.append(f"s{int(steps)}")
    return "_".join(p for p in parts if p)


_TRACE_STEPS_BUDGET_RE = re.compile(r"_s(?P<steps>\d+)$", re.IGNORECASE)
# Scheduler tokens as path segments (e.g. ``…_fdm_s128``, ``…_fhs_s512``).
_TRACE_FDM_RE = re.compile(r"(^|[_-])fdm([_-]|$)", re.IGNORECASE)
_TRACE_FHS_RE = re.compile(r"(^|[_-])fhs([_-]|$)", re.IGNORECASE)


def parse_trace_step_budget(name: str | Path | None) -> Optional[int]:
    """Parse step budget ``N`` from a trace stem/path ending in ``_s{N}``.

    Examples: ``llada_gsm8k_pd_s512.jsonl`` → 512,
    ``llada-100m-pcfg-pcfg_dus_s128.npz`` → 128.
    """
    if name is None:
        return None
    stem = Path(str(name)).stem
    m = _TRACE_STEPS_BUDGET_RE.search(stem)
    return int(m.group("steps")) if m else None


def is_fdm_trace_name(name: str | Path | None) -> bool:
    """True when the stem encodes the FDM scheduler (e.g. ``…_fdm_s128``)."""
    if name is None:
        return False
    return bool(_TRACE_FDM_RE.search(Path(str(name)).stem))


def is_fhs_trace_name(name: str | Path | None) -> bool:
    """True when the stem encodes the FHS scheduler (e.g. ``…_fhs_s512``)."""
    if name is None:
        return False
    return bool(_TRACE_FHS_RE.search(Path(str(name)).stem))


def squeeze_to_1d(data) -> np.ndarray:
    """Peel leading dims until 1D: (1, G) / nested list → length-G vector."""
    arr = np.asarray(data)
    while arr.ndim > 1:
        arr = arr[0]
    return np.asarray(arr).reshape(-1)


def squeeze_len(data) -> Optional[int]:
    if data is None:
        return None
    return int(squeeze_to_1d(data).shape[0])


def as_gen_vec(data, *, gen_length: int, dtype=np.bool_) -> Optional[np.ndarray]:
    """Squeeze a step field to length ``gen_length``, or None if missing/wrong length."""
    if data is None or gen_length <= 0:
        return None
    arr = squeeze_to_1d(data)
    if arr.shape[0] != gen_length:
        return None
    return arr.astype(dtype, copy=False)


def _is_tensor_like(obj):
    """Check if object is tensor-like without using isinstance to avoid recursion."""
    try:
        obj_type_name = type(obj).__name__
        obj_module = type(obj).__module__
        return obj_type_name == 'Tensor' and obj_module == 'torch'
    except Exception:
        return False



def extract_tensor_value(value, device=None, dtype=torch.long):
    """Extract scalar value and create 1D tensor from various input types."""
    if _is_tensor_like(value):
        if value.dim() == 0:
            val_item = value.item()
            val_tensor = value.unsqueeze(0) if device is None else torch.tensor([val_item], device=device, dtype=dtype)
        elif value.numel() == 1:
            val_item = value[0].item()
            val_tensor = value if device is None else value.to(device)
        elif value.numel() > 0:
            val_item = value[0].item()
            val_tensor = value[0:1]
            if device is not None:
                val_tensor = val_tensor.to(device)
        else:
            return None, None
    elif (isinstance(value, (int, float)) or (hasattr(type(value), '__module__') and type(value).__module__ == 'numpy')):
        val_item = int(value)
        val_tensor = torch.tensor([val_item], device=device, dtype=dtype) if device is not None else torch.tensor([val_item], dtype=dtype)
    else:
        return None, None
    return val_item, val_tensor

def ensure_2d_shape(tensor, batch_size=1):
    """Ensure tensor has [batch_size, L] shape."""
    if tensor.dim() == 1:
        return tensor.unsqueeze(0)
    elif tensor.dim() == 0:
        return tensor.unsqueeze(0).unsqueeze(0)
    return tensor

def get_value_item(value, mask_id):
    """Extract integer value from various types for comparison."""
    try:
        if _is_tensor_like(value):
            if value.dim() == 0:
                return value.item()
            elif value.numel() == 1:
                return value[0].item()
            elif value.numel() > 0:
                return value[0].item()
            return mask_id
        elif (isinstance(value, (int, float)) or (hasattr(type(value), '__module__') and type(value).__module__ == 'numpy')):
            return int(value)
        return mask_id
    except Exception:
        return mask_id

def is_negative_inf(value):
    """Check if value is negative infinity."""
    try:
        if isinstance(value, (int, float)):
            return (value == float('-inf')) or (value == np.NINF) or (value < -1e10)
        elif type(value).__name__ == 'ndarray':
            return np.all(np.isinf(value)) and np.all(value < 0) if value.size > 0 else False
        elif _is_tensor_like(value):
            return torch.all(value == -torch.inf) if value.numel() > 0 else False
        elif hasattr(value, '__float__'):
            val_float = float(value)
            return (val_float == float('-inf')) or (val_float < -1e10)
    except Exception:
        try:
            if isinstance(value, (int, float)):
                return value < -1e10
        except Exception:
            pass
    return False

def shapes_match(tensors):
    """Check if all tensors have matching [1, L] shape."""
    if not tensors:
        return False
    shape = tensors[0].shape
    return (all(t.shape == shape for t in tensors) and 
            len(shape) == 2 and shape[0] == 1)


def same_2d_batch_shape(tensors):
    """Check if all tensors share the same [B, L] shape (any batch size)."""
    if not tensors:
        return False
    shape = tensors[0].shape
    return len(shape) == 2 and all(t.shape == shape for t in tensors)

def was_position_masked(x, pos, mask_id):
    """Check if a single position was masked."""
    try:
        if x.dim() == 2:
            return (x[0, pos] == mask_id).item()
        return (x[pos] == mask_id).item()
    except:
        return False

def was_positions_masked(x, positions, mask_id):
    """Check which positions were masked."""
    if x.dim() == 2:
        return x[0, positions] == mask_id
    return x[positions] == mask_id

def value_matches_x0(x0_trace, pos, val_item):
    """Check if value matches x0 prediction at position."""
    if x0_trace.shape[1] <= pos:
        return False
    x0_pred = x0_trace[0, pos].item()
    return (abs(x0_pred - val_item) < 1e-6 or 
            (isinstance(val_item, int) and int(x0_pred) == val_item))

def get_x0_trace(tracer_instance, x, batch_index=None):
    """Get x0_trace, creating empty one if _last_x0 is None.
    When batch_index is not None and _last_x0 has batch dim > 1, returns that row only [1, L]."""
    if tracer_instance._last_x0 is not None:
        out = ensure_2d_shape(tracer_instance._last_x0.clone())
        if batch_index is not None and out.shape[0] > 1 and batch_index < out.shape[0]:
            out = out[batch_index : batch_index + 1]
        return out
    seq_len = x.shape[1] if x.dim() == 2 else x.shape[0]
    batch_dim = x.shape[0] if x.dim() >= 2 and x.shape[0] > 0 else 1
    return torch.zeros((batch_dim, seq_len), dtype=torch.long, device=x.device)

def build_transfer_index(transfer_positions, seq_len, device):
    """Build boolean transfer_index from integer positions."""
    transfer_index = torch.zeros((1, seq_len), dtype=torch.bool, device=device)
    if transfer_positions.dim() == 0:
        transfer_index[0, transfer_positions.item()] = True
    else:
        transfer_index[0, transfer_positions] = True
    return transfer_index

def fill_x0_trace_zeros(x0_for_trace, transfer_positions, transfer_values):
    """Fill zeros in x0_for_trace with values from transfer_values."""
    if transfer_positions.dim() == 0:
        pos = transfer_positions.item()
        if x0_for_trace[0, pos] == 0:
            val_item, _ = extract_tensor_value(transfer_values)
            if val_item is not None:
                x0_for_trace[0, pos] = int(val_item)
    else:
        for i, pos in enumerate(transfer_positions):
            if x0_for_trace[0, pos] == 0:
                if transfer_values.dim() == 0:
                    val_item, _ = extract_tensor_value(transfer_values)
                elif transfer_values.numel() > i:
                    val_item, _ = extract_tensor_value(
                        transfer_values[i] if transfer_values.dim() > 0 else transfer_values
                    )
                elif transfer_values.numel() > 0:
                    val_item, _ = extract_tensor_value(
                        transfer_values[0] if transfer_values.dim() > 0 else transfer_values
                    )
                else:
                    val_item, _ = extract_tensor_value(transfer_values)
                
                if val_item is not None:
                    x0_for_trace[0, pos] = int(val_item)

def build_x0_for_trace(x0_trace, transfer_positions, transfer_values):
    """Build x0_for_trace from x0_trace and actual transfer values."""
    x0_for_trace = x0_trace.clone()
    if transfer_values is None:
        return x0_for_trace
    
    if transfer_positions.dim() == 0:
        pos = transfer_positions.item()
        val_item, _ = extract_tensor_value(transfer_values)
        if val_item is not None:
            x0_for_trace[0, pos] = int(val_item)
    else:
        if transfer_values.numel() == transfer_positions.numel():
            if transfer_values.dim() == 0:
                val_item, _ = extract_tensor_value(transfer_values)
                if val_item is not None:
                    for pos in transfer_positions:
                        x0_for_trace[0, pos] = int(val_item)
            else:
                x0_for_trace[0, transfer_positions] = transfer_values
    
    # Fill any remaining zeros with actual values
    fill_x0_trace_zeros(x0_for_trace, transfer_positions, transfer_values)
    return x0_for_trace