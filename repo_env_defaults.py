"""Repo default credentials (mirrors ".env.example").
"""

from __future__ import annotations

import os

# Keep in sync with .env.example.
HF_TOKEN_DEFAULT = "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" # NOTE: DON'T COMMIT THIS TOKEN
WANDB_API_KEY_DEFAULT = (
    "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"
) # NOTE: DON'T COMMIT THIS TOKEN
WANDB_PROJECT_DEFAULT = "strace"


def apply_repo_env_defaults() -> None:
    os.environ.setdefault("HF_TOKEN", HF_TOKEN_DEFAULT)
    os.environ.setdefault("WANDB_API_KEY", WANDB_API_KEY_DEFAULT)
    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT_DEFAULT)


def is_offline_env(target: dict[str, str] | None = None) -> bool:
    env = target if target is not None else os.environ
    return env.get("HF_HUB_OFFLINE") == "1"


def apply_offline_env(target: dict[str, str] | None = None) -> None:
    """Use cached HF datasets/models and local W&B (no Hub/network for eval telemetry)."""
    env = target if target is not None else os.environ
    env["HF_DATASETS_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["WANDB_MODE"] = "offline"
    env["TRACE_HF_UPLOAD"] = "FALSE"


def resolve_eval_device(device: str | None = None) -> tuple[str, str | None]:
    """Map CLI/YAML device to harness ``device`` and optional ``CUDA_VISIBLE_DEVICES``.
    """
    if device is None or not str(device).strip():
        return "cuda", None

    s = str(device).strip()
    lower = s.lower()

    if lower == "cpu":
        return "cpu", ""

    if s.isdigit():
        return "cuda", s

    if lower == "cuda":
        return "cuda", None

    if lower.startswith("cuda:"):
        idx = s.split(":", 1)[1].strip()
        if idx.isdigit():
            return "cuda", idx
        return s, None

    return s, None
