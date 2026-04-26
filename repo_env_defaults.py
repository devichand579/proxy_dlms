"""Repo default credentials (mirrors ``.env.example``).

Real environment variables always win: we only ``setdefault`` so exports and
secrets override these fallbacks.
"""

from __future__ import annotations

import os

# Keep in sync with .env.example
HF_TOKEN_DEFAULT = "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"
WANDB_API_KEY_DEFAULT = (
    "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"
)
WANDB_PROJECT_DEFAULT = "strace"


def apply_repo_env_defaults() -> None:
    os.environ.setdefault("HF_TOKEN", HF_TOKEN_DEFAULT)
    os.environ.setdefault("WANDB_API_KEY", WANDB_API_KEY_DEFAULT)
    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT_DEFAULT)
