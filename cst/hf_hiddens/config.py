"""Resolve Hub repo ids and tokens from login / env, with optional config overrides."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Optional

from cst.hf_hiddens.naming import eval_hf_repo_id, train_hf_repo_id

logger = logging.getLogger("cst.hf_hiddens.config")


def _blank(value: Any) -> bool:
    return value is None or str(value).strip().lower() in ("", "null", "none")


def _explicit_repo(value: Any) -> Optional[str]:
    if _blank(value):
        return None
    return str(value).strip()


def hf_token(section: Any = None) -> bool | str | None:
    """Use ``huggingface-cli login`` / ``HF_TOKEN`` unless a config token is set."""
    t = None if section is None else section.get("hf_token")
    if _blank(t):
        env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        return env.strip() if env and env.strip() else True
    return t


@lru_cache(maxsize=4)
def _whoami_namespace(token_key: str) -> str:
    from huggingface_hub import HfApi

    token: bool | str | None = True
    if token_key.startswith("str:"):
        token = token_key[4:]
    info = HfApi(token=token).whoami()
    return str(info.get("name") or info.get("preferred_username") or "").strip()


def resolve_logged_in_namespace(token: bool | str | None = True) -> str:
    """Hub username/org from env or ``HfApi.whoami()``."""
    for key in ("HF_HUB_USERNAME", "TRACE_HF_NAMESPACE"):
        val = os.environ.get(key, "").strip()
        if val:
            return val
    token_key = "default" if token is True or token is None else f"str:{token}"
    try:
        ns = _whoami_namespace(token_key)
    except Exception as e:
        raise ValueError(
            "Could not infer Hugging Face username. Run `huggingface-cli login` "
            "or set HF_HUB_USERNAME."
        ) from e
    if not ns:
        raise ValueError(
            "Could not infer Hugging Face username. Run `huggingface-cli login` "
            "or set HF_HUB_USERNAME."
        )
    return ns


def resolve_org(section: Any = None) -> str:
    org = None if section is None else section.get("hf_hiddens_org")
    if not _blank(org):
        return str(org).strip()
    ns = resolve_logged_in_namespace(hf_token(section))
    logger.info("Hub namespace from login: %s", ns)
    return ns


def resolve_train_repo_id(section: Any = None) -> str:
    org = resolve_org(section)
    explicit = _explicit_repo(None if section is None else section.get("hf_hiddens_train_repo"))
    if explicit is not None:
        return train_hf_repo_id(org=org, repo_id=explicit)
    return train_hf_repo_id(org=org)


def resolve_eval_repo_id(
    section: Any = None,
    *,
    store: Any = None,
    eval_cfg: Any = None,
) -> str:
    org = resolve_org(section)
    explicit = _explicit_repo(store.get("hf_repo") if store is not None else None)
    if explicit is None and eval_cfg is not None:
        explicit = _explicit_repo(eval_cfg.get("hf_repo"))
    if explicit is None and section is not None:
        explicit = _explicit_repo(section.get("hf_hiddens_eval_repo"))
    if explicit is not None:
        return eval_hf_repo_id(org=org, repo_id=explicit)
    return eval_hf_repo_id(org=org)
