"""Resolve Azure credentials and container names from config or env."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from cst.azure_hiddens.naming import eval_azure_container, train_azure_container

logger = logging.getLogger("cst.azure_hiddens.config")


def _blank(value: Any) -> bool:
    return value is None or str(value).strip().lower() in ("", "null", "none")


def _explicit_container(value: Any) -> Optional[str]:
    if _blank(value):
        return None
    return str(value).strip()


def azure_connection_string(section: Any = None) -> str:
    """Use config azure_conn_str or AZURE_STORAGE_CONNECTION_STRING."""
    conn_str = None if section is None else section.get("azure_conn_str")
    if _blank(conn_str):
        env = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        if env and env.strip():
            return env.strip()
        raise ValueError(
            "Azure storage connection string not found. Set azure_conn_str in "
            "config or AZURE_STORAGE_CONNECTION_STRING env var."
        )
    return str(conn_str).strip()


def resolve_train_repo_id(section: Any = None) -> str:
    """Parallel to resolve_train_repo_id; returns the train container."""
    explicit = _explicit_container(None if section is None else section.get("azure_train_container"))
    return train_azure_container(explicit)


def resolve_eval_repo_id(
    section: Any = None,
    *,
    store: Any = None,
    eval_cfg: Any = None,
) -> str:
    """Parallel to resolve_eval_repo_id; returns the eval container."""
    explicit = _explicit_container(store.get("azure_container") if store is not None else None)
    if explicit is None and eval_cfg is not None:
        explicit = _explicit_container(eval_cfg.get("azure_container"))
    if explicit is None and section is not None:
        explicit = _explicit_container(section.get("azure_eval_container"))
    return eval_azure_container(explicit)
