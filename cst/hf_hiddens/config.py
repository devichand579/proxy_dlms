"""Resolve Hub repo ids and tokens from train config."""

from __future__ import annotations

from typing import Any, Optional

from cst.hf_hiddens.naming import eval_hf_repo_id, train_hf_repo_id


def _explicit_repo(value: Any) -> Optional[str]:
    if value is None or str(value).strip().lower() in ("", "null", "none"):
        return None
    return str(value).strip()


def hf_token(tr: Any) -> bool | str | None:
    t = tr.get("hf_token")
    return True if t is None else t


def resolve_org(tr: Any) -> str:
    org = tr.get("hf_hiddens_org")
    if org is None or not str(org).strip() or str(org).strip().lower() in ("null", "none"):
        raise ValueError("train.hf_hiddens_org is required (set in config/cst.yaml)")
    return str(org).strip()


def resolve_train_repo_id(tr: Any) -> str:
    org = resolve_org(tr)
    explicit = _explicit_repo(tr.get("hf_hiddens_train_repo"))
    if explicit is not None:
        return train_hf_repo_id(org=org, repo_id=explicit)
    return train_hf_repo_id(org=org)


def resolve_eval_repo_id(
    tr: Any,
    *,
    store: Any = None,
    eval_cfg: Any = None,
) -> str:
    org = resolve_org(tr)
    explicit = _explicit_repo(store.get("hf_repo") if store is not None else None)
    if explicit is None and eval_cfg is not None:
        explicit = _explicit_repo(eval_cfg.get("hf_repo"))
    if explicit is None:
        explicit = _explicit_repo(tr.get("hf_hiddens_eval_repo"))
    if explicit is not None:
        return eval_hf_repo_id(org=org, repo_id=explicit)
    return eval_hf_repo_id(org=org)
