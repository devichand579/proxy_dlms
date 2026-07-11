"""HuggingFace Hub helpers for PCFG corpora (aligned with Strace trace/CST conventions)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from repo_env_defaults import apply_repo_env_defaults
from traces.hf_trace_upload import ensure_dataset_repo, normalize_dataset_repo_id

DEFAULT_PCFG_DATASET = "pcfg-code"


def pcfg_dataset_repo() -> str:
    """Default short dataset name (override with PCFG_HF_DATASET)."""
    return os.environ.get("PCFG_HF_DATASET", DEFAULT_PCFG_DATASET).strip() or DEFAULT_PCFG_DATASET


def pcfg_namespace(namespace: Optional[str] = None) -> Optional[str]:
    if namespace and str(namespace).strip():
        return str(namespace).strip()
    for key in ("PCFG_HF_NAMESPACE", "TRACE_HF_NAMESPACE", "HF_HUB_USERNAME"):
        val = os.environ.get(key, "").strip()
        if val:
            return val
    return None


def resolve_pcfg_repo(
    repo_id: Optional[str] = None,
    *,
    dataset: Optional[str] = None,
    namespace: Optional[str] = None,
) -> str:
    """Return full ``org/repo`` for a PCFG corpus on the Hub."""
    name = (repo_id or dataset or pcfg_dataset_repo()).strip()
    if not name:
        raise ValueError("PCFG Hub dataset name is empty")
    return normalize_dataset_repo_id(name, pcfg_namespace(namespace))


def resolve_hf_token(token: Optional[str] = None) -> Optional[str]:
    return token or os.environ.get("HF_TOKEN") or None


def _push_dataset_card(full_repo: str, *, ddict, token: Optional[str] = None) -> None:
    """Push/update README.md with configs YAML so the HF dataset viewer indexes splits."""
    try:
        from huggingface_hub import DatasetCard, DatasetCardData
    except ImportError:
        return  # non-fatal: older hub versions without DatasetCard

    configs = [
        {
            "config_name": "default",
            "data_files": [
                {"split": split_name, "path": f"data/{split_name}-*.parquet"}
                for split_name in ddict.keys()
            ],
        }
    ]

    # Load existing card if present, otherwise start fresh
    try:
        card = DatasetCard.load(full_repo, token=token)
    except Exception:
        card = DatasetCard("")

    if card.data is None:
        card.data = DatasetCardData()

    card.data["configs"] = configs

    # Build a minimal description with split sizes
    split_lines = []
    for split_name, ds in ddict.items():
        split_lines.append(f"- **{split_name}**: {len(ds):,} examples")

    if not str(card).strip():
        card_text = "# PCFG Code Corpus\n\n"
        card_text += "Synthetically generated PCFG strings for language-model research.\n\n"
        card_text += "## Splits\n\n" + "\n".join(split_lines) + "\n"
        card = DatasetCard(card_text)
        card.data = DatasetCardData(configs=configs)

    card.push_to_hub(full_repo, token=token)


def push_pcfg_dataset(
    ddict,
    *,
    repo_id: Optional[str] = None,
    dataset: Optional[str] = None,
    namespace: Optional[str] = None,
    private: bool = False,
    token: Optional[str] = None,
) -> str:
    """Create dataset repo if needed, then push a DatasetDict."""
    full_repo = resolve_pcfg_repo(repo_id, dataset=dataset, namespace=namespace)
    tok = resolve_hf_token(token)
    ensure_dataset_repo(full_repo, token=tok, private=private)
    ddict.push_to_hub(full_repo, private=private, token=tok)
    _push_dataset_card(full_repo, ddict=ddict, token=tok)
    return full_repo


def load_pcfg_dataset(
    repo_id: Optional[str] = None,
    *,
    dataset: Optional[str] = None,
    namespace: Optional[str] = None,
    token: Optional[str] = None,
):
    from datasets import load_dataset

    full_repo = resolve_pcfg_repo(repo_id, dataset=dataset, namespace=namespace)
    return load_dataset(full_repo, token=resolve_hf_token(token))


def apply_pcfg_hf_env() -> None:
    """Apply repo-wide HF/W&B defaults (same as eval, CST, trace scripts)."""
    apply_repo_env_defaults()
