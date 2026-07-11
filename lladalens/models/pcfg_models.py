"""
Small LLaDA variants (100M / 200M / 300M) for PCFG grammar training.

Tokenizer and vocab align with ``pcfg_data.Vocab(code_grammar())``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer

from pcfg_data import ARCH_LEN, Vocab
from pcfg_grammar import code_grammar

from .configuration_llada import LLaDAConfig
from .modeling_llada import LLaDAModelLM

_ROOT = Path(__file__).resolve().parent
_VARIANTS = json.loads((_ROOT / "variants.json").read_text())


def list_variants() -> list[str]:
    return sorted(_VARIANTS["variants"])


def pcfg_vocab() -> Vocab:
    return Vocab(code_grammar().terminals)


def load_pcfg_tokenizer():
    return AutoTokenizer.from_pretrained(str(_ROOT))


def load_pcfg_llada(
    variant: str = "100m",
    *,
    vocab: Optional[Vocab] = None,
    init_params: bool = True,
    max_sequence_length: int = ARCH_LEN,
    device: Optional[str] = None,
):
    """Build a randomly-initialized LLaDA for the PCFG terminal vocab."""
    vocab = vocab or pcfg_vocab()
    key = str(variant).lower().replace("llada-", "").replace("-", "")
    if key not in _VARIANTS["variants"]:
        raise ValueError(f"unknown variant {variant!r}; choose from {list_variants()}")

    embedding_size = ((vocab.size + 127) // 128) * 128
    cfg = {**_VARIANTS["base"], **_VARIANTS["variants"][key], "variant": key}
    cfg.update(
        vocab_size=vocab.size,
        embedding_size=embedding_size,
        max_sequence_length=max_sequence_length,
        pad_token_id=vocab.pad_id,
        eos_token_id=vocab.eos_id,
        mask_token_id=vocab.mask_id,
        use_cache=False,
        init_device="cpu",
    )
    config = LLaDAConfig(**cfg)
    model = LLaDAModelLM(config, init_params=init_params)
    if device is not None:
        model = model.to(device)
    return model, config, vocab


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())
