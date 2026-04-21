"""
Prompt pools: :func:`load_pools` (full data per source), :func:`sample_uniform` (mixed draw, no replacement).

API: ``load_pools``, ``sample_uniform``, ``default_source_specs``, ``SourceSpec``, ``list_minerva_math_fewshot_samples``.
CLI: ``python -m cst.train_data``.
"""

from .compose import SourceSpec, default_source_specs, load_pools, sample_uniform
from .minerva_math import list_minerva_math_fewshot_samples

__all__ = [
    "SourceSpec",
    "default_source_specs",
    "list_minerva_math_fewshot_samples",
    "load_pools",
    "sample_uniform",
]
