from .logits_head import logits_from_prelogit_hidden
from .metrics import (
    compute_l0,
    compute_nmse,
    compute_ntf_logits,
    cosine_fidelity,
    prompt_gen_token_masks,
    temporal_coherence,
)
from .agg_persist import (
    cst_eval_agg_path,
    load_cst_eval_agg,
    save_cst_eval_agg,
)
from cst.naming import cst_resumable_eval_agg_basename, cst_resumable_eval_agg_path
from .run_eval_cst import (
    eval_summary_from_agg,
    evaluate_chunk_file,
    merge_eval_file_into_agg,
    new_eval_aggregator,
    run_eval_on_dir,
)

__all__ = [
    "logits_from_prelogit_hidden",
    "compute_l0",
    "compute_nmse",
    "compute_ntf_logits",
    "cosine_fidelity",
    "prompt_gen_token_masks",
    "temporal_coherence",
    "eval_summary_from_agg",
    "evaluate_chunk_file",
    "merge_eval_file_into_agg",
    "new_eval_aggregator",
    "run_eval_on_dir",
    "cst_eval_agg_path",
    "cst_resumable_eval_agg_path",
    "cst_resumable_eval_agg_basename",
    "load_cst_eval_agg",
    "save_cst_eval_agg",
]
