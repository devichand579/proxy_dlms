"""SchedulerLens generation tracing: live hooks, Hub audit/convert, NPZ storage.

Layout:
  trace_generator / trace_wrapper — live recording during generate
  utils — path helpers, gen-region vectors, setitem detection utils
  trace_checks — validate + repair + round-trip
  trace_npz — JSONL/JSON ↔ NPZ
  trace_pipeline / trace_audit — Hub audit/convert/repair CLI
  hf_trace_upload — Hub I/O
  trace_metrics — lm-eval / PCFG metric comparison
"""

from traces.utils import as_gen_vec, squeeze_len, squeeze_to_1d

__all__ = [
    "as_gen_vec",
    "squeeze_len",
    "squeeze_to_1d",
]
