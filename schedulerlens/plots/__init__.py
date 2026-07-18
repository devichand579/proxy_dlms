"""SchedulerLens plotting (lazy matplotlib imports)."""

from __future__ import annotations

__all__ = [
    "TraceSurfaces",
    "compute_trace_surfaces",
    "join_rows_with_strace",
    "load_strace_metrics",
    "save_bucket_plots",
    "save_compare_plots",
    "save_eval_join_plots",
    "save_pos_plots",
    "save_quad_plots",
    "save_trend_plots",
]


def __getattr__(name: str):
    if name in (
        "TraceSurfaces",
        "compute_trace_surfaces",
        "save_bucket_plots",
        "save_compare_plots",
        "save_eval_join_plots",
        "save_quad_plots",
        "save_trend_plots",
        "join_rows_with_strace",
        "load_strace_metrics",
    ):
        from . import viz
        return getattr(viz, name)
    if name == "save_pos_plots":
        from .pos import save_pos_plots
        return save_pos_plots
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
