"""How long to generate (``gen_length``) and how many diffusion ``steps`` to use.

Loaded by the lm-eval launcher and hidden-state capture.
"""

SHORT_GEN_LENGTH_STEPS = (128, 128)
DEFAULT_GEN_LENGTH_STEPS = (512, 512)


def get_gen_length_steps(task_name: str) -> tuple[int, int]:
    """Default (gen_length, steps) for this task when YAML does not set them.

    ARC-Easy, ARC-Challenge chat, and CommonsenseQA use 128/128 (short
    multiple-choice answers). Everything else is 512/512.
    """
    t = str(task_name or "").strip().lower().replace("-", "_")
    if "arc_easy" in t or "arc_challenge" in t:
        return SHORT_GEN_LENGTH_STEPS
    if "commonsense" in t or t in ("csqa",):
        return SHORT_GEN_LENGTH_STEPS
    return DEFAULT_GEN_LENGTH_STEPS


def effective_gen_length_steps(
    task_name: str,
    gen_length: int,
    steps: int,
) -> tuple[int, int]:
    """Keep YAML 512/512 except ARC-Easy / ARC-chat / CommonsenseQA, which stay 128/128."""
    task_g, task_s = get_gen_length_steps(task_name)
    if (task_g, task_s) == SHORT_GEN_LENGTH_STEPS:
        return task_g, task_s
    return int(gen_length), int(steps)


def resolve_gen_length_steps(cfg, task_name: str) -> tuple[int, int]:
    """Pick gen_length and steps from cfg, or use task defaults."""
    gen_d, steps_d = get_gen_length_steps(task_name)
    gl_raw = cfg.get("gen_length") if cfg is not None else None
    st_raw = cfg.get("steps") if cfg is not None else None
    gen_length = int(gl_raw) if gl_raw is not None else gen_d
    steps = int(st_raw) if st_raw is not None else steps_d
    if steps > gen_length:
        raise ValueError(
            f"steps ({steps}) must be <= gen_length ({gen_length}) for task={task_name!r}; "
            "set overrides in YAML/CLI so both constraints hold."
        )
    return gen_length, steps
