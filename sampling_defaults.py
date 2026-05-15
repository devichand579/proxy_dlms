"""How long to generate (``gen_length``) and how many diffusion ``steps`` to use.

Loaded by the lm-eval launcher and hidden-state capture.
"""


def get_gen_length_steps(task_name: str) -> tuple[int, int]:
    """Default (gen_length, steps) for this task when YAML does not set them."""
    if "arc_challenge_chat" in task_name:
        return 128, 128
    return 512, 512


def resolve_gen_length_steps(cfg, task_name: str) -> tuple[int, int]:
    """Pick ``gen_length`` and ``steps`` from ``cfg``, or use task defaults.

    Reads keys ``gen_length`` and ``steps`` (Hydra YAML, OmegaConf, or plain dict).
    Omit a key or set it to null to keep the default for that slot.
    Raises if ``steps > gen_length``.
    """
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
