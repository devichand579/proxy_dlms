"""Pace / DAS unmasking schedulers for LLaDA.

When das/vendor/ exists (standalone checkout), it goes first on sys.path so the copied
Strace modules (load_model, schedulers, schedulerlens) are used. Inside the Strace repo
there is no vendor/ folder and the repo's own modules are used.
"""
import os
import sys

_VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vendor')
if os.path.isdir(_VENDOR) and _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)


def model_path(path):
    """Resolve a model dir such as 'models/llada' against MODEL_ROOT (default /root/Strace)."""
    if os.path.isabs(path) or os.path.isdir(path):
        return path
    return os.path.join(os.environ.get('MODEL_ROOT', '/root/Strace'), path)
