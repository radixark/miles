"""Translate Transformer Engine's attention window into FlashAttention-4's at TE's FA4 call site.

TE spells an unbounded window side as -1 and hands its causal window (-1, 0) to FA4 together with
causal=True. FA4's cute interface spells an unbounded side as None. Since flash-attn-4 4.0.0b30
(Dao-AILab/flash-attention#2490) it clears a negative window only when both sides are negative, so
mixed-sign pairs can express intentionally empty offset windows. FA4 therefore resolves TE's (-1, 0)
to a local window with window_size_left=-1 instead of a causal mask: the forward attends with the
wrong mask, and the SM100 head_dim=256 backward rejects local attention outright.

TE never uses negative window sides for anything but "unbounded" (``check_set_window_size``), so
mapping them to None at TE's two FA4 entry points is exact. Non-negative sides, i.e. genuine sliding
windows, pass through unchanged, and other FA4 callers keep FA4's own window semantics.
"""

import functools
import importlib
import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)

_TE_BACKENDS_MODULE = "transformer_engine.pytorch.attention.dot_product_attention.backends"
_TE_FA4_ENTRY_POINTS = ("flash_attn_func_v4", "flash_attn_varlen_func_v4")


def te_window_to_fa4(window_size: tuple[int | None, int | None] | None) -> tuple[int | None, int | None] | None:
    """Map each negative (TE: unbounded) window side to None (FA4: unbounded)."""
    if window_size is None:
        return None
    return tuple(None if side is not None and side < 0 else side for side in window_size)


def patch_te_flash_attn_4_window() -> None:
    """Make TE's FA4 calls pass FA4's window convention. A no-op without TE or without FA4."""
    try:
        backends = importlib.import_module(_TE_BACKENDS_MODULE)
    except ImportError:
        return
    for name in _TE_FA4_ENTRY_POINTS:
        fa4_func = getattr(backends, name, None)  # config-access-exempt: TE releases before FA4 lack these names
        if fa4_func is None or getattr(fa4_func, "_miles_fa4_window", False):  # config-access-exempt: wrapper mark
            continue
        setattr(backends, name, _with_fa4_window(fa4_func))
        logger.info("Patched %s.%s to pass unbounded window sides as None", _TE_BACKENDS_MODULE, name)


def _with_fa4_window(fa4_func: Callable) -> Callable:
    @functools.wraps(fa4_func)
    def call(*args, **kwargs):
        if "window_size" in kwargs:
            kwargs["window_size"] = te_window_to_fa4(kwargs["window_size"])
        return fa4_func(*args, **kwargs)

    call._miles_fa4_window = True
    return call
