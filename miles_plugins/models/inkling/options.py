from __future__ import annotations

from functools import cache

_DEFAULTS = {
    "inkling_attn_backend": "fa4",
    "inkling_sconv_impl": "triton",
    "inkling_sconv_packed": False,
    "inkling_freeze_global_scale": "all",
    "inkling_mm_towers": False,
    "inkling_train_mm_towers": False,
}


@cache
def inkling_opt(name: str):
    """Resolve one knob; cached for kernel-selection hot paths."""
    try:
        from megatron.training import get_args

        return getattr(get_args(), name)
    except Exception:
        return _DEFAULTS[name]
