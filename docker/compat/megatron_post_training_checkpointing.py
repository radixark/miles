"""Compatibility shim for Miles' pinned GLM Megatron checkout.

The conversion utility only needs to know that the public HF checkpoint has no
NVIDIA ModelOpt state.  Newer Megatron training.py imports this helper even
when ModelOpt is not installed in the runtime image.
"""


def has_modelopt_state(*_args, **_kwargs) -> bool:
    return False
