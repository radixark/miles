"""Bind resolved hub kernels onto constructed models.

Binding only replaces callables the HF modeling code already looks up per forward, so parameters,
`state_dict` keys, `_no_split_modules` and the DTensor gather in `update_weight_utils.py` are
untouched -- which is why running before `apply_fsdp2` is safe, for the policy and the ref model alike.
"""

import functools
import inspect
import logging
import sys

from miles.backends.fsdp_utils.plugins.hf_kernels.loader import HubKernels
from miles.backends.fsdp_utils.plugins.hf_kernels.presets import (
    SLOT_CAUSAL_CONV1D,
    SLOT_FLASH_ATTN_VARLEN,
    SLOT_GATED_DELTA_RULE,
)

logger = logging.getLogger(__name__)

# Per slot: the module-level name a GatedDeltaNet forward calls in its transformers modeling module
# -> the function name in that Hub build. The delta-rule names keep their `torch_` prefix whichever
# implementation transformers resolved at import.
_GDN_KERNEL_SLOTS = (
    (
        SLOT_GATED_DELTA_RULE,
        {
            "torch_chunk_gated_delta_rule": "chunk_gated_delta_rule",
            "torch_recurrent_gated_delta_rule": "fused_recurrent_gated_delta_rule",
        },
    ),
    (
        SLOT_CAUSAL_CONV1D,
        {
            "causal_conv1d_fn": "causal_conv1d_fn",
            "causal_conv1d_update": "causal_conv1d_update",
        },
    ),
)


def bind_gated_deltanet(model, hub: HubKernels) -> int:
    """Point the kernels each GatedDeltaNet calls at the Hub builds; returns modules served.

    transformers' GatedDeltaNet forward calls module-level kernels of its modeling module (resolved
    at import to fla / causal_conv1d or a torch fallback) and hands them the packed-document
    boundaries it receives as kwargs (`seq_idx`, `cu_seq_lens_q`), so this rebinds those names.
    That is process-wide, which the policy and the ref model share. Slots resolve independently:
    one missing Hub build leaves the other bound.
    """
    modules = [m for m in model.modules() if type(m).__name__.endswith("GatedDeltaNet")]
    if not modules:
        return 0
    modeling_modules = list(dict.fromkeys(sys.modules[type(m).__module__] for m in modules))

    bound = []
    for slot, name_to_function in _GDN_KERNEL_SLOTS:
        functions = hub.resolve_slot(slot)
        if not functions:
            continue
        missing = [
            f"{mod.__name__}.{name}"
            for mod in modeling_modules
            for name in name_to_function
            if not hasattr(
                mod, name
            )  # config-access-exempt: the kernel binding table supplies optional third-party module symbol names
        ]
        if missing:
            message = f"slot {slot!r}: no module-level GatedDeltaNet kernel {missing} to rebind"
            if hub.strict:
                raise RuntimeError(f"--kernel-strict: {message}")
            logger.warning("[hf kernels] %s; keeping the native kernel", message)
            continue
        for mod in modeling_modules:
            for name, function_name in name_to_function.items():
                setattr(mod, name, _with_declared_kwargs(functions[function_name]))
        bound.append(slot)

    if not bound:
        return 0
    logger.info(
        "[hf kernels] GatedDeltaNet %s served from the Hub in %s (%d module(s))",
        " + ".join(bound),
        ", ".join(mod.__name__ for mod in modeling_modules),
        len(modules),
    )
    return len(modules)


def _with_declared_kwargs(function):
    """Call `function` with only the kwargs it declares, as transformers' own kernel wrappers do.

    The GatedDeltaNet forward hands every padding-free kwarg (`cu_seq_lens_k`, `max_length_q`, ...)
    to each kernel, while a Hub build such as `causal_conv1d_fn` declares no `**kwargs`.
    """
    declared = frozenset(inspect.signature(function).parameters)

    @functools.wraps(function)
    def call(*args, **kwargs):
        return function(*args, **{key: value for key, value in kwargs.items() if key in declared})

    return call


def bind_nemotron_h(model, hub: HubKernels) -> int:
    """Stash the Hub varlen-attention kernel on each NemotronH attention mixer; returns mixers patched.

    `_patch_attn_forward` in `models/nemotron_h.py` reads the handle per forward and otherwise
    falls back to the flash-attn wheel, so this can run before or after the packing patch.
    """
    from miles.backends.fsdp_utils.models.nemotron_h import HUB_VARLEN_ATTR

    mixers = [
        mod.mixer
        for mod in model.modules()
        if getattr(mod, "block_type", None) == "attention"
        and hasattr(
            mod, "mixer"
        )  # config-access-exempt: HF model traversal includes modules without mixer; HF model traversal includes non-attention modules without block_type or mixer
    ]
    if not mixers:
        return 0

    functions = hub.resolve_slot(SLOT_FLASH_ATTN_VARLEN)
    if not functions:
        return 0

    for mixer in mixers:
        setattr(mixer, HUB_VARLEN_ATTR, functions["flash_attn_varlen_func"])
    logger.info("[hf kernels] NemotronH varlen attention served from the Hub on %d mixer(s)", len(mixers))
    return len(mixers)
