"""Bind resolved hub kernels onto constructed model instances.

Rebinding happens on module instances, not on classes: it only replaces callables the HF modeling
code already looks up per forward, so parameters, `state_dict` keys, `_no_split_modules` and the
DTensor gather in `update_weight_utils.py` are untouched -- which is why running before
`apply_fsdp2` is safe, for the policy and the ref model alike.
"""

import logging

from miles.backends.fsdp_utils.plugins.hf_kernels.loader import HubKernels
from miles.backends.fsdp_utils.plugins.hf_kernels.presets import (
    SLOT_CAUSAL_CONV1D,
    SLOT_FLASH_ATTN_VARLEN,
    SLOT_GATED_DELTA_RULE,
)

logger = logging.getLogger(__name__)

# Per slot: the GatedDeltaNet instance attribute -> the function name in that Hub build. The names
# differ on the recurrent kernel, which transformers stores without the `fused_` prefix.
_GDN_KERNEL_SLOTS = (
    (
        SLOT_GATED_DELTA_RULE,
        {
            "chunk_gated_delta_rule": "chunk_gated_delta_rule",
            "recurrent_gated_delta_rule": "fused_recurrent_gated_delta_rule",
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
    """Point each GatedDeltaNet's kernel handles at the Hub builds; returns modules patched.

    `_patch_gdn_forward` in `models/qwen3_5.py` injects the packed-document boundaries into
    whatever callables the instance carries, so this only has to replace them. Slots resolve
    independently: one missing Hub build leaves the other bound.
    """
    modules = [m for m in model.modules() if type(m).__name__.endswith("GatedDeltaNet")]
    if not modules:
        return 0

    bound = []
    for slot, attr_to_function in _GDN_KERNEL_SLOTS:
        functions = hub.resolve_slot(slot)
        if not functions:
            continue
        for module in modules:
            for attr, function_name in attr_to_function.items():
                setattr(module, attr, functions[function_name])
        bound.append(slot)

    if not bound:
        return 0
    logger.info("[hf kernels] GatedDeltaNet %s served from the Hub on %d module(s)", " + ".join(bound), len(modules))
    return len(modules)


def bind_nemotron_h(model, hub: HubKernels) -> int:
    """Stash the Hub varlen-attention kernel on each NemotronH attention mixer; returns mixers patched.

    `_patch_attn_forward` in `models/nemotron_h.py` reads the handle per forward and otherwise
    falls back to the flash-attn wheel, so this can run before or after the packing patch.
    """
    from miles.backends.fsdp_utils.models.nemotron_h import HUB_VARLEN_ATTR

    mixers = [
        mod.mixer
        for mod in model.modules()
        if getattr(mod, "block_type", None) == "attention" and hasattr(mod, "mixer")
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
