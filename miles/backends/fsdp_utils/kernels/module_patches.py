"""Bind the active hub module-kernels onto a constructed model.

One entry point, called from ``FSDPTrainRayActor.init()`` and ``_create_ref_model()`` after
``apply_model_instance_patches`` and before ``apply_fsdp2``. Each per-arch binder walks the model
and returns 0 when its architecture isn't present, so the dispatch stays a flat list.

Rebinding happens on module instances, not on classes: it only replaces callables the HF modeling
code already looks up per forward, so parameters, ``state_dict`` keys, ``_no_split_modules`` and
the DTensor gather in ``update_weight_utils.py`` are all untouched. That is also why running it
before ``apply_fsdp2`` is safe.
"""

import logging

logger = logging.getLogger(__name__)


def apply_hub_module_kernels(model, args) -> dict[str, int]:
    """Rebind every arch's hub-backed kernels on ``model``. No-op unless ``--kernel-backend hub``.

    Returns the per-architecture count of patched modules, for logging and tests.
    """
    from miles.backends.fsdp_utils.kernels.hub import hub_kernels_enabled

    if not hub_kernels_enabled(args):
        return {}

    from miles.backends.fsdp_utils.models.nemotron_h import bind_nemotron_h_hub_kernels
    from miles.backends.fsdp_utils.models.qwen3_5 import bind_gated_deltanet_hub_kernels

    patched = {
        "gated_deltanet": bind_gated_deltanet_hub_kernels(model, args),
        "nemotron_h": bind_nemotron_h_hub_kernels(model, args),
    }
    patched = {arch: n for arch, n in patched.items() if n}
    if patched:
        logger.info("[fsdp hub kernels] bound module kernels: %s", patched)
    return patched
