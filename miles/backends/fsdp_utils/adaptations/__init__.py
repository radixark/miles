"""Per-arch adaptation layer for the FSDP backend.

Pure-mechanism registries (``weight_bridge``, ``class_patches``, ``packing``, ``post_load_fixups``,
``precision``) plus arch-centric ``specs/`` that register each arch's hooks across them. The actor drives
the registries; a new arch plugs in by adding ``specs/<arch>.py``.
"""

from .class_patches import (
    ModelInstancePatchHook,
    ModelPatchHook,
    apply_class_patches,
    apply_model_instance_patches,
    register_model_instance_patch,
    register_model_patch,
)
from .packing import PackingPatch, apply_packing, get_packing_patches, register_packing_patch
from .post_load_fixups import PostLoadFixup, apply_post_load_fixups, register_post_load_fixup
from .precision import (
    PrecisionPolicy,
    PrecisionPolicyHook,
    apply_fp32_master,
    precision_forward_context,
    register_precision_policy,
    resolve_precision_policy,
)
from .weight_bridge import ParamTransform, get_param_transform, register_param_transform

# MUST be last: importing the specs registers each arch's hooks into the mechanism registries above.
from . import specs  # noqa: F401,E402  # isort: skip

__all__ = [
    "ModelPatchHook",
    "apply_class_patches",
    "register_model_patch",
    "ModelInstancePatchHook",
    "apply_model_instance_patches",
    "register_model_instance_patch",
    "PackingPatch",
    "apply_packing",
    "get_packing_patches",
    "register_packing_patch",
    "PostLoadFixup",
    "apply_post_load_fixups",
    "register_post_load_fixup",
    "PrecisionPolicy",
    "PrecisionPolicyHook",
    "apply_fp32_master",
    "precision_forward_context",
    "register_precision_policy",
    "resolve_precision_policy",
    "ParamTransform",
    "get_param_transform",
    "register_param_transform",
]
