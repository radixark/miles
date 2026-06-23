"""Per-arch adaptation layer for the FSDP backend.

Pure-mechanism registries -- ``weight_bridge`` (train->rollout param contract), ``class_patches``
(config-time HF-compat/model patches), ``packing`` (packed-sequence layout), ``post_load_fixups``
(post-load weight fixups), ``precision`` (fp32-master + MixedPrecision dtypes) -- plus arch-centric
``specs/`` that register each arch's hooks across them. The actor drives the registries; a new arch
plugs in by adding ``specs/<arch>.py``.
"""

from .class_patches import ModelPatchHook, apply_class_patches, register_model_patch
from .packing import PackingPatch, apply_packing, get_packing_patches, register_packing_patch
from .post_load_fixups import PostLoadFixup, apply_post_load_fixups, register_post_load_fixup
from .precision import PrecisionPolicy, apply_fp32_master, register_fp32_master_type, resolve_precision_policy
from .weight_bridge import ParamTransform, get_param_transform, register_param_transform

# MUST be last: importing the specs registers each arch's hooks into the mechanism registries above.
from . import specs  # noqa: F401,E402  # isort: skip

__all__ = [
    "ModelPatchHook",
    "apply_class_patches",
    "register_model_patch",
    "PackingPatch",
    "apply_packing",
    "get_packing_patches",
    "register_packing_patch",
    "PostLoadFixup",
    "apply_post_load_fixups",
    "register_post_load_fixup",
    "PrecisionPolicy",
    "apply_fp32_master",
    "register_fp32_master_type",
    "resolve_precision_policy",
    "ParamTransform",
    "get_param_transform",
    "register_param_transform",
]
