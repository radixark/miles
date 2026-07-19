"""qwen3_moe adaptations: train->rollout expert-split weight transform and the config-time MoE-block patch."""

from ..class_patches import ModelPatchHook, register_model_patch
from ..weight_bridge import _qwen3_moe_expand, _qwen3_moe_matches, register_param_transform

register_param_transform("qwen3_moe", _qwen3_moe_matches, _qwen3_moe_expand)


def _is_qwen3_moe(hf_config) -> bool:
    return str(getattr(hf_config, "model_type", "") or "") == "qwen3_moe"


def _apply_moe_patch(hf_config, args) -> None:
    """MoE-block patch before construction; variant depends on run mode. Lazy imports (true-on-policy needs sglang)."""
    if getattr(args, "true_on_policy_mode", False):
        from ...models.qwen3_moe import apply_true_on_policy_patch_for_qwen3_moe

        apply_true_on_policy_patch_for_qwen3_moe()
    else:
        from ...models.qwen3_moe_hf import apply_fsdp_moe_patch

        apply_fsdp_moe_patch()


register_model_patch(ModelPatchHook("qwen3_moe_moe_patch", _is_qwen3_moe, _apply_moe_patch))
