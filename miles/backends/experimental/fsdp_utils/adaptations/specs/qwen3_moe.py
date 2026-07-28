"""qwen3_moe adaptations: the train->rollout weight transform (unfuse transformers>=5.6 batched experts
into the per-expert names sglang expects, via HF's own reverse conversion) and the true-on-policy
config-time MoE-block patch."""

from ..class_patches import ModelPatchHook, register_model_patch
from ..weight_bridge import _batched_experts_matches, _hf_unfuse_experts_expand, register_param_transform

register_param_transform("qwen3_moe", _batched_experts_matches, _hf_unfuse_experts_expand)


def _is_qwen3_moe(hf_config) -> bool:
    return str(getattr(hf_config, "model_type", "") or "") == "qwen3_moe"


def _apply_moe_patch(hf_config, args) -> None:
    """Patch the MoE block before construction in true-on-policy mode. The import stays lazy because it
    pulls sglang fused-MoE kernels that don't exist for every sglang build."""
    if not getattr(args, "true_on_policy_mode", False):
        return
    from ...models.qwen3_moe import apply_true_on_policy_patch_for_qwen3_moe

    apply_true_on_policy_patch_for_qwen3_moe()


register_model_patch(ModelPatchHook("qwen3_moe_moe_patch", _is_qwen3_moe, _apply_moe_patch))
