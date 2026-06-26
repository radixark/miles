"""qwen3_moe adaptations: the train->rollout weight transform (split transformers>=5.6 batched experts
into the per-expert names sglang expects) and the config-time MoE-block patch (true-on-policy swaps the
batch-invariant block, otherwise the legacy graph patch that no-ops on batched experts)."""

from ..class_patches import ModelPatchHook, register_model_patch
from ..weight_bridge import _qwen3_moe_expand, _qwen3_moe_matches, register_param_transform

register_param_transform("qwen3_moe", _qwen3_moe_matches, _qwen3_moe_expand)


def _is_qwen3_moe(hf_config) -> bool:
    return str(getattr(hf_config, "model_type", "") or "") == "qwen3_moe"


def _apply_moe_patch(hf_config, args) -> None:
    """MoE-block patch before construction; variant depends on the run mode. Imports are lazy because
    the true-on-policy path pulls sglang fused-MoE kernels that don't exist for every sglang build."""
    if getattr(args, "true_on_policy_mode", False):
        from ...models.qwen3_moe import apply_true_on_policy_patch_for_qwen3_moe

        apply_true_on_policy_patch_for_qwen3_moe()
    else:
        from ...models.qwen3_moe_hf import apply_fsdp_moe_patch

        apply_fsdp_moe_patch()


register_model_patch(ModelPatchHook("qwen3_moe_moe_patch", _is_qwen3_moe, _apply_moe_patch))
