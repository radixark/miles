"""glm4_moe_lite (GLM-4.7-Flash) train-to-rollout weight transform; its batched expert layout matches qwen3_moe, so it reuses the same HF-native unfuse."""

from ..weight_bridge import _batched_experts_matches, _hf_unfuse_experts_expand, register_param_transform

register_param_transform("glm4_moe_lite", _batched_experts_matches, _hf_unfuse_experts_expand)
