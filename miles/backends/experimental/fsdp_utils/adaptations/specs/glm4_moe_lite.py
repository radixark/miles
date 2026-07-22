"""glm4_moe_lite (GLM-4.7-Flash) train-to-rollout weight transform; its batched expert layout matches qwen3_moe, so it reuses the same split."""

from ..weight_bridge import _qwen3_moe_expand, _qwen3_moe_matches, register_param_transform

register_param_transform("glm4_moe_lite", _qwen3_moe_matches, _qwen3_moe_expand)
