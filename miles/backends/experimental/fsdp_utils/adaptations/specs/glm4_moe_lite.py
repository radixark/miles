"""glm4_moe_lite (GLM-4.7-Flash) adaptations: the train->rollout weight transform (its batched expert
layout is identical to qwen3_moe, so it reuses the same split) and an fp32 master (its FSDP2 bf16 reshard
perturbs weights ~1 ULP vs a clean disk load, downcast back at weight sync)."""

from ..precision import register_fp32_master_type
from ..weight_bridge import _qwen3_moe_expand, _qwen3_moe_matches, register_param_transform

register_param_transform("glm4_moe_lite", _qwen3_moe_matches, _qwen3_moe_expand)
register_fp32_master_type("glm4_moe_lite")
