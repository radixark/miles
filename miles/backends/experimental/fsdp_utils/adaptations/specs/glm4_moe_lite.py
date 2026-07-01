"""glm4_moe_lite (GLM-4.7-Flash) adaptations: reuse qwen3_moe expert-split weight transform plus an fp32 master."""

from ..precision import register_fp32_master_type
from ..weight_bridge import _qwen3_moe_expand, _qwen3_moe_matches, register_param_transform

register_param_transform("glm4_moe_lite", _qwen3_moe_matches, _qwen3_moe_expand)
register_fp32_master_type("glm4_moe_lite")
