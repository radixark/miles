from mbridge.core import register_model
from mbridge.models import DeepseekV3Bridge

from .deepseek_v32 import DeepseekV32Bridge
from .glm4 import GLM4Bridge
from .glm4moe import GLM4MoEBridge
from .glm4moe_lite import GLM4MoELiteBridge
from .mimo import MimoBridge
from .qwen3_next import Qwen3NextBridge

# Kimi-K2 is architecturally identical to DeepSeek-V3 (same weight layout, MLA + MoE),
# just with different hyperparameters (num_experts, num_attention_heads, etc.)
register_model("kimi_k2")(DeepseekV3Bridge)

__all__ = ["GLM4Bridge", "GLM4MoEBridge", "GLM4MoELiteBridge", "Qwen3NextBridge", "MimoBridge", "DeepseekV32Bridge"]
