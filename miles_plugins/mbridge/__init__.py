import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from miles.utils.deepseek_v32_patch import apply_deepseek_v32_patch
apply_deepseek_v32_patch(restore_model_type=True)

from .deepseekv32 import DeepseekV32Bridge
from .glm4 import GLM4Bridge
from .glm4moe import GLM4MoEBridge
from .mimo import MimoBridge
from .qwen3_next import Qwen3NextBridge

__all__ = ["DeepseekV32Bridge", "GLM4Bridge", "GLM4MoEBridge", "Qwen3NextBridge", "MimoBridge"]

from mbridge import AutoBridge

_original_from_config = AutoBridge.from_config

@classmethod
def _patched_from_config(cls, hf_config, **kwargs):
    if hf_config.model_type == "deepseek_v32":
        from mbridge.core.bridge import _MODEL_REGISTRY
        return _MODEL_REGISTRY['deepseek_v32'](hf_config, **kwargs)
    
    return _original_from_config(hf_config, **kwargs)

AutoBridge.from_config = _patched_from_config
