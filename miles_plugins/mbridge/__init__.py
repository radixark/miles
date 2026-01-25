import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from .deepseekv4 import DeepseekV4Bridge
from .deepseekv32 import DeepseekV32Bridge
from .glm4 import GLM4Bridge
from .glm4moe import GLM4MoEBridge
from .mimo import MimoBridge
from .qwen3_next import Qwen3NextBridge

__all__ = ["DeepseekV32Bridge", "DeepseekV4Bridge", "GLM4Bridge", "GLM4MoEBridge", "Qwen3NextBridge", "MimoBridge"]

from mbridge import AutoBridge

_original_from_config = AutoBridge.from_config


@classmethod
def _patched_from_config(cls, hf_config, **kwargs):
    from mbridge.core.bridge import _MODEL_REGISTRY

    # HACK, only support dsv4 temporarily
    return _MODEL_REGISTRY["deepseek_v4"](hf_config, **kwargs)

    # V3.2 has index_n_heads but no hc_mult
    if hasattr(hf_config, "index_n_heads"):
        return _MODEL_REGISTRY["deepseek_v32"](hf_config, **kwargs)

    return _original_from_config(hf_config, **kwargs)


AutoBridge.from_config = _patched_from_config
