from .deepseek_v32 import DeepseekV32Bridge
from .glm4 import GLM4Bridge
from .glm4moe import GLM4MoEBridge
from .glm4moe_lite import GLM4MoELiteBridge
from .mimo import MimoBridge
from .qwen3_next import Qwen3NextBridge

try:
    from .deepseekv4 import DeepseekV4Bridge
except ImportError:
    DeepseekV4Bridge = None

__all__ = ["GLM4Bridge", "GLM4MoEBridge", "GLM4MoELiteBridge", "Qwen3NextBridge", "MimoBridge", "DeepseekV32Bridge"]
if DeepseekV4Bridge is not None:
    __all__.append("DeepseekV4Bridge")

from mbridge import AutoBridge

_original_from_config = AutoBridge.from_config


@classmethod
def _patched_from_config(cls, hf_config, **kwargs):
    from mbridge.core.bridge import _MODEL_REGISTRY

    if hasattr(hf_config, "hc_mult"):
        return _MODEL_REGISTRY["deepseek_v4"](hf_config, **kwargs)
    if hasattr(hf_config, "index_n_heads"):
        return _MODEL_REGISTRY["deepseek_v32"](hf_config, **kwargs)

    return _original_from_config(hf_config, **kwargs)


AutoBridge.from_config = _patched_from_config
