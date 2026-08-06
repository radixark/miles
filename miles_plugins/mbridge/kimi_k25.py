"""mbridge bridge for Kimi-K2.5 (``model_type: kimi_k25``).

Kimi-K2.5's text stack is DeepSeek-V3-shaped MLA + MoE, wrapped in a
multimodal ``KimiK25ForConditionalGeneration`` shell: the HF config nests the
text fields under ``text_config`` and every text weight is prefixed with
``language_model.``. Loading is pull-based (each mcore param fetches its HF
tensors by name), so the vision tower and mm projector are skipped without any
explicit filtering — mirroring the Kimi-K3 bridge's handling of the same shell.
"""

from mbridge.core import register_model
from mbridge.models import DeepseekV3Bridge

_PREFIX = "language_model."


def _prefixed(mapping: dict) -> dict:
    return {
        mcore: (_PREFIX + hf if isinstance(hf, str) else [_PREFIX + name for name in hf])
        for mcore, hf in mapping.items()
    }


@register_model("kimi_k25")
class KimiK25Bridge(DeepseekV3Bridge):
    _DIRECT_MAPPING = _prefixed(DeepseekV3Bridge._DIRECT_MAPPING)
    _ATTENTION_MAPPING = _prefixed(DeepseekV3Bridge._ATTENTION_MAPPING)
    _MLP_MAPPING = _prefixed(DeepseekV3Bridge._MLP_MAPPING)
    _SHARED_STATE_DICT_MAPPING = _prefixed(DeepseekV3Bridge._SHARED_STATE_DICT_MAPPING)

    def __init__(self, hf_config, **kwargs):
        # All DeepseekV3Bridge config logic reads plain DSv3 field names, which
        # live on the nested text config; the wrapper keys are irrelevant here.
        super().__init__(hf_config.text_config, **kwargs)
