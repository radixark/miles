"""GPU-delta model mappings (CUDA receiver only): the one place a served SGLang model implementation plugs in.

Each module mirrors one ``sglang/srt/models/<module>.py`` implementation and is named after it;
architectures SGLang serves on that implementation share its module. To add one, write
``models/<module>.py`` with a class satisfying ``ModelMapping`` (``models/base.py``), list each
exact architecture name it serves in ``_MODEL_MAPPINGS``, the only list of admitted served
architectures, and add its checkpoints' exact config.json ``architectures[0]`` to
``_STATIC_POLICIES`` (``miles/utils/gpu_delta/publication.py``), the only list of admitted
checkpoint architectures; admission requires both. Names match exactly, never by subclass or
substring, and an unlisted architecture is rejected: there is no generic mapping, because a
shape-compatible binding can still address the wrong bytes.
"""

from __future__ import annotations

from miles.backends.sglang_utils.gpu_delta.models.base import ModelMapping
from miles.backends.sglang_utils.gpu_delta.models.deepseek_v2 import DeepseekV2Mapping
from miles.backends.sglang_utils.gpu_delta.models.deepseek_v4 import DeepseekV4Mapping
from miles.backends.sglang_utils.gpu_delta.models.kimi_k3 import KimiK3Mapping

_MODEL_MAPPINGS = {
    "DeepseekV2ForCausalLM": DeepseekV2Mapping,
    "DeepseekV3ForCausalLM": DeepseekV2Mapping,
    "DeepseekV32ForCausalLM": DeepseekV2Mapping,
    "GlmMoeDsaForCausalLM": DeepseekV2Mapping,
    "KimiK3ForConditionalGeneration": KimiK3Mapping,
    # SGLang serves DeepSeek-V4.1 checkpoints (DeepseekV41ForCausalLM) under this name; the mapping
    # admits model type deepseek_v41 only.
    "DeepseekV4ForCausalLM": DeepseekV4Mapping,
}


def model_mapping(model, shadows) -> ModelMapping:
    """The listed mapping of ``model``'s exact architecture, given the session's ``CanonicalShadows``."""
    architecture = model.config.architectures[0]
    if architecture not in _MODEL_MAPPINGS:
        raise ValueError(f"unsupported GPU delta model mapping: {architecture}; supported: {sorted(_MODEL_MAPPINGS)}")
    return _MODEL_MAPPINGS[architecture](model, shadows)
