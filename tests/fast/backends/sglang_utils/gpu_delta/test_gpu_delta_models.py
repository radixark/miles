"""Model-mapping registry: exact architecture names and no generic fallback."""

from types import SimpleNamespace

import pytest
import torch

from miles.backends.sglang_utils.gpu_delta.bindings import CanonicalShadows
from miles.backends.sglang_utils.gpu_delta.models import model_mapping
from miles.backends.sglang_utils.gpu_delta.models.deepseek_v2 import DeepseekV2Mapping


def _model(architecture):
    model = torch.nn.Module()
    model.config = SimpleNamespace(architectures=[architecture])
    return model


def test_registered_architecture_builds_its_runtime_mapping():
    shadows = CanonicalShadows({}, create=True)
    mapping = model_mapping(_model("GlmMoeDsaForCausalLM"), shadows)
    assert isinstance(mapping, DeepseekV2Mapping)
    # Every mapping is constructed with the session's shadows, which reach its binding context.
    assert mapping.parameters.shadows is shadows


# MistralLarge3 subclasses DeepseekV3ForCausalLM but renames checkpoint tensors, and Glm4MoeLite
# reuses the DeepSeek loader mixin without qualification: neither may match by subclass or name.
@pytest.mark.parametrize("architecture", ["MistralLarge3ForCausalLM", "Glm4MoeLiteForCausalLM"])
def test_unlisted_architecture_is_rejected_with_the_supported_list(architecture):
    with pytest.raises(ValueError, match=f"unsupported GPU delta model mapping: {architecture}; supported: "):
        model_mapping(_model(architecture), None)
