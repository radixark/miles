import pytest
from transformers import Qwen3MoeConfig

from miles.utils.hf_utils.weight_mapping import HfWeightMapping


@pytest.fixture(scope="module")
def hf_mapping():
    config = Qwen3MoeConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=4,
    )
    return HfWeightMapping.from_config(config)


_PREFIX = "model.layers.0.mlp.experts"
_TARGETS = [f"{_PREFIX}.gate_up_proj", f"{_PREFIX}.down_proj"]


def _unpacked_names():
    return {
        f"{_PREFIX}.{expert}.{projection}_proj.weight" for expert in range(2) for projection in ("gate", "up", "down")
    }


def test_checkpoint_formats_resolve_to_the_same_hf_parameters(hf_mapping):
    unpacked = _unpacked_names()
    assert {hf_mapping.model_parameter(name) for name in unpacked} == set(_TARGETS)
    assert {hf_mapping.model_parameter(name) for name in _TARGETS} == set(_TARGETS)
