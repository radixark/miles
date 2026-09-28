"""Raw-mode conversion of a megatron.bridge VL checkpoint must say how to fix it (#634)."""

from argparse import Namespace

import pytest
import torch

from miles.backends.megatron_utils.megatron_to_hf import _convert_to_hf_core


@pytest.mark.parametrize(
    "name",
    [
        "module.module.vision_model.patch_embed.proj.weight",
        "module.module.vision_projector.encoder.linear_fc1.weight",
        "module.module.language_model.embedding.word_embeddings.weight",
    ],
)
def test_bridge_only_param_names_point_at_bridge_mode(name):
    args = Namespace(hidden_size=4, kv_channels=2, num_attention_heads=2, num_query_groups=1)
    with pytest.raises(ValueError, match=r"--megatron-to-hf-mode bridge") as excinfo:
        _convert_to_hf_core(args, "qwen3", name, torch.zeros(1))
    assert f"Unknown parameter name: {name}" in str(excinfo.value)


def test_converter_that_knows_the_wrapper_is_left_alone():
    param = torch.zeros(1)
    assert _convert_to_hf_core(Namespace(), "kimivl", "module.module.vision_model.encoder.layers.0.w", param) == [
        ("vision_tower.encoder.layers.0.w", param)
    ]


def test_plain_decoder_param_still_reaches_the_model_converter():
    args = Namespace(hidden_size=4, kv_channels=2, num_attention_heads=2, num_query_groups=1)
    param = torch.ones(4)
    assert _convert_to_hf_core(args, "qwen3", "module.module.decoder.layers.0.input_layernorm.weight", param) == [
        ("model.layers.0.input_layernorm.weight", param)
    ]
