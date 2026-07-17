from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.megatron_to_hf.kimi_k3 import (
    convert_kimi_k3_to_hf,
    get_kimi_k3_atomic_update_groups,
)
from miles.backends.megatron_utils.update_weight.common import get_named_update_units


def _convert(name, shape=(4, 2)):
    param = torch.arange(torch.tensor(shape).prod()).reshape(shape)
    return convert_kimi_k3_to_hf(SimpleNamespace(), name, param)


@pytest.mark.parametrize(
    ("megatron_name", "hf_name"),
    [
        (
            "module.module.decoder.layers.0.self_attention.q_proj.weight",
            "language_model.model.layers.0.self_attn.q_proj.weight",
        ),
        (
            "module.module.decoder.layers.1.mlp.fc1_latent_proj.weight",
            "language_model.model.layers.1.block_sparse_moe.routed_expert_down_proj.weight",
        ),
        (
            "module.module.decoder.layers.3.output_attn_res_proj.weight",
            "language_model.model.output_attn_res_proj.weight",
        ),
    ],
)
def test_direct_kimi_k3_conversion(megatron_name, hf_name):
    converted = _convert(megatron_name)
    assert [name for name, _param in converted] == [hf_name]


def test_dense_fc1_splits_gate_and_up():
    converted = _convert("module.module.decoder.layers.0.mlp.linear_fc1.weight")
    assert [name for name, _param in converted] == [
        "language_model.model.layers.0.mlp.gate_proj.weight",
        "language_model.model.layers.0.mlp.up_proj.weight",
    ]
    assert converted[0][1].tolist() == [[0, 1], [2, 3]]
    assert converted[1][1].tolist() == [[4, 5], [6, 7]]


def test_expert_fc1_splits_w1_and_w3():
    converted = _convert("module.module.decoder.layers.2.mlp.experts.linear_fc1.weight17")
    assert [name for name, _param in converted] == [
        "language_model.model.layers.2.block_sparse_moe.experts.17.w1.weight",
        "language_model.model.layers.2.block_sparse_moe.experts.17.w3.weight",
    ]


def test_unknown_name_fails_explicitly():
    with pytest.raises(ValueError, match="Unknown Kimi K3 layer parameter"):
        _convert("module.module.decoder.layers.0.not_a_parameter")


def test_kimi_k3_atomic_group_keeps_fused_qkv_a_together():
    prefix = "module.module.decoder.layers.3.self_attention"
    names = [
        f"{prefix}.q_a_layernorm.weight",
        f"{prefix}.q_a_proj.weight",
        f"{prefix}.kv_a_proj_with_mqa.weight",
        f"{prefix}.q_b_proj.weight",
    ]

    units = get_named_update_units(names, get_kimi_k3_atomic_update_groups())

    assert [unit.names for unit in units] == [
        (f"{prefix}.q_a_layernorm.weight",),
        (f"{prefix}.q_a_proj.weight", f"{prefix}.kv_a_proj_with_mqa.weight"),
        (f"{prefix}.q_b_proj.weight",),
    ]
