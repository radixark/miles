from types import SimpleNamespace

import torch

from miles.backends.megatron_utils.megatron_to_hf.kimi_k3 import (
    convert_kimi_k3_to_hf,
    get_kimi_k3_atomic_update_groups,
)
from miles.backends.megatron_utils.update_weight.common import get_named_update_units


def _convert(name, shape=(4, 2)):
    param = torch.arange(torch.tensor(shape).prod()).reshape(shape)
    return convert_kimi_k3_to_hf(SimpleNamespace(), name, param)


def test_fused_fc1_splits_gate_before_up():
    """Megatron fuses gate and up into one linear_fc1; HF wants them separate.
    Which half is which is pure convention -- swapping them produces a model
    that runs and generates plausible-looking garbage, on both the dense MLP
    and the per-expert path.
    """
    dense = _convert("module.module.decoder.layers.0.mlp.linear_fc1.weight")
    assert [name for name, _param in dense] == [
        "language_model.model.layers.0.mlp.gate_proj.weight",
        "language_model.model.layers.0.mlp.up_proj.weight",
    ]
    assert dense[0][1].tolist() == [[0, 1], [2, 3]]
    assert dense[1][1].tolist() == [[4, 5], [6, 7]]

    expert = _convert("module.module.decoder.layers.2.mlp.experts.linear_fc1.weight17")
    assert [name for name, _param in expert] == [
        "language_model.model.layers.2.block_sparse_moe.experts.17.w1.weight",
        "language_model.model.layers.2.block_sparse_moe.experts.17.w3.weight",
    ]
    assert expert[0][1].tolist() == [[0, 1], [2, 3]]
    assert expert[1][1].tolist() == [[4, 5], [6, 7]]


def test_kimi_k3_atomic_group_keeps_fused_qkv_a_together():
    """q_a_proj and kv_a_proj_with_mqa are one fused GEMM on the rollout side,
    so they have to land in the same update unit. If the streaming chunker
    splits them the engine sees a half-updated fusion between the two chunks.
    """
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
