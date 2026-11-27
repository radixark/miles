from types import SimpleNamespace

import torch

from miles.backends.megatron_utils.megatron_to_hf.deepseekv3 import convert_deepseekv3_to_hf

ARGS = SimpleNamespace(kv_channels=None, hidden_size=64, num_attention_heads=4, num_query_groups=4)
PREFIX = "module.module.decoder.layers.4"


def test_sequential_mlp_expert_splits_gate_and_up():
    out = convert_deepseekv3_to_hf(ARGS, f"{PREFIX}.mlp.experts.local_experts.7.linear_fc1.weight", torch.zeros(4, 2))
    assert [name for name, _ in out] == [
        "model.layers.4.mlp.experts.7.gate_proj.weight",
        "model.layers.4.mlp.experts.7.up_proj.weight",
    ]


def test_standalone_q_and_kv_layernorms():
    q = convert_deepseekv3_to_hf(ARGS, f"{PREFIX}.self_attention.q_layernorm.weight", torch.ones(8))
    kv = convert_deepseekv3_to_hf(ARGS, f"{PREFIX}.self_attention.kv_layernorm.weight", torch.ones(8))
    assert [name for name, _ in q] == ["model.layers.4.self_attn.q_a_layernorm.weight"]
    assert [name for name, _ in kv] == ["model.layers.4.self_attn.kv_a_layernorm.weight"]
