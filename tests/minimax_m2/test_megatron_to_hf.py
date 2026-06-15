"""MiniMax-M2.5 Megatron-to-HF name/tensor conversion checks."""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_MODULE_PATH = (
    Path(__file__).parents[2]
    / "miles"
    / "backends"
    / "megatron_utils"
    / "megatron_to_hf"
    / "minimax_m2.py"
)
_SPEC = importlib.util.spec_from_file_location("minimax_m2_converter_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
convert_minimax_m2_to_hf = _MODULE.convert_minimax_m2_to_hf


def _args():
    return types.SimpleNamespace(
        hidden_size=8,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=2,
    )


def test_direct_top_level_mappings():
    args = _args()
    weight = torch.arange(6).reshape(2, 3)

    out = convert_minimax_m2_to_hf(args, "module.module.embedding.word_embeddings.weight", weight)
    assert out[0][0] == "model.embed_tokens.weight"
    assert out[0][1] is weight

    out = convert_minimax_m2_to_hf(args, "module.module.output_layer.weight", weight)
    assert out[0][0] == "lm_head.weight"
    assert out[0][1] is weight

    out = convert_minimax_m2_to_hf(args, "module.module.decoder.final_layernorm.weight", weight)
    assert out[0][0] == "model.norm.weight"
    assert out[0][1] is weight


def test_fused_qkv_splits_to_hf_q_k_v():
    args = _args()
    fused = torch.arange(16 * args.hidden_size).reshape(16, args.hidden_size)

    out = convert_minimax_m2_to_hf(
        args,
        "module.module.decoder.layers.3.self_attention.linear_qkv.weight",
        fused,
    )

    names = [name for name, _tensor in out]
    tensors = [tensor for _name, tensor in out]
    assert names == [
        "model.layers.3.self_attn.q_proj.weight",
        "model.layers.3.self_attn.k_proj.weight",
        "model.layers.3.self_attn.v_proj.weight",
    ]
    assert tensors[0].shape == (8, args.hidden_size)
    assert tensors[1].shape == (4, args.hidden_size)
    assert tensors[2].shape == (4, args.hidden_size)

    grouped = fused.view(args.num_query_groups, -1, args.kv_channels, args.hidden_size)
    expected_q, expected_k, expected_v = torch.split(grouped, [2, 1, 1], dim=1)
    assert torch.equal(tensors[0], expected_q.reshape(-1, args.hidden_size))
    assert torch.equal(tensors[1], expected_k.reshape(-1, args.hidden_size))
    assert torch.equal(tensors[2], expected_v.reshape(-1, args.hidden_size))


def test_moe_expert_fc1_splits_gate_and_up():
    args = _args()
    fused = torch.arange(12).reshape(6, 2)

    out = convert_minimax_m2_to_hf(
        args,
        "module.module.decoder.layers.5.mlp.experts.linear_fc1.weight42",
        fused,
    )

    gate, up = fused.chunk(2, dim=0)
    assert [name for name, _tensor in out] == [
        "model.layers.5.block_sparse_moe.experts.42.w1.weight",
        "model.layers.5.block_sparse_moe.experts.42.w3.weight",
    ]
    assert torch.equal(out[0][1], gate)
    assert torch.equal(out[1][1], up)


def test_moe_expert_fc2_and_router_bias_mappings():
    args = _args()
    weight = torch.arange(6).reshape(3, 2)

    out = convert_minimax_m2_to_hf(
        args,
        "module.module.decoder.layers.5.mlp.experts.linear_fc2.weight42",
        weight,
    )
    assert out[0][0] == "model.layers.5.block_sparse_moe.experts.42.w2.weight"
    assert out[0][1] is weight

    out = convert_minimax_m2_to_hf(
        args,
        "module.module.decoder.layers.5.mlp.router.expert_bias",
        weight,
    )
    assert out[0][0] == "model.layers.5.block_sparse_moe.e_score_correction_bias"
    assert out[0][1] is weight


def test_full_dimension_qk_norm_mappings():
    args = _args()
    weight = torch.ones(8)

    out = convert_minimax_m2_to_hf(
        args,
        "module.module.decoder.layers.0.self_attention.q_norm.weight",
        weight,
    )
    assert out[0][0] == "model.layers.0.self_attn.q_norm.weight"
    assert out[0][1] is weight

    out = convert_minimax_m2_to_hf(
        args,
        "module.module.decoder.layers.61.self_attention.k_norm.weight",
        weight,
    )
    assert out[0][0] == "model.layers.61.self_attn.k_norm.weight"
    assert out[0][1] is weight
