"""Unit tests for :mod:`miles_plugins.mbridge.minimax_m2` (HF ↔ MCore naming)."""

from __future__ import annotations

import types

import pytest

pytest.importorskip("mbridge")

import miles_plugins.mbridge  # noqa: F401
from mbridge.core import auto_bridge
from miles_plugins.mbridge.minimax_m2 import MinimaxM2Bridge


def _hf_config():
    return types.SimpleNamespace(
        vocab_size=200064,
        hidden_size=3072,
        intermediate_size=1536,
        num_hidden_layers=62,
        num_attention_heads=48,
        num_key_value_heads=8,
        head_dim=128,
        rotary_dim=64,
        rope_theta=5_000_000.0,
        num_local_experts=256,
        num_experts_per_tok=8,
        use_mtp=True,
        num_mtp_modules=3,
    )


def test_minimax_m2_registered_in_mbridge():
    assert "minimax_m2" in auto_bridge._MODEL_REGISTRY
    assert auto_bridge._MODEL_REGISTRY["minimax_m2"] is MinimaxM2Bridge


def _stub_bridge():
    b = MinimaxM2Bridge.__new__(MinimaxM2Bridge)
    b.hf_config = _hf_config()
    return b


def test_maps_embed_and_norm_and_lm_head():
    b = _stub_bridge()
    assert b._weight_name_mapping_mcore_to_hf("embedding.word_embeddings.weight") == [
        "model.embed_tokens.weight"
    ]
    assert b._weight_name_mapping_mcore_to_hf("decoder.final_layernorm.weight") == ["model.norm.weight"]
    assert b._weight_name_mapping_mcore_to_hf("output_layer.weight") == ["lm_head.weight"]


def test_maps_moe_router_and_expert_bias():
    b = _stub_bridge()
    assert b._weight_name_mapping_mcore_to_hf("decoder.layers.0.mlp.router.weight") == [
        "model.layers.0.block_sparse_moe.gate.weight"
    ]
    assert b._weight_name_mapping_mcore_to_hf("decoder.layers.3.mlp.router.expert_bias") == [
        "model.layers.3.block_sparse_moe.e_score_correction_bias"
    ]


def test_maps_qkv_attention():
    b = _stub_bridge()
    out = b._weight_name_mapping_mcore_to_hf("decoder.layers.0.self_attention.linear_qkv.weight")
    assert out == [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
    ]


def test_maps_mtp_head_tensors():
    b = _stub_bridge()
    assert b._weight_name_mapping_mcore_to_hf("mtp.layers.1.enorm.weight") == [
        "model.mtp.layers.1.enorm.weight"
    ]
