"""Integration checks for MiniMax-M2.5 + miles mbridge (offline, no weights)."""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

pytest.importorskip("mbridge")

import miles_plugins.mbridge  # noqa: F401
from mbridge.core import auto_bridge
from miles_plugins.mbridge.minimax_m2 import MiniMaxM2Bridge, MinimaxM2Bridge

M2_5_CONFIG_PATH = Path("/home/yangchengyi/data/models/MiniMax-M2.5/config.json")


@pytest.fixture
def real_hf_config():
    if not M2_5_CONFIG_PATH.exists():
        pytest.skip(f"{M2_5_CONFIG_PATH} not present; integration test skipped")
    data = json.loads(M2_5_CONFIG_PATH.read_text())
    return types.SimpleNamespace(**data)


def test_plugin_chain_imports_cleanly():
    import miles_plugins.mbridge.minimax_m2 as mb  # noqa: F401

    pytest.importorskip("megatron")
    import miles_plugins.models.minimax_m2 as ms  # noqa: F401

    assert hasattr(mb, "MiniMaxM2Bridge")
    assert hasattr(mb, "MinimaxM2Bridge")
    assert mb.MinimaxM2Bridge is mb.MiniMaxM2Bridge
    assert hasattr(ms, "MiniMaxM2SelfAttention")
    assert hasattr(ms, "get_minimax_m2_layer_spec")


def test_mbridge_registry_has_minimax_m2():
    assert auto_bridge._MODEL_REGISTRY.get("minimax_m2") is MiniMaxM2Bridge
    assert MinimaxM2Bridge is MiniMaxM2Bridge


def test_name_mapping_matches_released_ckpt_layout(real_hf_config):
    """Spot-check a few tensors against the public HF naming convention."""
    b = MinimaxM2Bridge.__new__(MinimaxM2Bridge)
    b.hf_config = real_hf_config
    assert real_hf_config.model_type == "minimax_m2"

    assert b._weight_name_mapping_mcore_to_hf("decoder.layers.61.mlp.router.weight") == [
        "model.layers.61.block_sparse_moe.gate.weight"
    ]
    assert b._weight_name_mapping_mcore_to_hf("decoder.layers.0.self_attention.q_norm.weight") == [
        "model.layers.0.self_attn.q_norm.weight"
    ]


def test_get_minimax_m2_layer_spec_replaces_attention_module(monkeypatch):
    """The slime path keeps MCore q_layernorm/k_layernorm specs but replaces attention."""
    pytest.importorskip("megatron")
    import miles_plugins.models.minimax_m2 as ms

    block_spec = types.SimpleNamespace(
        layer_specs=[
            types.SimpleNamespace(
                submodules=types.SimpleNamespace(
                    self_attention=types.SimpleNamespace(module=object())
                )
            )
        ]
    )

    def fake_get_gpt_decoder_block_spec(config, **kwargs):
        assert kwargs == {"use_transformer_engine": True}
        return block_spec

    monkeypatch.setattr(ms, "get_gpt_decoder_block_spec", fake_get_gpt_decoder_block_spec)

    args = types.SimpleNamespace(transformer_impl="transformer_engine")
    out = ms.get_minimax_m2_layer_spec(args, types.SimpleNamespace())

    assert out.layer_specs[0].submodules.self_attention.module is ms.MiniMaxM2SelfAttention
