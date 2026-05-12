"""Integration checks for MiniMax-M2.5 + miles mbridge (offline, no weights)."""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

pytest.importorskip("mbridge")

import miles_plugins.mbridge  # noqa: F401
from mbridge.core import auto_bridge
from miles_plugins.mbridge.minimax_m2 import MinimaxM2Bridge

M2_5_CONFIG_PATH = Path("/home/yangchengyi/data/models/MiniMax-M2.5/config.json")


@pytest.fixture
def real_hf_config():
    if not M2_5_CONFIG_PATH.exists():
        pytest.skip(f"{M2_5_CONFIG_PATH} not present; integration test skipped")
    data = json.loads(M2_5_CONFIG_PATH.read_text())
    return types.SimpleNamespace(**data)


def test_plugin_chain_imports_cleanly():
    import miles_plugins.mbridge.minimax_m2 as mb  # noqa: F401
    import miles_plugins.models.minimax_m2 as ms  # noqa: F401

    assert hasattr(mb, "MinimaxM2Bridge")
    assert hasattr(ms, "_PerLayerRMSNorm")
    assert hasattr(ms, "get_minimax_m2_spec")


def test_mbridge_registry_has_minimax_m2():
    assert auto_bridge._MODEL_REGISTRY.get("minimax_m2") is MinimaxM2Bridge


def test_name_mapping_matches_released_ckpt_layout(real_hf_config):
    """Spot-check a few tensors against the public HF naming convention."""
    b = MinimaxM2Bridge.__new__(MinimaxM2Bridge)
    b.hf_config = real_hf_config
    assert real_hf_config.model_type == "minimax_m2"

    assert b._weight_name_mapping_mcore_to_hf("decoder.layers.61.mlp.router.weight") == [
        "model.layers.61.block_sparse_moe.gate.weight"
    ]
    assert b._weight_name_mapping_mcore_to_hf(
        "decoder.layers.0.self_attention.q_layernorm.weight"
    ) == ["model.layers.0.self_attn.q_norm.weight"]


def test_get_minimax_m2_layer_spec_runs_without_megatron_dist():
    """Sanity: get_minimax_m2_layer_spec must work on CPU/no-dist (used by tests)."""
    from miles_plugins.models.minimax_m2 import get_minimax_m2_layer_spec

    cfg = types.SimpleNamespace(
        num_attention_heads=48,
        num_query_groups=8,
        kv_channels=128,
        hidden_size=3072,
        num_moe_experts=256,
        moe_grouped_gemm=True,
    )
    layer_spec = get_minimax_m2_layer_spec(cfg)
    sub = layer_spec.submodules.self_attention.submodules
    from miles_plugins.models.minimax_m2 import _PerLayerRMSNorm

    assert sub.q_layernorm.module is _PerLayerRMSNorm
    assert sub.k_layernorm.module is _PerLayerRMSNorm
