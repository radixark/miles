"""Unit tests for the MiniMax-M2.5 Megatron layer spec."""

from __future__ import annotations

import types

import pytest

pytest.importorskip("megatron")

import miles_plugins.models.minimax_m2 as minimax_m2


def _fake_block_spec(num_layers=2):
    return types.SimpleNamespace(
        layer_specs=[
            types.SimpleNamespace(
                submodules=types.SimpleNamespace(
                    self_attention=types.SimpleNamespace(module=object())
                )
            )
            for _ in range(num_layers)
        ]
    )


def test_get_minimax_m2_layer_spec_uses_custom_attention(monkeypatch):
    calls = []

    def fake_get_gpt_decoder_block_spec(config, **kwargs):
        calls.append((config, kwargs))
        return _fake_block_spec()

    monkeypatch.setattr(
        minimax_m2,
        "get_gpt_decoder_block_spec",
        fake_get_gpt_decoder_block_spec,
    )

    args = types.SimpleNamespace(transformer_impl="transformer_engine")
    config = types.SimpleNamespace()

    spec = minimax_m2.get_minimax_m2_layer_spec(args, config, vp_stage=3)

    assert calls == [(config, {"use_transformer_engine": True, "vp_stage": 3})]
    assert all(
        layer.submodules.self_attention.module is minimax_m2.MiniMaxM2SelfAttention
        for layer in spec.layer_specs
    )


def test_get_minimax_m2_layer_spec_passes_local_transformer_impl(monkeypatch):
    calls = []

    def fake_get_gpt_decoder_block_spec(config, **kwargs):
        calls.append(kwargs)
        return _fake_block_spec(num_layers=1)

    monkeypatch.setattr(
        minimax_m2,
        "get_gpt_decoder_block_spec",
        fake_get_gpt_decoder_block_spec,
    )

    args = types.SimpleNamespace(transformer_impl="local")

    minimax_m2.get_minimax_m2_layer_spec(args, types.SimpleNamespace())

    assert calls == [{"use_transformer_engine": False}]
