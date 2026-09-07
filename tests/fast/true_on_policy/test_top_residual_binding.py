"""Residual sites must share one paired path across family-specific specs."""

import copy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("megatron.core")

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer

from miles_plugins.top import spec as S
from miles_plugins.top.residual_norm import (
    ResidualTransformerLayer,
    deferred_residual_add,
    validate_residual_config,
)


def _config(**overrides):
    return SimpleNamespace(**{
        "hidden_size": 128, "pipeline_hidden_size": None,
        "fp32_residual_connection": False, "hidden_dropout": 0,
        "add_bias_linear": False, **overrides,
    })


def _layer():
    return get_gpt_layer_local_spec(num_experts=None, qk_layernorm=True, normalization="RMSNorm")


@pytest.fixture
def on_policy_server_args(monkeypatch):
    from sglang.srt.true_on_policy import config

    monkeypatch.setattr(
        config, "_get_global_server_args",
        lambda: SimpleNamespace(true_on_policy_contract="true_on_policy_v1"),
    )


def test_residual_binding_preserves_family_submodules():
    layer = _layer()
    attention, mlp = layer.submodules.self_attention, layer.submodules.mlp
    params, key_map = copy.deepcopy(layer.params), copy.deepcopy(layer.submodules.sharded_state_dict_keys_map)
    stamp = []
    S._bind_residual_layer(layer, stamp)
    assert layer.module is ResidualTransformerLayer
    assert layer.module.forward is TransformerLayer.forward
    assert layer.submodules.self_attn_bda is deferred_residual_add
    assert layer.submodules.mlp_bda is deferred_residual_add
    assert layer.submodules.self_attention is attention
    assert layer.submodules.mlp is mlp
    assert layer.params == params
    assert layer.submodules.sharded_state_dict_keys_map == key_map
    assert sum(role == "residual_add" for role, _, _ in stamp) == 1
    for role in ("input_layernorm", "pre_mlp_layernorm"):
        norm = getattr(layer.submodules, role)
        assert norm.module is S.TopRMSNorm
        assert norm.params == {"role": role}


@pytest.mark.parametrize("role", ["input_layernorm", "pre_mlp_layernorm"])
def test_unexposed_residual_norm_is_not_a_fallback(role):
    layer = _layer()
    setattr(layer.submodules, role, IdentityOp)
    with pytest.raises(NotImplementedError, match=role):
        S._bind_residual_layer(layer, [])


def test_custom_layer_forward_is_not_silently_replaced():
    with pytest.raises(NotImplementedError, match="stock TransformerLayer"):
        S._bind_residual_layer(ModuleSpec(module=torch.nn.Identity), [])


@pytest.mark.parametrize("role", ["input_layernorm", "pre_mlp_layernorm", "final_layernorm"])
@pytest.mark.parametrize("width", [None, 128, 384])
def test_residual_roles_require_paired_transport(role, width):
    with pytest.raises(RuntimeError, match="requires packed residual transport"):
        S.TopRMSNorm(_config(pipeline_hidden_size=width), 128, role=role)


@pytest.mark.parametrize("family", ["qwen3"])
def test_configuration_is_validated_before_family_builder(monkeypatch, family):
    monkeypatch.setattr(S, "_pin_sglang", lambda args: "true_on_policy_v1")
    monkeypatch.setattr(S, "_hf_model_type", lambda args: family)
    with pytest.raises(NotImplementedError, match="transformer_engine spec"):
        S.get_top_spec(SimpleNamespace(transformer_impl="transformer_engine"), _config(), None)


def test_supported_configuration_sets_transport_width():
    config = _config()
    validate_residual_config(config, use_te=False)
    assert config.pipeline_hidden_size == 256


@pytest.mark.parametrize("role", ["pre_mlp_layernorm", "final_layernorm"])
def test_unpaired_payload_at_residual_site_raises(role, on_policy_server_args):
    norm = S.TopRMSNorm(_config(pipeline_hidden_size=256), 128, role=role)
    with pytest.raises(ValueError, match="expected a packed residual pair"):
        norm(torch.empty(2, 128, dtype=torch.bfloat16))


def test_nonfirst_input_norm_requires_a_pair(on_policy_server_args):
    norm = S.TopRMSNorm(_config(pipeline_hidden_size=256), 128, role="input_layernorm")
    norm.requires_residual_pair = True
    with pytest.raises(ValueError, match="expected a packed residual pair"):
        norm(torch.empty(2, 128, dtype=torch.bfloat16))
