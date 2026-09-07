"""Structural role-binding tests for the supported local Qwen dense spec."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("megatron.core")

from miles_plugins.top import spec as S  # noqa: E402


def _local_spec():
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

    return get_gpt_layer_local_spec(num_experts=None, qk_layernorm=True, normalization="RMSNorm")


def _bind_all(layer_spec):

    stamp: list[tuple[str, str, str]] = []
    S._bind_roles(layer_spec, S._NORM_ROLES, S.TopRMSNorm, stamp, as_spec_param=True)
    S._bind_roles(
        layer_spec, S._ROW_LINEAR_ROLES, S.TopRowParallelLinear._build(), stamp,
        as_spec_param=True,
    )
    from miles_plugins.top.attention import TopAttention

    S._bind_roles(layer_spec, S._ATTENTION_ROLES, TopAttention, stamp)
    return {role for role, _, _ in stamp}


def test_local_path_binds_the_unfused_roles():
    bound = _bind_all(_local_spec())
    assert {"input_layernorm", "pre_mlp_layernorm"} <= bound, bound
    assert {"linear_proj", "linear_fc2", "core_attention"} <= bound, bound


def test_binding_something_is_mandatory():
    """A table that matches nothing is worse than no table."""
    assert _bind_all(_local_spec())
