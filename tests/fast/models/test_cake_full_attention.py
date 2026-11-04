"""--full-attention-backend cake: the layer-spec swap and the startup validation (CPU).

The numerics of the swapped layer run in tests/fast-gpu/test_cake_full_attention.py, the kernels' own in the
miles-ext-kernels submodule (3rdparty/miles-ext-kernels/tests/gqa).
"""

from types import SimpleNamespace

import pytest
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from tests.fast.fixtures.args_fixtures import parse_megatron_test_config

from miles_plugins.models.cake_full_attention import (
    CakeGQACoreAttention,
    apply_cake_full_attention,
    validate_full_attention_backend,
)


def _layer_spec(core_attention=TEDotProductAttention) -> ModuleSpec:
    attention = ModuleSpec(
        module=SelfAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=SelfAttentionSubmodules(
            linear_qkv="linear_qkv", core_attention=core_attention, linear_proj="linear_proj", q_layernorm="q_norm"
        ),
    )
    return ModuleSpec(
        module=TransformerLayer, submodules=TransformerLayerSubmodules(self_attention=attention, mlp="mlp")
    )


def test_the_swap_replaces_only_the_core_attention_of_a_copy():
    shared = _layer_spec()

    swapped = apply_cake_full_attention(shared)

    attention = swapped.submodules.self_attention
    assert attention.submodules.core_attention is CakeGQACoreAttention
    assert (attention.module, attention.params) == (SelfAttention, {"attn_mask_type": AttnMaskType.causal})
    assert (attention.submodules.linear_qkv, attention.submodules.linear_proj) == ("linear_qkv", "linear_proj")
    assert attention.submodules.q_layernorm == "q_norm"
    assert (swapped.module, swapped.submodules.mlp) == (TransformerLayer, "mlp")
    # Megatron shares one layer spec between layers: the original keeps Transformer Engine's attention
    assert shared.submodules.self_attention.submodules.core_attention is TEDotProductAttention


def test_the_swap_rejects_a_layer_without_transformer_engine_attention():
    with pytest.raises(ValueError, match="replaces TEDotProductAttention"):
        apply_cake_full_attention(_layer_spec(core_attention="DotProductAttention"))


def _args(**overrides) -> SimpleNamespace:
    """Parsed arguments of a Qwen3.5-27B run at TP4 (6 query heads over 1 key/value head per rank)."""
    values = dict(
        full_attention_backend="cake",
        train_backend="megatron",
        spec=["miles_plugins.models.qwen3_5", "get_qwen3_5_spec"],
        qkv_format="thd",
        megatron_to_hf_mode="raw",
        kv_channels=256,
        num_attention_heads=24,
        group_query_attention=True,
        num_query_groups=4,
        tensor_model_parallel_size=4,
        context_parallel_size=1,
        attention_dropout=0.0,
        window_size=None,
        deterministic_mode=False,
    )
    return SimpleNamespace(**(values | overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"num_attention_heads": 16, "num_query_groups": 2, "tensor_model_parallel_size": 1},  # 16 / 2
        {"num_attention_heads": 16, "num_query_groups": 2, "tensor_model_parallel_size": 2},  # 8 / 1
        {"num_attention_heads": 16, "num_query_groups": 4, "tensor_model_parallel_size": 2},  # 8 / 2
        {"spec": ["miles_plugins.models.qwen3_next", "get_qwen3_next_spec"]},
    ],
)
def test_the_supported_settings_validate(overrides):
    validate_full_attention_backend(_args(**overrides))


def test_the_megatron_backend_is_not_checked():
    validate_full_attention_backend(_args(full_attention_backend="megatron", kv_channels=128, context_parallel_size=2))


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"context_parallel_size": 2}, "--context-parallel-size 2"),
        ({"kv_channels": 128}, "--kv-channels 128"),
        ({"num_attention_heads": 24, "num_query_groups": 4, "tensor_model_parallel_size": 8}, "3 query heads over 1"),
        ({"num_attention_heads": 12, "num_query_groups": 4, "tensor_model_parallel_size": 1}, "12 query heads over 4"),
        # without --group-query-attention every query head has its own key/value head
        ({"group_query_attention": False, "num_query_groups": 1}, "6 query heads over 6"),
        ({"attention_dropout": 0.1}, "--attention-dropout 0.1"),
        ({"window_size": (128, 0)}, "--window-size"),
        ({"deterministic_mode": True}, "--deterministic-mode"),
        ({"qkv_format": "bshd"}, "--qkv-format bshd"),
        ({"megatron_to_hf_mode": "bridge"}, "--megatron-to-hf-mode bridge"),
        ({"spec": ["miles_plugins.models.glm4", "get_glm_spec"]}, "--spec miles_plugins.models.glm4"),
        ({"spec": None}, "--spec None"),
    ],
)
def test_unsupported_settings_are_rejected_at_startup(overrides, message):
    with pytest.raises(ValueError, match="--full-attention-backend cake does not support") as excinfo:
        validate_full_attention_backend(_args(**overrides))

    assert message in str(excinfo.value)


def test_every_unsupported_setting_is_named_at_once():
    with pytest.raises(ValueError) as excinfo:
        validate_full_attention_backend(_args(context_parallel_size=2, kv_channels=128))

    assert "--context-parallel-size 2" in str(excinfo.value) and "--kv-channels 128" in str(excinfo.value)


def test_another_training_backend_is_rejected():
    with pytest.raises(ValueError, match="needs --train-backend megatron, not fsdp"):
        validate_full_attention_backend(_args(train_backend="fsdp"))


def test_the_full_parse_runs_the_startup_rules():
    """parse_args calls validate_full_attention_backend: the test model is not a Qwen3.5 / Qwen3-Next spec."""
    with pytest.raises(ValueError, match="--full-attention-backend cake does not support"):
        parse_megatron_test_config("--use-miles-ext-kernels", "--full-attention-backend", "cake")
