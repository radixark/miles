"""Adapter HF names come from the converter that names the base weights — no GPU."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from tests.fast.fixtures.args_fixtures import make_trainer_config

from miles.backends.megatron_utils.lora.native.naming import ConverterNaming, DeclaredNaming
from miles.backends.megatron_utils.lora.native.registry import MODEL_SPECS


def _layer(layer_number, **weights):
    """A layer holding ``path=(shape, partition_dim or None, is_expert)`` weights, MCore-attributed."""
    layer = nn.Module()
    layer.layer_number = layer_number
    for path, (shape, partition_dim, is_expert) in weights.items():
        *modules, leaf = path.split("__")
        owner = layer
        for name in modules:
            if not hasattr(owner, name):
                owner.add_module(name, nn.Module())
            owner = getattr(owner, name)
        weight = nn.Parameter(torch.empty(shape))
        weight.tensor_model_parallel = partition_dim is not None
        weight.partition_dim = -1 if partition_dim is None else partition_dim
        if is_expert:
            weight.allreduce = False
        owner.register_parameter(leaf, weight)
    return layer


def _weight(layer, path):
    owner = layer
    for name in path.split("__"):
        owner = getattr(owner, name)
    return owner


_GQA_ARGS = make_trainer_config(
    hidden_size=8, num_attention_heads=4, num_query_groups=2, kv_channels=2, hf_checkpoint="/ckpt"
)


def _ANY(name):
    return True


def test_fused_qkv_and_gate_up_are_named_per_projection_from_their_global_shape():
    """The host is a TP shard; the converter splits the global [q;k;v] by query group."""
    layer = _layer(
        4,
        self_attention__linear_qkv__weight=((8, 8), 0, False),  # global (4 + 2*2) * 2 = 16 rows at TP=2
        self_attention__linear_proj__weight=((8, 4), 1, False),
        mlp__linear_fc1__weight=((6, 8), 0, False),
    )
    naming = ConverterNaming(_GQA_ARGS, "qwen3config", tp_size=2, is_hf_name=_ANY)

    qkv = _weight(layer, "self_attention__linear_qkv__weight")
    assert naming.block_prefix(layer, qkv, ["q_proj", "k_proj", "v_proj"]) == "model.layers.3.self_attn."
    proj = _weight(layer, "self_attention__linear_proj__weight")
    assert naming.block_prefix(layer, proj, ["o_proj"]) == "model.layers.3.self_attn."
    fc1 = _weight(layer, "mlp__linear_fc1__weight")
    assert naming.block_prefix(layer, fc1, ["gate_proj", "up_proj"]) == "model.layers.3.mlp."


def test_a_spec_that_disagrees_with_the_converter_fails():
    """GLM-4 keeps gate/up fused in HF: split adapters would be saved under names HF GLM-4 has no module for."""
    layer = _layer(1, mlp__linear_fc1__weight=((12, 8), 0, False))
    naming = ConverterNaming(_GQA_ARGS, "glm4config", tp_size=1, is_hf_name=_ANY)
    fc1 = _weight(layer, "mlp__linear_fc1__weight")

    with pytest.raises(AssertionError, match=r"\['model.layers.0.mlp.gate_up_proj'\].*\['gate_proj', 'up_proj'\]"):
        naming.block_prefix(layer, fc1, ["gate_proj", "up_proj"])
    assert naming.block_prefix(layer, fc1, ["gate_up_proj"]) == "model.layers.0.mlp."
    assert [binding.projection.hf_leaf for binding in MODEL_SPECS["glm4"].mlp.layout.singles] == [
        "gate_up_proj",
        "down_proj",
    ]


def test_kimi_k3_names_kda_experts_and_shared_experts():
    layer = _layer(
        5,
        self_attention__linear_attn__out_proj__weight=((8, 6), 1, False),
        mlp__experts__linear_fc1__weight0=((10, 4), 0, True),
        mlp__experts__linear_fc2__weight0=((4, 5), 1, True),
        mlp__shared_experts__linear_fc1__weight=((10, 8), 0, False),
    )
    naming = ConverterNaming(SimpleNamespace(), "kimi_k3", tp_size=4, is_hf_name=_ANY)
    prefix = "language_model.model.layers.4."

    out_proj = _weight(layer, "self_attention__linear_attn__out_proj__weight")
    assert naming.block_prefix(layer, out_proj, ["o_proj"]) == f"{prefix}self_attn."
    # expert weights shard over expert TP (pinned to 1), not the attention TP of 4
    fc1 = _weight(layer, "mlp__experts__linear_fc1__weight0")
    assert naming.block_prefix(layer, fc1, ["w1", "w3"]) == f"{prefix}block_sparse_moe.experts."
    fc2 = _weight(layer, "mlp__experts__linear_fc2__weight0")
    assert naming.block_prefix(layer, fc2, ["w2"]) == f"{prefix}block_sparse_moe.experts."
    shared = _weight(layer, "mlp__shared_experts__linear_fc1__weight")
    assert naming.block_prefix(layer, shared, ["gate_proj", "up_proj"]) == f"{prefix}block_sparse_moe.shared_experts."


def test_inkling_declares_sglang_module_names():
    layer = _layer(
        3,
        self_attention__linear_qkv__weight=((8, 8), 0, False),
        mlp__linear_fc1__weight=((8, 8), 0, False),
        mlp__experts__linear_fc1__weight0=((8, 8), 0, True),
        mlp__shared_experts__experts__0__linear_fc1__weight=((8, 8), 0, False),
    )
    naming = MODEL_SPECS["inkling_text"].declared_naming
    assert isinstance(naming, DeclaredNaming)

    for path, block in [
        ("self_attention__linear_qkv__weight", "attn."),
        ("mlp__linear_fc1__weight", "mlp."),
        ("mlp__experts__linear_fc1__weight0", "mlp.experts."),
        ("mlp__shared_experts__experts__0__linear_fc1__weight", "mlp.shared_experts."),
    ]:
        assert naming.block_prefix(layer, _weight(layer, path), []) == f"model.llm.layers.2.{block}"


def test_a_converter_name_outside_the_hf_namespace_fails_instead_of_never_being_selected():
    """Selectors are matched against the HF model's names: an adapter named outside them is silently
    never attached. Here the qwen3_5 converter's "model.language_model." prefix is not the model's."""
    hf_names = {f"model.layers.0.self_attn.{leaf}.weight" for leaf in ("q_proj", "k_proj", "v_proj")}
    # gated q: 2 groups x (2 q heads x 2 (gate) + k + v) x head_dim 2 = 24 rows
    layer = _layer(1, self_attention__linear_qkv__weight=((24, 8), 0, False))
    naming = ConverterNaming(_GQA_ARGS, "qwen3_5config", tp_size=1, is_hf_name=hf_names.__contains__)
    qkv = _weight(layer, "self_attention__linear_qkv__weight")

    with pytest.raises(AssertionError, match=r"model.language_model.layers.0.self_attn.q_proj.weight"):
        naming.block_prefix(layer, qkv, ["q_proj", "k_proj", "v_proj"])
