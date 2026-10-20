"""Muse Spark native LoRA: the gated, rotary-pair q/g/k/v layout against its converter, and its targets — no GPU."""

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from tests.fast.backends.megatron_utils.lora.native.naming_stub import named

from miles.backends.megatron_utils.lora.native.config import LoRAConfig
from miles.backends.megatron_utils.lora.native.merge import merge_lora_into_weights
from miles.backends.megatron_utils.lora.native.modules.linear import build_qgkv_rotary_pair_permutation
from miles.backends.megatron_utils.lora.native.registry import resolve_adapter_targets, validate_native_lora_args
from miles.backends.megatron_utils.lora.native.spec.attention import MuseSparkAttentionSpec
from miles.backends.megatron_utils.lora.native.spec.base import AttachContext
from miles.utils.lora.hf_lora_targets import expand_hf_lora_targets, get_hf_lora_targets, resolve_hf_lora_targets
from miles_plugins.models.muse_spark.weight_mapping import MuseSpark1xParamMapping

_SPARK = SimpleNamespace(num_attention_heads=4, num_key_value_heads=2, head_dim=4, hidden_size=8, use_attn_o_gate=True)
_PREFIX = "model.language_model.layers.0.self_attn."
_LAYERS = "model.language_model.layers.*."
_HF_CONFIG = {
    "model_type": "muse_spark",
    "text_config": {"model_type": "muse_spark_text", "num_hidden_layers": 2, "mlp_layer_types": ["dense", "sparse"]},
}


def _qgkv_mapping(tp: int = 1) -> MuseSpark1xParamMapping:
    mapping = MuseSpark1xParamMapping("linear_qkv", {key: key for key in "qgkv"}, _SPARK, hf_specs={}, kind="qgkv")
    mapping.partition_size = tp
    return mapping


@pytest.mark.parametrize("tp", [1, 2])
def test_packed_checkpoint_rows_land_where_the_converter_puts_them(tp):
    """A TP shard of the checkpoint's q/g/k/v rows, packed by the permutation, is the converter's MCore shard."""
    q_rows = _SPARK.num_attention_heads // tp * _SPARK.head_dim
    kv_rows = _SPARK.num_key_value_heads // tp * _SPARK.head_dim
    shapes = dict(q=q_rows, g=q_rows, k=kv_rows, v=kv_rows)
    hf_rows = {key: torch.randn(rows, _SPARK.hidden_size) for key, rows in shapes.items()}
    permutation = build_qgkv_rotary_pair_permutation(
        _SPARK.num_attention_heads // tp, _SPARK.num_key_value_heads // tp, _SPARK.head_dim, "cpu"
    )

    packed = torch.cat([hf_rows[key] for key in "qgkv"]).index_select(0, permutation)

    assert torch.equal(packed, _qgkv_mapping(tp).hf_slices_to_megatron(hf_rows))


class _Linear(nn.Module):
    def __init__(self, out_features, in_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        return x @ self.weight.t(), None


def _attention_with_lora():
    attention = nn.Module()
    q_rows, kv_rows = _SPARK.num_attention_heads * _SPARK.head_dim, _SPARK.num_key_value_heads * _SPARK.head_dim
    attention.linear_qkv = _Linear(2 * q_rows + 2 * kv_rows, _SPARK.hidden_size)
    attention.linear_proj = _Linear(_SPARK.hidden_size, q_rows)
    attention.num_attention_heads_per_partition = _SPARK.num_attention_heads
    attention.num_query_groups_per_partition = _SPARK.num_key_value_heads
    attention.hidden_size_per_attention_head = _SPARK.head_dim
    context = AttachContext(
        lora=LoRAConfig(rank=2, alpha=6, dropout=0.0, target_modules=None),
        transformer_config=SimpleNamespace(
            hidden_size=_SPARK.hidden_size, sequence_parallel=False, attention_output_gate=True
        ),
        tp_size=1,
        tp_rank=0,
        naming=None,
    )
    MuseSparkAttentionSpec().attach(attention, None, named(context, _PREFIX))
    torch.manual_seed(1)
    with torch.no_grad():
        for name, parameter in attention.named_parameters():
            if "lora" in name:
                parameter.normal_()
    return attention, context.lora.scale


def test_each_adapter_is_its_checkpoint_projections_update():
    """Merged into the fused host and converted, each HF projection moves by exactly scale * B @ A."""
    attention, scale = _attention_with_lora()
    adapter = attention.lora_qkv_adapter
    weights = {name: parameter for name, parameter in attention.named_parameters() if "lora" not in name}
    merged = merge_lora_into_weights([attention], weights)["linear_qkv.weight"]

    base = _qgkv_mapping().megatron_to_hf(attention.linear_qkv.weight.detach())
    updated = _qgkv_mapping().megatron_to_hf(merged)

    exports = list(adapter.exports())
    assert [export.hf_name for export in exports] == [
        f"{_PREFIX}{leaf}" for leaf in ("q_proj", "gate_proj", "k_proj", "v_proj")
    ]
    for key, export in zip("qgkv", exports, strict=True):
        torch.testing.assert_close(updated[key] - base[key], scale * export.b.detach() @ export.a.detach())


def test_the_fused_forward_is_the_merged_weights_forward():
    attention, _ = _attention_with_lora()
    weights = {name: parameter for name, parameter in attention.named_parameters() if "lora" not in name}
    merged = merge_lora_into_weights([attention], weights)
    with torch.no_grad():
        for host in ("linear_qkv", "linear_proj"):
            x = torch.randn(5, getattr(attention, host).weight.shape[1])
            torch.testing.assert_close(
                getattr(attention, host)(x)[0], x @ merged[f"{host}.weight"].t(), rtol=1e-4, atol=1e-4
            )


@pytest.mark.parametrize(
    "groups, modules",
    [
        (
            ["attn"],
            ["self_attn.q_proj", "self_attn.gate_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
        ),
        (
            ["mlp"],
            [
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
                "mlp.experts.*.gate_proj",
                "mlp.experts.*.up_proj",
                "mlp.experts.*.down_proj",
            ],
        ),
    ],
    ids=["attention", "mlp-and-routed-experts"],
)
def test_target_groups_resolve_to_projections_the_spec_attaches(groups, modules):
    targets = expand_hf_lora_targets(
        resolve_hf_lora_targets(_HF_CONFIG, target_modules=groups), get_hf_lora_targets(_HF_CONFIG)
    )
    assert targets == [f"{_LAYERS}{module}" for module in modules]
    # the checkpoint is a custom HF implementation, so the target selectors themselves are checked
    assert (
        resolve_adapter_targets(_HF_CONFIG, targets, hf_modules=[], lora_type="lora", experts_shared_outer_loras=True)
        == targets
    )


def test_the_lm_head_and_per_expert_adapters_are_refused():
    with pytest.raises(AssertionError, match="does not implement adapters"):
        resolve_adapter_targets(
            _HF_CONFIG, ["lm_head"], hf_modules=[], lora_type="lora", experts_shared_outer_loras=True
        )
    with pytest.raises(AssertionError, match="--experts-shared-outer-loras"):
        resolve_adapter_targets(
            _HF_CONFIG,
            [f"{_LAYERS}self_attn.q_proj"],
            hf_modules=[],
            lora_type="lora",
            experts_shared_outer_loras=False,
        )


def test_routed_expert_adapters_need_expert_tp_1(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_HF_CONFIG))
    args = SimpleNamespace(
        hf_checkpoint=str(tmp_path),
        lora_rank=8,
        colocate=True,
        overlap_param_gather=False,
        moe_shared_expert_overlap=False,
        overlap_grad_reduce=False,
        expert_tensor_parallel_size=1,
    )
    validate_native_lora_args(args)
    args.expert_tensor_parallel_size = 2
    with pytest.raises(AssertionError, match="expert-tensor-parallel-size"):
        validate_native_lora_args(args)
