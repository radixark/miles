"""MiMo-V2 LoRA through the Bridge plugin: the HF target groups reach every adapted module, and a
``linear_qkv`` LoRA B splits back into the HF q/k/v rows of both global and SWA layers."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.lora.target_modules import resolve_megatron_lora_targets
from miles.utils.hf_utils.weight_mapping import HfWeightMapping
from miles.utils.lora.hf_lora_targets import resolve_hf_lora_targets

_PLUGIN = Path(__file__).resolve().parents[4] / "miles_plugins/megatron_bridge/mimo_v2.py"
# MiMo-V2.6-Flash: 64 query heads of 192, value heads of 128; global layers have 4 KV heads, SWA layers 8.
_HEADS, _QK_DIM, _V_DIM, _GLOBAL_GROUPS, _SWA_GROUPS = 64, 192, 128, 4, 8


class _Mapping:
    def __init__(self, megatron_param, hf_param=None, **projections):
        self.megatron_param = megatron_param
        self.hf_param = projections if hf_param is None else hf_param

    def resolve(self, captures):
        def expand(name):
            for value in captures:
                name = name.replace("*", value, 1)
            return name

        if isinstance(self.hf_param, str):
            return _Mapping(expand(self.megatron_param), expand(self.hf_param))
        return _Mapping(expand(self.megatron_param), {key: expand(name) for key, name in self.hf_param.items()})

    @classmethod
    def register_module_type(cls, *args):
        pass


class _Registry(list):
    def __init__(self, *mappings):
        super().__init__(mappings)


class _ModelBridge:
    @classmethod
    def register_bridge(cls, **kwargs):
        return lambda bridge: bridge


@pytest.fixture
def plugin(monkeypatch):
    stubs = {
        "megatron.bridge.models.conversion.mapping_registry": {"MegatronMappingRegistry": _Registry},
        "megatron.bridge.models.conversion.model_bridge": {"MegatronModelBridge": _ModelBridge},
        "megatron.bridge.models.conversion.param_mapping": {"AutoMapping": _Mapping, "GatedMLPMapping": _Mapping},
        "megatron.bridge.models.gpt_provider": {"GPTModelProvider": type("GPTModelProvider", (), {})},
        "megatron.bridge.models.mimo_v2_flash.mimo_v2_flash_bridge": {"MiMoV2FlashQKVMapping": _Mapping},
        "megatron.core.extensions.transformer_engine": {
            "TEDotProductAttention": type("TEDotProductAttention", (), {})
        },
        "megatron.core.models.gpt.gpt_layer_specs": {"get_gpt_decoder_block_spec": None},
        "megatron.core.models.gpt.gpt_model": {"GPTModel": object},
        "megatron.core.transformer": {"ModuleSpec": object},
        "megatron.core.transformer.attention": {"SelfAttention": type("SelfAttention", (), {})},
    }
    for name, attributes in stubs.items():
        stub = ModuleType(name)
        stub.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, stub)
    spec = importlib.util.spec_from_file_location("mimo_v2_bridge_under_test", _PLUGIN)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve the plugin's string annotations through sys.modules
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _model():
    config = SimpleNamespace(
        num_attention_heads=_HEADS,
        kv_channels=_QK_DIM,
        v_head_dim=_V_DIM,
        full_attn_num_query_groups=_GLOBAL_GROUPS,
        swa_num_query_groups=_SWA_GROUPS,
    )
    return SimpleNamespace(config=config)


def _megatron_qkv_rows(query, key, value, groups):
    """The fused linear_qkv rows, merged group by group as MiMoV2FlashQKVMapping.hf_to_megatron does."""
    heads_per_group = _HEADS // groups
    rows = []
    for group in range(groups):
        rows.append(query[group * heads_per_group * _QK_DIM : (group + 1) * heads_per_group * _QK_DIM])
        rows.append(key[group * _QK_DIM : (group + 1) * _QK_DIM])
        rows.append(value[group * _V_DIM : (group + 1) * _V_DIM])
    return torch.cat(rows)


def test_hf_groups_reach_every_adapted_module(plugin):
    targets = resolve_hf_lora_targets(
        {"model_type": "mimo_v2", "moe_layer_freq": [0, 1]}, target_modules=["attn", "mlp", "unembed"]
    )
    parameters = {
        *(
            f"decoder.layers.{layer}.self_attention.{name}.weight"
            for layer in (0, 1)
            for name in ("linear_qkv", "linear_proj")
        ),
        "decoder.layers.0.mlp.linear_fc1.weight",
        "decoder.layers.0.mlp.linear_fc2.weight",
        "decoder.layers.1.mlp.router.weight",
        "decoder.layers.1.mlp.experts.linear_fc1.weight0",
        "decoder.layers.1.mlp.experts.linear_fc2.weight0",
        "embedding.word_embeddings.weight",
        "output_layer.weight",
    }

    selected = resolve_megatron_lora_targets(
        targets,
        plugin.MiMoV2Bridge().mapping_registry(),
        parameter_names=parameters,
        hf_mapping=HfWeightMapping(frozenset()),
        canonical=False,
    )

    assert set(selected) == {
        "decoder.layers.*.self_attention.linear_qkv",
        "decoder.layers.*.self_attention.linear_proj",
        "decoder.layers.*.mlp.linear_fc1",
        "decoder.layers.*.mlp.linear_fc2",
        "decoder.layers.*.mlp.experts.linear_fc1",
        "decoder.layers.*.mlp.experts.linear_fc2",
        "output_layer",
    }


@pytest.mark.parametrize("groups", [_GLOBAL_GROUPS, _SWA_GROUPS], ids=["global", "swa"])
def test_qkv_lora_b_splits_back_into_hf_rows(plugin, groups):
    rank = 4
    query = torch.randn(_HEADS * _QK_DIM, rank)
    key = torch.randn(groups * _QK_DIM, rank)
    value = torch.randn(groups * _V_DIM, rank)

    split = plugin.MiMoV2Bridge()._split_qkv_linear_out_weight(
        [_model()], _megatron_qkv_rows(query, key, value, groups)
    )

    assert torch.equal(split["q_proj"], query)
    assert torch.equal(split["k_proj"], key)
    assert torch.equal(split["v_proj"], value)


def test_qkv_lora_b_of_an_unknown_layer_is_rejected(plugin):
    # The generic split's assumption: V as wide as K, with the global KV heads.
    rows = (_HEADS + 2 * _GLOBAL_GROUPS) * _QK_DIM

    with pytest.raises(AssertionError, match="neither a global nor a SWA layer"):
        plugin.MiMoV2Bridge()._split_qkv_linear_out_weight(_model(), torch.zeros(rows, 4))


def test_attention_split_matches_the_grouped_view(plugin):
    groups, heads_per_group = 2, _HEADS // _GLOBAL_GROUPS  # one rank of a global layer at TP2
    sizes = [heads_per_group * _QK_DIM, _QK_DIM, _V_DIM]
    mixed_qkv = torch.randn(3, 2, groups * sum(sizes))

    actual = plugin._split_qkv(
        mixed_qkv, dim=-1, groups=groups, heads_per_group=heads_per_group, qk_dim=_QK_DIM, v_dim=_V_DIM
    )

    expected = torch.split(mixed_qkv.view(3, 2, groups, sum(sizes)), sizes, dim=3)
    for actual_part, expected_part in zip(actual, expected, strict=True):
        assert torch.equal(actual_part, expected_part)
