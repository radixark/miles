"""Unit tests for miles.backends.megatron_utils.lora_utils.

Tests cover module name conversion, LoRA detection helpers, parameter identification,
and LoRA sync config building — all without GPU.
"""

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from miles.backends.megatron_utils.lora_utils import (
    _adapter_shard_name,
    _get_lora_class_name,
    _is_adapter_param_name,
    _is_canonical_shard_writer,
    build_lora_sync_config,
    convert_target_modules_to_hf,
    convert_target_modules_to_megatron,
    is_lora_enabled,
    is_lora_weight_name,
    reduce_marked_lora_grads,
    resolve_lora_provider,
)
from miles.utils.lora import LORA_ADAPTER_NAME
from miles_plugins.lora.lora import export_lora_hf_named, load_lora_adapter_hf, wrap_model_provider_with_lora

# ---------------------------------------------------------------------------
# _get_lora_class_name
# ---------------------------------------------------------------------------


class TestGetLoraClassName:
    def test_none_returns_canonical(self):
        assert _get_lora_class_name(None) == "CanonicalLoRA"

    def test_type_returns_class_name(self):
        class FakeLoRA:
            pass

        assert _get_lora_class_name(FakeLoRA) == "FakeLoRA"

    def test_instance_returns_class_name(self):
        class FakeLoRA:
            pass

        assert _get_lora_class_name(FakeLoRA()) == "FakeLoRA"


# ---------------------------------------------------------------------------
# convert_target_modules_to_megatron
# ---------------------------------------------------------------------------


def _make_lora_type(name: str):
    """Helper to create a mock lora_type whose class name matches *name*."""
    mock = MagicMock()
    type(mock).__name__ = name
    return mock


class TestConvertTargetModulesToMegatron:
    def test_gdn_hf_names_collapse_to_fused_in_proj(self):
        lora = _make_lora_type("LoRA")
        result = convert_target_modules_to_megatron(["in_proj_qkvz", "in_proj_ba", "out_proj"], lora_type=lora)
        assert result == ["in_proj", "out_proj"]

    # --- "all-linear" variants ------------------------------------------------

    @pytest.mark.parametrize("shorthand", ["all", "all-linear", "all_linear"])
    def test_all_linear_string_canonical(self, shorthand):
        result = convert_target_modules_to_megatron(shorthand, lora_type=None)
        assert result == [
            "linear_q",
            "linear_k",
            "linear_v",
            "linear_proj",
            "linear_fc1_up",
            "linear_fc1_gate",
            "linear_fc2",
        ]

    @pytest.mark.parametrize("shorthand", ["all", "all-linear", "all_linear"])
    def test_all_linear_string_standard(self, shorthand):
        lora_type = _make_lora_type("LoRA")
        result = convert_target_modules_to_megatron(shorthand, lora_type=lora_type)
        assert result == ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]

    @pytest.mark.parametrize("shorthand", ["all", "all-linear", "all_linear"])
    def test_all_linear_single_element_list(self, shorthand):
        result = convert_target_modules_to_megatron([shorthand], lora_type=None)
        assert len(result) == 7  # CanonicalLoRA has 7 modules

    # --- HF -> Megatron conversion (standard LoRA) ----------------------------

    def test_hf_to_megatron_standard_dedup(self):
        lora = _make_lora_type("LoRA")
        result = convert_target_modules_to_megatron(["q_proj", "k_proj", "v_proj"], lora_type=lora)
        assert result == ["linear_qkv"]

    def test_hf_to_megatron_standard_all_modules(self):
        lora = _make_lora_type("LoRA")
        modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        result = convert_target_modules_to_megatron(modules, lora_type=lora)
        assert result == ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]

    # --- HF -> Megatron conversion (CanonicalLoRA) ----------------------------

    def test_hf_to_megatron_canonical_split(self):
        result = convert_target_modules_to_megatron(["q_proj", "k_proj", "v_proj"], lora_type=None)
        assert result == ["linear_q", "linear_k", "linear_v"]

    def test_hf_to_megatron_canonical_gate_up(self):
        result = convert_target_modules_to_megatron(["gate_proj", "up_proj"], lora_type=None)
        assert result == ["linear_fc1_gate", "linear_fc1_up"]

    # --- Already in Megatron format -------------------------------------------

    def test_megatron_format_passthrough(self):
        modules = ["linear_qkv", "linear_proj"]
        result = convert_target_modules_to_megatron(modules, lora_type=None)
        assert result == modules

    def test_megatron_format_passthrough_canonical(self):
        modules = ["linear_q", "linear_k", "linear_v"]
        result = convert_target_modules_to_megatron(modules, lora_type=None)
        assert result == modules

    # --- Single string input --------------------------------------------------

    def test_single_hf_string_input(self):
        lora = _make_lora_type("LoRA")
        result = convert_target_modules_to_megatron("o_proj", lora_type=lora)
        assert result == ["linear_proj"]


# ---------------------------------------------------------------------------
# convert_target_modules_to_hf
# ---------------------------------------------------------------------------


class TestConvertTargetModulesToHf:
    def test_standard_linear_qkv(self):
        assert convert_target_modules_to_hf(["linear_qkv"]) == ["q_proj", "k_proj", "v_proj"]

    def test_standard_linear_proj(self):
        assert convert_target_modules_to_hf(["linear_proj"]) == ["o_proj"]

    def test_standard_linear_fc1(self):
        assert convert_target_modules_to_hf(["linear_fc1"]) == ["gate_proj", "up_proj"]

    def test_standard_linear_fc2(self):
        assert convert_target_modules_to_hf(["linear_fc2"]) == ["down_proj"]

    def test_gdn_in_proj_expands_to_sglang_modules(self):
        assert convert_target_modules_to_hf(["in_proj"]) == ["in_proj_qkvz", "in_proj_ba"]

    @pytest.mark.parametrize(
        "module,expected",
        [
            ("out_proj", ["out_proj"]),  # same-name passthrough
            ("language_model.decoder.layers.*.self_attention.out_proj", ["out_proj"]),  # wildcard path
            ("language_model.decoder.layers.0.self_attention.out_proj", ["out_proj"]),  # dotted path, passthrough
            (
                "language_model.decoder.layers.0.self_attention.linear_qkv",  # dotted path, table-mapped
                ["q_proj", "k_proj", "v_proj"],
            ),
        ],
    )
    def test_paths_reduce_to_leaf_before_mapping(self, module, expected):
        assert convert_target_modules_to_hf([module]) == expected

    def test_canonical_split_modules(self):
        result = convert_target_modules_to_hf(["linear_q", "linear_k", "linear_v"])
        assert result == ["q_proj", "k_proj", "v_proj"]

    def test_canonical_fc1_gate_up(self):
        result = convert_target_modules_to_hf(["linear_fc1_gate", "linear_fc1_up"])
        assert result == ["gate_proj", "up_proj"]

    def test_unknown_module_passthrough(self):
        assert convert_target_modules_to_hf(["some_custom_module"]) == ["some_custom_module"]

    def test_roundtrip_canonical_all_linear(self):
        megatron = convert_target_modules_to_megatron("all-linear", lora_type=None)
        hf = convert_target_modules_to_hf(megatron)
        assert set(hf) == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}

    def test_roundtrip_standard_all_linear(self):
        lora = _make_lora_type("LoRA")
        megatron = convert_target_modules_to_megatron("all-linear", lora_type=lora)
        hf = convert_target_modules_to_hf(megatron)
        assert set(hf) == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


# ---------------------------------------------------------------------------
# is_lora_enabled
# ---------------------------------------------------------------------------


class TestIsLoraEnabled:
    def test_enabled_by_rank(self):
        args = Namespace(lora_rank=32, lora_adapter_path=None)
        assert is_lora_enabled(args) is True

    def test_enabled_by_adapter_path(self):
        args = Namespace(lora_rank=0, lora_adapter_path="/some/path")
        assert is_lora_enabled(args) is True

    def test_enabled_by_both(self):
        args = Namespace(lora_rank=16, lora_adapter_path="/some/path")
        assert is_lora_enabled(args) is True

    def test_disabled(self):
        args = Namespace(lora_rank=0, lora_adapter_path=None)
        assert is_lora_enabled(args) is False

    def test_disabled_missing_attrs(self):
        args = Namespace()
        assert is_lora_enabled(args) is False


# ---------------------------------------------------------------------------
# is_lora_weight_name / _is_adapter_param_name
# ---------------------------------------------------------------------------


class TestIsLoraWeightName:
    @pytest.mark.parametrize(
        "name",
        [
            "model.layers.0.self_attn.q_proj.lora_A.weight",
            "model.layers.0.self_attn.q_proj.lora_B.weight",
            "base_model.model.layers.5.mlp.gate_proj.lora_A.default.weight",
            "base_model.model.layers.5.mlp.gate_proj.lora_B.default.weight",
        ],
    )
    def test_positive(self, name):
        assert is_lora_weight_name(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            "model.layers.0.self_attn.q_proj.weight",
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.layers.0.mlp.gate_proj.weight",
        ],
    )
    def test_negative(self, name):
        assert is_lora_weight_name(name) is False


class TestIsAdapterParamName:
    @pytest.mark.parametrize(
        "name",
        [
            "module.decoder.layers.0.self_attention.linear_qkv.lora_A.weight",
            "module.decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight",
            "module.decoder.layers.0.self_attention.linear_qkv.adapter.linear_out.weight",
        ],
    )
    def test_positive(self, name):
        assert _is_adapter_param_name(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            "module.decoder.layers.0.self_attention.linear_qkv.weight",
            "module.decoder.layers.0.mlp.linear_fc1.weight",
            "module.embedding.word_embeddings.weight",
        ],
    )
    def test_negative(self, name):
        assert _is_adapter_param_name(name) is False


# ---------------------------------------------------------------------------
# build_lora_sync_config
# ---------------------------------------------------------------------------


class TestBuildLoraSyncConfig:
    def test_basic_config(self):
        args = Namespace(
            lora_rank=32,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
        )
        config = build_lora_sync_config(args)
        assert config["peft_type"] == "LORA"
        assert config["r"] == 32
        assert config["lora_alpha"] == 32
        assert config["lora_dropout"] == 0.0
        assert config["bias"] == "none"
        assert config["task_type"] == "CAUSAL_LM"
        assert set(config["target_modules"]) == {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        }

    def test_no_target_modules_uses_default(self):
        args = Namespace(lora_rank=16, lora_alpha=16, lora_dropout=0.0, target_modules=None)
        config = build_lora_sync_config(args)
        assert len(config["target_modules"]) == 7

    def test_canonical_target_modules(self):
        args = Namespace(
            lora_rank=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=["linear_q", "linear_k"],
            megatron_to_hf_mode="raw",
        )
        config = build_lora_sync_config(args)
        assert config["target_modules"] == ["k_proj", "q_proj", "v_proj"]
        assert config["r"] == 8


# ---------------------------------------------------------------------------
# LORA_ADAPTER_NAME constant
# ---------------------------------------------------------------------------


def test_lora_adapter_name_constant():
    assert LORA_ADAPTER_NAME == "miles_lora"


def _fake_model(num_layers=2, *, output_gate=False, mla=False, with_qkv=True, num_query_groups=8, q_lora_rank=1536):
    layers = []
    for i in range(num_layers):
        attn = SimpleNamespace(layer_number=i + 1)
        if with_qkv:
            attn.linear_qkv = object()
        layers.append(SimpleNamespace(layer_number=i + 1, self_attention=attn))
    cfg = SimpleNamespace(
        attention_output_gate=output_gate,
        multi_latent_attention=mla,
        num_query_groups=num_query_groups,
        q_lora_rank=q_lora_rank,
    )
    return SimpleNamespace(config=cfg, decoder=SimpleNamespace(layers=layers))


class TestResolveLoraProvider:
    def test_default_is_the_plugin(self):
        mod = resolve_lora_provider(Namespace())
        assert mod.wrap_model_provider_with_lora is wrap_model_provider_with_lora
        assert mod.export_lora_hf_named is export_lora_hf_named
        assert mod.load_lora_adapter_hf is load_lora_adapter_hf

    @pytest.mark.parametrize("path", ["miles_plugins.lora.lora"])
    def test_native_provider_paths_are_imported(self, path):
        provider = resolve_lora_provider(Namespace(lora_provider_path=path))
        assert provider.wrap_model_provider_with_lora is wrap_model_provider_with_lora
        assert provider.export_lora_hf_named is export_lora_hf_named

    def test_module_without_protocol_is_rejected(self):
        args = Namespace(lora_provider_path="json")
        with pytest.raises(AssertionError, match="wrap_model_provider_with_lora"):
            resolve_lora_provider(args)


class TestWrapModelProvider:
    def test_provider_args_are_forwarded_and_result_wrapped(self):
        seen = {}

        def provider(pre_process, post_process, vp_stage=None):
            seen.update(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
            return _fake_model()

        calls = []
        wrapped = wrap_model_provider_with_lora(provider, Namespace(lora_rank=8))
        import miles_plugins.lora.lora as ln

        orig = ln.apply_native_lora
        ln.apply_native_lora = lambda m, a: calls.append((m, a)) or m
        try:
            out = wrapped(True, False, vp_stage=1)
        finally:
            ln.apply_native_lora = orig

        assert seen == {"pre_process": True, "post_process": False, "vp_stage": 1}
        assert out is calls[0][0]


class TestAdapterShardName:
    def test_native_name_is_ep_invariant(self):
        """Routed experts carry no native adapter and the shared expert shards over attention TP,
        so every EP rank holds identical state for a given (tp, pp) — one file serves them all."""
        names = {_adapter_shard_name(1, 2, ep, ep_sharded=False) for ep in range(4)}
        assert names == {"adapter_megatron_tp1_pp2.pt"}

    def test_bridge_name_keys_on_ep(self):
        """Bridge PEFT can attach genuinely expert-parallel adapters, so its shards differ per EP rank."""
        assert _adapter_shard_name(1, 2, 0, ep_sharded=True) == "adapter_megatron_tp1_pp2.pt"
        assert _adapter_shard_name(1, 2, 3, ep_sharded=True) == "adapter_megatron_tp1_pp2_ep3.pt"
        names = {_adapter_shard_name(0, 0, ep, ep_sharded=True) for ep in range(4)}
        assert len(names) == 4

    def test_writer_election_is_a_noop_without_distributed(self):
        assert _is_canonical_shard_writer("adapter_megatron_tp0_pp0.pt")


class TestReduceMarkedLoraGrads:
    def test_no_marked_params_is_a_noop_without_distributed(self):
        chunk = torch.nn.Linear(2, 2)
        reduce_marked_lora_grads([chunk])

    def test_empty_model_list_is_a_noop(self):
        reduce_marked_lora_grads([])
