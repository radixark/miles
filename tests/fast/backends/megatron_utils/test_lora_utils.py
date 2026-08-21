"""Unit tests for miles.backends.megatron_utils.lora_utils.

Tests cover module name conversion, LoRA detection helpers, parameter identification,
exclude-module parsing, and LoRA sync config building — all without GPU.
"""

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import miles.backends.megatron_utils.lora_checkpoint as lora_checkpoint
import miles.backends.megatron_utils.lora_utils as lora_utils
from miles.backends.megatron_utils.lora_utils import (
    _get_lora_class_name,
    _is_adapter_param_name,
    build_lora_sync_config,
    convert_target_modules_to_hf,
    convert_target_modules_to_megatron,
    is_lora_enabled,
    is_lora_weight_name,
    parse_exclude_modules,
)
from miles.utils.lora import LORA_ADAPTER_NAME

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
# parse_exclude_modules
# ---------------------------------------------------------------------------


class TestParseExcludeModules:
    def test_none(self):
        args = Namespace(exclude_modules=None)
        assert parse_exclude_modules(args) == []

    def test_single_module_string(self):
        args = Namespace(exclude_modules="o_proj")
        result = parse_exclude_modules(args, lora_type=_make_lora_type("LoRA"))
        assert result == ["linear_proj"]

    def test_comma_separated(self):
        args = Namespace(exclude_modules="o_proj, down_proj")
        result = parse_exclude_modules(args, lora_type=_make_lora_type("LoRA"))
        assert set(result) == {"linear_proj", "linear_fc2"}

    def test_list_input(self):
        args = Namespace(exclude_modules=["o_proj", "down_proj"])
        result = parse_exclude_modules(args, lora_type=_make_lora_type("LoRA"))
        assert set(result) == {"linear_proj", "linear_fc2"}

    def test_missing_attr(self):
        args = Namespace()
        assert parse_exclude_modules(args) == []


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
        )
        config = build_lora_sync_config(args)
        assert config["target_modules"] == ["q_proj", "k_proj"]
        assert config["r"] == 8


# ---------------------------------------------------------------------------
# Native adapter checkpoints
# ---------------------------------------------------------------------------


def _parallel_state(tp=0, pp=0, ep=0, ep_size=1):
    return SimpleNamespace(
        tp=SimpleNamespace(rank=tp),
        pp=SimpleNamespace(rank=pp),
        ep=SimpleNamespace(rank=ep, size=ep_size),
    )


def _adapter_model(name="module.lora_A.weight"):
    model = MagicMock()
    model.named_parameters.return_value = [(name, torch.nn.Parameter(torch.ones(2, 2)))]
    return model


class TestNativeAdapterCheckpoint:
    def test_writer_saves_model_parallel_shard(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lora_utils, "get_parallel_state", lambda: _parallel_state(tp=1, pp=2, ep=3, ep_size=4))
        monkeypatch.setattr(lora_utils, "adapter_shard_topology", lambda: (True, ((1, 2, 3),)))

        path = lora_utils._save_native_adapter_checkpoint([_adapter_model()], tmp_path)

        assert path == tmp_path / "adapter_megatron_tp1_pp2_ep3.pt"
        assert path.exists()
        assert not list(tmp_path.glob("adapter_megatron_rank*.pt"))

    def test_replica_does_not_write_duplicate_shard(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lora_utils, "adapter_shard_topology", lambda: (False, ((0, 0, 0),)))

        assert lora_utils._save_native_adapter_checkpoint([_adapter_model()], tmp_path) is None
        assert not list(tmp_path.iterdir())

    def test_writer_failure_is_reported_before_later_collectives(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lora_utils, "adapter_shard_topology", lambda: (True, ((0, 0, 0),)))
        monkeypatch.setattr(lora_utils, "get_parallel_state", lambda: _parallel_state())
        monkeypatch.setattr(lora_utils.torch, "save", MagicMock(side_effect=OSError("disk full")))

        with pytest.raises(RuntimeError, match="Native LoRA checkpoint save.*disk full"):
            lora_utils._save_native_adapter_checkpoint([_adapter_model()], tmp_path)

    def test_replica_observes_peer_writer_failure(self, monkeypatch):
        monkeypatch.setattr(lora_checkpoint.dist, "is_initialized", lambda: True)
        monkeypatch.setattr(lora_checkpoint.dist, "get_world_size", lambda group: 2)
        monkeypatch.setattr(lora_checkpoint, "get_gloo_group", MagicMock(return_value=object()))

        def gather(messages, _local_message, group):
            messages[:] = ["OSError: disk full", None]

        monkeypatch.setattr(lora_checkpoint.dist, "all_gather_object", gather)

        with pytest.raises(RuntimeError, match="Native LoRA checkpoint save.*disk full"):
            lora_checkpoint.raise_if_any_rank_failed(None, "Native LoRA checkpoint save")

    def test_loads_model_parallel_shard(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lora_utils, "get_parallel_state", lambda: _parallel_state())
        expected = torch.full((2, 2), 3.0)
        torch.save({"module.lora_A.weight": expected}, tmp_path / "adapter_megatron_tp0_pp0.pt")
        model = _adapter_model()

        loaded, iteration = lora_utils.load_lora_adapter([model], str(tmp_path))

        assert loaded
        assert iteration is None
        assert torch.equal(model.named_parameters.return_value[0][1], expected)

    def test_loads_legacy_global_rank_shard(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lora_utils, "get_parallel_state", lambda: _parallel_state())
        expected = torch.full((2, 2), 4.0)
        torch.save({"module.lora_A.weight": expected}, tmp_path / "adapter_megatron_rank0.pt")
        model = _adapter_model()

        loaded, _ = lora_utils.load_lora_adapter([model], str(tmp_path))

        assert loaded
        assert torch.equal(model.named_parameters.return_value[0][1], expected)


# ---------------------------------------------------------------------------
# LORA_ADAPTER_NAME constant
# ---------------------------------------------------------------------------


def test_lora_adapter_name_constant():
    assert LORA_ADAPTER_NAME == "miles_lora"
