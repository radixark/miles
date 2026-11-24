"""LoRA detection, adapter parameters, and training checkpoint state."""

from argparse import Namespace

import pytest
import torch

from miles.backends.megatron_utils.lora.native.modules.linear import NativeLoRAAdapter
from miles.backends.megatron_utils.lora.utils import adapter_named_parameters, is_lora_enabled
from miles.utils.lora.utils import LORA_ADAPTER_NAME, is_lora_weight_name

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


# ---------------------------------------------------------------------------
# is_lora_weight_name / adapter_named_parameters
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


class _NativeAdapter(NativeLoRAAdapter):
    def __init__(self):
        super().__init__("model.layers.0.self_attn.q_proj.", (), tp_rank=0)
        self.lora_A = torch.nn.Parameter(torch.zeros(1, 2))
        self.lora_B = torch.nn.Parameter(torch.zeros(2, 1))

    def export_plan(self, gather):
        return []


class TestAdapterNamedParameters:
    """Adapter parameters are whatever the adapter modules own; their names are irrelevant."""

    def test_native_adapter_parameters_are_found_whatever_they_are_called(self):
        model = torch.nn.Module()
        model.base = torch.nn.Linear(2, 2)
        model.anything = _NativeAdapter()
        model.anything.register_parameter("magnitude", torch.nn.Parameter(torch.ones(2)))

        assert sorted(adapter_named_parameters([model])[0]) == [
            "anything.lora_A",
            "anything.lora_B",
            "anything.magnitude",
        ]

    def test_bridge_wrappers_contribute_everything_but_the_wrapped_base(self):
        AdapterWrapper = pytest.importorskip("megatron.bridge.peft.adapter_wrapper").AdapterWrapper

        adapter = torch.nn.Module()
        adapter.linear_in = torch.nn.Linear(2, 1, bias=False)
        model = torch.nn.Module()
        model.linear_qkv = AdapterWrapper(torch.nn.Linear(2, 2), adapter)

        assert sorted(adapter_named_parameters([model])[0]) == ["linear_qkv.adapter.linear_in.weight"]

    def test_base_parameters_named_like_adapters_are_not_adapters(self):
        model = torch.nn.Module()
        model.mm_adapter = torch.nn.Linear(2, 2)
        model.register_parameter("lora_scale", torch.nn.Parameter(torch.ones(1)))

        assert adapter_named_parameters([model]) == [{}]


# ---------------------------------------------------------------------------
# LORA_ADAPTER_NAME constant
# ---------------------------------------------------------------------------


def test_lora_adapter_name_constant():
    assert LORA_ADAPTER_NAME == "miles_lora"
