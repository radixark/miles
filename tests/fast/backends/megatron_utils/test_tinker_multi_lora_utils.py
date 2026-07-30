import sys
import types
from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.multi_lora_optimizer import named_adapter_slot_parameters
from miles.backends.megatron_utils.tinker import _merge_retained_grads
from miles.backends.megatron_utils.multi_lora_utils import (
    _multi_lora_module_name,
    _tinker_module_group,
)
from miles.ray.tinker.protocol import TinkerError


def test_multi_lora_module_name_supports_dense_bridge_wrapper():
    module = SimpleNamespace(adapters=[SimpleNamespace(base_linear_name="decoder.layers.0.self_attention.linear_qkv")])

    assert _multi_lora_module_name(module) == "decoder.layers.0.self_attention.linear_qkv"


def test_multi_lora_module_name_prefers_wrapper_attribute():
    module = SimpleNamespace(
        base_linear_name="decoder.layers.0.mlp.experts.linear_fc1",
        adapters=[SimpleNamespace(base_linear_name="different")],
    )

    assert _multi_lora_module_name(module) == "decoder.layers.0.mlp.experts.linear_fc1"


def test_multi_lora_module_name_rejects_unknown_wrapper():
    with pytest.raises(RuntimeError, match="cannot determine wrapped linear name"):
        _multi_lora_module_name(SimpleNamespace(adapters=[]))


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("decoder.output_layer", "unembed"),
        ("decoder.layers.0.mlp.linear_fc1", "mlp"),
        ("decoder.layers.0.self_attention.linear_qkv", "attn"),
    ],
)
def test_tinker_module_group(name, expected):
    assert _tinker_module_group(name) == expected


def test_named_adapter_slot_parameters_are_stable_across_slots(monkeypatch):
    class FakeMultiLoRALinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.adapters = torch.nn.ModuleList(
                [
                    torch.nn.Linear(2, 3, bias=False),
                    torch.nn.Linear(2, 3, bias=False),
                ]
            )

    stub = types.ModuleType("megatron.bridge.peft.multi_lora_layers")
    stub.MultiLoRALinear = FakeMultiLoRALinear
    monkeypatch.setitem(sys.modules, "megatron.bridge.peft.multi_lora_layers", stub)

    model = torch.nn.Module()
    model.projection = FakeMultiLoRALinear()

    slot0 = named_adapter_slot_parameters(model, 0)
    slot1 = named_adapter_slot_parameters(model, 1)

    assert [name for name, _ in slot0] == ["model0.projection.weight"]
    assert [name for name, _ in slot1] == ["model0.projection.weight"]
    assert slot0[0][1] is not slot1[0][1]


def test_retained_gradients_follow_the_optimizer_owner():
    rank0 = {
        "optimizer_state": {"layer_a.weight": {}},
        "retained_grads": {"layer_a.weight": torch.tensor([1.0])},
    }
    rank1 = {
        "optimizer_state": {"layer_b.weight": {}},
        "retained_grads": {"layer_b.weight": torch.tensor([2.0])},
    }

    merged = _merge_retained_grads([rank0, rank1])

    torch.testing.assert_close(merged["layer_a.weight"], torch.tensor([1.0]))
    torch.testing.assert_close(merged["layer_b.weight"], torch.tensor([2.0]))


def test_retained_gradient_owner_must_be_unique():
    duplicate = {
        "optimizer_state": {"layer.weight": {}},
        "retained_grads": {"layer.weight": torch.tensor([1.0])},
    }

    with pytest.raises(TinkerError, match="inconsistent retained-gradient"):
        _merge_retained_grads([duplicate, duplicate])
