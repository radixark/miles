"""zero_optimizer_state_for_adapter must reset a retired slot's Adam moments and step clock
(group-level FusedAdam or per-param torch AdamW) while leaving co-tenant slots untouched."""

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils.multi_lora_utils import zero_optimizer_state_for_adapter

MLL_MODULE = "megatron.bridge.peft.multi_lora_layers"


class FakeAdapter:
    def __init__(self, params):
        self._params = list(params)

    def parameters(self):
        return self._params


class FakeMultiLoRALinear:
    def __init__(self, adapters):
        self.adapters = adapters


@pytest.fixture()
def rig(monkeypatch):
    # Stub the lazily imported bridge module so the test needs no bridge build that ships multi-LoRA.
    p0 = torch.nn.Parameter(torch.ones(4))
    p1 = torch.nn.Parameter(torch.ones(4))
    module = FakeMultiLoRALinear({0: FakeAdapter([p0]), 1: FakeAdapter([p1])})
    stub = types.ModuleType(MLL_MODULE)
    stub.MultiLoRALinear = FakeMultiLoRALinear
    stub._iter_multi_lora_modules = lambda model: [module]
    monkeypatch.setitem(sys.modules, MLL_MODULE, stub)
    return SimpleNamespace(p0=p0, p1=p1, model=object())


def make_optimizer(groups, state):
    inner = SimpleNamespace(param_groups=groups, state=state)
    return inner, SimpleNamespace(chained_optimizers=[SimpleNamespace(optimizer=inner)])


def test_group_level_fused_adam_clock_resets_only_for_the_retired_slot(rig):
    inner, optimizer = make_optimizer(
        groups=[
            {"params": [rig.p0], "miles_multi_lora_slot": 0, "step": 50},
            {"params": [rig.p1], "miles_multi_lora_slot": 1, "step": 50},
        ],
        state={
            rig.p0: {"exp_avg": torch.ones(4), "exp_avg_sq": torch.ones(4)},
            rig.p1: {"exp_avg": torch.ones(4), "exp_avg_sq": torch.ones(4)},
        },
    )

    zero_optimizer_state_for_adapter(optimizer, rig.model, 0)

    assert inner.param_groups[0]["step"] == 0
    assert inner.param_groups[1]["step"] == 50  # co-tenant slot untouched
    assert float(inner.state[rig.p0]["exp_avg"].abs().sum()) == 0.0
    assert float(inner.state[rig.p0]["exp_avg_sq"].abs().sum()) == 0.0
    assert float(inner.state[rig.p1]["exp_avg"].abs().sum()) == 4.0


def test_tensor_valued_group_clock_resets_in_place(rig):
    step = torch.tensor(50)
    inner, optimizer = make_optimizer(
        groups=[{"params": [rig.p0], "miles_multi_lora_slot": 0, "step": step}],
        state={rig.p0: {"exp_avg": torch.ones(4), "exp_avg_sq": torch.ones(4)}},
    )

    zero_optimizer_state_for_adapter(optimizer, rig.model, 0)

    assert int(step) == 0  # zeroed in place, no rebinding needed


def test_per_param_adamw_fallback_clock_resets(rig):
    # torch.optim.AdamW keeps the clock per param; groups carry no "step".
    inner, optimizer = make_optimizer(
        groups=[{"params": [rig.p0], "miles_multi_lora_slot": 0}],
        state={
            rig.p0: {"exp_avg": torch.ones(4), "exp_avg_sq": torch.ones(4), "step": torch.tensor(50.0)},
        },
    )

    zero_optimizer_state_for_adapter(optimizer, rig.model, 0)

    assert float(inner.state[rig.p0]["step"]) == 0.0
