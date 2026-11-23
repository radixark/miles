import sys
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch


class _MultiLoRALinear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Linear(2, 2)
        self.adapters = torch.nn.ModuleList([torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)])


@dataclass
class _OptimizerConstruction:
    model: _MultiLoRALinear
    use_gloo: list[bool] = field(default_factory=list)
    inner: Any = None
    config: Any = None


@pytest.fixture
def optimizer_construction(monkeypatch: pytest.MonkeyPatch, bridge_sdk: Mock) -> _OptimizerConstruction:

    from miles.backends.megatron_utils.lora import optimizer

    construction = _OptimizerConstruction(model=_MultiLoRALinear())

    def build(config: Any, model: list[torch.nn.Module], *, use_gloo_process_groups: bool) -> Any:
        parameters = [parameter for chunk in model for parameter in chunk.parameters() if parameter.requires_grad]
        adam = torch.optim.Adam(
            parameters,
            lr=config.lr,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
        )
        child = SimpleNamespace(optimizer=adam, get_parameters=lambda: parameters)
        construction.use_gloo.append(use_gloo_process_groups)
        return SimpleNamespace(chained_optimizers=[child])

    def wrap(base_optimizers: list, config: Any, groups: object, **kwargs: Any) -> Any:
        construction.config = config
        construction.inner = SimpleNamespace(
            chained_optimizers=[
                SimpleNamespace(param_groups=base.param_groups, grad_stats_parallel_group=object())
                for base in base_optimizers
            ]
        )
        return construction.inner

    peft = ModuleType("megatron.bridge.peft")
    multi_lora_layers = ModuleType("megatron.bridge.peft.multi_lora_layers")
    multi_lora_layers.MultiLoRALinear = _MultiLoRALinear
    peft.multi_lora_layers = multi_lora_layers
    for module in (peft, multi_lora_layers):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(optimizer, "get_megatron_optimizer", build)
    monkeypatch.setattr(optimizer, "LayerWiseDistributedOptimizer", wrap)
    monkeypatch.setattr(optimizer.ProcessGroupCollection, "use_mpu_process_groups", lambda: object())
    return construction
