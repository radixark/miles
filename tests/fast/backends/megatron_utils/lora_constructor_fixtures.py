import sys
from dataclasses import dataclass
from types import ModuleType

import pytest


@dataclass(kw_only=True)
class _AdapterConfig:
    target_modules: list[str]
    dim: int
    alpha: int
    dropout: float
    lora_A_init_method: str
    lora_B_init_method: str


@dataclass(kw_only=True)
class _LoRA(_AdapterConfig):
    share_expert_adapters: bool = True
    experts_shared_outer_loras: bool = False


@dataclass(kw_only=True)
class _CanonicalLoRA(_AdapterConfig):
    pass


@dataclass(kw_only=True)
class _MultiLoRA(_AdapterConfig):
    n_adapters: int


@dataclass(frozen=True)
class _BridgeConstructors:
    lora: type[_LoRA] = _LoRA
    canonical_lora: type[_CanonicalLoRA] = _CanonicalLoRA
    multi_lora: type[_MultiLoRA] = _MultiLoRA


@pytest.fixture
def bridge_constructors(monkeypatch: pytest.MonkeyPatch) -> _BridgeConstructors:
    constructors = _BridgeConstructors()
    bridge = ModuleType("megatron.bridge")
    bridge.__path__ = []
    peft = ModuleType("megatron.bridge.peft")
    peft.__path__ = []
    lora = ModuleType("megatron.bridge.peft.lora")
    lora.LoRA = constructors.lora
    canonical = ModuleType("megatron.bridge.peft.canonical_lora")
    canonical.CanonicalLoRA = constructors.canonical_lora
    multi = ModuleType("megatron.bridge.peft.multi_lora")
    multi.MultiLoRA = constructors.multi_lora
    bridge.peft = peft
    peft.lora = lora
    peft.canonical_lora = canonical
    peft.multi_lora = multi
    for module in (bridge, peft, lora, canonical, multi):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return constructors
