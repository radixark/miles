import sys
from collections.abc import Callable
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def raw_mbridge_loader(monkeypatch: pytest.MonkeyPatch) -> Callable[[bool], list[set[str]]]:
    def configure(fail: bool) -> list[set[str]]:
        observed: list[set[str]] = []

        def load_weights(models: list[torch.nn.Module], path: str, *, memory_efficient: bool) -> None:
            for model in models:
                observed.append(set(model.state_dict()))
                model.weight.data.fill_(5)
            if fail:
                raise RuntimeError("HF loading failed")

        bridge = Mock(load_weights=load_weights)
        mbridge = ModuleType("mbridge")
        mbridge.AutoBridge = Mock(from_pretrained=Mock(return_value=bridge))
        monkeypatch.setitem(sys.modules, "mbridge", mbridge)
        monkeypatch.setitem(sys.modules, "miles_plugins.mbridge", ModuleType("miles_plugins.mbridge"))
        monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
        return observed

    return configure
