import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


class _UnavailableRMSNorm(torch.nn.Module):
    def __init__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("GDN kernel configuration must not construct Transformer Engine layers")


@pytest.fixture
def gdn(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    importlib.import_module("megatron.core.models.gpt.gpt_layer_specs")
    engine = ModuleType("transformer_engine")
    engine.pytorch = ModuleType("transformer_engine.pytorch")
    engine.pytorch.RMSNorm = _UnavailableRMSNorm
    monkeypatch.setitem(sys.modules, "transformer_engine", engine)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", engine.pytorch)

    root = Path(__file__).resolve().parents[5]
    for name, relative_path in (
        ("miles_plugins.models.qwen3_5", "miles_plugins/models/qwen3_5.py"),
        ("_test_qwen3_8_gdn", "miles_plugins/models/qwen3_8_next/gdn.py"),
    ):
        spec = importlib.util.spec_from_file_location(name, root / relative_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    return module
