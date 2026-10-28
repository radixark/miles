import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _unexpected_gpu_call(*args: object, **kwargs: object) -> None:
    raise AssertionError("Configuration tests must not construct GPU layers")


@pytest.fixture
def inkling_model(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    importlib.import_module("megatron.core.models.gpt.gpt_model")
    importlib.import_module("megatron.core.models.gpt.gpt_layer_specs")
    moe_utils = importlib.import_module("megatron.core.transformer.moe.moe_utils")
    for name in (
        "fused_permute",
        "fused_permute_with_probs",
        "fused_sort_chunks_by_index",
        "fused_sort_chunks_by_index_with_probs",
        "fused_unpermute",
    ):
        monkeypatch.setattr(moe_utils, name, _unexpected_gpu_call)
    engine = ModuleType("transformer_engine")
    engine.pytorch = ModuleType("transformer_engine.pytorch")
    layers = ModuleType("miles_plugins.models.inkling.layers")
    for name in (
        "InklingDenseMLP",
        "InklingMoELayer",
        "InklingRouter",
        "InklingSelfAttention",
        "InklingSharedExperts",
    ):
        setattr(layers, name, _unexpected_gpu_call)
    monkeypatch.setitem(sys.modules, "transformer_engine", engine)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", engine.pytorch)
    monkeypatch.setitem(sys.modules, "miles_plugins.models.inkling.layers", layers)

    path = Path(__file__).resolve().parents[5] / "miles_plugins/models/inkling/model.py"
    spec = importlib.util.spec_from_file_location("_test_inkling_model", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
