"""Dispatch timing is independent of optional quantization backends."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from miles.utils.timer import Timer


@pytest.fixture
def processor(monkeypatch):
    package = "_quantization_timing_test"
    path = Path(__file__).resolve().parents[6] / "miles/backends/megatron_utils/megatron_to_hf/processors/__init__.py"
    elapsed = [0.0]
    monkeypatch.setattr("miles.utils.timer.perf_counter", lambda: elapsed[0])
    instance = Timer()
    monkeypatch.setattr(instance, "timers", {})
    monkeypatch.setattr(instance, "start_time", {"update_weights": 0.0})

    def quantize(*args):
        elapsed[0] += 0.25
        return args

    for name, function in (
        ("padding_remover", "remove_padding"),
        ("quantizer_compressed_tensors", "quantize_params_compressed_tensors"),
        ("quantizer_fp8", "quantize_params_fp8"),
        ("quantizer_mxfp8", "quantize_params_mxfp8"),
        ("quantizer_nvfp4", "quantize_params_nvfp4"),
    ):
        stub = ModuleType(f"{package}.{name}")
        setattr(stub, function, quantize)
        monkeypatch.setitem(sys.modules, stub.__name__, stub)
    spec = importlib.util.spec_from_file_location(package, path, submodule_search_locations=[str(path.parent)])
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, package, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("method", ["fp8", "mxfp8", "nvfp4", "compressed-tensors"])
def test_quantized_dispatch_accumulates_once_per_call(processor, method):
    params, packed = object(), {"expert.weight"}
    config = {"quant_method": method}
    for _ in range(2):
        actual = processor.quantize_params(None, "expert", params, config, packed)
        assert params in actual
    assert Timer().log_dict() == {"update_weights_quantization": 0.5}


def test_unquantized_dispatch_bypasses_timer(processor):
    params = object()
    assert processor.quantize_params(None, "expert", params, None) is params
    assert Timer().log_dict() == {}


def test_snapshot_quantization_does_not_count_as_weight_update(processor, monkeypatch):
    monkeypatch.setattr(Timer(), "start_time", {"save_model": 0.0})
    params = object()

    actual = processor.quantize_params(None, "expert", params, {"quant_method": "nvfp4"})

    assert params in actual
    assert Timer().log_dict() == {}
