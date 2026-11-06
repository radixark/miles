"""CPU coverage of export scheduling and wire metadata, not NVFP4 kernel numerics."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import miles.utils.nvfp4 as nvfp4
from miles.utils.timer import Timer


@pytest.fixture
def quantizer(monkeypatch):
    # Import this CPU-safe processor without importing optional SGLang/Triton
    # dependencies from the converter package's __init__.
    path = (
        Path(__file__).resolve().parents[4]
        / "miles/backends/megatron_utils/megatron_to_hf/processors/quantizer_nvfp4.py"
    )
    spec = importlib.util.spec_from_file_location("_nvfp4_expert_batch_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _args(*, extra=(), **backend):
    return SimpleNamespace(extra_high_precision_layers_megatron=extra, backend=SimpleNamespace(**backend))


def _weight(value, rows=3, cols=32):
    return torch.full((rows, cols), value, dtype=torch.float32)


def _fc1(expert, *, layer=1, suffixes=("gate_proj", "up_proj"), reverse=False):
    prefix = f"model.layers.{layer}.mlp.experts.{expert}"
    params = [
        (f"{prefix}.{suffixes[0]}.weight", _weight(2 * expert + 1)),
        (f"{prefix}.{suffixes[1]}.weight", _weight(2 * expert + 2)),
    ]
    if reverse:
        params.reverse()
    return f"module.module.decoder.layers.{layer}.mlp.experts.linear_fc1.weight{expert}", params


def _fc2(expert, *, layer=1):
    return (
        f"module.module.decoder.layers.{layer}.mlp.experts.linear_fc2.weight{expert}",
        [(f"model.layers.{layer}.mlp.experts.{expert}.down_proj.weight", _weight(10 + expert, rows=5))],
    )


def _fake_quantize(weight):
    """Encode content plus AMAX distinctly so schedule/scale mixups are visible."""
    assert weight.ndim == 2
    amax = weight.abs().amax()
    qweight = (weight[:, ::2] + weight[:, 1::2]).round().to(torch.uint8)
    block_scale = weight.reshape(weight.shape[0], -1, 16).abs().amax(-1).to(torch.float8_e4m3fn)
    return qweight, block_scale, nvfp4.nvfp4_global_decode_scale_te(amax)


def _fake_pair(first, second):
    qweight, block_scale, scale = _fake_quantize(torch.cat((first, second), dim=0))
    split = first.shape[0]
    return (
        (qweight[:split], block_scale[:split], scale.clone()),
        (qweight[split:], block_scale[split:], scale.clone()),
    )


@pytest.fixture
def batch_calls(quantizer, monkeypatch):
    calls = []

    def quantize_batch(weights):
        weights = list(weights)
        calls.append(weights)
        return [_fake_quantize(weight) for weight in weights]

    monkeypatch.setattr(quantizer, "nvfp4_quantize_1d", _fake_quantize)
    monkeypatch.setattr(quantizer, "nvfp4_quantize_1d_pair", _fake_pair)
    monkeypatch.setattr(quantizer, "nvfp4_quantize_1d_batch", quantize_batch)
    monkeypatch.setattr(nvfp4, "nvfp4_quantize_1d_batch", quantize_batch)
    return calls


def _scalar_export(quantizer, args, units, config):
    return [quantizer.quantize_params_nvfp4(args, name, params, config) for name, params in units]


def test_batch_records_one_quantization_interval(quantizer, monkeypatch):
    instance = Timer()
    monkeypatch.setattr(instance, "timers", {})
    monkeypatch.setattr(instance, "start_time", {"update_weights": 0.0})
    clock = iter((10.0, 12.0))
    monkeypatch.setattr("miles.utils.timer.perf_counter", lambda: next(clock))
    expected = object()
    monkeypatch.setattr(quantizer, "_quantize_expert_params_nvfp4", lambda *args: expected)

    assert quantizer.quantize_expert_params_nvfp4(None, [], None) is expected
    assert instance.log_dict() == {"update_weights_quantization": 2.0}


def test_snapshot_batch_does_not_count_as_weight_update(quantizer, monkeypatch):
    instance = Timer()
    monkeypatch.setattr(instance, "timers", {})
    monkeypatch.setattr(instance, "start_time", {"save_model": 0.0})
    expected = object()
    monkeypatch.setattr(quantizer, "_quantize_expert_params_nvfp4", lambda *args: expected)

    assert quantizer.quantize_expert_params_nvfp4(None, [], None) is expected
    assert instance.log_dict() == {}


def _assert_units_equal(actual, expected):
    assert len(actual) == len(expected)
    for actual_unit, expected_unit in zip(actual, expected, strict=True):
        assert [name for name, _ in actual_unit] == [name for name, _ in expected_unit]
        for (_, actual_tensor), (_, expected_tensor) in zip(actual_unit, expected_unit, strict=True):
            assert actual_tensor.dtype == expected_tensor.dtype
            assert actual_tensor.shape == expected_tensor.shape
            # float8 CPU comparison is not supported on every torch build.
            torch.testing.assert_close(
                actual_tensor.contiguous().reshape(-1).view(torch.uint8),
                expected_tensor.contiguous().reshape(-1).view(torch.uint8),
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize("suffixes", [("gate_proj", "up_proj"), ("w1", "w3")])
def test_batch_preserves_units_order_and_per_expert_gate_up_scale(quantizer, batch_calls, suffixes):
    units = [_fc2(4), _fc1(2, suffixes=suffixes, reverse=True), _fc1(0, suffixes=suffixes), _fc2(1)]
    extra = ("model.layers.1.mlp.experts.2.input_scale", torch.ones(1))
    units[1][1].insert(1, extra)
    args, config = _args(), {"quant_method": "nvfp4"}
    expected = _scalar_export(quantizer, args, units, config)

    actual = quantizer.quantize_expert_params_nvfp4(args, units, config)

    _assert_units_equal(actual, expected)
    assert [len(weights) for weights in batch_calls] == [2, 2]  # FC1 pairs and FC2 tensors are batched separately.
    assert sorted(weight.abs().amax().item() for weights in batch_calls for weight in weights) == [2, 6, 11, 14]
    scales = dict(name_tensor for unit in actual for name_tensor in unit if name_tensor[0].endswith("scale_2"))
    for expert in (0, 2):
        prefix = f"model.layers.1.mlp.experts.{expert}"
        torch.testing.assert_close(
            scales[f"{prefix}.{suffixes[0]}.weight_scale_2"],
            scales[f"{prefix}.{suffixes[1]}.weight_scale_2"],
            rtol=0,
            atol=0,
        )
    assert not torch.equal(
        scales[f"model.layers.1.mlp.experts.0.{suffixes[0]}.weight_scale_2"],
        scales[f"model.layers.1.mlp.experts.2.{suffixes[0]}.weight_scale_2"],
    )
    assert dict(actual[1])[extra[0]] is extra[1]


@pytest.mark.parametrize("policy", ["ignore", "exclude_modules", "regex"])
def test_batch_keeps_skipped_layers_and_experts_in_place(quantizer, batch_calls, policy):
    units = [_fc1(0, layer=0), _fc1(1), _fc1(2), _fc1(3, layer=3), _fc2(4)]
    mtp = _fc1(5)
    units.append((mtp[0].replace("decoder", "mtp"), mtp[1]))
    vision = _fc1(6)
    units.append((vision[0].replace("module.module.", "module.module.vision_model."), vision[1]))
    shared = _fc2(7)
    units.append(
        (shared[0].replace("mlp.experts.linear_fc2.weight7", "mlp.shared_experts.linear_fc2.weight"), shared[1])
    )
    language = _fc1(8)
    units.append((language[0].replace("module.module.", "module.module.language_model."), language[1]))
    args = _args(
        extra=("linear_fc2.weight4",),
        first_last_layers_bf16=True,
        num_layers=4,
        num_layers_at_start_in_bf16=1,
        num_layers_at_end_in_bf16=1,
    )
    config = {"quant_method": "nvfp4"}
    if policy == "regex":
        config["ignore"] = [r"re:model\.layers\.1\.mlp\.experts\.2\."]
    else:
        config[policy] = "model.layers.1.mlp.experts.2"
    expected = _scalar_export(quantizer, args, units, config)

    actual = quantizer.quantize_expert_params_nvfp4(args, units, config)

    _assert_units_equal(actual, expected)
    assert len(batch_calls) == 1
    assert len(batch_calls[0]) == 2
    for index in (0, 2, 3, 4, 5, 6, 7):
        for (_, result), (_, source) in zip(actual[index], units[index][1], strict=True):
            assert result is source


@pytest.mark.parametrize("case", ["missing", "duplicate", "ignore_half", "bad_k"])
def test_batch_preserves_pair_validation(quantizer, batch_calls, case):
    name, params = _fc1(0)
    config = {"quant_method": "nvfp4"}
    if case == "missing":
        params.pop()
    elif case == "duplicate":
        params.append(params[0])
    elif case == "ignore_half":
        config["ignore"] = [params[1][0]]
    elif case == "bad_k":
        params[0] = params[0][0], _weight(1, cols=17)

    with pytest.raises(ValueError):
        quantizer.quantize_params_nvfp4(_args(), name, params, config)
    with pytest.raises(ValueError):
        quantizer.quantize_expert_params_nvfp4(_args(), [(name, params)], config)
    assert not batch_calls


def test_batch_preserves_non_matrix_tensors_and_three_dimensional_down(quantizer, batch_calls):
    name, params = _fc2(0)
    params[0] = params[0][0], torch.stack((_weight(2), _weight(9)))
    params.extend(
        [
            ("model.layers.1.mlp.experts.0.bias.weight", torch.ones(3)),
            ("model.layers.1.mlp.experts.0.packed.weight", torch.zeros((3, 32), dtype=torch.uint8)),
        ]
    )
    units = [(name, params), _fc1(1)]
    config = {"quant_method": "nvfp4"}
    expected = _scalar_export(quantizer, _args(), units, config)

    actual = quantizer.quantize_expert_params_nvfp4(_args(), units, config)

    _assert_units_equal(actual, expected)
    assert dict(actual[0])[params[1][0]] is params[1][1]
    assert dict(actual[0])[params[2][0]] is params[2][1]


def test_empty_batch_does_not_invoke_quantization(quantizer, batch_calls):
    assert quantizer.quantize_expert_params_nvfp4(_args(), [], {"quant_method": "nvfp4"}) == []
    assert not batch_calls


def test_fully_ignored_batch_does_not_invoke_quantization(quantizer, batch_calls):
    units = [_fc1(0), _fc2(1)]
    config = {"quant_method": "nvfp4", "ignore": ["model.layers.1.mlp.experts"]}

    actual = quantizer.quantize_expert_params_nvfp4(_args(), units, config)

    assert not batch_calls
    for actual_unit, (_, original_unit) in zip(actual, units, strict=True):
        for (actual_name, actual_weight), (original_name, original_weight) in zip(
            actual_unit, original_unit, strict=True
        ):
            assert actual_name == original_name
            assert actual_weight is original_weight


@pytest.mark.parametrize("field", ["fp4_param", "fp4_param_gather"])
def test_batch_rejects_unsupported_fp4_parameter_storage(quantizer, batch_calls, field):
    with pytest.raises(NotImplementedError, match="fp4-param-gather"):
        quantizer.quantize_expert_params_nvfp4(_args(**{field: True}), [_fc1(0)], {"quant_method": "nvfp4"})
    assert not batch_calls
