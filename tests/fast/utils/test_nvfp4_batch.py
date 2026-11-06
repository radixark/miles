"""CPU coverage of NVFP4 batch orchestration; TE kernels are deliberately mocked."""

import sys
from contextlib import contextmanager, nullcontext
from types import ModuleType, SimpleNamespace

import pytest
import torch

import miles.utils.nvfp4 as nvfp4


@pytest.fixture
def fake_te(monkeypatch):
    calls = SimpleNamespace(split=[], scalar=[], configs=[], scale_shapes=[], device=None)

    @contextmanager
    def device_context(device):
        previous = calls.device
        calls.device = device
        try:
            yield
        finally:
            calls.device = previous

    monkeypatch.setattr(torch.cuda, "device", device_context)

    def fake_output(weight):
        rows, cols = weight.shape
        # These bytes only identify the input and output layout, not FP4 numerics.
        data = weight[:, ::2].abs().to(torch.uint8)
        scales = torch.full((((rows + 127) // 128) * 128, ((cols // 16 + 3) // 4) * 4), 56, dtype=torch.uint8)
        amax = weight.float().abs().amax().reshape(1) if weight.numel() else torch.zeros(1)
        return SimpleNamespace(_rowwise_data=data, _rowwise_scale_inv=scales, _amax_rowwise=amax)

    class FakeQuantizer:
        def __init__(self, **kwargs):
            calls.configs.append(kwargs)

        def quantize(self, weight):
            assert calls.device == weight.device
            calls.scalar.append(weight)
            return fake_output(weight)

    def split_quantize(weight, split_rows, quantizers):
        assert weight.is_contiguous()
        assert calls.device == weight.device
        assert weight.dtype == torch.bfloat16
        assert weight.shape[1] % 128 == 0
        assert all(rows % 64 == 0 for rows in split_rows)
        assert len(split_rows) == len(quantizers)
        assert len(split_rows) <= 64
        calls.split.append((weight, split_rows))
        return [fake_output(part) for part in weight.split(split_rows)]

    quantizer_module = ModuleType("transformer_engine.pytorch.tensor.nvfp4_tensor")
    quantizer_module.NVFP4Quantizer = FakeQuantizer
    extension = ModuleType("transformer_engine_torch")
    extension.split_quantize = split_quantize
    monkeypatch.setitem(sys.modules, quantizer_module.__name__, quantizer_module)
    monkeypatch.setitem(sys.modules, extension.__name__, extension)

    decode_scale = nvfp4.nvfp4_global_decode_scale_te

    def record_scale_shape(amax, e4m3_max):
        calls.scale_shapes.append(tuple(amax.shape))
        return decode_scale(amax, e4m3_max)

    monkeypatch.setattr(nvfp4, "nvfp4_global_decode_scale_te", record_scale_shape)
    monkeypatch.setenv("NVTE_NVFP4_4OVER6", "none")
    return calls


def _assert_output_matches_input(output, weight, *, e4m3_max=448):
    data, scales, global_scale = output
    torch.testing.assert_close(data, weight[:, ::2].abs().to(torch.uint8), rtol=0, atol=0)
    assert scales.shape == (weight.shape[0], weight.shape[1] // 16)
    assert scales.dtype == torch.float8_e4m3fn
    assert scales.is_contiguous()
    assert global_scale.shape == ()
    # Preserve TE's rounded encode-then-decode arithmetic, rather than replacing
    # it with a mathematically equivalent single division.
    expected_scale = torch.div(1.0, torch.div(float(e4m3_max * 6), weight.float().abs().amax()))
    torch.testing.assert_close(global_scale, expected_scale, rtol=0, atol=0)


def test_batch_partitions_dtype_and_k_preserves_order_and_vectorizes_scales(fake_te):
    specifications = [
        (128, torch.float32),
        (256, torch.bfloat16),
        (128, torch.float32),
        (256, torch.bfloat16),
        (64, torch.bfloat16),
        (64, torch.bfloat16),
        (16, torch.bfloat16),
        (16, torch.bfloat16),
        (128, torch.float16),
        (128, torch.float16),
    ]
    weights = [torch.full((16, k), 2**i, dtype=dtype) for i, (k, dtype) in enumerate(specifications)]
    outputs = nvfp4.nvfp4_quantize_1d_batch(weights)

    assert fake_te.device is None
    assert [(weight.shape[1], weight.dtype) for weight, _ in fake_te.split] == [
        (256, torch.bfloat16),
    ]
    assert [weight.shape[1] for weight in fake_te.scalar] == [128, 128, 64, 64, 16, 16, 128, 128]
    assert fake_te.scale_shapes == [(), (), (2,), (), (), (), (), (), ()]
    for output, weight in zip(outputs, weights, strict=True):
        _assert_output_matches_input(output, weight)


def test_batch_partitions_devices_before_dispatch(monkeypatch):
    # Meta tensors test grouping without claiming to execute CUDA or TE kernels.
    weights = [
        torch.empty((64, 128), device=device, dtype=torch.bfloat16) for device in ("cpu", "meta", "cpu", "meta")
    ]
    calls = []
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())

    def split(inputs):
        calls.append(inputs)
        return inputs

    monkeypatch.setattr(nvfp4, "_nvfp4_quantize_split", split)
    outputs = nvfp4.nvfp4_quantize_1d_batch(weights)
    assert [[weight.device.type for weight in batch] for batch in calls] == [["cpu", "cpu"], ["meta", "meta"]]
    assert all(actual is expected for actual, expected in zip(outputs, weights, strict=True))


@pytest.mark.parametrize("layout", ["adjacent", "disjoint", "strided", "padded", "gap"])
def test_batch_packing_preserves_values_and_only_reuses_adjacent_storage(fake_te, layout):
    backing = torch.arange(256 * 128).reshape(256, 128).remainder(127).add(1).to(torch.bfloat16)
    if layout == "padded":
        weights = [backing[:3], backing[3:20]]
    elif layout == "gap":
        weights = [backing[:64], backing[128:192]]
    elif layout == "strided":
        weights = [backing[:128:2], backing[128::2]]
    else:
        weights = list(backing[:192].split(64))
        if layout == "disjoint":
            weights = [weight.clone() for weight in weights]
    snapshots = [weight.clone() for weight in weights]

    outputs = nvfp4.nvfp4_quantize_1d_batch(weights)
    assert len(fake_te.split) == 1
    packed, split_rows = fake_te.split[0]
    assert (packed.untyped_storage().data_ptr() == backing.untyped_storage().data_ptr()) is (layout == "adjacent")
    assert split_rows == [((weight.shape[0] + 63) // 64) * 64 for weight in weights]
    for packed_weight, weight, snapshot, output in zip(
        packed.split(split_rows), weights, snapshots, outputs, strict=True
    ):
        torch.testing.assert_close(packed_weight[: weight.shape[0]], snapshot, rtol=0, atol=0)
        assert torch.count_nonzero(packed_weight[weight.shape[0] :]) == 0
        torch.testing.assert_close(weight, snapshot, rtol=0, atol=0)
        _assert_output_matches_input(output, weight)


@pytest.mark.parametrize(
    ("enabled", "error_mode", "e4m3_max"),
    [(False, "MAE", 448), (True, "MAE", 448), (True, "MAE", 256), (True, "MSE", 448), (True, "MSE", 256)],
)
def test_batch_preserves_quantizer_policy(fake_te, monkeypatch, enabled, error_mode, e4m3_max):
    monkeypatch.setenv("NVTE_NVFP4_4OVER6", "weights" if enabled else "none")
    monkeypatch.setenv("NVTE_NVFP4_4OVER6_E4M3_USE_256", "weights" if e4m3_max == 256 else "none")
    monkeypatch.setenv("NVTE_NVFP4_4OVER6_ERR_MODE", error_mode)
    weights = [torch.full((16, 128), value, dtype=torch.bfloat16) for value in (1.0, 16.0)]
    outputs = nvfp4.nvfp4_quantize_1d_batch(weights)
    assert len(fake_te.split) == 1
    expected_config = dict(
        rowwise=True,
        columnwise=False,
        with_amax_reduction=False,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=False,
        stochastic_rounding=False,
        row_scaled_nvfp4=False,
        nvfp4_use_4over6=enabled,
        nvfp4_e4m3_max=e4m3_max,
        nvfp4_4over6_err_mode=error_mode,
        with_random_sign_mask=False,
    )
    assert fake_te.configs == [expected_config, expected_config]
    for output, weight in zip(outputs, weights, strict=True):
        _assert_output_matches_input(output, weight, e4m3_max=e4m3_max)


def test_empty_and_singleton_batches_skip_split(fake_te):
    assert nvfp4.nvfp4_quantize_1d_batch([]) == []
    assert not fake_te.configs
    weight = torch.ones((3, 128), dtype=torch.bfloat16)
    outputs = nvfp4.nvfp4_quantize_1d_batch([weight])
    assert len(outputs) == len(fake_te.scalar) == 1
    assert not fake_te.split
    assert fake_te.scalar[0].shape == (16, 128)
    _assert_output_matches_input(outputs[0], weight)


@pytest.mark.parametrize("count,split_counts,scalar_count", [(64, [64], 0), (65, [64], 1), (130, [64, 64, 2], 0)])
def test_batch_respects_te_split_capacity(fake_te, count, split_counts, scalar_count):
    weights = [torch.full((3, 128), i + 1, dtype=torch.bfloat16) for i in range(count)]
    outputs = nvfp4.nvfp4_quantize_1d_batch(weights)
    assert [len(rows) for _, rows in fake_te.split] == split_counts
    assert len(fake_te.scalar) == scalar_count
    for output, weight in zip(outputs, weights, strict=True):
        _assert_output_matches_input(output, weight)


def test_batch_initializes_te_before_importing_extension(fake_te, monkeypatch):
    extension = sys.modules["transformer_engine_torch"]
    monkeypatch.delitem(sys.modules, "transformer_engine_torch")
    make_quantizer = nvfp4._make_nvfp4_quantizer

    def initialize_te(e4m3_max):
        quantizer = make_quantizer(e4m3_max)
        monkeypatch.setitem(sys.modules, "transformer_engine_torch", extension)
        return quantizer

    monkeypatch.setattr(nvfp4, "_make_nvfp4_quantizer", initialize_te)
    weights = [torch.ones((64, 128), dtype=torch.bfloat16) for _ in range(2)]
    outputs = nvfp4.nvfp4_quantize_1d_batch(weights)
    assert len(fake_te.split) == 1
    assert len(outputs) == 2


@pytest.mark.parametrize("shape", [(2, 3, 128), (16, 127)])
def test_batch_rejects_invalid_shapes_before_te(fake_te, shape):
    with pytest.raises(ValueError, match="2D weights with K divisible by 16"):
        nvfp4.nvfp4_quantize_1d_batch([torch.empty(shape)])
    assert not fake_te.configs
