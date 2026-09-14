from types import SimpleNamespace

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

from miles.utils.mxfp4 import E2M1_VALUES, MXFP4_GROUP_SIZE, mxfp4_quantize
from miles_plugins.models.deepseek_v4.ops.mxfp4_qat import (
    _wrap_get_weight_tensors,
    mxfp4_fake_quantize_ste,
    mxfp4_quantize_dequantize,
)

register_cuda_ci(est_time=30, suite="stage-b-2-gpu-h200", labels=["precision"])


def _dequantize_reference(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    logical_shape = (*packed.shape[:-1], packed.shape[-1] * 2)
    codes = packed.view(torch.uint8)
    nibbles = torch.stack((codes & 0x0F, (codes >> 4) & 0x0F), dim=-1).flatten(-2)
    table = torch.tensor(E2M1_VALUES + tuple(-value for value in E2M1_VALUES), device=packed.device)
    values = table[nibbles.long()].reshape(-1, MXFP4_GROUP_SIZE)
    scale_factor = torch.exp2(scale.view(torch.uint8).float() - 127).reshape(-1, 1)
    return (values * scale_factor).reshape(logical_shape)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_mxfp4_qat_matches_rollout_codec(dtype):
    torch.manual_seed(0)
    weight = torch.randn((67, 4 * MXFP4_GROUP_SIZE), device="cuda", dtype=dtype)
    original = weight.clone()

    actual = mxfp4_quantize_dequantize(weight)
    packed, scale = mxfp4_quantize(weight)
    expected = _dequantize_reference(packed, scale).to(dtype)

    assert torch.equal(actual, expected)
    assert torch.equal(weight, original)


@pytest.mark.parametrize(
    ("linear_name", "shape"),
    [
        ("linear_fc1", (6 * MXFP4_GROUP_SIZE, 4 * MXFP4_GROUP_SIZE)),
        ("linear_fc2", (4 * MXFP4_GROUP_SIZE, 3 * MXFP4_GROUP_SIZE)),
    ],
)
def test_mxfp4_qat_grid_matches_rollout_after_native_layout_conversion(linear_name, shape):
    torch.manual_seed(1)
    weight = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    def convert_native_layout(value):
        return value.chunk(2, dim=0) if linear_name == "linear_fc1" else (value,)

    qat_converted = convert_native_layout(mxfp4_quantize_dequantize(weight))
    rollout_converted = convert_native_layout(weight)

    for qat_weight, rollout_weight in zip(qat_converted, rollout_converted, strict=True):
        packed, scale = mxfp4_quantize(rollout_weight)
        expected = _dequantize_reference(packed, scale).to(weight.dtype)
        assert torch.equal(qat_weight, expected)


def test_mxfp4_qat_ste_passes_gradient_through_and_preserves_main_grad():
    weight = torch.randn((4, 2 * MXFP4_GROUP_SIZE), device="cuda", requires_grad=True)
    weight.main_grad = torch.empty_like(weight)
    grad = torch.randn_like(weight)

    output = mxfp4_fake_quantize_ste(weight)
    output.backward(grad)

    assert torch.equal(weight.grad, grad)
    assert output.main_grad is weight.main_grad


def test_grouped_linear_patch_is_config_gated(monkeypatch):
    weight = torch.randn((2, MXFP4_GROUP_SIZE), device="cuda")
    module = SimpleNamespace(config=SimpleNamespace(dsv4_mxfp4_qat=False))
    wrapped = _wrap_get_weight_tensors(lambda _: [weight])

    assert wrapped(module) == [weight]

    sentinel = torch.zeros_like(weight)
    monkeypatch.setattr(
        "miles_plugins.models.deepseek_v4.ops.mxfp4_qat.mxfp4_fake_quantize_ste",
        lambda _: sentinel,
    )
    module.config.dsv4_mxfp4_qat = True
    assert wrapped(module) == [sentinel]
