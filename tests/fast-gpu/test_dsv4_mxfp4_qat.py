from types import MethodType, SimpleNamespace

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

from miles.utils.mxfp4 import E2M1_VALUES, MXFP4_GROUP_SIZE, mxfp4_quantize
from miles_plugins.models.deepseek_v4.ops.mxfp4_qat import (
    _wrap_get_weight_tensors,
    mxfp4_fake_quantize_ste,
    mxfp4_quantize_dequantize,
)

register_cuda_ci(
    est_time=30, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"], num_gpus=1
)


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


def _boundary_blocks() -> torch.Tensor:
    """Blocks whose amax/6 sits on or next to a power of two, with values on rounding thresholds."""
    blocks = []
    for exponent in range(-24, 9, 3):
        scale = 2.0**exponent
        for amax_factor in (1.0, 1.0 + 2.0**-23, 1.0 - 2.0**-24):
            for threshold in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
                block = torch.full((MXFP4_GROUP_SIZE,), threshold * scale, dtype=torch.float32)
                block[0] = 6.0 * scale * amax_factor
                block[1::2] *= -1
                block[2] = torch.nextafter(block[2], torch.tensor(float("inf")))
                blocks.append(block)
    return torch.stack(blocks).cuda()


def test_mxfp4_qat_matches_rollout_codec_at_scale_and_rounding_boundaries():
    weight = _boundary_blocks()

    packed, scale = mxfp4_quantize(weight)

    assert torch.equal(mxfp4_quantize_dequantize(weight), _dequantize_reference(packed, scale))


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
        lambda *_, **__: sentinel,
    )
    module.config.dsv4_mxfp4_qat = True
    assert wrapped(module) == [sentinel]


@pytest.mark.parametrize("wgrad_in_main_grad", [False, True])
def test_mxfp4_qat_ste_marks_fused_weight_gradients_as_added(wgrad_in_main_grad):
    weight = torch.randn((4, 2 * MXFP4_GROUP_SIZE), device="cuda", requires_grad=True)
    weight.main_grad = torch.zeros_like(weight)
    weight.grad_added_to_main_grad = False
    placeholder = torch.randn_like(weight)

    output = mxfp4_fake_quantize_ste(weight, wgrad_in_main_grad=wgrad_in_main_grad)
    output.backward(placeholder)

    assert output.main_grad is weight.main_grad
    assert output.grad_added_to_main_grad is False
    assert weight.grad_added_to_main_grad is wgrad_in_main_grad
    assert torch.equal(weight.grad, placeholder)


def _megatron_ddp_hook(param):
    """Megatron DDP's backward post-hook for one parameter."""
    if param.grad is not None and (not param.grad_added_to_main_grad or getattr(param, "zero_out_wgrad", False)):
        param.main_grad.add_(param.grad.data)
    param.grad = None


@pytest.mark.parametrize("fuse_wgrad_accumulation", [False, True])
def test_te_grouped_linear_adds_the_qat_weight_gradient_once(fuse_wgrad_accumulation):
    te = pytest.importorskip("transformer_engine.pytorch")
    torch.manual_seed(3)
    num_gemms, in_features, out_features = 2, 4 * MXFP4_GROUP_SIZE, 64
    m_splits = [24, 40]
    inp = torch.randn((sum(m_splits), in_features), device="cuda", dtype=torch.bfloat16)
    grad_out = torch.randn((sum(m_splits), out_features), device="cuda", dtype=torch.bfloat16)

    def grouped_linear(fused):
        return te.GroupedLinear(
            num_gemms,
            in_features,
            out_features,
            bias=False,
            params_dtype=torch.bfloat16,
            device="cuda",
            fuse_wgrad_accumulation=fused,
        )

    module = grouped_linear(fuse_wgrad_accumulation)
    module.config = SimpleNamespace(dsv4_mxfp4_qat=True, gradient_accumulation_fusion=fuse_wgrad_accumulation)
    module._get_weight_tensors = MethodType(_wrap_get_weight_tensors(type(module)._get_weight_tensors), module)
    params = list(module._get_weight_tensors.__wrapped__(module))
    for param in params:
        param.main_grad = torch.zeros(param.shape, dtype=torch.float32, device="cuda")
        param.grad_added_to_main_grad = False
        param.register_post_accumulate_grad_hook(_megatron_ddp_hook)

    reference = grouped_linear(False)
    with torch.no_grad():
        for ref_param, param in zip(reference._get_weight_tensors(), params, strict=True):
            ref_param.copy_(mxfp4_quantize_dequantize(param.detach()))

    module(inp, m_splits).backward(grad_out)
    reference(inp, m_splits).backward(grad_out)

    for param, ref_param in zip(params, reference._get_weight_tensors(), strict=True):
        assert param.grad is None
        torch.testing.assert_close(param.main_grad, ref_param.grad.float(), rtol=1e-2, atol=1e-2)
