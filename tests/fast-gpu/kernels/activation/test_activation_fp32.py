"""The fp32 SwiGLU and residual short conv Triton kernels against float64 torch references, forward and
backward, including the per-token scale and packed-sequence boundaries."""

import sys

import pytest
import torch
import torch.nn.functional as F
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"], num_gpus=1
)

pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from miles.kernels.activation.short_conv_fp32 import short_conv_fp32  # noqa: E402
from miles.kernels.activation.swiglu_fp32 import swiglu_fp32  # noqa: E402


def _rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.double() - expected).norm() / expected.norm()).item()


def _swiglu_reference(fc1_out: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
    gate, up = fc1_out.chunk(2, dim=-1)
    out = F.silu(gate) * up
    return out if scale is None else out * scale[:, None]


@pytest.mark.parametrize("with_scale", [False, True])
def test_swiglu_matches_float64_reference(with_scale):
    torch.manual_seed(0)
    tokens, width = 257, 384
    fc1_out = torch.randn(tokens, 2 * width, device="cuda", dtype=torch.bfloat16)
    scale = torch.rand(tokens, device="cuda", dtype=torch.float32) + 0.5 if with_scale else None
    grad = torch.randn(tokens, width, device="cuda", dtype=torch.bfloat16)

    fc1_kernel = fc1_out.clone().requires_grad_()
    scale_kernel = scale.clone().requires_grad_() if with_scale else None
    out = swiglu_fp32(fc1_kernel, scale_kernel)
    out.backward(grad)

    fc1_ref = fc1_out.double().requires_grad_()
    scale_ref = scale.double().requires_grad_() if with_scale else None
    out_ref = _swiglu_reference(fc1_ref, scale_ref)
    out_ref.backward(grad.double())

    assert out.dtype == fc1_out.dtype
    assert _rel_err(out, out_ref) < 5e-3
    assert _rel_err(fc1_kernel.grad, fc1_ref.grad) < 5e-3
    if with_scale:
        assert _rel_err(scale_kernel.grad, scale_ref.grad) < 5e-3


_SWIGLU_LIMIT = 7.0


def _clamped_swiglu_reference(fc1_out: torch.Tensor) -> torch.Tensor:
    gate, up = fc1_out.chunk(2, dim=-1)
    return F.silu(gate).clamp(max=_SWIGLU_LIMIT) * up.clamp(-_SWIGLU_LIMIT, _SWIGLU_LIMIT)


def test_clamped_swiglu_bounds_both_halves():
    gate = torch.tensor([[-3.0, 0.5, 20.0]], device="cuda")
    up = torch.tensor([[9.0, -0.25, -12.0]], device="cuda")
    fc1_out = torch.cat([gate, up], dim=-1).bfloat16()

    out = swiglu_fp32(fc1_out, limit=_SWIGLU_LIMIT)

    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out, _clamped_swiglu_reference(fc1_out.float()).bfloat16(), rtol=0, atol=0)


def test_clamped_swiglu_matches_float64_reference():
    torch.manual_seed(0)
    tokens, width = 257, 384
    # wide enough that both clamps engage on a good share of the elements
    fc1_out = (torch.randn(tokens, 2 * width, device="cuda") * 6).bfloat16()
    grad = torch.randn(tokens, width, device="cuda", dtype=torch.bfloat16)

    fc1_kernel = fc1_out.clone().requires_grad_()
    out = swiglu_fp32(fc1_kernel, limit=_SWIGLU_LIMIT)
    out.backward(grad)

    fc1_ref = fc1_out.double().requires_grad_()
    out_ref = _clamped_swiglu_reference(fc1_ref)
    out_ref.backward(grad.double())

    gate, up = fc1_out.float().chunk(2, dim=-1)
    assert (F.silu(gate) > _SWIGLU_LIMIT).any() and (up.abs() > _SWIGLU_LIMIT).any()
    assert _rel_err(out, out_ref) < 5e-3
    assert _rel_err(fc1_kernel.grad, fc1_ref.grad) < 5e-3


def test_clamped_swiglu_is_bit_identical_to_sglang():
    """The rollout's dense MLPs and shared experts run sglang's SiluClampAndMul; training must round alike."""
    sglang_op = pytest.importorskip("sglang.kernels.ops.activation.silu_clamp_and_mul_triton")
    torch.manual_seed(0)
    fc1_out = (torch.randn(513, 2 * 1536, device="cuda") * 6).bfloat16()
    expected = torch.empty(513, 1536, device="cuda", dtype=torch.bfloat16)
    sglang_op._silu_clamp_and_mul(fc1_out, expected, limit=_SWIGLU_LIMIT)

    assert torch.equal(swiglu_fp32(fc1_out, limit=_SWIGLU_LIMIT), expected)


def _sconv_reference(x: torch.Tensor, weight: torch.Tensor, seqlens: list[int]) -> torch.Tensor:
    kernel_size = weight.shape[-1]
    segments = [
        segment + F.conv1d(F.pad(segment.t()[None], (kernel_size - 1, 0)), weight, groups=weight.shape[0])[0].t()
        for segment in x.split(seqlens)
    ]
    return torch.cat(segments)


def test_swiglu_addresses_rows_past_int32_offsets():
    """Rows whose element offset ``row * 2N`` passes 2**31 are read and written where they belong, not wrapped."""
    width = 8192
    tokens = 2**31 // (2 * width) + 2
    fc1_out = torch.zeros(tokens, 2 * width, device="cuda", dtype=torch.bfloat16)
    rows = torch.tensor([0, tokens - 2, tokens - 1], device="cuda")
    fc1_out[rows] = torch.randn(3, 2 * width, device="cuda", dtype=torch.bfloat16)
    fc1_out.requires_grad_()

    out = swiglu_fp32(fc1_out)
    grad = torch.zeros_like(out)
    grad[rows] = torch.randn(3, width, device="cuda", dtype=torch.bfloat16)
    out.backward(grad)

    tail = fc1_out.detach()[rows].double().requires_grad_()
    expected = _swiglu_reference(tail, None)
    expected.backward(grad[rows].double())
    assert _rel_err(out[rows], expected.detach()) < 1e-2
    assert _rel_err(fc1_out.grad[rows], tail.grad) < 1e-2


@pytest.mark.parametrize("seqlens", [[300], [70, 1, 129, 100]], ids=["single", "packed"])
def test_sconv_matches_float64_reference(seqlens):
    torch.manual_seed(0)
    tokens, channels, kernel_size = sum(seqlens), 256, 4
    x = torch.randn(tokens, channels, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(channels, 1, kernel_size, device="cuda", dtype=torch.float32) * 0.3
    grad = torch.randn_like(x)

    x_kernel = x.clone().requires_grad_()
    weight_kernel = weight.clone().requires_grad_()
    out = short_conv_fp32(x_kernel, weight_kernel, seqlens)
    out.backward(grad)

    x_ref = x.double().requires_grad_()
    weight_ref = weight.double().requires_grad_()
    out_ref = _sconv_reference(x_ref, weight_ref, seqlens)
    out_ref.backward(grad.double())

    assert out.dtype == x.dtype
    assert _rel_err(out, out_ref) < 5e-3
    assert _rel_err(x_kernel.grad, x_ref.grad) < 5e-3
    assert _rel_err(weight_kernel.grad, weight_ref.grad) < 5e-3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
