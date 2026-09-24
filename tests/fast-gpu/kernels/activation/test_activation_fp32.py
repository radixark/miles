"""The fp32 SwiGLU and residual short conv Triton kernels against float64 torch references, forward and
backward, including the per-token scale and packed-sequence boundaries."""

import pytest
import torch
import torch.nn.functional as F
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"])

pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from miles.kernels.activation.short_conv_fp32 import sconv_fp32_triton  # noqa: E402
from miles.kernels.activation.swiglu_fp32 import swiglu_fp32_triton  # noqa: E402


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
    out = swiglu_fp32_triton(fc1_kernel, scale_kernel)
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


def _sconv_reference(x: torch.Tensor, weight: torch.Tensor, seqlens: list[int]) -> torch.Tensor:
    kernel_size = weight.shape[-1]
    segments = [
        segment + F.conv1d(F.pad(segment.t()[None], (kernel_size - 1, 0)), weight, groups=weight.shape[0])[0].t()
        for segment in x.split(seqlens)
    ]
    return torch.cat(segments)


@pytest.mark.parametrize("seqlens", [[300], [70, 1, 129, 100]], ids=["single", "packed"])
def test_sconv_matches_float64_reference(seqlens):
    torch.manual_seed(0)
    tokens, channels, kernel_size = sum(seqlens), 256, 4
    x = torch.randn(tokens, channels, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(channels, 1, kernel_size, device="cuda", dtype=torch.float32) * 0.3
    grad = torch.randn_like(x)

    x_kernel = x.clone().requires_grad_()
    weight_kernel = weight.clone().requires_grad_()
    out = sconv_fp32_triton(x_kernel, weight_kernel, seqlens)
    out.backward(grad)

    x_ref = x.double().requires_grad_()
    weight_ref = weight.double().requires_grad_()
    out_ref = _sconv_reference(x_ref, weight_ref, seqlens)
    out_ref.backward(grad.double())

    assert out.dtype == x.dtype
    assert _rel_err(out, out_ref) < 5e-3
    assert _rel_err(x_kernel.grad, x_ref.grad) < 5e-3
    assert _rel_err(weight_kernel.grad, weight_ref.grad) < 5e-3
