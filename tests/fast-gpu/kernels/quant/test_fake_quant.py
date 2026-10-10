"""The block fake-quant Triton kernels against a torch reference: fp8 and fp4 with power-of-two
scales over 32-wide blocks, fp4 with e4m3 scales over 16-wide blocks, and the straight-through
backward."""

import sys

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"], num_gpus=1
)

pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from miles.kernels.quant.fake_quant import fake_quant_compressed_kv, fake_quant_fp4, fake_quant_fp8  # noqa: E402

FP8_MAX = 448.0
FP4_MAX = 6.0


def _ceil_pow2(value: torch.Tensor) -> torch.Tensor:
    mantissa, exponent = torch.frexp(value)
    return torch.ldexp(torch.ones_like(value), torch.where(mantissa == 0.5, exponent - 1, exponent))


def _round_to_fp4_grid(scaled: torch.Tensor) -> torch.Tensor:
    magnitude = scaled.abs()
    step = torch.where(magnitude < 2.0, 0.5, torch.where(magnitude < 4.0, 1.0, 2.0))
    return torch.round(magnitude / step) * step * torch.sign(scaled)


def _reference(x: torch.Tensor, kind: str) -> torch.Tensor:
    block = 16 if kind == "compressed_kv" else 32
    blocks = x.float().reshape(-1, block)
    amax = blocks.abs().amax(dim=1, keepdim=True)
    if kind == "compressed_kv":
        scale = (amax / FP4_MAX).clamp(2.0**-9, FP8_MAX).to(torch.float8_e4m3fn).float()
        scaled = (blocks / scale).clamp(-FP4_MAX, FP4_MAX)
        quantized = _round_to_fp4_grid(scaled)
    elif kind == "fp4":
        scale = _ceil_pow2(amax.clamp(min=6 * 2.0**-126) / FP4_MAX)
        quantized = _round_to_fp4_grid((blocks / scale).clamp(-FP4_MAX, FP4_MAX))
    else:
        scale = _ceil_pow2(amax.clamp(min=1e-4) / FP8_MAX)
        quantized = (blocks / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()
    return (quantized * scale).reshape(x.shape).to(x.dtype)


KERNELS = {"fp8": fake_quant_fp8, "fp4": fake_quant_fp4, "compressed_kv": fake_quant_compressed_kv}


@pytest.mark.parametrize("kind", list(KERNELS))
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fake_quant_matches_reference(kind, dtype):
    torch.manual_seed(0)
    x = torch.randn(3, 37, 512, device="cuda", dtype=dtype) * torch.logspace(-3, 3, 512, device="cuda").to(dtype)
    x[0, 0, :32] = 0
    torch.testing.assert_close(KERNELS[kind](x), _reference(x, kind), rtol=0, atol=0)


@pytest.mark.parametrize("kind", list(KERNELS))
def test_fake_quant_backward_is_straight_through(kind):
    torch.manual_seed(0)
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    grad = torch.randn_like(x)
    KERNELS[kind](x).backward(grad)
    torch.testing.assert_close(x.grad, grad, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
