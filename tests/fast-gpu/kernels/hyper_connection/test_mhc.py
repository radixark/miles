"""The DeepSeek-V4.1 hyper-connection residual mix and stream aggregation Triton kernels against
float64 torch references, forward and backward."""

import sys

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"], num_gpus=1
)

pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from miles.kernels.hyper_connection.mhc import mhc_aggregate, mhc_mix  # noqa: E402

NUM_STREAMS = 4


def _rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.double() - expected).norm() / expected.norm()).item()


def _mix_reference(x, residual, h_res, h_post, n):
    s, b, _ = residual.shape
    streams = residual.view(s, b, n, -1)
    mixed = torch.einsum("sbij,sbid->sbjd", h_res, streams)
    return (h_post.unsqueeze(-1) * x.unsqueeze(2) + mixed).view(s, b, -1)


def _aggregate_reference(x, pre, n):
    s, b, nc = x.shape
    return (pre.unsqueeze(-1) * x.view(s, b, n, nc // n)).sum(dim=2)


def _leaves(*tensors):
    kernel = [t.clone().requires_grad_() for t in tensors]
    reference = [t.double().requires_grad_() for t in tensors]
    return kernel, reference


@pytest.mark.parametrize("hidden", [1000, 2048])
def test_mix_matches_float64_reference(hidden):
    torch.manual_seed(0)
    seq, batch, n = 37, 2, NUM_STREAMS
    x = torch.randn(seq, batch, hidden, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(seq, batch, n * hidden, device="cuda", dtype=torch.bfloat16)
    h_res = torch.rand(seq, batch, n, n, device="cuda", dtype=torch.float32)
    h_post = torch.rand(seq, batch, n, device="cuda", dtype=torch.float32)
    grad = torch.randn(seq, batch, n * hidden, device="cuda", dtype=torch.bfloat16)

    kernel, reference = _leaves(x, residual, h_res, h_post)
    out = mhc_mix(*kernel, n)
    expected = _mix_reference(*reference, n)
    assert out.dtype == residual.dtype
    assert _rel_err(out, expected) < 5e-3

    out.backward(grad)
    expected.backward(grad.double())
    for actual, ref in zip(kernel, reference, strict=True):
        assert actual.grad.dtype == actual.dtype
        assert _rel_err(actual.grad, ref.grad) < 5e-3


@pytest.mark.parametrize("hidden", [1000, 2048])
def test_aggregate_matches_float64_reference(hidden):
    torch.manual_seed(0)
    seq, batch, n = 37, 2, NUM_STREAMS
    x = torch.randn(seq, batch, n * hidden, device="cuda", dtype=torch.bfloat16)
    pre = torch.rand(seq, batch, n, device="cuda", dtype=torch.float32)
    grad = torch.randn(seq, batch, hidden, device="cuda", dtype=torch.bfloat16)

    kernel, reference = _leaves(x, pre)
    out = mhc_aggregate(*kernel, n)
    expected = _aggregate_reference(*reference, n)
    assert out.dtype == x.dtype
    assert _rel_err(out, expected) < 5e-3

    out.backward(grad)
    expected.backward(grad.double())
    for actual, ref in zip(kernel, reference, strict=True):
        assert _rel_err(actual.grad, ref.grad) < 5e-3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
