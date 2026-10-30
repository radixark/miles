"""On a TPU host with the patched torchtitan: `_TpuGroupedMM`'s gradients equal a per-group fp32 reference.

torch_tpu's grouped weight-gradient kernels are wrong below 256 rows a group and leave empty groups unwritten,
so the patch pads every group; rows past the last offset (a dispatcher's padding) belong to no expert.

    python -m pytest -q tests/manual/tpu/test_tpu_grouped_mm.py
"""

import pytest
import torch

torch_tpu = pytest.importorskip("torch_tpu")
moe = pytest.importorskip("torchtitan.models.common.moe")


def _reference_grads(a, b_t, offs, grad_out):
    """Per-group fp32 matmuls on the host; rows past offs[-1] contribute to nothing."""
    a, b_t, grad_out = a.float().cpu(), b_t.float().cpu(), grad_out.float().cpu()
    grad_a, grad_b = torch.zeros_like(a), torch.zeros_like(b_t)
    start = 0
    for group, end in enumerate(offs.tolist()):
        grad_a[start:end] = grad_out[start:end] @ b_t[group].T
        grad_b[group] = a[start:end].T @ grad_out[start:end]
        start = end
    return grad_a, grad_b


@pytest.mark.parametrize(
    "sizes,tail_rows",
    [
        ([3, 17, 40, 1], 0),  # every group far below 256 rows
        ([0, 50, 0, 300], 0),  # empty groups, and one above 256
        ([3, 17, 40, 1], 64),  # rows past the last offset, as a padding dispatcher leaves them
        ([256, 512, 7, 0], 33),
    ],
)
def test_the_gradients_match_a_per_group_reference(sizes, tail_rows):
    torch.manual_seed(0)
    device = torch.device("tpu", 0)
    num_rows, k, n = sum(sizes) + tail_rows, 64, 32
    offs = torch.tensor(sizes, dtype=torch.int32).cumsum(0).to(torch.int32)
    a = torch.randn(num_rows, k, dtype=torch.bfloat16)
    b_t = torch.randn(len(sizes), k, n, dtype=torch.bfloat16)
    grad_out = torch.randn(num_rows, n, dtype=torch.bfloat16)

    a_dev = a.to(device).requires_grad_()
    b_dev = b_t.to(device).requires_grad_()
    moe._TpuGroupedMM.apply(a_dev, b_dev, offs.to(device)).backward(grad_out.to(device))

    expected_a, expected_b = _reference_grads(a, b_t, offs, grad_out)
    rows = offs[-1].item()
    # bf16 inputs, fp32 accumulation: compare against the reference's scale
    torch.testing.assert_close(b_dev.grad.float().cpu(), expected_b, rtol=0, atol=2e-2 * expected_b.abs().max().item())
    torch.testing.assert_close(
        a_dev.grad.float().cpu()[:rows], expected_a[:rows], rtol=0, atol=2e-2 * expected_a.abs().max().item()
    )
