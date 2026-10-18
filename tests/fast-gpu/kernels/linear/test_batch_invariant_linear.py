"""`batch_invariant_linear` must be batch-invariant and match the FP64 reference."""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=45, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"], num_gpus=1
)

import pytest
import torch

from miles.kernels.linear.batch_invariant_linear import batch_invariant_linear

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for the Triton kernel")


def _reference(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (hidden.double() @ weight.double().T).to(torch.bfloat16)


@pytest.mark.parametrize("tokens, hidden_size", [(1, 256), (37, 256), (2048, 256), (6000, 4096)])
@pytest.mark.parametrize("heads", [4, 8, 16])
def test_matches_fp64_reference(tokens, hidden_size, heads):
    torch.manual_seed(tokens + hidden_size + heads)
    hidden = torch.randn(tokens, hidden_size, device="cuda").to(torch.bfloat16)
    weight = (0.02 * torch.randn(heads, hidden_size, device="cuda")).to(torch.bfloat16)

    got = batch_invariant_linear(hidden, weight)
    want = _reference(hidden, weight)

    assert torch.equal(got, want)


def test_is_invariant_to_batch_slices_and_row_padding():
    torch.manual_seed(0)
    tokens, hidden_size, heads = 4096, 4096, 4
    buffer = torch.randn(tokens, hidden_size + 256, device="cuda").to(torch.bfloat16)
    hidden = buffer[:, :hidden_size]
    weight = (0.02 * torch.randn(heads, hidden_size, device="cuda")).to(torch.bfloat16)

    full = batch_invariant_linear(hidden, weight)
    pieces = [
        batch_invariant_linear(hidden[start : start + count], weight) for start, count in ((0, 1), (5, 37), (64, 2048))
    ]

    assert torch.equal(pieces[0], full[:1])
    assert torch.equal(pieces[1], full[5:42])
    assert torch.equal(pieces[2], full[64:2112])
