"""Unit tests for the FSDP2 DTensor materialization primitive.

``gather_full_param`` always moves to CUDA first (a CPU DTensor picks the wrong collective
backend), so these are GPU-gated. The sharded-DTensor all-gather path is exercised end to
end by the per-model weight-export (lp_diff) runs; here we pin the non-DTensor contract.
"""

import pytest
import torch

from miles.backends.experimental.fsdp_utils.dtensor import gather_full_param


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gather_full_param moves to CUDA")
def test_non_dtensor_passthrough_to_cuda():
    p = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    out = gather_full_param(p)
    assert not isinstance(out, torch.distributed.tensor.DTensor)
    assert out.is_cuda
    torch.testing.assert_close(out.cpu(), p)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gather_full_param moves to CUDA")
def test_async_passthrough_is_noop_for_non_dtensor():
    # A plain tensor has no async handle to drain; async_op must not change the result.
    p = torch.ones(2, 2)
    out = gather_full_param(p, async_op=True)
    assert not hasattr(out, "wait")
    torch.testing.assert_close(out.cpu(), p)
