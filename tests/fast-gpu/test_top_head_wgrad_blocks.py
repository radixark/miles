"""The TOP deferred LM head's weight gradient, accumulated in vocab-row blocks, has the bytes of one unblocked
batch-invariant mm per chunk, and runs on the batch-invariant kernel, not cuBLAS."""

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"], num_gpus=1
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from miles.backends.training_utils.loss.hub.math_utils import _accumulate_head_wgrad  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _batch_invariant():
    from sglang.srt.batch_invariant_ops.batch_invariant_ops import enable_batch_invariant_mode

    enable_batch_invariant_mode()


@pytest.mark.parametrize(("vocab", "hidden", "rows"), [(248320, 1024, 777), (151936, 2048, 4097), (40000, 512, 64)])
def test_blocked_head_wgrad_has_the_unblocked_bytes(vocab, hidden, rows):
    g = torch.Generator(device="cuda").manual_seed(0)
    chunks = [
        (
            torch.randn(rows, vocab, generator=g, device="cuda", dtype=torch.bfloat16),
            torch.randn(rows, hidden, generator=g, device="cuda", dtype=torch.bfloat16),
        )
        for _ in range(2)
    ]
    expected = torch.zeros(vocab + 3, hidden, device="cuda", dtype=torch.float32)
    blocked = expected.clone()
    for d, h in chunks:
        expected[:vocab] += torch.mm(d.t(), h, out_dtype=torch.float32)
        _accumulate_head_wgrad(blocked, d, h)
    assert torch.equal(blocked.view(torch.int32), expected.view(torch.int32))


def test_blocked_head_wgrad_runs_the_batch_invariant_kernel():
    from torch.profiler import ProfilerActivity, profile

    d = torch.randn(256, 20000, device="cuda", dtype=torch.bfloat16)
    h = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
    grad_weight = torch.zeros(20000, 512, device="cuda", dtype=torch.float32)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        _accumulate_head_wgrad(grad_weight, d, h)
        torch.cuda.synchronize()
    kernels = {e.key for e in prof.key_averages() if e.device_type.name == "CUDA"}
    assert any("matmul_kernel_persistent" in k for k in kernels), kernels
    assert not any("nvjet" in k or "gemm" in k.lower() and "persistent" not in k for k in kernels), kernels
