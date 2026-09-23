"""Regressions of the DeepSeek-V4 sparse-MLA kernels, each reproducing a failure they had.

These run the real kernels, so they need tilelang and a GPU; nothing here depends on the device.
"""

import pytest
import torch

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

register_cuda_ci(est_time=60, suite="stage-b-2-gpu-h200", labels=["miles-plugin"], hardware=["hopper", "blackwell"])
register_rocm_ci(est_time=60, suite="stage-c-4-gpu-mi350", labels=["miles-plugin"])

tilelang = pytest.importorskip("tilelang", reason="the DeepSeek-V4 kernels are tilelang modules")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="runs the kernels on a GPU")

from miles_plugins.models.deepseek_v4.ops.kernel.tilelang_sparse_mla import sparse_attn_tilelang  # noqa: E402

B, S, H, D, S_KV, TOPK = 1, 128, 16, 512, 160, 64
SM_SCALE = D**-0.5


def _inputs(seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(B, S_KV, D, device="cuda", dtype=torch.bfloat16)
    attn_sink = torch.randn(H, device="cuda", dtype=torch.float32)
    rows = [torch.randperm(S_KV, device="cuda")[:TOPK] for _ in range(S)]
    topk_idxs = torch.stack(rows).to(torch.int32).unsqueeze(0)
    return q, kv, attn_sink, topk_idxs


def _run(q, kv, attn_sink, topk_idxs, backward):
    """The output and (dq, dkv, d_attn_sink); `backward` gets the output and runs autograd."""
    leaves = [t.detach().clone().requires_grad_(True) for t in (q, kv, attn_sink)]
    o = sparse_attn_tilelang(leaves[0], leaves[1], leaves[2], topk_idxs, SM_SCALE)
    backward(o)
    return o.detach(), [t.grad for t in leaves]


def test_a_broadcast_output_gradient_is_read_as_a_broadcast():
    """o.sum().backward() hands the backward a stride-0 do; read as dense, it faulted the GPU."""
    inputs = _inputs()
    _, broadcast = _run(*inputs, lambda o: o.sum().backward())
    _, dense = _run(*inputs, lambda o: o.backward(torch.ones_like(o)))
    for got, want in zip(broadcast, dense, strict=True):
        # Two backwards over the same do agree only up to the order of float atomics.
        torch.testing.assert_close(got, want, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
