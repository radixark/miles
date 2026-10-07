"""The head-sharded short conv splits inputs past fla's int32 indexing limit into channel chunks. The
conv is per channel, so the chunked forward matches the single call bit for bit. fla picks its backward
tiling from the channel count, so on Blackwell the gradients differ from the single call by fp32
accumulation order (~1e-6 relative); a wrong channel mapping is off by ~1."""

import sys

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"])

pytest.importorskip("fla")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from miles_plugins.models import linear_attn  # noqa: E402
from miles_plugins.models.linear_attn import ShardedShortConv  # noqa: E402


@pytest.mark.parametrize("channels", [768, 1000])
def test_chunked_conv_matches_single_call(monkeypatch, channels):
    torch.manual_seed(0)
    tokens = 2048
    conv = ShardedShortConv(channels, 4, tp_group=None, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(1, tokens, channels, device="cuda", dtype=torch.bfloat16)
    grad = torch.randn_like(x)
    cu_seqlens = torch.tensor([0, 700, tokens], device="cuda", dtype=torch.int32)

    def run():
        conv.weight.grad = None
        xi = x.clone().requires_grad_()
        out = conv(xi, cu_seqlens=cu_seqlens)
        out.backward(grad)
        return out.detach(), xi.grad, conv.weight.grad.clone()

    whole = run()
    monkeypatch.setattr(linear_attn, "INT32_ELEMENTS", tokens * 300)
    monkeypatch.setattr(linear_attn, "_CHUNK_ELEMENTS", tokens * 300)
    chunked = run()
    assert torch.equal(whole[0], chunked[0])
    for a, b in zip(whole[1:], chunked[1:], strict=True):
        assert ((a.float() - b.float()).norm() / b.float().norm()).item() < 1e-4


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
