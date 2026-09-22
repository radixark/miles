"""The head-sharded short conv splits inputs past fla's int32 indexing limit into channel chunks; the
chunked path must match the single call bit for bit."""

import pytest
import torch

pytest.importorskip("fla")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from miles_plugins.models.layers import delta_rule_attention  # noqa: E402


@pytest.mark.parametrize("channels", [768, 1000])
def test_chunked_conv_is_bitwise_equal(monkeypatch, channels):
    torch.manual_seed(0)
    tokens = 2048
    conv = delta_rule_attention._ShardedShortConvolution(
        hidden_size=channels,
        kernel_size=4,
        bias=False,
        activation="silu",
        device="cuda",
        dtype=torch.bfloat16,
        tp_group=None,
    )
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
    monkeypatch.setattr(delta_rule_attention, "INT32_ELEMENTS", tokens * 300)
    chunked = run()
    for a, b in zip(whole, chunked, strict=True):
        assert torch.equal(a, b)
