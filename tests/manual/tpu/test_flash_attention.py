"""miles.kernels.tpu.flash_attention against fp64 dense attention, as close as bf16 dense SDPA on the same TPU gets.

Run on a TPU host: python -m pytest tests/manual/tpu/test_flash_attention.py
"""

import pytest
import torch
import torch.nn.functional as F

torch_tpu = pytest.importorskip("torch_tpu")

from miles.kernels.tpu.flash_attention import flash_attention  # noqa: E402

# segments of 1500, 1300 and 900 tokens, then a 396-token padding tail: most cross a 1024-token kernel block
_SEGMENT_LENGTHS = [1500, 1300, 900, 396]


def _errors(out, grads, ref, ref_grads):
    return [
        (got.float().cpu().double() - want).abs().max().item()
        for got, want in zip([out, *grads], [ref, *ref_grads], strict=True)
    ]


@pytest.mark.parametrize("scale", [None, 0.05], ids=["default-scale", "custom-scale"])
def test_matches_dense_causal_attention_within_each_segment(scale):
    device = torch.device("tpu")
    positions = torch.cat([torch.arange(n) for n in _SEGMENT_LENGTHS])
    seq_len = positions.shape[0]
    segment_ids = (torch.arange(seq_len) - positions).to(torch.int32).unsqueeze(0)
    mask = (segment_ids[:, :, None] == segment_ids[:, None, :]) & torch.ones(seq_len, seq_len, dtype=torch.bool).tril()
    torch.manual_seed(0)
    q, grad_out = torch.randn(1, 8, seq_len, 128), torch.randn(1, 8, seq_len, 128)
    k, v = torch.randn(1, 2, seq_len, 128), torch.randn(1, 2, seq_len, 128)

    ref_inputs = [t.double().requires_grad_() for t in (q, k, v)]
    ref = F.scaled_dot_product_attention(*ref_inputs, attn_mask=mask.unsqueeze(1), enable_gqa=True, scale=scale)
    ref.backward(grad_out.double())
    ref_grads = [t.grad for t in ref_inputs]

    def run(attention):
        inputs = [t.to(device, torch.bfloat16).requires_grad_() for t in (q, k, v)]
        out = attention(*inputs)
        out.backward(grad_out.to(device, torch.bfloat16))
        return _errors(out, [t.grad for t in inputs], ref, ref_grads)

    kernel = run(
        torch.compile(
            lambda q, k, v: flash_attention(q, k, v, segment_ids=segment_ids.to(device), scale=scale),
            backend="tpu",
            fullgraph=True,
        )
    )
    dense = run(
        lambda q, k, v: F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask.unsqueeze(1).to(device), enable_gqa=True, scale=scale
        )
    )
    # out, dq, dk, dv: within bf16 dense SDPA's own error, with room for a different reduction order
    for name, got, bound in zip(("out", "dq", "dk", "dv"), kernel, dense, strict=True):
        assert got <= 1.5 * bound + 1e-3, f"{name}: {got} vs bf16 dense SDPA's {bound}"
