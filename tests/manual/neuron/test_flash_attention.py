"""miles.kernels.neuron.flash_attention against fp64 dense attention, as close as an fp32 reference on bf16 inputs gets.

Run on a trn host: python -m pytest tests/manual/neuron/test_flash_attention.py
"""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torch_neuronx")

from miles.kernels.neuron.flash_attention import packed_causal_attention  # noqa: E402

# packed segments, the last one a padding tail
_SEGMENT_LENGTHS = [300, 800, 800, 148]
# the kernels run bf16 matmuls; the reference only rounds its inputs to bf16
_TOLERANCE_FACTOR = 2.5


def _relative_error(got: torch.Tensor, want: torch.Tensor) -> float:
    got = got.detach().cpu().double()
    return ((got - want).norm() / want.norm()).item()


def test_matches_dense_causal_attention_within_each_segment():
    positions = torch.cat([torch.arange(n) for n in _SEGMENT_LENGTHS])
    num_tokens = positions.shape[0]
    segment_start_T = (torch.arange(num_tokens) - positions).to(torch.int32)
    mask = (segment_start_T[:, None] == segment_start_T[None, :]) & torch.ones(num_tokens, num_tokens).tril().bool()
    scale = 128**-0.5
    torch.manual_seed(0)
    q, grad_out = torch.randn(8, num_tokens, 128), torch.randn(8, num_tokens, 128)
    k, v = torch.randn(2, num_tokens, 128), torch.randn(2, num_tokens, 128)

    def dense(inputs):
        out = F.scaled_dot_product_attention(*inputs, attn_mask=mask, scale=scale, enable_gqa=True)
        out.backward(grad_out.to(out.dtype))
        return [out, *(t.grad for t in inputs)]

    ref = dense([t.double().requires_grad_() for t in (q, k, v)])
    bf16_inputs = dense([t.bfloat16().double().requires_grad_() for t in (q, k, v)])

    device = torch.device("neuron", 0)
    inputs = [t.to(device, torch.bfloat16).requires_grad_() for t in (q, k, v)]
    out, _ = packed_causal_attention(*inputs, segment_start_T.to(device), scale)
    out.float().backward(grad_out.to(device))
    kernel = [out, *(t.grad for t in inputs)]

    for name, got, floor, want in zip(["out", "dq", "dk", "dv"], kernel, bf16_inputs, ref, strict=True):
        error, floor_error = _relative_error(got, want), _relative_error(floor, want)
        assert error <= _TOLERANCE_FACTOR * floor_error, f"{name}: {error:.4%} vs bf16-input floor {floor_error:.4%}"
