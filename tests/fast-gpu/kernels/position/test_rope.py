"""The DeepSeek-V4.1 triton RoPE must match the torch reference, forward and backward."""

import sys

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="stage-b-2-gpu-h200", labels=["miles-plugin"], hardware=["hopper", "blackwell"])

import pytest
import torch

from miles.kernels.position.rope import apply_rotary_emb
from miles_plugins.models.deepseek_v4.ops.rope import apply_rotary_emb as apply_rotary_emb_reference

ROPE_DIM = 64
HEAD_DIM = 512


def _freqs_cis(seqlen: int) -> torch.Tensor:
    angles = torch.rand(seqlen, ROPE_DIM // 2, device="cuda") * 6.28
    return torch.polar(torch.ones_like(angles), angles)


def _rotate_tail(rope_fn, base: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool, permute: bool):
    x = base.clone()
    if permute:
        x = x.transpose(1, 2).contiguous().transpose(1, 2)
    rope_fn(x[..., -ROPE_DIM:], freqs_cis, inverse=inverse)
    return x


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("shape", [(2, 37, 4, HEAD_DIM), (2, 37, HEAD_DIM)])
@pytest.mark.parametrize("permute", [False, True])
def test_rope_matches_reference(shape, inverse, permute):
    if permute and len(shape) == 3:
        pytest.skip("a 3-d tensor has no head axis to permute")
    torch.manual_seed(0)
    freqs_cis = _freqs_cis(shape[1])
    weight = torch.randn(shape, device="cuda", dtype=torch.float32)
    grad_out = torch.randn(shape, device="cuda", dtype=torch.float32)

    base_t = torch.randn(shape, device="cuda", dtype=torch.float32, requires_grad=True)
    base_r = base_t.detach().clone().requires_grad_(True)

    out_t = _rotate_tail(apply_rotary_emb, base_t * weight, freqs_cis, inverse, permute)
    out_r = _rotate_tail(apply_rotary_emb_reference, base_r * weight, freqs_cis, inverse, permute)
    torch.testing.assert_close(out_t, out_r, rtol=1e-5, atol=1e-5)

    out_t.backward(grad_out)
    out_r.backward(grad_out)
    torch.testing.assert_close(base_t.grad, base_r.grad, rtol=1e-5, atol=1e-5)


def test_rope_backward_is_not_identity():
    torch.manual_seed(0)
    freqs_cis = _freqs_cis(37)
    base = torch.randn(2, 37, 4, HEAD_DIM, device="cuda", requires_grad=True)
    out = _rotate_tail(apply_rotary_emb, base * 1.0, freqs_cis, inverse=False, permute=False)
    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    assert not torch.allclose(base.grad[..., -ROPE_DIM:], grad_out[..., -ROPE_DIM:])
    torch.testing.assert_close(base.grad[..., :-ROPE_DIM], grad_out[..., :-ROPE_DIM])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
