from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from miles_plugins.top import glm_rope
from miles_plugins.top.glm_ops import TopGLMOps

BASE = 10000.0


def _cos_sin_cache(rows, dim):
    inv_freq = 1.0 / (BASE ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    freqs = torch.outer(torch.arange(rows, dtype=torch.float32), inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], -1)


def _rotate_rows(rope, positions, x, *, key):
    """A CPU stand-in for serving's row-wise fused RoPE kernel."""
    cache, neox = rope.cos_sin_cache, rope.is_neox_style
    cos, sin = cache[positions].chunk(2, -1)
    cos, sin = cos[:, None, :], sin[:, None, :]
    xf = x.float()
    if neox:
        x1, x2 = xf.chunk(2, -1)
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).to(x.dtype)
    x1, x2 = xf[..., 0::2], xf[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).flatten(-2).to(x.dtype)


@pytest.fixture(autouse=True)
def _cpu_rope():
    caches = {}

    def serving_rope(head_dim, rows, base, interleaved, device_index):
        cache = caches.setdefault((head_dim, rows), _cos_sin_cache(rows, head_dim))
        return SimpleNamespace(cos_sin_cache=cache, is_neox_style=not interleaved)

    with patch.object(glm_rope, "_serving_rope", serving_rope), patch.object(
        glm_rope, "_serving_rotate", _rotate_rows
    ):
        yield


def _rotate(q, cu, interleaved, cp_size, cp_rank, grad):
    q = q.clone().requires_grad_(True)
    with patch("megatron.core.parallel_state.get_context_parallel_world_size", return_value=cp_size), patch(
        "megatron.core.parallel_state.get_context_parallel_rank", return_value=cp_rank
    ):
        out = TopGLMOps.rotary(q, cu, BASE, interleaved, gathered=False)
    out.backward(grad)
    return out.detach(), q.grad


@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize(("cp_size", "lengths"), [(2, [700, 1300, 48]), (4, [4096]), (2, [1, 1535, 1024, 512])])
def test_each_cp_rank_rotates_its_rows_as_cp1_does(interleaved, cp_size, lengths):
    """Under all-gather CP a rank holds a contiguous slice of the packing; RoPE is row-wise, so its rows rotate and
    back-propagate exactly as the same rows of the unsplit packing."""
    torch.manual_seed(0)
    total = sum(lengths)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    q = torch.randn(total, 4, 64, dtype=torch.bfloat16)
    grad = torch.randn(total, 4, 64, dtype=torch.bfloat16)
    full_out, full_grad = _rotate(q, cu, interleaved, 1, 0, grad)
    rows = total // cp_size
    for cp_rank in range(cp_size):
        local = slice(cp_rank * rows, (cp_rank + 1) * rows)
        out, q_grad = _rotate(q[local], cu, interleaved, cp_size, cp_rank, grad[local])
        assert torch.equal(out, full_out[local])
        assert torch.equal(q_grad, full_grad[local])


def test_a_negative_zero_gradient_leaves_a_cp_rank_as_positive_zero():
    q = torch.randn(8, 1, 4, dtype=torch.bfloat16).requires_grad_(True)
    (glm_rope._PositiveZeroGrad.apply(q) * 1.0).backward(torch.full_like(q, -0.0))
    assert not torch.signbit(q.grad).any()
