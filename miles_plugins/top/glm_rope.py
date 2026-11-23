"""Delegate unscaled GLM rotary blocks to serving's fused RoPE kernel for either precision."""

from functools import lru_cache

import torch


@lru_cache(maxsize=16)
def _serving_rope(head_dim, rows, base, interleaved, device_index):
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding

    with torch.device("cuda", device_index):
        rope = RotaryEmbedding(head_dim, head_dim, rows, base, not interleaved, torch.bfloat16)
    if rope._forward_method != rope.forward_cuda:
        raise RuntimeError("TOP RoPE requires serving's fused RoPE kernel under TOP (BF16, head size 64/128/256/512)")
    return rope


class _PositiveZeroGrad(torch.autograd.Function):
    """Identity whose backward adds +0.0, turning a -0.0 gradient element into +0.0."""

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad + 0.0


def _serving_rotate(rope, positions, x, *, key):
    """Serving's fused RoPE kernel on a copy of ``x`` [rows, ..., dim], in the query or key slot serving rotates it
    in; the other slot is empty."""
    rows = x.reshape(x.shape[0], -1, x.shape[-1]).clone(memory_format=torch.contiguous_format)
    if rows.numel():
        empty = rows.new_empty((rows.shape[0], 0, rows.shape[-1]))
        query, keys = (empty, rows) if key else (rows, empty)
        rope.forward_cuda(positions, query, keys)
    return rows.view(x.shape)


def rotary(x, cu_seqlens, base, interleaved, *, key, cp_size=1, cp_rank=0):
    """Delegate serving's fused RoPE, including its cache and BF16 rounding. ``key`` rotates ``x`` in the kernel's key
    slot, as serving rotates keys.

    Under all-gather CP, ``x`` holds rank ``cp_rank``'s rows of the ``cu_seqlens`` packing, which rotate at their
    global positions with the cache of the whole packing."""
    rows = x.shape[0]
    total, first = rows * cp_size, rows * cp_rank
    rope = _serving_rope(x.shape[-1], total, base, interleaved, x.device.index)
    starts = torch.repeat_interleave(cu_seqlens[:-1], cu_seqlens[1:] - cu_seqlens[:-1], output_size=total)
    # int64, as serving's positions.
    positions = torch.arange(first, first + rows, device=x.device, dtype=torch.int64) - starts[first : first + rows]
    with torch.no_grad():
        out = _serving_rotate(rope, positions, x.detach(), key=key)
    if not (torch.is_grad_enabled() and x.requires_grad):
        return out
    from miles_plugins.top.kernels.surrogate import attach, reference_mode

    if cp_size > 1:
        # Match the gradient bytes of rotating x.repeat(cp_size) and keeping this rank's slice: repeat's backward sums
        # the copies, which turns a -0.0 gradient into +0.0.
        x = _PositiveZeroGrad.apply(x)
    with reference_mode():
        cos, sin = rope.cos_sin_cache[positions].float().chunk(2, -1)
        shape = (rows,) + (1,) * (x.dim() - 2) + (cos.shape[-1],)
        cos, sin = cos.view(shape), sin.view(shape)
        xf = x.float()
        if interleaved:
            x1, x2 = xf[..., 0::2], xf[..., 1::2]
            ref = torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).flatten(-2)
        else:
            x1, x2 = xf.chunk(2, -1)
            ref = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)
    return attach(out, ref, "glm_rope")
