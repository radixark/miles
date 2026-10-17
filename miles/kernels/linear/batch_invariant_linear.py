# SPDX-License-Identifier: Apache-2.0
"""A batch-invariant BF16 ``x @ w.T`` for a few output columns, rounded as an FP64 GEMM."""

import torch
import triton
import triton.language as tl

# Each (token, head) dot product runs over 256 lanes: lane l sums products l, l+256,
# ... error-free, then a fixed pairwise tree joins the lanes. That order depends on
# neither the batch nor the launch shape below, so no token depends on its batch.
_LANES = 256
_LANE_LEVELS = 8
# Measured on GB300 with 4 heads per rank: below about 2K outputs, one head per
# program spreads latency-bound work over more SMs; above, heads share a token row.
_SPLIT_HEADS_BELOW = 2048


@triton.jit
def _two_sum(a, b):
    total = a + b
    b_part = total - a
    return total, (a - (total - b_part)) + (b - b_part)


@triton.jit
def _join_lanes(s, c, HEADS: tl.constexpr, LANES: tl.constexpr, LEVELS: tl.constexpr):
    # Pairs lanes by logical index, so the tree ignores the register layout.
    for level in tl.static_range(LEVELS):
        s0, s1 = tl.split(tl.reshape(s, (HEADS, LANES >> (level + 1), 2)))
        c0, c1 = tl.split(tl.reshape(c, (HEADS, LANES >> (level + 1), 2)))
        high, low = _two_sum(s0, s1)
        s, c = _two_sum(high, low + (c0 + c1))
    return tl.reshape(s, (HEADS,)), tl.reshape(c, (HEADS,))


@triton.jit
def _batch_invariant_linear_kernel(
    X,
    W,
    Out,
    stride_x,
    stride_w,
    stride_out,
    K: tl.constexpr,
    HEADS: tl.constexpr,
    LANES: tl.constexpr,
    LANE_LEVELS: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    heads = tl.program_id(1) * HEADS + tl.arange(0, HEADS)
    lanes = tl.arange(0, LANES)
    s = tl.zeros((HEADS, LANES), tl.float32)
    c = tl.zeros((HEADS, LANES), tl.float32)
    for step in tl.static_range(K // LANES):
        k = step * LANES + lanes
        x = tl.load(X + token * stride_x + k).to(tl.float32)
        w = tl.load(W + heads[:, None] * stride_w + k[None, :]).to(tl.float32)
        # Two 8-bit BF16 significands multiply exactly in FP32.
        s, err = _two_sum(s, x[None, :] * w)
        c += err
    s, c = _join_lanes(s, c, HEADS, LANES, LANE_LEVELS)
    # Round the double-float sum to FP32, then BF16: torch's float64 -> bfloat16.
    tl.store(Out + token * stride_out + heads, (s + c).to(tl.bfloat16))


def batch_invariant_linear(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Return BF16 ``hidden_states @ weight.T`` with batch-invariant FP64-equivalent rounding.

    Inputs are BF16 ``[T, K]`` and ``[H, K]`` tensors with unit inner stride;
    ``H`` is a power of two up to 16 and ``K`` is a multiple of 256 on CUDA.
    CPU callers use the FP64 expression as a portable reference fallback.
    """
    if hidden_states.device.type != "cuda":
        return (hidden_states.double() @ weight.double().T).to(torch.bfloat16)
    if torch.version.hip is not None:
        raise ValueError("batch_invariant_linear requires NVIDIA CUDA")
    if (
        hidden_states.ndim != 2
        or weight.ndim != 2
        or hidden_states.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or hidden_states.shape[1] != weight.shape[1]
        or hidden_states.shape[1] % _LANES
        or hidden_states.stride(1) != 1
        or weight.stride(1) != 1
        or weight.device != hidden_states.device
    ):
        raise ValueError("batch_invariant_linear requires BF16 [T, K] and [H, K] rows")
    tokens, k = hidden_states.shape
    heads = weight.shape[0]
    if heads not in (1, 2, 4, 8, 16):
        raise ValueError("batch_invariant_linear supports at most 16 output columns, a power of two")
    out = torch.empty((tokens, heads), dtype=torch.bfloat16, device=hidden_states.device)
    if tokens:
        heads_per_program = 1 if tokens * heads < _SPLIT_HEADS_BELOW else min(heads, 8)
        _batch_invariant_linear_kernel[(tokens, heads // heads_per_program)](
            hidden_states,
            weight,
            out,
            hidden_states.stride(0),
            weight.stride(0),
            out.stride(0),
            K=k,
            HEADS=heads_per_program,
            LANES=_LANES,
            LANE_LEVELS=_LANE_LEVELS,
            num_warps=4,
        )
    return out
