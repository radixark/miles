"""Region-sparse attention for training: forward and backward in Triton.

A program owns one query block and one KV group -- the block's tokens times the group's 16
query heads form a 128-row tile -- and walks the union of the block's selected regions a chunk
at a time, masking each (token, key) pair to that token's selection and to causality. The
backward makes one pass over the same tiles; dK and dV go into fp32 buffers with atomics,
since many blocks share a region.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

from miles.kernels.attention.region_sparse.layout import (
    ENTRY_REGION_SHIFT,
    QUERY_BLOCK,
    REGION_SIZE,
    QueryBlocks,
    RegionUnions,
    query_blocks,
    region_unions,
    token_region_bitmap,
)

HEAD_DIM = 192
# head dim 192 is not a power of two, which tl.arange needs: every tile is split 128 + 64
_HEAD_DIM_SPLIT = (128, 64)
_LOG2E = 1.4426950408889634
# regions per chunk: 64 keys forward; the backward's extra accumulators leave tensor memory for 32
_FWD_CHUNK = 8
_BWD_CHUNK = 4


@triton.jit
def _load_rows(ptr, rows_ok, D0: tl.constexpr, D1: tl.constexpr):
    first = tl.load(ptr[:, None] + tl.arange(0, D0)[None, :], mask=rows_ok[:, None], other=0.0)
    second = tl.load(ptr[:, None] + D0 + tl.arange(0, D1)[None, :], mask=rows_ok[:, None], other=0.0)
    return first, second


@triton.jit
def _block_tile(
    BLOCK_FIRST_TOKEN,
    BLOCK_NUM_TOKENS,
    BLOCK_SEQ_START,
    BLOCK_UNION_OFFSET,
    UNION_COUNT,
    block,
    group,
    num_blocks,
    group_stride,
    HEADS: tl.constexpr,
    QBLOCK: tl.constexpr,
):
    """The tile's rows (token-major: row = token * HEADS + head) and where its union lies."""
    n_tok = tl.load(BLOCK_NUM_TOKENS + block)
    seq_start = tl.load(BLOCK_SEQ_START + block)
    union_start = group.to(tl.int64) * group_stride + tl.load(BLOCK_UNION_OFFSET + block)
    union_count = tl.load(UNION_COUNT + group * num_blocks + block)
    rows = tl.arange(0, QBLOCK * HEADS)
    row_tok = rows // HEADS
    row_head = group * HEADS + rows % HEADS
    q_pos = tl.load(BLOCK_FIRST_TOKEN + block) + row_tok
    return seq_start, union_start, union_count, row_tok, row_head, row_tok < n_tok, q_pos


@triton.jit
def _chunk_keys(
    UNION_ENTRIES,
    union_start,
    union_count,
    chunk0,
    seq_start,
    q_pos,
    row_tok,
    num_tokens,
    CHUNK: tl.constexpr,
    REGION: tl.constexpr,
    ENTRY_SHIFT: tl.constexpr,
):
    """The chunk's key positions, which of them exist, and the [rows, keys] visibility mask."""
    cols = tl.arange(0, CHUNK * REGION)
    slot = chunk0 + cols // REGION
    in_union = slot < union_count
    entry = tl.load(UNION_ENTRIES + union_start + slot, mask=in_union, other=0)
    k_pos = seq_start + (entry >> ENTRY_SHIFT) * REGION + cols % REGION
    k_ok = in_union & (k_pos < num_tokens)
    chosen = ((entry[None, :] >> row_tok[:, None]) & 1) != 0
    visible = chosen & k_ok[None, :] & (k_pos[None, :] <= q_pos[:, None])
    return k_pos, k_ok, visible


@triton.jit
def _sparse_region_attn_fwd_kernel(
    Q,
    K,
    V,
    OUT,
    LSE,
    BLOCK_FIRST_TOKEN,
    BLOCK_NUM_TOKENS,
    BLOCK_SEQ_START,
    BLOCK_UNION_OFFSET,
    UNION_COUNT,
    UNION_ENTRIES,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_kg,
    stride_ot,
    stride_oh,
    num_blocks,
    group_stride,
    num_tokens,
    num_heads,
    scale_log2,
    HEADS: tl.constexpr,
    QBLOCK: tl.constexpr,
    REGION: tl.constexpr,
    CHUNK: tl.constexpr,
    ENTRY_SHIFT: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
):
    # the last blocks of a sequence have the longest unions: start them first
    block = num_blocks - 1 - tl.program_id(0)
    group = tl.program_id(1)
    if tl.load(BLOCK_NUM_TOKENS + block) == 0:
        return
    seq_start, union_start, union_count, row_tok, row_head, row_ok, q_pos = _block_tile(
        BLOCK_FIRST_TOKEN,
        BLOCK_NUM_TOKENS,
        BLOCK_SEQ_START,
        BLOCK_UNION_OFFSET,
        UNION_COUNT,
        block,
        group,
        num_blocks,
        group_stride,
        HEADS,
        QBLOCK,
    )
    # q and out pass 2^31 elements at 175K tokens of 64 heads: step to the block's first row in int64
    first = tl.load(BLOCK_FIRST_TOKEN + block).to(tl.int64)
    Q += first * stride_qt
    OUT += first * stride_ot
    q0, q1 = _load_rows(Q + row_tok * stride_qt + row_head * stride_qh, row_ok, D0, D1)

    m_i = tl.full((QBLOCK * HEADS,), float("-inf"), tl.float32)
    l_i = tl.zeros((QBLOCK * HEADS,), tl.float32)
    acc0 = tl.zeros((QBLOCK * HEADS, D0), tl.float32)
    acc1 = tl.zeros((QBLOCK * HEADS, D1), tl.float32)
    for chunk0 in range(0, union_count, CHUNK):
        k_pos, k_ok, visible = _chunk_keys(
            UNION_ENTRIES,
            union_start,
            union_count,
            chunk0,
            seq_start,
            q_pos,
            row_tok,
            num_tokens,
            CHUNK,
            REGION,
            ENTRY_SHIFT,
        )
        k0, k1 = _load_rows(K + k_pos * stride_kt + group * stride_kg, k_ok, D0, D1)
        s = tl.dot(q1, tl.trans(k1), acc=tl.dot(q0, tl.trans(k0)))
        s = tl.where(visible, s * scale_log2, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        # a row may see nothing yet; shift it by 0 so -inf - -inf never appears
        m_shift = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp2(s - m_shift[:, None])
        alpha = tl.exp2(m_i - m_shift)
        l_i = l_i * alpha + tl.sum(p, 1)
        v0, v1 = _load_rows(V + k_pos * stride_kt + group * stride_kg, k_ok, D0, D1)
        p = p.to(v0.dtype)
        acc0 = tl.dot(p, v0, acc=acc0 * alpha[:, None])
        acc1 = tl.dot(p, v1, acc=acc1 * alpha[:, None])
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    out = OUT + row_tok[:, None] * stride_ot + row_head[:, None] * stride_oh
    tl.store(out + tl.arange(0, D0)[None, :], (acc0 / l_safe[:, None]).to(OUT.dtype.element_ty), mask=row_ok[:, None])
    tl.store(
        out + D0 + tl.arange(0, D1)[None, :], (acc1 / l_safe[:, None]).to(OUT.dtype.element_ty), mask=row_ok[:, None]
    )
    # log2-domain log-sum-exp of the scaled scores, which is what the backward exponentiates against
    lse = tl.where(l_i == 0.0, float("inf"), m_i + tl.log2(l_safe))
    tl.store(LSE + q_pos * num_heads + row_head, lse, mask=row_ok)


@triton.jit
def _sparse_region_attn_bwd_kernel(
    Q,
    K,
    V,
    OUT,
    DOUT,
    LSE,
    DQ,
    DK,
    DV,
    BLOCK_FIRST_TOKEN,
    BLOCK_NUM_TOKENS,
    BLOCK_SEQ_START,
    BLOCK_UNION_OFFSET,
    UNION_COUNT,
    UNION_ENTRIES,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_kg,
    stride_ot,
    stride_oh,
    stride_dkt,
    stride_dkg,
    num_blocks,
    group_stride,
    num_tokens,
    num_heads,
    scale,
    scale_log2,
    HEADS: tl.constexpr,
    QBLOCK: tl.constexpr,
    REGION: tl.constexpr,
    CHUNK: tl.constexpr,
    ENTRY_SHIFT: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
):
    block = num_blocks - 1 - tl.program_id(0)
    group = tl.program_id(1)
    if tl.load(BLOCK_NUM_TOKENS + block) == 0:
        return
    seq_start, union_start, union_count, row_tok, row_head, row_ok, q_pos = _block_tile(
        BLOCK_FIRST_TOKEN,
        BLOCK_NUM_TOKENS,
        BLOCK_SEQ_START,
        BLOCK_UNION_OFFSET,
        UNION_COUNT,
        block,
        group,
        num_blocks,
        group_stride,
        HEADS,
        QBLOCK,
    )
    first = tl.load(BLOCK_FIRST_TOKEN + block).to(tl.int64)
    Q += first * stride_qt
    DQ += first * stride_qt
    OUT += first * stride_ot
    DOUT += first * stride_ot
    q0, q1 = _load_rows(Q + row_tok * stride_qt + row_head * stride_qh, row_ok, D0, D1)
    do0, do1 = _load_rows(DOUT + row_tok * stride_ot + row_head * stride_oh, row_ok, D0, D1)
    o0, o1 = _load_rows(OUT + row_tok * stride_ot + row_head * stride_oh, row_ok, D0, D1)
    delta = tl.sum(do0.to(tl.float32) * o0.to(tl.float32), 1) + tl.sum(do1.to(tl.float32) * o1.to(tl.float32), 1)
    lse = tl.load(LSE + q_pos * num_heads + row_head, mask=row_ok, other=float("inf"))

    dq0 = tl.zeros((QBLOCK * HEADS, D0), tl.float32)
    dq1 = tl.zeros((QBLOCK * HEADS, D1), tl.float32)
    for chunk0 in range(0, union_count, CHUNK):
        k_pos, k_ok, visible = _chunk_keys(
            UNION_ENTRIES,
            union_start,
            union_count,
            chunk0,
            seq_start,
            q_pos,
            row_tok,
            num_tokens,
            CHUNK,
            REGION,
            ENTRY_SHIFT,
        )
        k0, k1 = _load_rows(K + k_pos * stride_kt + group * stride_kg, k_ok, D0, D1)
        v0, v1 = _load_rows(V + k_pos * stride_kt + group * stride_kg, k_ok, D0, D1)
        s = tl.dot(q1, tl.trans(k1), acc=tl.dot(q0, tl.trans(k0)))
        p = tl.where(visible, tl.exp2(s * scale_log2 - lse[:, None]), 0.0)
        dp = tl.dot(do1, tl.trans(v1), acc=tl.dot(do0, tl.trans(v0)))
        ds = (p * (dp - delta[:, None])).to(q0.dtype)
        p = p.to(q0.dtype)
        dq0 = tl.dot(ds, k0, acc=dq0)
        dq1 = tl.dot(ds, k1, acc=dq1)

        # dK and dV as [dims, keys]: 32 keys are too few rows for a tensor-core tile
        dkv = k_pos[None, :] * stride_dkt + group * stride_dkg
        dkv_ok = k_ok[None, :]
        d0 = tl.arange(0, D0)[:, None]
        d1 = D0 + tl.arange(0, D1)[:, None]
        tl.atomic_add(DK + dkv + d0, tl.dot(tl.trans(q0), ds) * scale, mask=dkv_ok, sem="relaxed")
        tl.atomic_add(DK + dkv + d1, tl.dot(tl.trans(q1), ds) * scale, mask=dkv_ok, sem="relaxed")
        tl.atomic_add(DV + dkv + d0, tl.dot(tl.trans(do0), p), mask=dkv_ok, sem="relaxed")
        tl.atomic_add(DV + dkv + d1, tl.dot(tl.trans(do1), p), mask=dkv_ok, sem="relaxed")

    dq = DQ + row_tok[:, None] * stride_qt + row_head[:, None] * stride_qh
    tl.store(dq + tl.arange(0, D0)[None, :], (dq0 * scale).to(DQ.dtype.element_ty), mask=row_ok[:, None])
    tl.store(dq + D0 + tl.arange(0, D1)[None, :], (dq1 * scale).to(DQ.dtype.element_ty), mask=row_ok[:, None])


def _layout_args(blocks: QueryBlocks, unions: RegionUnions) -> tuple[Tensor, ...]:
    return blocks.first_token, blocks.num_tokens, blocks.seq_start, blocks.union_offset, unions.count, unions.entries


def _tile_constants(heads_per_group: int) -> dict[str, int]:
    return dict(
        HEADS=heads_per_group,
        QBLOCK=QUERY_BLOCK,
        REGION=REGION_SIZE,
        ENTRY_SHIFT=ENTRY_REGION_SHIFT,
        D0=_HEAD_DIM_SPLIT[0],
        D1=_HEAD_DIM_SPLIT[1],
    )


def _forward(
    q: Tensor, k: Tensor, v: Tensor, blocks: QueryBlocks, unions: RegionUnions, softmax_scale: float
) -> tuple[Tensor, Tensor]:
    num_tokens, num_heads, _ = q.shape
    num_groups = k.shape[1]
    out = torch.empty_like(q)
    lse = torch.empty(num_tokens, num_heads, dtype=torch.float32, device=q.device)
    _sparse_region_attn_fwd_kernel[(blocks.num_blocks, num_groups)](
        q,
        k,
        v,
        out,
        lse,
        *_layout_args(blocks, unions),
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        out.stride(0),
        out.stride(1),
        blocks.num_blocks,
        blocks.group_stride,
        num_tokens,
        num_heads,
        softmax_scale * _LOG2E,
        CHUNK=_FWD_CHUNK,
        **_tile_constants(num_heads // num_groups),
        num_warps=8,
        num_stages=3,
    )
    return out, lse


def _backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    dout: Tensor,
    lse: Tensor,
    blocks: QueryBlocks,
    unions: RegionUnions,
    softmax_scale: float,
) -> tuple[Tensor, Tensor, Tensor]:
    num_tokens, num_heads, _ = q.shape
    num_groups = k.shape[1]
    dq = torch.empty_like(q)
    dk = torch.zeros(k.shape, dtype=torch.float32, device=k.device)
    dv = torch.zeros(v.shape, dtype=torch.float32, device=v.device)
    _sparse_region_attn_bwd_kernel[(blocks.num_blocks, num_groups)](
        q,
        k,
        v,
        out,
        dout,
        lse,
        dq,
        dk,
        dv,
        *_layout_args(blocks, unions),
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        out.stride(0),
        out.stride(1),
        dk.stride(0),
        dk.stride(1),
        blocks.num_blocks,
        blocks.group_stride,
        num_tokens,
        num_heads,
        softmax_scale,
        softmax_scale * _LOG2E,
        CHUNK=_BWD_CHUNK,
        **_tile_constants(num_heads // num_groups),
        num_warps=8,
        num_stages=3,
    )
    return dq, dk.to(k.dtype), dv.to(v.dtype)


class _SparseRegionAttention(torch.autograd.Function):
    """Keeps the selection for backward as its bitmap (T x G x T/8 bits) and rebuilds the unions there."""

    @staticmethod
    def forward(
        ctx, q: Tensor, k: Tensor, v: Tensor, region_ids: Tensor, cu_seqlens: Tensor, softmax_scale: float
    ) -> Tensor:
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        blocks = query_blocks(cu_seqlens, q.shape[0], max_selected=region_ids.shape[-1])
        bitmap = token_region_bitmap(region_ids)
        out, lse = _forward(q, k, v, blocks, region_unions(blocks, bitmap), softmax_scale)
        ctx.save_for_backward(
            q, k, v, out, lse, blocks.first_token, blocks.num_tokens, blocks.seq_start, blocks.union_offset, bitmap
        )
        ctx.group_stride = blocks.group_stride
        ctx.softmax_scale = softmax_scale
        return out

    @staticmethod
    def backward(ctx, dout: Tensor):
        q, k, v, out, lse, first_token, num_tokens, seq_start, union_offset, bitmap = ctx.saved_tensors
        blocks = QueryBlocks(first_token, num_tokens, seq_start, union_offset, ctx.group_stride)
        unions = region_unions(blocks, bitmap)
        dq, dk, dv = _backward(q, k, v, out, dout.contiguous(), lse, blocks, unions, ctx.softmax_scale)
        return dq, dk, dv, None, None, None


def sparse_region_attention(
    q: Tensor, k: Tensor, v: Tensor, region_ids: Tensor, cu_seqlens: Tensor, softmax_scale: float
) -> Tensor:
    """Attention of each query over its KV group's selected regions, THD-packed.

    ``q`` is [T, Hq, 192] and ``k``/``v`` are [T, G, 192], bf16, with the query heads of group
    ``g`` at ``g * Hq // G`` onwards. ``cu_seqlens`` (int32 [num_sequences + 1], ending at T)
    holds the THD boundaries. ``region_ids`` (int32 [T, G, S]) lists, per token and group,
    ascending region ids padded with -1; region r covers tokens [8r, 8r + 8) of the token's own
    sequence, and the region holding the token is its last valid entry, visible only up to the
    token. Returns [T, Hq, 192] in ``q``'s dtype. Gradients flow to q, k and v.
    """
    num_tokens, num_heads, head_dim = q.shape
    num_groups = k.shape[1]
    assert head_dim == HEAD_DIM and k.shape == v.shape == (
        num_tokens,
        num_groups,
        HEAD_DIM,
    ), f"q {tuple(q.shape)}, k {tuple(k.shape)} and v {tuple(v.shape)} are not [T, Hq, 192] and [T, G, 192]"
    assert num_heads % num_groups == 0, f"{num_heads} query heads do not divide into {num_groups} KV groups"
    assert (
        region_ids.shape[:2] == (num_tokens, num_groups) and region_ids.dtype == torch.int32
    ), f"region_ids must be int32 [{num_tokens}, {num_groups}, S], got {region_ids.dtype} {tuple(region_ids.shape)}"
    assert region_ids.stride(-1) == 1, "each token's region ids must be contiguous"
    return _SparseRegionAttention.apply(q, k, v, region_ids, cu_seqlens, softmax_scale)
