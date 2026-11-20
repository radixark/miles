"""The union of consecutive queries' top-k keys, for kernels that run a block of queries over one key list.

Consecutive queries of a DSA layer select mostly the same keys (a GLM-5 indexer shares ~90% between
neighbours, and 8 consecutive queries pick ~16% of 8 x topk distinct keys), so a kernel that holds a block
of queries per CTA gathers and reduces each key once instead of once per query. Each block's rows are
sorted as (key, query) pairs and deduplicated into an ascending key list, with a bitmask per key that says
which queries of the block selected it.
"""

import torch
import triton
import triton.language as tl

UNION_PAD = 32  # -1 slots written after each block's keys, so a kernel may read whole 32-key tiles
_SENTINEL = tl.constexpr(2**31 - 1)


@triton.jit
def _topk_union_kernel(
    indices_ptr,
    union_ptr,
    mask_ptr,
    lengths_ptr,
    topk,
    stride_row,
    BQ: tl.constexpr,
    CAP: tl.constexpr,
    PAD: tl.constexpr,
):
    block = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, CAP)
    query = offs // topk
    in_rows = offs < BQ * topk
    key = tl.load(indices_ptr + (block * BQ + query) * stride_row + offs % topk, mask=in_rows, other=-1)
    # (key, query) pairs sort by key with the block's queries in order; padding sorts last
    pair = tl.where(key >= 0, key * BQ + query, _SENTINEL)
    pair = tl.sort(pair)
    row = union_ptr + block * CAP
    tl.store(row + offs, pair)  # the block's own output row is the scratch for the neighbour lookup
    tl.store(mask_ptr + block * CAP + offs, tl.zeros([CAP], tl.int32))
    tl.debug_barrier()
    prev = tl.load(row + offs - 1, mask=offs > 0, other=_SENTINEL)
    valid = pair != _SENTINEL
    first = valid & (pair // BQ != prev // BQ)
    slot = tl.cumsum(first.to(tl.int32)) - 1
    length = tl.sum(first.to(tl.int32))
    tl.debug_barrier()
    tl.store(row + slot, pair // BQ, mask=first)
    tl.store(row + length + tl.arange(0, PAD), tl.full([PAD], -1, tl.int32), mask=length + tl.arange(0, PAD) < CAP)
    tl.atomic_or(mask_ptr + block * CAP + slot, (1 << (pair % BQ)).to(tl.int32), mask=valid, sem="relaxed")
    tl.store(lengths_ptr + block, length)


def topk_union(indices: torch.Tensor, query_block: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge every ``query_block`` consecutive rows of ``indices`` [rows, topk] int32 (-1 = padding).

    Returns ``union`` [rows / query_block, cap] int32 (each block's distinct keys ascending, then -1 to at
    least ``UNION_PAD`` slots past its length), ``mask`` [rows / query_block, cap] int32 (bit q set when
    row q of the block selected the key), and ``lengths`` [rows / query_block] int32.
    """
    rows, topk = indices.shape
    assert indices.dtype == torch.int32 and indices.is_cuda
    assert rows % query_block == 0, f"{rows} rows do not split into blocks of {query_block}"
    assert 1 < query_block <= 31, "the membership mask holds one bit per query of the block"
    # the sort needs a power of two; a block with no shared keys fills the row, and then needs no padding
    cap = max(triton.next_power_of_2(query_block * topk), UNION_PAD)
    blocks = rows // query_block
    union = torch.empty(blocks, cap, dtype=torch.int32, device=indices.device)
    mask = torch.empty(blocks, cap, dtype=torch.int32, device=indices.device)
    lengths = torch.empty(blocks, dtype=torch.int32, device=indices.device)
    _topk_union_kernel[(blocks,)](
        indices,
        union,
        mask,
        lengths,
        topk,
        indices.stride(0),
        BQ=query_block,
        CAP=cap,
        PAD=UNION_PAD,
        num_warps=min(32, max(4, cap // 512)),  # the in-register sort wants few elements per thread (H200)
    )
    return union, mask, lengths
