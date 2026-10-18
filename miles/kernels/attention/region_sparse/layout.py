"""The form of a region selection that the sparse region attention kernels walk.

A region is 8 consecutive tokens of one sequence in a THD pack. A query block is the up to 8 tokens of one region of queries, which share their candidate
regions. Each block lists the union of its tokens' selected regions, every entry tagged with
the tokens that chose it. Every buffer is sized from tensor shapes, so no length is read back
to the host.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch import Tensor

# tokens per region; region r of a sequence covers its tokens [8r, 8r + 8)
REGION_SIZE = 8
QUERY_BLOCK = REGION_SIZE
# an entry is region << ENTRY_REGION_SHIFT | token_mask, bit i set when the block's token i chose it
ENTRY_REGION_SHIFT = QUERY_BLOCK
_REGIONS_PER_WORD = 32
_UNION_WORDS_PER_STEP = 16


@dataclass(frozen=True)
class QueryBlocks:
    """The query blocks of a THD pack, numbered sequence by sequence.

    ``num_blocks`` is a bound computed from shapes; blocks past the real ones have
    ``num_tokens == 0``. Block ``b``'s union for group ``g`` has capacity for its largest
    possible size from ``g * group_stride + union_offset[b]`` on.
    """

    first_token: Tensor  # int32 [num_blocks]
    num_tokens: Tensor  # int32 [num_blocks]
    seq_start: Tensor  # int32 [num_blocks], the first token of the block's sequence
    union_offset: Tensor  # int64 [num_blocks]
    group_stride: int

    @property
    def num_blocks(self) -> int:
        return self.first_token.shape[0]


@dataclass(frozen=True)
class RegionUnions:
    """Block ``b``'s union for group ``g``: ``count[g, b]`` ascending entries at its offset."""

    count: Tensor  # int32 [num_kv_groups, num_blocks], set for the real blocks only
    entries: Tensor  # int32 [num_kv_groups * group_stride]


@triton.jit
def _sequence_union_capacity(length, QUERY_BLOCK: tl.constexpr, MAX_UNION: tl.constexpr):
    """Block ``b`` of a sequence holds at most ``b + 1`` regions, and ``MAX_UNION // QUERY_BLOCK`` per token."""
    full = (length // QUERY_BLOCK).to(tl.int64)
    capped = tl.minimum(full, MAX_UNION)
    capacity = capped * (capped + 1) // 2 + (full - capped) * MAX_UNION
    tail = (length % QUERY_BLOCK).to(tl.int64)
    return capacity + tl.where(tail > 0, tl.minimum(full + 1, tail * (MAX_UNION // QUERY_BLOCK)), 0)


@triton.jit
def _query_block_table_kernel(
    CU_SEQLENS,
    FIRST_TOKEN,
    NUM_TOKENS,
    SEQ_START,
    UNION_OFFSET,
    SEQS_POW2: tl.constexpr,
    BLOCKS_PER_STEP: tl.constexpr,
    QUERY_BLOCK: tl.constexpr,
    MAX_UNION: tl.constexpr,
):
    """One program per sequence; where its blocks and unions start is summed over the sequences before it."""
    seq = tl.program_id(0)
    before = tl.arange(0, SEQS_POW2)
    prior = before < seq
    prior_len = tl.load(CU_SEQLENS + before + 1, mask=prior, other=0) - tl.load(
        CU_SEQLENS + before, mask=prior, other=0
    )
    block0 = tl.sum((prior_len + QUERY_BLOCK - 1) // QUERY_BLOCK, 0)
    offset0 = tl.sum(_sequence_union_capacity(prior_len, QUERY_BLOCK, MAX_UNION), 0)

    start = tl.load(CU_SEQLENS + seq)
    end = tl.load(CU_SEQLENS + seq + 1)
    for b0 in range(0, (end - start + QUERY_BLOCK - 1) // QUERY_BLOCK, BLOCKS_PER_STEP):
        b = b0 + tl.arange(0, BLOCKS_PER_STEP)
        first = start + b * QUERY_BLOCK
        real = first < end
        num_tokens = tl.minimum(end - first, QUERY_BLOCK)
        capacity = tl.where(real, tl.minimum(b + 1, num_tokens * (MAX_UNION // QUERY_BLOCK)).to(tl.int64), 0)
        tl.store(FIRST_TOKEN + block0 + b, first, mask=real)
        tl.store(NUM_TOKENS + block0 + b, num_tokens, mask=real)
        tl.store(SEQ_START + block0 + b, start + tl.zeros_like(b), mask=real)
        tl.store(UNION_OFFSET + block0 + b, offset0 + tl.cumsum(capacity, 0) - capacity, mask=real)
        offset0 += tl.sum(capacity, 0)


def query_blocks(cu_seqlens: Tensor, num_tokens: int, max_selected: int) -> QueryBlocks:
    """The query blocks of a THD pack, each with room for its tokens' union of at most ``max_selected`` regions."""
    num_seqs = cu_seqlens.shape[0] - 1
    bound = num_tokens // QUERY_BLOCK + num_seqs
    device = cu_seqlens.device
    first_token = torch.empty(bound, dtype=torch.int32, device=device)
    num_tokens_in_block = torch.zeros(bound, dtype=torch.int32, device=device)
    seq_start = torch.empty(bound, dtype=torch.int32, device=device)
    union_offset = torch.empty(bound, dtype=torch.int64, device=device)
    _query_block_table_kernel[(num_seqs,)](
        cu_seqlens,
        first_token,
        num_tokens_in_block,
        seq_start,
        union_offset,
        SEQS_POW2=triton.next_power_of_2(num_seqs),
        BLOCKS_PER_STEP=1024,
        QUERY_BLOCK=QUERY_BLOCK,
        MAX_UNION=QUERY_BLOCK * max_selected,
        num_warps=4,
    )
    return QueryBlocks(
        first_token=first_token,
        num_tokens=num_tokens_in_block,
        seq_start=seq_start,
        union_offset=union_offset,
        group_stride=_union_capacity(num_tokens, max_selected),
    )


def _union_capacity(num_tokens: int, max_selected: int) -> int:
    """The union entries any packing of ``num_tokens`` tokens can need, per group.

    A sequence of length L needs at most ``(L + 7)(L + 15) / 128``, convex in L, so one sequence
    plus one entry per other sequence bounds the sum; and no union outgrows its tokens' ``max_selected`` each.
    """
    one_sequence = triton.cdiv((num_tokens + QUERY_BLOCK - 1) * (num_tokens + 2 * QUERY_BLOCK - 1), 2 * QUERY_BLOCK**2)
    return min(max_selected * num_tokens, one_sequence + num_tokens)


@triton.jit
def _segmented_or(word_a, bits_a, word_b, bits_b):
    # the ids are ascending, so equal words are adjacent and this is associative
    return word_b, tl.where(word_a == word_b, bits_a | bits_b, bits_b)


@triton.jit
def _token_region_bitmap_kernel(
    REGION_IDS,
    BITMAP,
    stride_it,
    stride_ig,
    stride_bt,
    stride_bg,
    SLOTS: tl.constexpr,
    SLOTS_POW2: tl.constexpr,
):
    """Bit ``r`` of token ``t``'s row is set when ``t`` selected region ``r`` for the group."""
    # the bitmap passes 2^31 words at 370K tokens of 4 groups, so row offsets are int64
    token = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1)
    offs = tl.arange(0, SLOTS_POW2)
    ids = REGION_IDS + token * stride_it + group * stride_ig
    region = tl.load(ids + offs, mask=offs < SLOTS, other=-1)
    region_next = tl.load(ids + offs + 1, mask=offs + 1 < SLOTS, other=-1)
    valid = region >= 0
    # padding sorts after every word, so it never splits a run of equal words
    word = tl.where(valid, region >> 5, 1 << 30)
    word_next = tl.where(region_next >= 0, region_next >> 5, 1 << 30)
    bits = tl.where(valid, 1 << (region & 31), 0)
    _, bits = tl.associative_scan((word, bits), 0, _segmented_or)
    last_of_word = valid & (word_next != word)
    tl.store(BITMAP + token * stride_bt + group * stride_bg + word, bits, mask=last_of_word)


def token_region_bitmap(region_ids: Tensor) -> Tensor:
    """int32 [T, G, ceil(ceil(T / 8) / 32)]: the selection as one bit per (token, group, region)."""
    num_tokens, num_groups, slots = region_ids.shape
    num_words = triton.cdiv(triton.cdiv(num_tokens, REGION_SIZE), _REGIONS_PER_WORD)
    bitmap = torch.zeros(num_tokens, num_groups, num_words, dtype=torch.int32, device=region_ids.device)
    _token_region_bitmap_kernel[(num_tokens, num_groups)](
        region_ids,
        bitmap,
        region_ids.stride(0),
        region_ids.stride(1),
        bitmap.stride(0),
        bitmap.stride(1),
        SLOTS=slots,
        SLOTS_POW2=triton.next_power_of_2(slots),
        num_warps=4,
    )
    return bitmap


@triton.jit
def _block_region_union_kernel(
    BITMAP,
    BLOCK_FIRST_TOKEN,
    BLOCK_NUM_TOKENS,
    BLOCK_SEQ_START,
    BLOCK_UNION_OFFSET,
    UNION_COUNT,
    UNION_ENTRIES,
    stride_bt,
    stride_bg,
    num_blocks,
    group_stride,
    QUERY_BLOCK: tl.constexpr,
    WORDS: tl.constexpr,
):
    """Compact the OR of a block's token bitmaps into ascending ``region << 8 | token_mask`` entries."""
    block = tl.program_id(0)
    group = tl.program_id(1)
    n_tok = tl.load(BLOCK_NUM_TOKENS + block)
    if n_tok == 0:
        return
    first = tl.load(BLOCK_FIRST_TOKEN + block)
    own_region = (first - tl.load(BLOCK_SEQ_START + block)) // QUERY_BLOCK
    out = UNION_ENTRIES + group.to(tl.int64) * group_stride + tl.load(BLOCK_UNION_OFFSET + block)

    tok = tl.arange(0, QUERY_BLOCK)
    bit = tl.arange(0, 32)
    rows = BITMAP + (first.to(tl.int64) + tok) * stride_bt + group * stride_bg
    last_word = own_region // 32
    count = tl.full((), 0, tl.int32)
    for word0 in range(0, last_word + 1, WORDS):
        word = word0 + tl.arange(0, WORDS)
        words = tl.load(
            rows[:, None] + word[None, :], mask=(tok < n_tok)[:, None] & (word <= last_word)[None, :], other=0
        )
        chosen = (words[:, :, None] >> bit[None, None, :]) & 1
        token_mask = tl.sum(chosen << tok[:, None, None], axis=0)
        region = word[:, None] * 32 + bit[None, :]
        # regions past the block's own cannot be visible to it
        token_mask = tl.reshape(tl.where(region <= own_region, token_mask, 0), (WORDS * 32,))
        present = token_mask != 0
        slot = count + tl.cumsum(present.to(tl.int32), 0) - 1
        tl.store(out + slot, (tl.reshape(region, (WORDS * 32,)) << QUERY_BLOCK) | token_mask, mask=present)
        count += tl.sum(present.to(tl.int32), 0)
    tl.store(UNION_COUNT + group * num_blocks + block, count)


def region_unions(blocks: QueryBlocks, bitmap: Tensor) -> RegionUnions:
    num_groups = bitmap.shape[1]
    count = torch.empty(num_groups, blocks.num_blocks, dtype=torch.int32, device=bitmap.device)
    entries = torch.empty(num_groups * blocks.group_stride, dtype=torch.int32, device=bitmap.device)
    _block_region_union_kernel[(blocks.num_blocks, num_groups)](
        bitmap,
        blocks.first_token,
        blocks.num_tokens,
        blocks.seq_start,
        blocks.union_offset,
        count,
        entries,
        bitmap.stride(0),
        bitmap.stride(1),
        blocks.num_blocks,
        blocks.group_stride,
        QUERY_BLOCK=QUERY_BLOCK,
        WORDS=_UNION_WORDS_PER_STEP,
        num_warps=4,
    )
    return RegionUnions(count=count, entries=entries)
