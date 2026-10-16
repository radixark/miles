"""Tile lists of QSA block attention: per query tile the blocks it attends, per key tile the queries.

Built without sorting. One pass sets, for every attended (query, block) pair, one bit in each of two
bitmaps: bit ``block % 32`` of word ``block // 32`` in the query tile's row, and bit ``query % 32`` of word
``query // 32`` in the key tile's row. A pass over each row counts the bits in the words before each word,
so a set bit's rank in its row, which is the entry's place in the ascending list, is that count plus the
bits below it in its word. A last pass writes every pair at that place: the entry's bit (an atomic OR, as
all pairs of an entry meet there) and its first key or query.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch import Tensor

# Rows of a forward / dQ tile and of a dK / dV step: T = _TILE_ROWS // G queries x G heads.
_TILE_ROWS = 64
# Blocks per dK / dV tile per arch major, measured together with block_attention's launch parameters.
_KEY_TILE_BLOCKS = {9: 4, 10: 16}
_SCAN_CHUNK = 1024


def _key_tile_blocks(device: torch.device) -> int:
    major = torch.cuda.get_device_capability(device)[0] if device.type == "cuda" else 9
    return _KEY_TILE_BLOCKS.get(major, _KEY_TILE_BLOCKS[9])


@dataclass(frozen=True)
class QSABlockIndex:
    """Tile lists consumed by :func:`qsa_block_attention`; build with :func:`build_qsa_block_index`.

    ``q_*`` list, per forward / dQ tile, the blocks its queries selected in ascending order (first key,
    and one bit per query of the tile); ``k_*`` list, per dK / dV tile, the queries that selected its
    blocks in ascending order (query, and one bit per block of the tile). The order inside a list is the
    kernels' summation order. ``*_order`` is the launch order: forward / dQ tiles busiest first, dK / dV
    tiles by position, so that neighbouring tiles, which mostly read the same queries, run together.
    """

    query_tile: int
    key_tile: int
    ratio: int
    q_offsets: Tensor
    q_key_start: Tensor
    q_bits: Tensor
    q_order: Tensor
    k_offsets: Tensor
    k_query: Tensor
    k_bits: Tensor
    k_order: Tensor
    block_start: Tensor
    block_len: Tensor
    last_key: Tensor


@triton.jit
def _seq_blocks_kernel(CU, FIRST_BLOCK, n_seqs, RATIO: tl.constexpr, CH: tl.constexpr):
    """FIRST_BLOCK[s]: blocks of the sequences before s, for s = 0 .. n_seqs (the last is the batch's count)."""
    total = 0
    tl.store(FIRST_BLOCK, 0)
    for start in range(0, n_seqs, CH):
        seq = start + tl.arange(0, CH)
        mask = seq < n_seqs
        length = tl.load(CU + seq + 1, mask=mask, other=0) - tl.load(CU + seq, mask=mask, other=0)
        n_blocks = (length + RATIO - 1) // RATIO
        tl.store(FIRST_BLOCK + seq + 1, total + tl.cumsum(n_blocks, 0), mask=mask)
        total += tl.sum(n_blocks, 0)


@triton.jit
def _count_not_above(ARR, value, n, n_steps):
    """Per lane, how many of the ascending ``ARR[0 .. n)`` are <= ``value`` (binary search)."""
    lo = tl.zeros_like(value)
    hi = lo + n
    for _ in range(n_steps):
        active = lo < hi
        mid = (lo + hi) // 2
        not_above = tl.load(ARR + mid, mask=active, other=0) <= value
        lo = tl.where(active & not_above, mid + 1, lo)
        hi = tl.where(active & ~not_above, mid, hi)
    return lo


@triton.jit
def _grid_kernel(
    CU,
    FIRST_BLOCK,
    BLOCK_START,
    BLOCK_LEN,
    TAIL,
    LAST_KEY,
    INSIDE,
    ROW_FIRST,
    ROW_WHOLE,
    n_seqs,
    n_queries,
    block_cap,
    n_steps,
    RATIO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Per row: its own block when that block is incomplete, its last key, its sequence's first block and how many
    whole blocks precede it; per block: first key and length."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    last_seq = tl.maximum(n_seqs - 1, 0)
    row_mask = offs < n_queries
    seq = _count_not_above(CU + 1, offs, n_seqs, n_steps)
    inside = row_mask & (seq < n_seqs)
    seq = tl.minimum(seq, last_seq)
    pos = offs - tl.load(CU + seq, mask=row_mask, other=0)
    first = tl.load(FIRST_BLOCK + seq, mask=row_mask, other=0)
    tl.store(TAIL + offs, tl.where(inside & ((pos + 1) % RATIO != 0), first + pos // RATIO, -1), mask=row_mask)
    tl.store(LAST_KEY + offs, tl.where(inside, offs, -1), mask=row_mask)
    tl.store(INSIDE + offs, inside.to(tl.int8), mask=row_mask)
    tl.store(ROW_FIRST + offs, tl.where(inside, first, -1), mask=row_mask)
    tl.store(ROW_WHOLE + offs, tl.where(inside, (pos + 1) // RATIO, 0), mask=row_mask)
    block_mask = offs < block_cap
    seq = _count_not_above(FIRST_BLOCK + 1, offs, n_seqs, n_steps)
    real = block_mask & (seq < n_seqs)
    seq = tl.minimum(seq, last_seq)
    start = tl.load(CU + seq, mask=block_mask, other=0)
    start += (offs - tl.load(FIRST_BLOCK + seq, mask=block_mask, other=0)) * RATIO
    end = tl.load(CU + seq + 1, mask=block_mask, other=0)
    tl.store(BLOCK_START + offs, tl.where(real, start, 0), mask=block_mask)
    tl.store(BLOCK_LEN + offs, tl.where(real, tl.minimum(end - start, RATIO), 0), mask=block_mask)


@triton.jit
def _span_bits(lo, hi, base, width: tl.constexpr):
    """Bits ``[lo, hi) - base`` of a ``width``-bit word starting at ``base``, as int32."""
    one = tl.full(lo.shape, 1, tl.int64)
    start = tl.minimum(tl.maximum(lo - base, 0), width).to(tl.int64)
    end = tl.minimum(tl.maximum(hi - base, 0), width).to(tl.int64)
    return (((one << end) - 1) & ~((one << start) - 1)).to(tl.int32)


@triton.jit
def _selectable(block, first, whole):
    """Ids a query may select: the whole blocks of its own sequence before it. Any other id is ignored."""
    return (block >= first) & (block < first + whole)


@triton.jit
def _mark_kernel(
    BLOCKS,
    TAIL,
    INSIDE,
    ROW_FIRST,
    ROW_WHOLE,
    QMAP,
    KMAP,
    DENSE,
    stride_b,
    q_words,
    k_words,
    K: tl.constexpr,
    KP: tl.constexpr,
    WP: tl.constexpr,
    QT: tl.constexpr,
    NUK: tl.constexpr,
):
    """Sets, for each block a query attends, bit ``block % 32`` of its tile's word ``block // 32`` in QMAP and
    bit ``query % 32`` of word ``query // 32`` of the block's key tile in KMAP. A row that selected every whole
    block before it (the first ``budget`` rows of a sequence) attends one range of blocks: it sets words and key
    tiles of that range at once, and is flagged in DENSE."""
    query = tl.program_id(0)
    if tl.load(INSIDE + query) != 0:
        offs = tl.arange(0, KP)
        block = tl.load(BLOCKS + query.to(tl.int64) * stride_b + offs, mask=offs < K, other=-1)
        first = tl.load(ROW_FIRST + query)
        whole = tl.load(ROW_WHOLE + query)
        valid = _selectable(block, first, whole)
        dense = tl.sum(valid.to(tl.int32), 0) == whole
        tl.store(DENSE + query, dense.to(tl.int8))
        tail = tl.load(TAIL + query)
        end = first + whole + (tail >= 0).to(tl.int32)  # a dense row attends blocks [first, end)
        q_row = QMAP + (query // QT).to(tl.int64) * q_words
        k_bit = 1 << (query % 32)
        k_word = query // 32
        if dense:
            word = first // 32 + tl.arange(0, WP)
            bits = _span_bits(first, end, word * 32, 32)
            tl.atomic_or(q_row + word, bits, mask=bits != 0, sem="relaxed")
            for kt0 in range(first // NUK, (end - 1) // NUK + 1, KP):
                kt = kt0 + offs
                kt_mask = kt <= (end - 1) // NUK
                tl.atomic_or(KMAP + kt.to(tl.int64) * k_words + k_word, k_bit, mask=kt_mask, sem="relaxed")
        else:
            block = tl.where(valid, block, 0)
            tl.atomic_or(q_row + block // 32, 1 << (block % 32), mask=valid, sem="relaxed")
            tl.atomic_or(KMAP + (block // NUK).to(tl.int64) * k_words + k_word, k_bit, mask=valid, sem="relaxed")
            if tail >= 0:
                tl.atomic_or(q_row + tail // 32, 1 << (tail % 32), sem="relaxed")
                tl.atomic_or(KMAP + (tail // NUK).to(tl.int64) * k_words + k_word, k_bit, sem="relaxed")


@triton.jit
def _popc(word):
    """Set bits of each int32 lane, in bit arithmetic: ROCm's libdevice has no popc."""
    x = word.to(tl.uint32, bitcast=True)
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    return ((x * 0x01010101) >> 24).to(tl.int32)


@triton.jit
def _rank_row(MAP, BEFORE, COUNT, row, width, CH: tl.constexpr):
    base = row.to(tl.int64) * width
    total = 0
    for start in range(0, width, CH):
        offs = start + tl.arange(0, CH)
        ones = _popc(tl.load(MAP + base + offs, mask=offs < width, other=0))
        tl.store(BEFORE + base + offs, total + tl.cumsum(ones, 0) - ones, mask=offs < width)
        total += tl.sum(ones, 0)
    tl.store(COUNT + row, total)


@triton.jit
def _rank_kernel(QMAP, Q_BEFORE, Q_COUNT, KMAP, K_BEFORE, K_COUNT, n_q_tiles, q_words, k_words, CH: tl.constexpr):
    """Per bitmap row (query tiles, then key tiles): set bits in the words before each word, and in the row."""
    row = tl.program_id(0)
    if row < n_q_tiles:
        _rank_row(QMAP, Q_BEFORE, Q_COUNT, row, q_words, CH)
    else:
        _rank_row(KMAP, K_BEFORE, K_COUNT, row - n_q_tiles, k_words, CH)


@triton.jit
def _offsets(COUNT, OFFSETS, n, CH: tl.constexpr):
    tl.store(OFFSETS, 0)
    total = 0
    for start in range(0, n, CH):
        offs = start + tl.arange(0, CH)
        count = tl.load(COUNT + offs, mask=offs < n, other=0)
        tl.store(OFFSETS + offs + 1, total + tl.cumsum(count, 0), mask=offs < n)
        total += tl.sum(count, 0)
    return total


@triton.jit
def _offsets_kernel(
    Q_COUNT, Q_OFFSETS, K_COUNT, K_OFFSETS, FIRST_BLOCK, SIZES, n_q_tiles, n_k_tiles, n_seqs, CH: tl.constexpr
):
    """CSR offsets of both lists, and SIZES = (query-side entries, key-side entries, blocks)."""
    tl.store(SIZES, _offsets(Q_COUNT, Q_OFFSETS, n_q_tiles, CH))
    tl.store(SIZES + 1, _offsets(K_COUNT, K_OFFSETS, n_k_tiles, CH))
    tl.store(SIZES + 2, tl.load(FIRST_BLOCK + n_seqs))


@triton.jit
def _rank_in(MAP, BEFORE, OFFSETS, row, bit, words, mask):
    """Position of bit ``bit`` of row ``row`` in its CSR list: the set bits before it in the row's bitmap."""
    at = row.to(tl.int64) * words + bit // 32
    below = ((tl.full(bit.shape, 1, tl.int64) << (bit % 32)) - 1).to(tl.int32)
    word = tl.load(MAP + at, mask=mask, other=0)
    rank = tl.load(BEFORE + at, mask=mask, other=0) + _popc(word & below)
    return tl.load(OFFSETS + row, mask=mask, other=0) + rank


@triton.jit
def _fill_query_side(block, query, BLOCK_START, QMAP, Q_BEFORE, Q_OFFSETS, Q_KEY_START, Q_BITS, q_words, QT):
    valid = block >= 0
    block = tl.where(valid, block, 0)
    pos = _rank_in(QMAP, Q_BEFORE, Q_OFFSETS, query // QT + tl.zeros_like(block), block, q_words, valid)
    tl.atomic_or(Q_BITS + pos, 1 << (query % QT), mask=valid, sem="relaxed")
    tl.store(Q_KEY_START + pos, tl.load(BLOCK_START + block, mask=valid, other=0), mask=valid)


@triton.jit
def _fill_key_side(block, query, KMAP, K_BEFORE, K_OFFSETS, K_QUERY, K_BITS, k_words, NUK):
    valid = block >= 0
    block = tl.where(valid, block, 0)
    pos = _rank_in(KMAP, K_BEFORE, K_OFFSETS, block // NUK, query + tl.zeros_like(block), k_words, valid)
    tl.atomic_or(K_BITS + pos, 1 << (block % NUK), mask=valid, sem="relaxed")
    tl.store(K_QUERY + pos, query + tl.zeros_like(block), mask=valid)


@triton.jit
def _fill_kernel(
    BLOCKS,
    TAIL,
    INSIDE,
    ROW_FIRST,
    ROW_WHOLE,
    DENSE,
    BLOCK_START,
    QMAP,
    Q_BEFORE,
    Q_OFFSETS,
    Q_KEY_START,
    Q_BITS,
    KMAP,
    K_BEFORE,
    K_OFFSETS,
    K_QUERY,
    K_BITS,
    stride_b,
    n_queries,
    q_words,
    k_words,
    K: tl.constexpr,
    KP: tl.constexpr,
    QT: tl.constexpr,
    QTP: tl.constexpr,
    NUK: tl.constexpr,
):
    """Writes each (query, block) pair into both lists: its bit (OR, all pairs of an entry meet there) and the
    entry's first key / query. A dense row writes its key-side entries whole; a query tile of dense rows of one
    sequence, whose union is one range of blocks, is written whole by its first row."""
    query = tl.program_id(0)
    if tl.load(INSIDE + query) != 0:
        offs = tl.arange(0, KP)
        first = tl.load(ROW_FIRST + query)
        whole = tl.load(ROW_WHOLE + query)
        tail = tl.load(TAIL + query)
        end = first + whole + (tail >= 0).to(tl.int32)
        dense = tl.load(DENSE + query) != 0
        block = tl.load(BLOCKS + query.to(tl.int64) * stride_b + offs, mask=offs < K, other=-1)
        block = tl.where(_selectable(block, first, whole), block, -1)
        tail_lane = tl.where(offs == 0, tail, -1)
        # the query tile: dense rows of one sequence?
        tile = query // QT
        slot = tl.arange(0, QTP)
        row = tile * QT + slot
        row_mask = (slot < QT) & (row < n_queries)
        row_inside = tl.load(INSIDE + row, mask=row_mask, other=0) != 0
        row_dense = tl.load(DENSE + row, mask=row_mask, other=0) != 0
        row_first = tl.load(ROW_FIRST + row, mask=row_mask, other=-1)
        row_end = row_first + tl.load(ROW_WHOLE + row, mask=row_mask, other=0)
        row_end += (tl.load(TAIL + row, mask=row_mask, other=-1) >= 0).to(tl.int32)
        tile_dense = tl.sum((row_inside & ~(row_dense & (row_first == first))).to(tl.int32), 0) == 0
        leader = tl.min(tl.where(row_inside, slot, QTP), 0) == query % QT
        if tile_dense:
            if leader:
                for e0 in range(0, tl.max(tl.where(row_inside, row_end - first, 0), 0), KP):
                    e = e0 + offs
                    covered = row_inside[None, :] & (e[:, None] < (row_end - first)[None, :])
                    bits = tl.sum(covered.to(tl.int32) << slot[None, :], 1)
                    pos = tl.load(Q_OFFSETS + tile) + e
                    tl.store(Q_BITS + pos, bits, mask=bits != 0)
                    tl.store(
                        Q_KEY_START + pos, tl.load(BLOCK_START + first + e, mask=bits != 0, other=0), mask=bits != 0
                    )
        else:
            q_lists = (BLOCK_START, QMAP, Q_BEFORE, Q_OFFSETS, Q_KEY_START, Q_BITS, q_words, QT)
            if dense:
                for b0 in range(first, end, KP):
                    _fill_query_side(tl.where(b0 + offs < end, b0 + offs, -1), query, *q_lists)
            else:
                _fill_query_side(block, query, *q_lists)
                if tail >= 0:
                    _fill_query_side(tail_lane, query, *q_lists)
        if dense:
            for kt0 in range(first // NUK, (end - 1) // NUK + 1, KP):
                kt = kt0 + offs
                kt_mask = kt <= (end - 1) // NUK
                pos = _rank_in(KMAP, K_BEFORE, K_OFFSETS, kt, query + tl.zeros_like(kt), k_words, kt_mask)
                tl.store(K_BITS + pos, _span_bits(first, end, kt * NUK, NUK), mask=kt_mask)
                tl.store(K_QUERY + pos, query + tl.zeros_like(kt), mask=kt_mask)
        else:
            k_lists = (KMAP, K_BEFORE, K_OFFSETS, K_QUERY, K_BITS, k_words, NUK)
            _fill_key_side(block, query, *k_lists)
            if tail >= 0:
                _fill_key_side(tail_lane, query, *k_lists)


@torch.no_grad()
def build_qsa_block_index(
    blocks: Tensor, cu_seqlens: Tensor, *, compress_ratio: int, num_heads: int, num_kv_heads: int
) -> QSABlockIndex:
    """Tile lists for one packed (thd) batch.

    ``blocks`` ``[T, K]`` holds the distinct ids of the whole blocks each query selected, ``-1`` where unused.
    Every sequence of ``cu_seqlens`` (``[B + 1]``) is cut into blocks of ``compress_ratio`` keys from its
    own first token and the ids run over the batch, as ``PackedBlockLayout`` numbers them. An id that is not
    a whole block of the query's own sequence before it is ignored. On top of its blocks a query always
    attends the causal tail of its own block; a row at or past ``cu_seqlens[-1]`` attends nothing.
    Synchronizes with the host once, for the list sizes.
    """
    group = num_heads // num_kv_heads
    assert num_heads % num_kv_heads == 0 and group <= _TILE_ROWS, (num_heads, num_kv_heads)
    query_tile = min(32, _TILE_ROWS // group)  # one int32 bit per query of a tile
    key_tile = _key_tile_blocks(blocks.device)
    n_queries, width = blocks.shape
    blocks = blocks.to(torch.int32)
    cu = cu_seqlens.to(device=blocks.device, dtype=torch.int32)
    n_seqs = cu.numel() - 1
    # sequences that fit in the rows cut at most this many blocks; the grid is cut to the real count below
    block_cap = (n_queries + (compress_ratio - 1) * n_seqs) // compress_ratio
    grid = _block_grid(cu, n_queries, block_cap, compress_ratio)
    n_q_tiles = -(-n_queries // query_tile)
    shape = _ListShape(query_tile, key_tile, n_q_tiles, -(-block_cap // key_tile), -(-block_cap // 32))
    maps = _mark(blocks, grid, shape)
    counts, before = _rank(maps, shape)
    q_order = torch.argsort(counts[:n_q_tiles], descending=True, stable=True).to(torch.int32)
    offsets, sizes = _list_offsets(counts, grid["first_block"], shape)
    n_q, n_k, n_blocks = sizes.tolist()
    if n_blocks > block_cap:
        raise ValueError(f"cu_seqlens {cu_seqlens.tolist()} reaches past the {n_queries} rows of blocks")
    lists = _fill(blocks, grid, shape, maps, before, offsets, n_q, n_k)
    n_k_tiles = -(-n_blocks // key_tile)
    return QSABlockIndex(
        query_tile=query_tile,
        key_tile=key_tile,
        ratio=compress_ratio,
        q_offsets=offsets[: n_q_tiles + 1],
        q_key_start=lists["q_key_start"],
        q_bits=lists["q_bits"],
        q_order=q_order,
        k_offsets=offsets[n_q_tiles + 1 : n_q_tiles + 2 + n_k_tiles],
        k_query=lists["k_query"],
        k_bits=lists["k_bits"],
        k_order=torch.arange(n_k_tiles, dtype=torch.int32, device=blocks.device),
        block_start=grid["block_start"][:n_blocks],
        block_len=grid["block_len"][:n_blocks],
        last_key=grid["last_key"],
    )


@dataclass(frozen=True)
class _ListShape:
    query_tile: int
    key_tile: int
    n_q_tiles: int
    n_k_tiles: int  # key tiles of the block cap; the ones past the real blocks stay empty
    q_words: int  # 32-bit words per query tile's block bitmap

    def k_words(self, n_queries: int) -> int:
        return -(-n_queries // 32)


def _block_grid(cu: Tensor, n_queries: int, block_cap: int, ratio: int) -> dict:
    n_seqs = cu.numel() - 1
    i32 = dict(dtype=torch.int32, device=cu.device)
    first_block = torch.empty(n_seqs + 1, **i32)
    _seq_blocks_kernel[(1,)](cu, first_block, n_seqs, RATIO=ratio, CH=_SCAN_CHUNK)
    buf = torch.empty(2 * block_cap + 4 * n_queries, **i32)
    rows = buf[2 * block_cap :].view(4, n_queries)
    flags = torch.empty(2, n_queries, dtype=torch.int8, device=cu.device)
    grid = {
        "first_block": first_block,
        "block_start": buf[:block_cap],
        "block_len": buf[block_cap : 2 * block_cap],
        "tail": rows[0],
        "last_key": rows[1],
        "row_first": rows[2],
        "row_whole": rows[3],
        "inside": flags[0],
        "dense": flags[1],
    }
    span = max(n_queries, block_cap)
    if span:
        _grid_kernel[(triton.cdiv(span, 1024),)](
            cu,
            first_block,
            grid["block_start"],
            grid["block_len"],
            grid["tail"],
            grid["last_key"],
            grid["inside"],
            grid["row_first"],
            grid["row_whole"],
            n_seqs,
            n_queries,
            block_cap,
            max(1, n_seqs.bit_length()),
            RATIO=ratio,
            BLOCK=1024,
        )
    return grid


def _mark(blocks: Tensor, grid: dict, shape: _ListShape) -> tuple[Tensor, Tensor]:
    n_queries, width = blocks.shape
    k_words = shape.k_words(n_queries)
    q_size = shape.n_q_tiles * shape.q_words
    maps = torch.zeros(q_size + shape.n_k_tiles * k_words, dtype=torch.int32, device=blocks.device)
    if n_queries:
        _mark_kernel[(n_queries,)](
            blocks,
            grid["tail"],
            grid["inside"],
            grid["row_first"],
            grid["row_whole"],
            maps[:q_size],
            maps[q_size:],
            grid["dense"],
            blocks.stride(0),
            shape.q_words,
            k_words,
            K=width,
            KP=triton.next_power_of_2(max(width, 1)),
            WP=triton.next_power_of_2(width // 32 + 3),
            QT=shape.query_tile,
            NUK=shape.key_tile,
        )
    return maps[:q_size], maps[q_size:]


def _rank(maps: tuple[Tensor, Tensor], shape: _ListShape) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    qmap, kmap = maps
    before = torch.empty(qmap.numel() + kmap.numel(), dtype=torch.int32, device=qmap.device)
    counts = torch.empty(shape.n_q_tiles + shape.n_k_tiles, dtype=torch.int32, device=qmap.device)
    k_words = kmap.numel() // max(shape.n_k_tiles, 1)
    if counts.numel():
        _rank_kernel[(counts.numel(),)](
            qmap,
            before[: qmap.numel()],
            counts[: shape.n_q_tiles],
            kmap,
            before[qmap.numel() :],
            counts[shape.n_q_tiles :],
            shape.n_q_tiles,
            shape.q_words,
            k_words,
            CH=_SCAN_CHUNK,
        )
    return counts, (before[: qmap.numel()], before[qmap.numel() :])


def _list_offsets(counts: Tensor, first_block: Tensor, shape: _ListShape) -> tuple[Tensor, Tensor]:
    """Both lists' CSR offsets in one buffer (query tiles, then key tiles), and their sizes with the block count."""
    buf = torch.empty(counts.numel() + 2 + 3, dtype=torch.int32, device=counts.device)
    n_q = shape.n_q_tiles
    _offsets_kernel[(1,)](
        counts[:n_q],
        buf[: n_q + 1],
        counts[n_q:],
        buf[n_q + 1 : counts.numel() + 2],
        first_block,
        buf[counts.numel() + 2 :],
        n_q,
        shape.n_k_tiles,
        first_block.numel() - 1,
        CH=_SCAN_CHUNK,
    )
    return buf[: counts.numel() + 2], buf[counts.numel() + 2 :]


def _fill(blocks, grid, shape: _ListShape, maps, before, offsets, n_q: int, n_k: int) -> dict:
    n_queries, width = blocks.shape
    buf = torch.empty(2 * (n_q + n_k), dtype=torch.int32, device=blocks.device)
    bits = buf[n_q + n_k :]
    bits.zero_()
    lists = {"q_key_start": buf[:n_q], "k_query": buf[n_q : n_q + n_k], "q_bits": bits[:n_q], "k_bits": bits[n_q:]}
    if n_queries:
        _fill_kernel[(n_queries,)](
            blocks,
            grid["tail"],
            grid["inside"],
            grid["row_first"],
            grid["row_whole"],
            grid["dense"],
            grid["block_start"],
            maps[0],
            before[0],
            offsets[: shape.n_q_tiles + 1],
            lists["q_key_start"],
            lists["q_bits"],
            maps[1],
            before[1],
            offsets[shape.n_q_tiles + 1 :],
            lists["k_query"],
            lists["k_bits"],
            blocks.stride(0),
            n_queries,
            shape.q_words,
            shape.k_words(n_queries),
            K=width,
            KP=triton.next_power_of_2(max(width, 1)),
            QT=shape.query_tile,
            QTP=triton.next_power_of_2(shape.query_tile),
            NUK=shape.key_tile,
        )
    return lists
