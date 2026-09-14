"""Tensor-core QSA sparse attention for training: forward + backward.

Same semantics as ``qsa_sparse_attn.py`` -- each query attends exactly the tokens in
its selection row -- but reached a different way, because the gather-per-query form
cannot use ``tl.dot``: with a distinct key set per query it has to materialise a
``[BQ, BK, D]`` tile and reduce it with ALU math. Measured at the production shape
(T=25k, 12 q-heads, D=256, budget 2048) that costs 2.5 s forward and 13.9 s
forward+backward per layer, which is ~330x slower than a DENSE causal flash kernel over
the same tensors (9.4 ms / 39.2 ms) even though dense does 6x more FLOPs. QSA at 25k
tokens only removes ~6x of the work, so paying 300x for the privilege is a large loss:
on the 16-node agentic run those three QSA layers per pipeline stage were ~49 s of a
~52 s micro-batch.

So this kernel walks key tiles with ``tl.dot`` and masks each (query, key) pair to the
query's own selection. Tiles nobody in the query tile selected are skipped entirely via
a CSR-style per-query-tile list, so it beats a dense sweep by the coverage ratio rather
than merely matching it. The result is exact, not an approximation:

- selection membership is tested at BLOCK granularity (the indexer picks blocks of
  ``BLK`` consecutive tokens), which is a superset of the row's token list, because
  expanding a block drops only the tokens past the query's own position;
- the dropped ones come back out via ``lo``/``hi``, the inclusive key range the caller
  already computes (sequence start .. query position).

bf16 inputs feed the dots with fp32 accumulation, the flash-attention convention, which
is also what the sglang kernel this mirrors does -- the old kernel's fp32 ALU path was
the odd one out.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _qsa_bs_fwd_kernel(
    Q,
    K,
    V,
    SEL,
    LO,
    HI,
    BLKBASE,
    TOKBASE,
    KLIST,
    KCNT,
    OUT,
    LSE,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    stride_st,
    stride_kl,
    stride_ot,
    stride_oh,
    T,
    NB,
    scale,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    BQ: tl.constexpr,
    BK: tl.constexpr,
    BLK: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    kv_head = pid_h // GROUP

    offs_q = pid_t * BQ + tl.arange(0, BQ)
    offs_d = tl.arange(0, D)
    q_mask = offs_q < T

    q = tl.load(
        Q + offs_q[:, None] * stride_qt + pid_h * stride_qh + offs_d[None, :],
        mask=q_mask[:, None],
        other=0.0,
    )
    lo = tl.load(LO + offs_q, mask=q_mask, other=0)
    hi = tl.load(HI + offs_q, mask=q_mask, other=-1)
    blk_base = tl.load(BLKBASE + offs_q, mask=q_mask, other=0)
    tok_base = tl.load(TOKBASE + offs_q, mask=q_mask, other=0)

    m_i = tl.full((BQ,), float("-inf"), tl.float32)
    l_i = tl.zeros((BQ,), tl.float32)
    acc = tl.zeros((BQ, D), tl.float32)

    # Only the key tiles this query tile actually selected, so the kernel beats a dense
    # sweep by the coverage ratio instead of merely matching it. The list is the union
    # over the tile's queries; per-query exactness comes from the mask below.
    n_tiles = tl.load(KCNT + pid_t)
    for i in range(0, n_tiles):
        kt = tl.load(KLIST + pid_t * stride_kl + i)
        offs_k = kt * BK + tl.arange(0, BK)
        k_in = offs_k < T

        # per-sequence block grid: after the packed-indexer fix a sequence's blocks start
        # at its own first token, which is not a multiple of BLK in a packed batch.
        # Looked up per TOKEN, not per 4-token group: a sequence whose start is not a
        # multiple of BLK has its per-sequence blocks straddling the global groups, so a
        # group-granular lookup would assign some tokens to the wrong block.
        blk = blk_base[:, None] + (offs_k[None, :] - tok_base[:, None]) // BLK
        sel = tl.load(
            SEL + offs_q[:, None] * stride_st + blk,
            mask=q_mask[:, None] & k_in[None, :] & (blk >= 0) & (blk < NB),
            other=0,
        )
        ok = (sel != 0) & (offs_k[None, :] <= hi[:, None]) & (offs_k[None, :] >= lo[:, None]) & k_in[None, :]

        k_tile = tl.load(
            K + offs_k[:, None] * stride_kt + kv_head * stride_kh + offs_d[None, :],
            mask=k_in[:, None],
            other=0.0,
        )
        s = tl.dot(q, tl.trans(k_tile)) * scale
        s = tl.where(ok, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(s - m_use[:, None])
        p = tl.where(ok, p, 0.0)
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i - m_use))
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_tile = tl.load(
            V + offs_k[:, None] * stride_vt + kv_head * stride_vh + offs_d[None, :],
            mask=k_in[:, None],
            other=0.0,
        )
        acc += tl.dot(p.to(v_tile.dtype), v_tile)
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / l_safe[:, None]
    tl.store(
        OUT + offs_q[:, None] * stride_ot + pid_h * stride_oh + offs_d[None, :],
        out,
        mask=q_mask[:, None],
    )
    lse = tl.where(m_i == float("-inf"), float("-inf"), m_i + tl.log(l_safe))
    tl.store(LSE + pid_h * T + offs_q, lse, mask=q_mask)


@triton.jit
def _qsa_bs_dq_kernel(
    Q,
    K,
    V,
    SEL,
    LO,
    HI,
    BLKBASE,
    TOKBASE,
    KLIST,
    KCNT,
    OUT,
    LSE,
    DO,
    DQ,
    DELTA,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    stride_st,
    stride_kl,
    stride_ot,
    stride_oh,
    T,
    NB,
    scale,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    BQ: tl.constexpr,
    BK: tl.constexpr,
    BLK: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    kv_head = pid_h // GROUP

    offs_q = pid_t * BQ + tl.arange(0, BQ)
    offs_d = tl.arange(0, D)
    q_mask = offs_q < T

    q = tl.load(Q + offs_q[:, None] * stride_qt + pid_h * stride_qh + offs_d[None, :], mask=q_mask[:, None], other=0.0)
    do = tl.load(
        DO + offs_q[:, None] * stride_ot + pid_h * stride_oh + offs_d[None, :], mask=q_mask[:, None], other=0.0
    )
    lse = tl.load(LSE + pid_h * T + offs_q, mask=q_mask, other=0.0)
    delta = tl.load(DELTA + pid_h * T + offs_q, mask=q_mask, other=0.0)
    lse_safe = tl.where(lse == float("-inf"), 0.0, lse)
    alive = lse != float("-inf")

    lo = tl.load(LO + offs_q, mask=q_mask, other=0)
    hi = tl.load(HI + offs_q, mask=q_mask, other=-1)
    blk_base = tl.load(BLKBASE + offs_q, mask=q_mask, other=0)
    tok_base = tl.load(TOKBASE + offs_q, mask=q_mask, other=0)

    dq = tl.zeros((BQ, D), tl.float32)
    n_tiles = tl.load(KCNT + pid_t)
    for i in range(0, n_tiles):
        kt = tl.load(KLIST + pid_t * stride_kl + i)
        offs_k = kt * BK + tl.arange(0, BK)
        k_in = offs_k < T

        # per-sequence block grid: after the packed-indexer fix a sequence's blocks start
        # at its own first token, which is not a multiple of BLK in a packed batch.
        # Looked up per TOKEN, not per 4-token group: a sequence whose start is not a
        # multiple of BLK has its per-sequence blocks straddling the global groups, so a
        # group-granular lookup would assign some tokens to the wrong block.
        blk = blk_base[:, None] + (offs_k[None, :] - tok_base[:, None]) // BLK
        sel = tl.load(
            SEL + offs_q[:, None] * stride_st + blk,
            mask=q_mask[:, None] & k_in[None, :] & (blk >= 0) & (blk < NB),
            other=0,
        )
        ok = (
            (sel != 0)
            & (offs_k[None, :] <= hi[:, None])
            & (offs_k[None, :] >= lo[:, None])
            & k_in[None, :]
            & alive[:, None]
        )

        k_tile = tl.load(
            K + offs_k[:, None] * stride_kt + kv_head * stride_kh + offs_d[None, :], mask=k_in[:, None], other=0.0
        )
        v_tile = tl.load(
            V + offs_k[:, None] * stride_vt + kv_head * stride_vh + offs_d[None, :], mask=k_in[:, None], other=0.0
        )

        s = tl.dot(q, tl.trans(k_tile)) * scale
        p = tl.exp(s - lse_safe[:, None])
        p = tl.where(ok, p, 0.0)

        dp = tl.dot(do, tl.trans(v_tile))
        ds = (p * (dp - delta[:, None]) * scale).to(k_tile.dtype)

        dq += tl.dot(ds, k_tile)

    tl.store(DQ + offs_q[:, None] * stride_qt + pid_h * stride_qh + offs_d[None, :], dq, mask=q_mask[:, None])


@triton.jit
def _qsa_bs_dkdv_kernel(
    Q,
    K,
    V,
    SEL,
    LO,
    HI,
    BLKBASE,
    TOKBASE,
    QLIST,
    QCNT,
    LSE,
    DO,
    DELTA,
    DK,
    DV,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    stride_st,
    stride_ql,
    stride_ot,
    stride_oh,
    T,
    NB,
    scale,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    BQ: tl.constexpr,
    BK: tl.constexpr,
    BLK: tl.constexpr,
):
    """dK/dV keyed on the KEY tile, so nothing needs atomics.

    Launched per KV head, not per query head: with GQA the ``GROUP`` query heads sharing a
    KV head all contribute to the same dK/dV, so they have to be summed here. Writing them
    from separate programs would have them overwrite each other (the gather kernel got away
    with it only because it used atomic_add).
    """
    pid_k = tl.program_id(0)
    kv_head = tl.program_id(1)

    offs_k = pid_k * BK + tl.arange(0, BK)
    offs_d = tl.arange(0, D)
    k_in = offs_k < T

    k_tile = tl.load(
        K + offs_k[:, None] * stride_kt + kv_head * stride_kh + offs_d[None, :], mask=k_in[:, None], other=0.0
    )
    v_tile = tl.load(
        V + offs_k[:, None] * stride_vt + kv_head * stride_vh + offs_d[None, :], mask=k_in[:, None], other=0.0
    )
    dk = tl.zeros((BK, D), tl.float32)
    dv = tl.zeros((BK, D), tl.float32)

    n_q = tl.load(QCNT + pid_k)
    for i in range(0, n_q):
        qt = tl.load(QLIST + pid_k * stride_ql + i)
        offs_q = qt * BQ + tl.arange(0, BQ)
        q_mask = offs_q < T
        lo = tl.load(LO + offs_q, mask=q_mask, other=0)
        hi = tl.load(HI + offs_q, mask=q_mask, other=-1)
        blk_base = tl.load(BLKBASE + offs_q, mask=q_mask, other=0)
        tok_base = tl.load(TOKBASE + offs_q, mask=q_mask, other=0)

        blk = blk_base[:, None] + (offs_k[None, :] - tok_base[:, None]) // BLK
        sel = tl.load(
            SEL + offs_q[:, None] * stride_st + blk,
            mask=q_mask[:, None] & k_in[None, :] & (blk >= 0) & (blk < NB),
            other=0,
        )
        ok = (
            (sel != 0)
            & (offs_k[None, :] <= hi[:, None])
            & (offs_k[None, :] >= lo[:, None])
            & k_in[None, :]
            & q_mask[:, None]
        )

        for gh in range(0, GROUP):
            qh = kv_head * GROUP + gh
            q = tl.load(
                Q + offs_q[:, None] * stride_qt + qh * stride_qh + offs_d[None, :],
                mask=q_mask[:, None],
                other=0.0,
            )
            do = tl.load(
                DO + offs_q[:, None] * stride_ot + qh * stride_oh + offs_d[None, :],
                mask=q_mask[:, None],
                other=0.0,
            )
            lse = tl.load(LSE + qh * T + offs_q, mask=q_mask, other=0.0)
            delta = tl.load(DELTA + qh * T + offs_q, mask=q_mask, other=0.0)
            okh = ok & (lse[:, None] != float("-inf"))

            lse_safe = tl.where(lse == float("-inf"), 0.0, lse)
            sc = tl.dot(q, tl.trans(k_tile)) * scale
            p = tl.exp(sc - lse_safe[:, None])
            p = tl.where(okh, p, 0.0)

            dp = tl.dot(do, tl.trans(v_tile))
            ds = (p * (dp - delta[:, None]) * scale).to(k_tile.dtype)

            dk += tl.dot(tl.trans(ds), q)
            dv += tl.dot(tl.trans(p.to(do.dtype)), do)

    tl.store(DK + offs_k[:, None] * stride_kt + kv_head * stride_kh + offs_d[None, :], dk, mask=k_in[:, None])
    tl.store(DV + offs_k[:, None] * stride_vt + kv_head * stride_vh + offs_d[None, :], dv, mask=k_in[:, None])


def selection_to_block_bitmap(indices: Tensor, num_tokens: int, block_size: int) -> Tensor:
    """``[T, K]`` token indices (``-1`` pad) -> ``[T, ceil(T / block_size)]`` uint8 flags.

    A block is flagged when any of its tokens appears in the row. Tokens that the caller
    clamped away inside an otherwise selected block are re-excluded by the ``lo``/``hi``
    range test in the kernel, so this stays exact while being ``block_size``x smaller.
    """
    num_blocks = -(-num_tokens // block_size)
    flags = torch.zeros(indices.shape[0], num_blocks, dtype=torch.uint8, device=indices.device)
    valid = indices >= 0
    rows = torch.arange(indices.shape[0], device=indices.device).unsqueeze(1).expand_as(indices)
    blocks = torch.where(valid, indices // block_size, torch.zeros_like(indices))
    flags[rows[valid], blocks[valid].long()] = 1
    return flags


def build_tile_index(sel: Tensor, bq: int, bk: int, block_size: int) -> tuple[Tensor, Tensor]:
    """``[T, NB]`` block flags -> (``klist`` [NQT, maxc] int32, ``kcnt`` [NQT] int32).

    ``klist[i]`` lists, ascending, the key tiles that at least one query in query-tile
    ``i`` selected. Per-query exactness still comes from the in-kernel mask; this only
    decides which tiles are worth visiting.
    """
    T, nb = sel.shape
    bpt = bk // block_size
    nqt = -(-T // bq)
    nkt = -(-nb // bpt)
    pad_b = nkt * bpt - nb
    pad_q = nqt * bq - T
    if pad_b or pad_q:
        sel = torch.nn.functional.pad(sel, (0, pad_b, 0, pad_q))
    tile = sel.view(nqt, bq, nkt, bpt).amax(dim=3).amax(dim=1) > 0  # [nqt, nkt]
    kcnt = tile.sum(dim=1).to(torch.int32)
    maxc = max(int(kcnt.max().item()), 1)
    order = torch.argsort((~tile).to(torch.int8), dim=1, stable=True)
    klist = order[:, :maxc].contiguous().to(torch.int32)
    return klist, kcnt


def build_tile_index_pair(sel: Tensor, bq: int, bk: int, block_size: int):
    """Both directions of the tile map: (klist, kcnt) per query tile, (qlist, qcnt) per key tile.

    The transposed half is what lets dK/dV be keyed on the key tile and so avoid atomics.
    """
    T, nb = sel.shape
    bpt = bk // block_size
    nqt = -(-T // bq)
    nkt = -(-nb // bpt)
    pad_b = nkt * bpt - nb
    pad_q = nqt * bq - T
    padded = sel
    if pad_b or pad_q:
        padded = torch.nn.functional.pad(sel, (0, pad_b, 0, pad_q))
    tile = padded.view(nqt, bq, nkt, bpt).amax(dim=3).amax(dim=1) > 0

    def compact(mat):
        cnt = mat.sum(dim=1).to(torch.int32)
        maxc = max(int(cnt.max().item()), 1)
        order = torch.argsort((~mat).to(torch.int8), dim=1, stable=True)
        return order[:, :maxc].contiguous().to(torch.int32), cnt

    klist, kcnt = compact(tile)
    qlist, qcnt = compact(tile.t().contiguous())
    return klist, kcnt, qlist, qcnt


class _QSABlockSparseAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sel, lo, hi, blk_base, tok_base, scale, block_size):
        T, Hq, D = q.shape
        S, Hkv, _ = k.shape
        assert Hq % Hkv == 0
        group = Hq // Hkv
        qc, kc, vc = q.contiguous(), k.contiguous(), v.contiguous()
        selc = sel.contiguous()
        BQ_, BK_ = 64, 64
        klist, kcnt = build_tile_index(selc, BQ_, BK_, block_size)
        o = torch.empty(T, Hq, D, device=q.device, dtype=torch.float32)
        lse = torch.empty(Hq, T, device=q.device, dtype=torch.float32)
        BQ, BK = 64, 64
        grid = (triton.cdiv(T, BQ), Hq)
        _qsa_bs_fwd_kernel[grid](
            qc,
            kc,
            vc,
            selc,
            lo,
            hi,
            blk_base,
            tok_base,
            klist,
            kcnt,
            o,
            lse,
            qc.stride(0),
            qc.stride(1),
            kc.stride(0),
            kc.stride(1),
            vc.stride(0),
            vc.stride(1),
            selc.stride(0),
            klist.stride(0),
            o.stride(0),
            o.stride(1),
            T,
            selc.shape[1],
            scale,
            GROUP=group,
            D=D,
            BQ=BQ,
            BK=BK,
            BLK=block_size,
            num_warps=8,
            num_stages=2,
        )
        ctx.save_for_backward(qc, kc, vc, selc, lo, hi, blk_base, tok_base, klist, kcnt, o, lse)
        ctx.scale = scale
        ctx.group = group
        ctx.block_size = block_size
        return o.to(q.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        qc, kc, vc, selc, lo, hi, blk_base, tok_base, _klist_fwd, _kcnt_fwd, o, lse = ctx.saved_tensors
        T, Hq, D = qc.shape
        do = grad_out.contiguous().to(qc.dtype)
        # delta once, in torch: both backward kernels need it and neither should redo it
        delta = (do.float() * o.float()).sum(-1).transpose(0, 1).contiguous()
        dq = torch.empty(T, Hq, D, device=qc.device, dtype=torch.float32)
        dk = torch.zeros(kc.shape, device=kc.device, dtype=torch.float32)
        dv = torch.zeros(vc.shape, device=vc.device, dtype=torch.float32)

        BQ, BK = 64, 32
        klist, kcnt, qlist, qcnt = build_tile_index_pair(selc, BQ, BK, ctx.block_size)
        common = (
            qc.stride(0),
            qc.stride(1),
            kc.stride(0),
            kc.stride(1),
            vc.stride(0),
            vc.stride(1),
            selc.stride(0),
        )
        _qsa_bs_dq_kernel[(triton.cdiv(T, BQ), Hq)](
            qc,
            kc,
            vc,
            selc,
            lo,
            hi,
            blk_base,
            tok_base,
            klist,
            kcnt,
            o,
            lse,
            do,
            dq,
            delta,
            *common,
            klist.stride(0),
            do.stride(0),
            do.stride(1),
            T,
            selc.shape[1],
            ctx.scale,
            GROUP=ctx.group,
            D=D,
            BQ=BQ,
            BK=BK,
            BLK=ctx.block_size,
            num_warps=8,
            num_stages=1,
        )
        _qsa_bs_dkdv_kernel[(triton.cdiv(T, BK), kc.shape[1])](
            qc,
            kc,
            vc,
            selc,
            lo,
            hi,
            blk_base,
            tok_base,
            qlist,
            qcnt,
            lse,
            do,
            delta,
            dk,
            dv,
            *common,
            qlist.stride(0),
            do.stride(0),
            do.stride(1),
            T,
            selc.shape[1],
            ctx.scale,
            GROUP=ctx.group,
            D=D,
            BQ=BQ,
            BK=BK,
            BLK=ctx.block_size,
            num_warps=8,
            num_stages=1,
        )
        return dq.to(qc.dtype), dk.to(kc.dtype), dv.to(vc.dtype), None, None, None, None, None, None, None


def qsa_block_sparse_attention_triton(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    sel_blocks: Tensor,
    lo: Tensor,
    hi: Tensor,
    blk_base: Tensor,
    tok_base: Tensor,
    scale: float,
    block_size: int = 4,
) -> Tensor:
    """``q`` [T, Hq, D], ``k``/``v`` [S, Hkv, D], ``sel_blocks`` [T, NB] uint8.

    ``lo``/``hi`` are the inclusive key range per query; ``blk_base``/``tok_base`` place
    the query's sequence in the packed block grid (both zero for a single sequence).
    """
    return _QSABlockSparseAttn.apply(q, k, v, sel_blocks, lo, hi, blk_base, tok_base, scale, block_size)


def qsa_sparse_attention_from_indices(
    q: Tensor, k: Tensor, v: Tensor, indices: Tensor, scale: float, block_size: int = 4
) -> Tensor:
    """Drop-in for the gather kernel: derives the bitmap and range from ``indices``."""
    T = q.shape[0]
    sel = selection_to_block_bitmap(indices, T, block_size)
    valid = indices >= 0
    big = torch.iinfo(torch.int32).max
    lo = torch.where(valid, indices, torch.full_like(indices, big)).min(dim=1).values.to(torch.int32)
    hi = torch.where(valid, indices, torch.full_like(indices, -1)).max(dim=1).values.to(torch.int32)
    zeros = torch.zeros(T, dtype=torch.int32, device=q.device)
    return qsa_block_sparse_attention_triton(q, k, v, sel, lo, hi, zeros, zeros, scale, block_size)
