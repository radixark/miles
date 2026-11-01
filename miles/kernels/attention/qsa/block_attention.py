"""QSA attention over the selected key blocks, forward and backward (Triton).

QSA lets every query attend up to ``indexer_budget / compress_ratio`` whole blocks of ``compress_ratio``
keys from its own sequence, plus the causal tail of the block it sits in, and all query heads share that
choice. These kernels compute only those keys:

* forward and dQ: a program is ``T`` consecutive queries x the ``G`` query heads of one KV head, so a
  tensor-core tile holds ``T * G <= 64`` rows. It walks the sorted union of its queries' blocks, ``NU``
  blocks per step, and masks each (query, key) pair with a per-(query, block) bit and ``key <= query``.
  Consecutive queries pick mostly the same blocks, so the union stays close to one query's selection.
* dK / dV: a program owns ``NUK`` consecutive blocks of one KV head and walks the queries that selected
  any of them, ``64 / G`` queries per step. Every key row is written by one program, so the backward
  needs no atomics and is deterministic.

The tile lists come from :func:`~miles.kernels.attention.qsa.block_index.build_qsa_block_index`.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from torch import Tensor

from miles.kernels.attention.qsa.block_index import _TILE_ROWS, QSABlockIndex

_LOG2E = 1.4426950408889634


@dataclass(frozen=True)
class _Launch:
    fwd: dict
    dq: dict
    dkdv: dict


# Measured per arch major at head_dim 256, 12 query heads per KV head, 8K and 64K tokens of indexer selections,
# with block_index's key tiles: sm_90 on H20 and H100, sm_100 on B200. Other archs take the sm_90 row.
_TUNED = {
    9: _Launch(
        fwd={"NU": 8, "num_warps": 4, "num_stages": 3},
        dq={"NU": 8, "num_warps": 4, "num_stages": 1},
        dkdv={"num_warps": 4, "num_stages": 3},
    ),
    10: _Launch(
        fwd={"NU": 16, "num_warps": 8, "num_stages": 3},
        dq={"NU": 16, "num_warps": 8, "num_stages": 3},
        dkdv={"num_warps": 8, "num_stages": 1},
    ),
}


def _launch(device: torch.device) -> _Launch:
    major = torch.cuda.get_device_capability(device)[0] if device.type == "cuda" else 9
    return _TUNED.get(major, _TUNED[9])


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    OUT,
    LSE,
    Q_OFFSETS,
    Q_KEY_START,
    Q_BITS,
    Q_ORDER,
    LAST_KEY,
    stride_qs,
    stride_qh,
    stride_ks,
    stride_kh,
    stride_vs,
    stride_vh,
    stride_os,
    stride_oh,
    n_queries,
    n_keys,
    qk_scale,
    HQ: tl.constexpr,
    G: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NU: tl.constexpr,
    RATIO: tl.constexpr,
):
    tile = tl.load(Q_ORDER + tl.program_id(0))
    kv_head = tl.program_id(1)
    BLOCK_N: tl.constexpr = NU * RATIO
    offs_m = tl.arange(0, BLOCK_M)
    slot = offs_m // G
    head = kv_head * G + offs_m % G
    query = tile * T + slot
    row_mask = (slot < T) & (query < n_queries)
    offs_d = tl.arange(0, D)
    query64 = query[:, None].to(tl.int64)
    q = tl.load(
        Q + query64 * stride_qs + head[:, None] * stride_qh + offs_d[None, :], mask=row_mask[:, None], other=0.0
    )
    last_key = tl.load(LAST_KEY + query, mask=row_mask, other=-1)
    lo = tl.load(Q_OFFSETS + tile)
    hi = tl.load(Q_OFFSETS + tile + 1)
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    block_of_n = offs_n // RATIO
    key_in_block = offs_n % RATIO
    for start in range(lo, hi, NU):
        entry = start + block_of_n
        entry_mask = entry < hi
        bits = tl.load(Q_BITS + entry, mask=entry_mask, other=0)
        key = tl.load(Q_KEY_START + entry, mask=entry_mask, other=0) + key_in_block
        key_mask = entry_mask & (key < n_keys)
        key64 = key[:, None].to(tl.int64)
        k = tl.load(K + kv_head * stride_kh + key64 * stride_ks + offs_d[None, :], mask=key_mask[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k))
        allowed = ((bits[None, :] >> slot[:, None]) & 1) != 0
        allowed = allowed & (key[None, :] <= last_key[:, None]) & key_mask[None, :]
        qk = tl.where(allowed, qk * qk_scale, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.math.exp2(m_i - m_use)
        p = tl.math.exp2(qk - m_use[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(V + kv_head * stride_vh + key64 * stride_vs + offs_d[None, :], mask=key_mask[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    no_key = l_i == 0.0
    l_safe = tl.where(no_key, 1.0, l_i)
    acc = acc / l_safe[:, None]
    # a row without any key gets output 0 and an infinite LSE, so its backward is 0 too
    lse = tl.where(no_key, float("inf"), m_i + tl.math.log2(l_safe))
    o_ptrs = OUT + query64 * stride_os + head[:, None] * stride_oh + offs_d[None, :]
    tl.store(o_ptrs, acc.to(OUT.dtype.element_ty), mask=row_mask[:, None])
    tl.store(LSE + query.to(tl.int64) * HQ + head, lse, mask=row_mask)


@triton.jit
def _delta_kernel(
    OUT,
    DO,
    DELTA,
    stride_os,
    stride_oh,
    stride_dos,
    stride_doh,
    n_rows,
    HQ: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row_mask = row < n_rows
    query = (row // HQ).to(tl.int64)
    head = row % HQ
    offs_d = tl.arange(0, D)
    o_ptrs = OUT + query[:, None] * stride_os + head[:, None] * stride_oh + offs_d[None, :]
    do_ptrs = DO + query[:, None] * stride_dos + head[:, None] * stride_doh + offs_d[None, :]
    o = tl.load(o_ptrs, mask=row_mask[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=row_mask[:, None], other=0.0)
    tl.store(DELTA + row, tl.sum(o.to(tl.float32) * do.to(tl.float32), 1), mask=row_mask)


@triton.jit
def _dq_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    DQ,
    Q_OFFSETS,
    Q_KEY_START,
    Q_BITS,
    Q_ORDER,
    LAST_KEY,
    stride_qs,
    stride_qh,
    stride_ks,
    stride_kh,
    stride_vs,
    stride_vh,
    stride_dos,
    stride_doh,
    stride_dqs,
    stride_dqh,
    n_queries,
    n_keys,
    qk_scale,
    sm_scale,
    HQ: tl.constexpr,
    G: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NU: tl.constexpr,
    RATIO: tl.constexpr,
):
    tile = tl.load(Q_ORDER + tl.program_id(0))
    kv_head = tl.program_id(1)
    BLOCK_N: tl.constexpr = NU * RATIO
    offs_m = tl.arange(0, BLOCK_M)
    slot = offs_m // G
    head = kv_head * G + offs_m % G
    query = tile * T + slot
    row_mask = (slot < T) & (query < n_queries)
    offs_d = tl.arange(0, D)
    query64 = query[:, None].to(tl.int64)
    q = tl.load(
        Q + query64 * stride_qs + head[:, None] * stride_qh + offs_d[None, :], mask=row_mask[:, None], other=0.0
    )
    do_ptrs = DO + query64 * stride_dos + head[:, None] * stride_doh + offs_d[None, :]
    do = tl.load(do_ptrs, mask=row_mask[:, None], other=0.0)
    lse = tl.load(LSE + query.to(tl.int64) * HQ + head, mask=row_mask, other=float("inf"))
    delta = tl.load(DELTA + query.to(tl.int64) * HQ + head, mask=row_mask, other=0.0)
    last_key = tl.load(LAST_KEY + query, mask=row_mask, other=-1)
    lo = tl.load(Q_OFFSETS + tile)
    hi = tl.load(Q_OFFSETS + tile + 1)
    dq = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    block_of_n = offs_n // RATIO
    key_in_block = offs_n % RATIO
    for start in range(lo, hi, NU):
        entry = start + block_of_n
        entry_mask = entry < hi
        bits = tl.load(Q_BITS + entry, mask=entry_mask, other=0)
        key = tl.load(Q_KEY_START + entry, mask=entry_mask, other=0) + key_in_block
        key_mask = entry_mask & (key < n_keys)
        key64 = key[:, None].to(tl.int64)
        k = tl.load(K + kv_head * stride_kh + key64 * stride_ks + offs_d[None, :], mask=key_mask[:, None], other=0.0)
        v = tl.load(V + kv_head * stride_vh + key64 * stride_vs + offs_d[None, :], mask=key_mask[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        allowed = ((bits[None, :] >> slot[:, None]) & 1) != 0
        allowed = allowed & (key[None, :] <= last_key[:, None]) & key_mask[None, :]
        p = tl.where(allowed, tl.math.exp2(qk - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(k.dtype), k)
    dq_ptrs = DQ + query64 * stride_dqs + head[:, None] * stride_dqh + offs_d[None, :]
    tl.store(dq_ptrs, (dq * sm_scale).to(DQ.dtype.element_ty), mask=row_mask[:, None])


@triton.jit
def _dkdv_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    DK,
    DV,
    K_OFFSETS,
    K_QUERY,
    K_BITS,
    K_ORDER,
    BLOCK_START,
    BLOCK_LEN,
    LAST_KEY,
    stride_qs,
    stride_qh,
    stride_ks,
    stride_kh,
    stride_vs,
    stride_vh,
    stride_dos,
    stride_doh,
    stride_dks,
    stride_dkh,
    stride_dvs,
    stride_dvh,
    n_blocks,
    qk_scale,
    sm_scale,
    HQ: tl.constexpr,
    G: tl.constexpr,
    TB: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NUK: tl.constexpr,
    RATIO: tl.constexpr,
):
    key_tile = tl.load(K_ORDER + tl.program_id(0))
    kv_head = tl.program_id(1)
    BLOCK_N: tl.constexpr = NUK * RATIO
    offs_n = tl.arange(0, BLOCK_N)
    block_of_n = offs_n // RATIO
    key_in_block = offs_n % RATIO
    block = key_tile * NUK + block_of_n
    block_mask = block < n_blocks
    key = tl.load(BLOCK_START + block, mask=block_mask, other=0) + key_in_block
    key_mask = block_mask & (key_in_block < tl.load(BLOCK_LEN + block, mask=block_mask, other=0))
    offs_d = tl.arange(0, D)
    key64 = key[:, None].to(tl.int64)
    k = tl.load(K + kv_head * stride_kh + key64 * stride_ks + offs_d[None, :], mask=key_mask[:, None], other=0.0)
    v = tl.load(V + kv_head * stride_vh + key64 * stride_vs + offs_d[None, :], mask=key_mask[:, None], other=0.0)
    dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    lo = tl.load(K_OFFSETS + key_tile)
    hi = tl.load(K_OFFSETS + key_tile + 1)
    offs_m = tl.arange(0, BLOCK_M)
    slot = offs_m // G
    head = kv_head * G + offs_m % G
    for start in range(lo, hi, TB):
        entry = start + slot
        entry_mask = (slot < TB) & (entry < hi)
        query = tl.load(K_QUERY + entry, mask=entry_mask, other=0)
        bits = tl.load(K_BITS + entry, mask=entry_mask, other=0)
        query64 = query[:, None].to(tl.int64)
        q_ptrs = Q + query64 * stride_qs + head[:, None] * stride_qh + offs_d[None, :]
        do_ptrs = DO + query64 * stride_dos + head[:, None] * stride_doh + offs_d[None, :]
        q = tl.load(q_ptrs, mask=entry_mask[:, None], other=0.0)
        do = tl.load(do_ptrs, mask=entry_mask[:, None], other=0.0)
        lse = tl.load(LSE + query.to(tl.int64) * HQ + head, mask=entry_mask, other=float("inf"))
        delta = tl.load(DELTA + query.to(tl.int64) * HQ + head, mask=entry_mask, other=0.0)
        last_key = tl.load(LAST_KEY + query, mask=entry_mask, other=-1)
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        allowed = ((bits[:, None] >> block_of_n[None, :]) & 1) != 0
        allowed = allowed & (key[None, :] <= last_key[:, None]) & key_mask[None, :] & entry_mask[:, None]
        p = tl.where(allowed, tl.math.exp2(qk - lse[:, None]), 0.0)
        dv += tl.dot(tl.trans(p.to(do.dtype)), do)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dk += tl.dot(tl.trans(ds.to(q.dtype)), q)
    dk_ptrs = DK + kv_head * stride_dkh + key64 * stride_dks + offs_d[None, :]
    dv_ptrs = DV + kv_head * stride_dvh + key64 * stride_dvs + offs_d[None, :]
    tl.store(dk_ptrs, (dk * sm_scale).to(DK.dtype.element_ty), mask=key_mask[:, None])
    tl.store(dv_ptrs, dv.to(DV.dtype.element_ty), mask=key_mask[:, None])


def _query_rows(index: QSABlockIndex, group: int) -> int:
    return max(16, triton.next_power_of_2(index.query_tile * group))


def _forward(q: Tensor, k: Tensor, v: Tensor, index: QSABlockIndex, scale: float) -> tuple[Tensor, Tensor]:
    n_queries, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group = num_heads // num_kv_heads
    out = torch.empty_like(q)
    lse = torch.empty(n_queries, num_heads, device=q.device, dtype=torch.float32)
    _fwd_kernel[(index.q_order.numel(), num_kv_heads)](
        q,
        k,
        v,
        out,
        lse,
        index.q_offsets,
        index.q_key_start,
        index.q_bits,
        index.q_order,
        index.last_key,
        *q.stride()[:2],
        *k.stride()[:2],
        *v.stride()[:2],
        *out.stride()[:2],
        n_queries,
        k.shape[0],
        scale * _LOG2E,
        HQ=num_heads,
        G=group,
        T=index.query_tile,
        D=head_dim,
        BLOCK_M=_query_rows(index, group),
        RATIO=index.ratio,
        **_launch(q.device).fwd,
    )
    return out, lse


def _backward_delta(out: Tensor, grad_out: Tensor) -> Tensor:
    n_queries, num_heads, head_dim = out.shape
    delta = torch.empty(n_queries, num_heads, device=out.device, dtype=torch.float32)
    rows_per_program = 32
    _delta_kernel[(triton.cdiv(n_queries * num_heads, rows_per_program),)](
        out,
        grad_out,
        delta,
        *out.stride()[:2],
        *grad_out.stride()[:2],
        n_queries * num_heads,
        HQ=num_heads,
        D=head_dim,
        BLOCK=rows_per_program,
    )
    return delta


def _backward_dq(q, k, v, grad_out, lse, delta, index: QSABlockIndex, scale: float) -> Tensor:
    n_queries, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group = num_heads // num_kv_heads
    grad_q = torch.empty_like(q)
    _dq_kernel[(index.q_order.numel(), num_kv_heads)](
        q,
        k,
        v,
        grad_out,
        lse,
        delta,
        grad_q,
        index.q_offsets,
        index.q_key_start,
        index.q_bits,
        index.q_order,
        index.last_key,
        *q.stride()[:2],
        *k.stride()[:2],
        *v.stride()[:2],
        *grad_out.stride()[:2],
        *grad_q.stride()[:2],
        n_queries,
        k.shape[0],
        scale * _LOG2E,
        scale,
        HQ=num_heads,
        G=group,
        T=index.query_tile,
        D=head_dim,
        BLOCK_M=_query_rows(index, group),
        RATIO=index.ratio,
        **_launch(q.device).dq,
    )
    return grad_q


def _backward_dkdv(q, k, v, grad_out, lse, delta, index: QSABlockIndex, scale: float) -> tuple[Tensor, Tensor]:
    num_heads, head_dim = q.shape[1:]
    num_kv_heads = k.shape[1]
    group = num_heads // num_kv_heads
    # keys outside every block (rows past the last sequence) get no gradient
    grad_k = torch.zeros_like(k)
    grad_v = torch.zeros_like(v)
    _dkdv_kernel[(index.k_order.numel(), num_kv_heads)](
        q,
        k,
        v,
        grad_out,
        lse,
        delta,
        grad_k,
        grad_v,
        index.k_offsets,
        index.k_query,
        index.k_bits,
        index.k_order,
        index.block_start,
        index.block_len,
        index.last_key,
        *q.stride()[:2],
        *k.stride()[:2],
        *v.stride()[:2],
        *grad_out.stride()[:2],
        *grad_k.stride()[:2],
        *grad_v.stride()[:2],
        index.block_start.numel(),
        scale * _LOG2E,
        scale,
        HQ=num_heads,
        G=group,
        TB=_TILE_ROWS // group,
        D=head_dim,
        BLOCK_M=_TILE_ROWS,
        NUK=index.key_tile,
        RATIO=index.ratio,
        **_launch(q.device).dkdv,
    )
    return grad_k, grad_v


def _last_dim_contiguous(x: Tensor) -> Tensor:
    return x if x.stride(-1) == 1 else x.contiguous()


class _QSABlockAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, index, scale):
        q, k, v = (_last_dim_contiguous(t) for t in (q, k, v))
        out, lse = _forward(q, k, v, index, scale)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.index = index
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, out, lse = ctx.saved_tensors
        grad_out = _last_dim_contiguous(grad_out)
        delta = _backward_delta(out, grad_out)
        grad_q = _backward_dq(q, k, v, grad_out, lse, delta, ctx.index, ctx.scale)
        grad_k, grad_v = _backward_dkdv(q, k, v, grad_out, lse, delta, ctx.index, ctx.scale)
        return grad_q, grad_k, grad_v, None, None


def qsa_block_attention(q: Tensor, k: Tensor, v: Tensor, index: QSABlockIndex, scale: float) -> Tensor:
    """``q`` [T, Hq, D], ``k`` / ``v`` [T, Hkv, D] -> [T, Hq, D] over the blocks of ``index``.

    ``D`` must be a power of two of at least 16. A query that sees no key gets zeros.
    """
    head_dim = q.shape[-1]
    if head_dim < 16 or head_dim & (head_dim - 1):
        raise ValueError(f"QSA block attention needs a power-of-two head_dim >= 16, got {head_dim}")
    return _QSABlockAttention.apply(q, k, v, index, scale)
