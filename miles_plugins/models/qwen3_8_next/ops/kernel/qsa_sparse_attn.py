"""Triton QSA sparse attention for training: forward + backward.

Semantics match sglang's ``_sparse_gqa_prefill`` (qsa/sparse_attn.py): each query
attends exactly the token indices in its selection row (``-1`` = unused). The
selection rows are produced torch-side (indexer top-k expansion + the query's own
partial-block tail), so the kernel itself needs no causal or segment logic -- the
"""

import torch
import triton
import triton.language as tl

from torch import Tensor


@triton.jit(do_not_specialize=["T", "TOPK"])
def _qsa_fwd_kernel(
    Q,
    K,
    V,
    IDX,
    OUT,
    LSE,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    stride_it,
    stride_ot,
    stride_oh,
    T,
    TOPK,
    scale,
    GROUP: tl.constexpr,  # query heads per kv head
    D: tl.constexpr,
    BQ: tl.constexpr,  # queries per program
    BK: tl.constexpr,  # selection indices per tile
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)  # query head index
    kv_head = pid_h // GROUP

    offs_q = pid_t * BQ + tl.arange(0, BQ)
    offs_d = tl.arange(0, D)
    q_mask = offs_q < T

    q = tl.load(
        Q + offs_q[:, None] * stride_qt + pid_h * stride_qh + offs_d[None, :],
        mask=q_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    m_i = tl.full((BQ,), float("-inf"), tl.float32)
    l_i = tl.zeros((BQ,), tl.float32)
    acc = tl.zeros((BQ, D), tl.float32)

    for start in range(0, TOPK, BK):
        offs_k = start + tl.arange(0, BK)
        k_mask = offs_k < TOPK
        idx = tl.load(
            IDX + offs_q[:, None] * stride_it + offs_k[None, :],
            mask=q_mask[:, None] & k_mask[None, :],
            other=-1,
        )
        valid = idx >= 0
        idx_safe = tl.where(valid, idx, 0)

        k_tile = tl.load(
            K + idx_safe[:, :, None] * stride_kt + kv_head * stride_kh + offs_d[None, None, :],
            mask=valid[:, :, None],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(q[:, None, :] * k_tile, axis=2) * scale
        scores = tl.where(valid, scores, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(scores - m_use[:, None])
        p = tl.where(valid, p, 0.0)
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i - m_use))
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_tile = tl.load(
            V + idx_safe[:, :, None] * stride_vt + kv_head * stride_vh + offs_d[None, None, :],
            mask=valid[:, :, None],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(p[:, :, None] * v_tile, axis=1)
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


@triton.jit(do_not_specialize=["T", "TOPK"])
def _qsa_bwd_kernel(
    Q,
    K,
    V,
    IDX,
    OUT,
    LSE,
    DO,
    DQ,
    DK,
    DV,
    stride_qt,
    stride_qh,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    stride_it,
    stride_ot,
    stride_oh,
    T,
    TOPK,
    scale,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    BQ: tl.constexpr,
    BK: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    kv_head = pid_h // GROUP

    offs_q = pid_t * BQ + tl.arange(0, BQ)
    offs_d = tl.arange(0, D)
    q_mask = offs_q < T

    q = tl.load(
        Q + offs_q[:, None] * stride_qt + pid_h * stride_qh + offs_d[None, :], mask=q_mask[:, None], other=0.0
    ).to(tl.float32)
    do = tl.load(
        DO + offs_q[:, None] * stride_ot + pid_h * stride_oh + offs_d[None, :], mask=q_mask[:, None], other=0.0
    ).to(tl.float32)
    o = tl.load(
        OUT + offs_q[:, None] * stride_ot + pid_h * stride_oh + offs_d[None, :], mask=q_mask[:, None], other=0.0
    ).to(tl.float32)
    lse = tl.load(LSE + pid_h * T + offs_q, mask=q_mask, other=0.0)
    delta = tl.sum(do * o, axis=1)

    dq = tl.zeros((BQ, D), tl.float32)
    for start in range(0, TOPK, BK):
        offs_k = start + tl.arange(0, BK)
        k_mask = offs_k < TOPK
        idx = tl.load(
            IDX + offs_q[:, None] * stride_it + offs_k[None, :], mask=q_mask[:, None] & k_mask[None, :], other=-1
        )
        valid = idx >= 0
        idx_safe = tl.where(valid, idx, 0)

        k_tile = tl.load(
            K + idx_safe[:, :, None] * stride_kt + kv_head * stride_kh + offs_d[None, None, :],
            mask=valid[:, :, None],
            other=0.0,
        ).to(tl.float32)
        v_tile = tl.load(
            V + idx_safe[:, :, None] * stride_vt + kv_head * stride_vh + offs_d[None, None, :],
            mask=valid[:, :, None],
            other=0.0,
        ).to(tl.float32)

        scores = tl.sum(q[:, None, :] * k_tile, axis=2) * scale
        lse_safe = tl.where(lse == float("-inf"), 0.0, lse)
        p = tl.exp(scores - lse_safe[:, None])
        p = tl.where(valid & (lse[:, None] != float("-inf")), p, 0.0)

        dp = tl.sum(do[:, None, :] * v_tile, axis=2)
        ds = p * (dp - delta[:, None]) * scale

        dq += tl.sum(ds[:, :, None] * k_tile, axis=1)

        dk_c = ds[:, :, None] * q[:, None, :]
        dv_c = p[:, :, None] * do[:, None, :]
        ptrs_k = DK + idx_safe[:, :, None] * stride_kt + kv_head * stride_kh + offs_d[None, None, :]
        ptrs_v = DV + idx_safe[:, :, None] * stride_vt + kv_head * stride_vh + offs_d[None, None, :]
        tl.atomic_add(ptrs_k, dk_c, mask=valid[:, :, None])
        tl.atomic_add(ptrs_v, dv_c, mask=valid[:, :, None])

    tl.store(DQ + offs_q[:, None] * stride_qt + pid_h * stride_qh + offs_d[None, :], dq, mask=q_mask[:, None])


class _QSASparseAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, indices, scale):
        T, Hq, D = q.shape
        S, Hkv, _ = k.shape
        assert Hq % Hkv == 0
        group = Hq // Hkv
        topk = indices.shape[1]
        qc, kc, vc = q.contiguous(), k.contiguous(), v.contiguous()
        idx = indices.contiguous()
        o = torch.empty(T, Hq, D, device=q.device, dtype=torch.float32)
        lse = torch.empty(Hq, T, device=q.device, dtype=torch.float32)
        BQ, BK = 32, 64
        grid = (triton.cdiv(T, BQ), Hq)
        _qsa_fwd_kernel[grid](
            qc,
            kc,
            vc,
            idx,
            o,
            lse,
            qc.stride(0),
            qc.stride(1),
            kc.stride(0),
            kc.stride(1),
            vc.stride(0),
            vc.stride(1),
            idx.stride(0),
            o.stride(0),
            o.stride(1),
            T,
            topk,
            scale,
            GROUP=group,
            D=D,
            BQ=BQ,
            BK=BK,
        )
        ctx.save_for_backward(qc, kc, vc, idx, o, lse)
        ctx.scale = scale
        ctx.group = group
        return o.to(q.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        qc, kc, vc, idx, o, lse = ctx.saved_tensors
        T, Hq, D = qc.shape
        topk = idx.shape[1]
        do = grad_out.contiguous()
        dq = torch.empty(T, Hq, D, device=qc.device, dtype=torch.float32)
        dk = torch.zeros(kc.shape, device=kc.device, dtype=torch.float32)
        dv = torch.zeros(vc.shape, device=vc.device, dtype=torch.float32)
        BQ, BK = 32, 64
        grid = (triton.cdiv(T, BQ), Hq)
        _qsa_bwd_kernel[grid](
            qc,
            kc,
            vc,
            idx,
            o,
            lse,
            do,
            dq,
            dk,
            dv,
            qc.stride(0),
            qc.stride(1),
            kc.stride(0),
            kc.stride(1),
            vc.stride(0),
            vc.stride(1),
            idx.stride(0),
            do.stride(0),
            do.stride(1),
            T,
            topk,
            ctx.scale,
            GROUP=ctx.group,
            D=D,
            BQ=BQ,
            BK=BK,
        )
        return dq.to(qc.dtype), dk.to(kc.dtype), dv.to(vc.dtype), None, None


def qsa_sparse_attention_triton(q: Tensor, k: Tensor, v: Tensor, indices: Tensor, scale: float) -> Tensor:
    """``q`` [T, Hq, D], ``k``/``v`` [S, Hkv, D], ``indices`` [T, K] int (-1 pad)."""
    return _QSASparseAttn.apply(q, k, v, indices, scale)
