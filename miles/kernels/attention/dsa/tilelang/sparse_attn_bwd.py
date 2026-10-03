# ruff: noqa
# Adapt from https://github.com/tile-ai/tilelang/blob/4ff81c7d40803d269569e157e847623e84553f78/examples/deepseek_v32/sparse_mla_bwd.py
import os

import tilelang
import torch
from tilelang import language as T


@tilelang.jit(out_idx=[-1])
def preprocess(
    H,
    D,
    block_ND=32,
    num_stages=5,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32
    B = T.dynamic("batch")
    S = T.dynamic("seq_len")
    shape = [B, S, H, D]

    @T.prim_func
    def preprocess_kernel(
        O: T.Tensor(shape, dtype),
        dO: T.Tensor(shape, dtype),
        Delta: T.Tensor([B, S, H], accum_dtype),
    ):
        with T.Kernel(H, T.ceildiv(S, block_ND), B) as (bx, by, bz):
            o = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            do = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            delta = T.alloc_fragment([block_ND], accum_dtype)
            acc = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            T.clear(acc)
            for k in T.Pipelined(T.ceildiv(D, block_ND), num_stages=num_stages):
                T.copy(O[bz, by * block_ND : (by + 1) * block_ND, bx, k * block_ND : (k + 1) * block_ND], o)
                T.copy(dO[bz, by * block_ND : (by + 1) * block_ND, bx, k * block_ND : (k + 1) * block_ND], do)
                for i, j in T.Parallel(block_ND, block_ND):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, Delta[bz, by * block_ND : (by + 1) * block_ND, bx])

    return preprocess_kernel


@tilelang.jit(out_idx=[-1])
def postprocess(
    D,
    D_tail,
    kv_group=1,
    block_N=64,
    threads=128,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32
    B = T.dynamic("batch")
    S_kv = T.dynamic("seq_len_kv")
    dkv_shape = [B, S_kv, kv_group, D + D_tail]

    @T.prim_func
    def postprocess_kernel(
        dKV: T.Tensor(dkv_shape, accum_dtype),
        dKV_out: T.Tensor(dkv_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(S_kv, block_N), kv_group, B, threads=threads) as (bx, by, bz):
            T.copy(
                dKV[bz, bx * block_N : (bx + 1) * block_N, by, :],
                dKV_out[bz, bx * block_N : (bx + 1) * block_N, by, :],
            )

    return postprocess_kernel


@tilelang.jit(
    out_idx=[-2],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        # Aggressive smem merge aliases buffers that are still live, giving NaN
        # dQ/dKV. Hopper's scheduling happens to hide it; that is luck, not
        # safety, so keep it off everywhere. Costs ~5% on this kernel.
        tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: False,
    },
)
def bwd(
    H,
    D,
    D_tail,
    topk,
    kv_group=1,
    sm_scale=None,
    block_size=32,
    num_stages=0,
    indices_dtype=T.int32,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    assert topk % block_size == 0, "otherwise will load some index=0 thus causing wrong kv to be loaded"
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32
    assert indices_dtype == T.int32

    if sm_scale is None:
        sm_scale = (D + D_tail) ** (-0.5)
    sm_scale_mul_reciprocal_log2 = sm_scale * 1.44269504  # log2(e)

    H_kv = H // kv_group
    B = T.dynamic("batch")
    S = T.dynamic("seq_len")
    S_kv = T.dynamic("seq_len_kv")
    q_shape = [B, S, H, D + D_tail]
    k_shape = [B, S_kv, kv_group, D + D_tail]
    o_shape = [B, S, H, D]
    indices_shape = [B, S, kv_group, topk]
    delta_shape = [B, S, H]
    lse_shape = [B, S, H]
    assert indices_dtype == T.int32
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32

    H = H_kv
    padded_H = max(tilelang.math.next_power_of_2(H_kv), 16)
    max_block_H = 32 if os.getenv("MILES_HARDWARE_PLATFORM") == "rocm" and padded_H >= 64 else 64
    block_H = min(max_block_H, padded_H)
    assert padded_H % block_H == 0
    NH = padded_H // block_H
    threads = 256 if block_H >= 64 else 128
    BS = block_size
    NS = tilelang.cdiv(topk, block_size)

    split_store = 2

    @T.prim_func
    def sparse_mla_bwd_kernel(
        Q: T.Tensor(q_shape, dtype),
        KV: T.Tensor(k_shape, dtype),
        dO: T.Tensor(o_shape, dtype),
        Indices: T.Tensor(indices_shape, indices_dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
        Delta: T.Tensor(delta_shape, accum_dtype),
        dQ: T.Tensor(q_shape, dtype),
        dKV: T.Tensor(k_shape, accum_dtype),
    ):
        with T.Kernel(S, B, kv_group * NH, threads=threads) as (s_i, by, bz):
            Q_shared = T.alloc_shared([block_H, D], dtype)
            KV_shared = T.alloc_shared([BS, D], dtype)
            dO_shared = T.alloc_shared([block_H, D], dtype)
            mask = T.alloc_shared([BS], "bool")
            kv_i = T.alloc_shared([BS], indices_dtype)

            P_shared_cast = T.alloc_shared([block_H, BS], dtype)
            dP_shared_cast = T.alloc_shared([block_H, BS], dtype)
            dQ_shared = T.alloc_shared([block_H, D], dtype)

            acc_p = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dp = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dq = T.alloc_fragment([block_H, D], accum_dtype)
            acc_dkv = T.alloc_fragment([BS, D], accum_dtype)
            acc_dkv_shared = T.alloc_shared([BS // split_store, D], accum_dtype)
            if D_tail > 0:
                Q_tail_shared = T.alloc_shared([block_H, D_tail], dtype)
                KV_tail_shared = T.alloc_shared([BS, D_tail], dtype)
                dQ_tail_shared = T.alloc_shared([block_H, D_tail], dtype)
                acc_dq_tail = T.alloc_fragment([block_H, D_tail], accum_dtype)
                acc_dkv_tail = T.alloc_fragment([BS, D_tail], accum_dtype)
                acc_dkv_tail_shared = T.alloc_shared([BS // split_store, D_tail], accum_dtype)

            T.copy(Q[by, s_i, bz * block_H : (bz + 1) * block_H, :D], Q_shared)
            T.copy(dO[by, s_i, bz * block_H : (bz + 1) * block_H, :D], dO_shared)
            T.clear(acc_dq)
            if D_tail > 0:
                T.copy(Q[by, s_i, bz * block_H : (bz + 1) * block_H, D:], Q_tail_shared)
                T.clear(acc_dq_tail)

            # Process each block of indices
            for i_i in T.Pipelined(NS, num_stages=num_stages):
                # Check which indices are valid
                for bi_i in T.Parallel(BS):
                    # Changed here for thd
                    mask[bi_i] = Indices[by, s_i, bz // NH, i_i * BS + bi_i] != -1
                # A padded slot holds -1, which addresses the element *before* the tensor. The
                # forward absorbs whatever it reads into an -inf score, but here the same bytes
                # also reach acc_dp through dO @ KV, and acc_dp is then multiplied by an exactly
                # zero acc_p -- so one overflowing garbage dot product is 0 * inf = NaN. Clamp
                # the gather in range and substitute a true zero key instead.
                for bi_i in T.Parallel(BS):
                    kv_i[bi_i] = T.max(Indices[by, s_i, bz // NH, i_i * BS + bi_i], 0)

                # Compute attention scores
                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_p.dtype))

                # Load KV, V for this block of indices
                for bi_i, d_i in T.Parallel(BS, D):
                    KV_shared[bi_i, d_i] = T.if_then_else(mask[bi_i], KV[by, kv_i[bi_i], bz // NH, d_i], 0)

                T.gemm(Q_shared, KV_shared, acc_p, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

                if D_tail > 0:
                    for bi_i, d_i in T.Parallel(BS, D_tail):
                        KV_tail_shared[bi_i, d_i] = T.if_then_else(
                            mask[bi_i], KV[by, kv_i[bi_i], bz // NH, D + d_i], 0
                        )
                    T.gemm(Q_tail_shared, KV_tail_shared, acc_p, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = T.exp2(
                        acc_p[h_i, bi_i] * sm_scale_mul_reciprocal_log2 - Lse[by, s_i, bz * block_H + h_i]
                    )

                T.copy(acc_p, P_shared_cast)

                T.gemm(
                    dO_shared, KV_shared, acc_dp, transpose_B=True, policy=T.GemmWarpPolicy.FullCol, clear_accum=True
                )

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_dp[h_i, bi_i] = (
                        acc_p[h_i, bi_i] * (acc_dp[h_i, bi_i] - Delta[by, s_i, bz * block_H + h_i]) * sm_scale
                    )

                T.copy(acc_dp, dP_shared_cast)
                T.gemm(dP_shared_cast, KV_shared, acc_dq, policy=T.GemmWarpPolicy.FullCol)
                if D_tail > 0:
                    T.gemm(dP_shared_cast, KV_tail_shared, acc_dq_tail, policy=T.GemmWarpPolicy.FullCol)

                T.gemm(
                    dP_shared_cast,
                    Q_shared,
                    acc_dkv,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=True,
                )
                T.gemm(P_shared_cast, dO_shared, acc_dkv, transpose_A=True, policy=T.GemmWarpPolicy.FullCol)

                if D_tail > 0:
                    T.clear(acc_dkv_tail)
                    T.gemm(
                        dP_shared_cast, Q_tail_shared, acc_dkv_tail, transpose_A=True, policy=T.GemmWarpPolicy.FullCol
                    )

                for s in range(split_store):
                    for bi_i, d_i in T.Parallel(BS, D):
                        if bi_i < BS // split_store:
                            acc_dkv_shared[bi_i, d_i] = acc_dkv[bi_i + s * (BS // split_store), d_i]

                    if D_tail > 0:
                        for bi_i, d_i in T.Parallel(BS, D_tail):
                            if bi_i < BS // split_store:
                                acc_dkv_tail_shared[bi_i, d_i] = acc_dkv_tail[bi_i + s * (BS // split_store), d_i]

                    # Padded slots contribute an exact zero here (their P and dP columns are
                    # zero), so the clamped address makes the atomic a no-op rather than an
                    # out-of-bounds write into whatever allocation precedes dKV.
                    for bi_i, d_i in T.Parallel(BS // split_store, D // 4):
                        T.atomic_addx4(
                            dKV[by, kv_i[bi_i + s * (BS // split_store)], bz // NH, d_i * 4],
                            acc_dkv_shared[bi_i, d_i * 4],
                        )

                    if D_tail > 0:
                        for bi_i, d_i in T.Parallel(BS // split_store, D_tail // 4):
                            T.atomic_addx4(
                                dKV[by, kv_i[bi_i + s * (BS // split_store)], bz // NH, D + d_i * 4],
                                acc_dkv_tail_shared[bi_i, d_i * 4],
                            )

            T.copy(acc_dq, dQ_shared)
            T.copy(dQ_shared, dQ[by, s_i, bz * block_H : (bz + 1) * block_H, :D])
            if D_tail > 0:
                T.copy(acc_dq_tail, dQ_tail_shared)
                T.copy(dQ_tail_shared, dQ[by, s_i, bz * block_H : (bz + 1) * block_H, D:])

    return sparse_mla_bwd_kernel


def sparse_attn_bwd_interface(q, kv, o, do, indices, lse, d_v, sm_scale=None):
    """Shapes as in sparse_attn_fwd_interface, plus o/do [B, S, H, d_v] and lse [B, S, H].
    Returns dq [B, S, H, d_v + d_tail] bf16, dkv [B, S_kv, G, d_v + d_tail] bf16 and delta [B, S, H] fp32
    (rowsum(o * do), which the caller needs for the attention-sink gradient)."""
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous() and lse.is_contiguous()
    B, S, H, dim_plus_tail_dim = q.shape
    _, S_kv, kv_group, _ = kv.shape
    assert kv.shape[-1] == dim_plus_tail_dim
    assert kv.shape[0] == B
    D_tail = dim_plus_tail_dim - d_v
    topk = indices.shape[-1]
    assert indices.shape == (B, S, kv_group, topk)
    assert lse.shape == (B, S, H)

    delta = preprocess(H, d_v)(o, do)
    dkv = torch.zeros_like(kv, dtype=torch.float32)
    dq = bwd(H, d_v, D_tail, topk, kv_group, sm_scale)(q, kv, do, indices, lse, delta, dkv)
    dkv = postprocess(d_v, D_tail, kv_group)(dkv)
    return dq, dkv, delta
