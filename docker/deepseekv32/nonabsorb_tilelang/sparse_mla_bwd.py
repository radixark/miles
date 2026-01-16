# ruff: noqa
import tilelang
from tilelang import language as T
import torch

@tilelang.jit(out_idx=[-1])
def preprocess(
    b,
    s,
    heads,
    dim_v,
    block_ND=32,
    num_stages=5,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32
    shape = [b, s, heads, dim_v]

    @T.prim_func
    def preprocess_kernel(
        O: T.Tensor(shape, dtype),
        dO: T.Tensor(shape, dtype),
        Delta: T.Tensor([b, s, heads], accum_dtype),
    ):
        with T.Kernel(heads, T.ceildiv(s, block_ND), b) as (bx, by, bz):
            o = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            do = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            delta = T.alloc_fragment([block_ND], accum_dtype)
            acc = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            T.clear(acc)
            for k in T.Pipelined(T.ceildiv(dim_v, block_ND), num_stages=num_stages):
                T.copy(O[bz, by * block_ND : (by + 1) * block_ND, bx, k * block_ND : (k + 1) * block_ND], o)
                T.copy(dO[bz, by * block_ND : (by + 1) * block_ND, bx, k * block_ND : (k + 1) * block_ND], do)
                for i, j in T.Parallel(block_ND, block_ND):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, Delta[bz, by * block_ND : (by + 1) * block_ND, bx])

    return preprocess_kernel


@tilelang.jit(out_idx=[-1])
def postprocess(
    b,
    skv,
    dim,
    kv_groups,
    block_N=64,
    threads=256,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32
    d_shape = [b, skv, kv_groups, dim]

    @T.prim_func
    def postprocess_kernel(
        din: T.Tensor(d_shape, accum_dtype),
        dout: T.Tensor(d_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(skv, block_N), kv_groups, b, threads=threads) as (bx, by, bz):
            T.copy(
                din[bz, bx * block_N : (bx + 1) * block_N, by, :],
                dout[bz, bx * block_N : (bx + 1) * block_N, by, :],
            )

    return postprocess_kernel


@tilelang.jit(
    out_idx=[-3],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
    },
)
def bwd(
    b,
    s,
    skv,
    heads,
    kv_groups,
    dim,
    dim_v,
    topk,
    sm_scale,
    block_size=32,
    num_stages=0,
    threads=128,
    indices_dtype=T.int32,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
    masks_dtype=T.bool,
):
    assert topk % block_size == 0, "otherwise will load some index=0 thus causing wrong kv to be loaded"
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32
    assert indices_dtype == T.int32

    sm_scale_mul_reciprocal_log2 = sm_scale * 1.44269504  # log2(e)

    H_kv = heads // kv_groups
    q_shape = [b, s, heads, dim]
    k_shape = [b, skv, kv_groups, dim]
    v_shape = [b, skv, kv_groups, dim_v]
    o_shape = [b, s, heads, dim_v]
    indices_shape = [b, s, kv_groups, topk]
    masks_shape = [b, s, kv_groups, skv]
    delta_shape = [b, s, heads]
    lse_shape = [b, s, heads]

    assert indices_dtype == T.int32
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32

    padded_H = max(tilelang.math.next_power_of_2(H_kv), 16)
    block_H = min(64, padded_H)
    assert padded_H % block_H == 0
    NH = padded_H // block_H
    BS = block_size
    NS = tilelang.cdiv(topk, block_size)

    split_store = 2

    @T.prim_func
    def sparse_mla_bwd_kernel(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        V: T.Tensor(v_shape, dtype),
        dO: T.Tensor(o_shape, dtype),
        Indices: T.Tensor(indices_shape, indices_dtype),
        Masks: T.Tensor(masks_shape, masks_dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
        Delta: T.Tensor(delta_shape, accum_dtype),
        dQ: T.Tensor(q_shape, dtype),
        dK: T.Tensor(k_shape, accum_dtype),
        dV: T.Tensor(v_shape, accum_dtype),
    ):
        with T.Kernel(s, b, kv_groups * NH, threads=threads) as (s_i, by, bz):
            Q_shared = T.alloc_shared([block_H, dim], dtype)
            K_shared = T.alloc_shared([BS, dim], dtype)
            V_shared = T.alloc_shared([BS, dim_v], dtype)
            dO_shared = T.alloc_shared([block_H, dim_v], dtype)
            mask = T.alloc_fragment([BS], "bool")

            P_shared_cast = T.alloc_shared([block_H, BS], dtype)
            dP_shared_cast = T.alloc_shared([block_H, BS], dtype)
            dQ_shared = T.alloc_shared([block_H, dim], dtype)

            acc_p = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dp = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dq = T.alloc_fragment([block_H, dim], accum_dtype)
            acc_dk = T.alloc_fragment([BS, dim], accum_dtype)
            acc_dv = T.alloc_fragment([BS, dim_v], accum_dtype)
            acc_dk_shared = T.alloc_shared([BS // split_store, dim], accum_dtype)
            acc_dv_shared = T.alloc_shared([BS // split_store, dim_v], accum_dtype)

            T.copy(Q[by, s_i, bz * block_H : (bz + 1) * block_H, :], Q_shared)
            T.copy(dO[by, s_i, bz * block_H : (bz + 1) * block_H, :], dO_shared)

            T.clear(acc_dq)

            # Process each block of indices
            for i_i in T.Pipelined(NS, num_stages=num_stages):
                # Compute attention scores
                for bi_i in T.Parallel(BS):
                    mask[bi_i] = Masks[by, s_i, bz // NH, Indices[by, s_i, bz // NH, i_i * BS + bi_i]]

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = T.if_then_else(mask[bi_i], -T.infinity(acc_p.dtype), 0)

                # Load KV, V for this block of indices
                for bi_i, d_i in T.Parallel(BS, dim):
                    K_shared[bi_i, d_i] = K[by, Indices[by, s_i, bz // NH, i_i * BS + bi_i], bz // NH, d_i]

                for bi_i, d_i in T.Parallel(BS, dim_v):
                    V_shared[bi_i, d_i] = V[by, Indices[by, s_i, bz // NH, i_i * BS + bi_i], bz // NH, d_i]

                T.gemm(Q_shared, K_shared, acc_p, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = T.exp2(acc_p[h_i, bi_i] * sm_scale_mul_reciprocal_log2 - Lse[by, s_i, bz * block_H + h_i])

                T.copy(acc_p, P_shared_cast)

                T.gemm(dO_shared, V_shared, acc_dp, transpose_B=True, policy=T.GemmWarpPolicy.FullCol, clear_accum=True)

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_dp[h_i, bi_i] = acc_p[h_i, bi_i] * (acc_dp[h_i, bi_i] - Delta[by, s_i, bz * block_H + h_i]) * sm_scale

                T.copy(acc_dp, dP_shared_cast)
                T.gemm(dP_shared_cast, K_shared, acc_dq, policy=T.GemmWarpPolicy.FullCol)

                T.gemm(dP_shared_cast, Q_shared, acc_dk, transpose_A=True, policy=T.GemmWarpPolicy.FullCol, clear_accum=True)
                T.gemm(P_shared_cast, dO_shared, acc_dv, transpose_A=True, policy=T.GemmWarpPolicy.FullCol, clear_accum=True)

                for split_i in range(split_store):
                    for bi_i, d_i in T.Parallel(BS, dim):
                        if bi_i < BS // split_store:
                            acc_dk_shared[bi_i, d_i] = acc_dk[bi_i + split_i * (BS // split_store), d_i]

                    for bi_i, d_i in T.Parallel(BS // split_store, dim // 4):
                        T.atomic_addx4(
                            dK[by, Indices[by, s_i, bz // NH, i_i * BS + bi_i + split_i * (BS // split_store)], bz // NH, d_i * 4],
                            acc_dk_shared[bi_i, d_i * 4],
                        )

                    for bi_i, d_i in T.Parallel(BS, dim_v):
                        if bi_i < BS // split_store:
                            acc_dv_shared[bi_i, d_i] = acc_dv[bi_i + split_i * (BS // split_store), d_i]

                    for bi_i, d_i in T.Parallel(BS // split_store, dim_v // 4):
                        T.atomic_addx4(
                            dV[by, Indices[by, s_i, bz // NH, i_i * BS + bi_i + split_i * (BS // split_store)], bz // NH, d_i * 4],
                            acc_dv_shared[bi_i, d_i * 4],
                        )

            # Store the accumulated dQ
            T.copy(acc_dq, dQ_shared)
            T.copy(dQ_shared, dQ[by, s_i, bz * block_H : (bz + 1) * block_H, :])

    return sparse_mla_bwd_kernel


def sparse_mla_bwd(q, k, v, o, do, indices, masks, lse, sm_scale, delta=None):
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and indices.is_contiguous() and masks.is_contiguous() and lse.is_contiguous()
    
    b, s, heads, dim = q.shape
    _, skv, kv_groups, dim_v = v.shape
    topk = indices.shape[-1]

    # Get kernels
    preprocess_kernel = preprocess(b, s, heads, dim_v)
    bwd_kernel = bwd(b, s, skv, heads, kv_groups, dim, dim_v, topk, sm_scale)
    postprocess_kernel_k = postprocess(b, skv, dim, kv_groups)
    postprocess_kernel_v = postprocess(b, skv, dim_v, kv_groups)

    if delta is None:
        delta = preprocess_kernel(o, do)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)
    dq = bwd_kernel(q, k, v, do, indices, masks, lse, delta, dk, dv)
    dk = postprocess_kernel_k(dk)
    dv = postprocess_kernel_v(dv)

    return dq, dk, dv
