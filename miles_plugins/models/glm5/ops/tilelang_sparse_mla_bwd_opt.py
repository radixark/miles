# ruff: noqa
# GLM-5 DSA sparse-MLA backward, optimized variant of tilelang_sparse_mla_bwd.py
# (pipelined split-D + manualy swizzle + GEMM-operand-reuse ; fp32-accumulated, cosine ~1.0 vs the reference).
# Adapt from https://github.com/tile-ai/tilelang/blob/4ff81c7d40803d269569e157e847623e84553f78/examples/deepseek_v32/sparse_mla_bwd.py
import tilelang
import torch
from tilelang import language as T

_BWD_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
}


@tilelang.jit(out_idx=[-1])
def preprocess(B, S, H, D, block_ND=32, num_stages=5, dtype=T.bfloat16, accum_dtype=T.float32):
    if dtype != T.bfloat16:
        raise ValueError("dtype must be T.bfloat16")
    if accum_dtype != T.float32:
        raise ValueError("accum_dtype must be T.float32")
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
def postprocess(B, S_kv, D, D_tail, kv_group=1, block_N=64, threads=128, dtype=T.bfloat16, accum_dtype=T.float32):
    if dtype != T.bfloat16:
        raise ValueError("dtype must be T.bfloat16")
    if accum_dtype != T.float32:
        raise ValueError("accum_dtype must be T.float32")
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


@tilelang.jit(out_idx=[-2], pass_configs=_BWD_PASS_CONFIGS)
def bwd(
    B,
    S,
    S_kv,
    H,
    D,
    D_tail,
    topk,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    block_size=32,
    num_stages=1,
    threads=None,
    indices_dtype=T.int32,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    from tilelang.layout import make_swizzled_layout

    if not is_causal:
        raise ValueError("non-causal is not supported now")
    if topk % block_size != 0:
        raise ValueError("otherwise will load some index=0 thus causing wrong kv to be loaded")
    if dtype != T.bfloat16:
        raise ValueError("dtype must be T.bfloat16")
    if accum_dtype != T.float32:
        raise ValueError("accum_dtype must be T.float32")
    if indices_dtype != T.int32:
        raise ValueError("indices_dtype must be T.int32")

    if sm_scale is None:
        sm_scale = (D + D_tail) ** (-0.5)
    sm_scale_mul_reciprocal_log2 = sm_scale * 1.44269504  # log2(e)

    H_kv = H // kv_group
    q_shape = [B, S, H, D + D_tail]
    k_shape = [B, S_kv, kv_group, D + D_tail]
    o_shape = [B, S, H, D]
    indices_shape = [B, S, kv_group, topk]
    delta_shape = [B, S, H]
    lse_shape = [B, S, H]

    H = H_kv
    padded_H = max(tilelang.math.next_power_of_2(H_kv), 16)
    block_H = min(64, padded_H)
    # adaptive: 256 threads (~2.3x on this bandwidth-bound kernel) need block_H>=64 for
    # GEMM warp tiling; smaller head blocks fall back to 128 so tiny shapes still build.
    if threads is None:
        threads = 256 if block_H >= 64 else 128
    assert padded_H % block_H == 0
    NH = padded_H // block_H
    BS = block_size
    NS = tilelang.cdiv(topk, block_size)
    split_store = 2
    half = BS // split_store
    D_CHUNK = 256  # split-D: D=512 as two D_CHUNK chunks, D_tail=64 as a third

    @T.macro
    def scatter_dkv(dKV, Indices, acc, staging, s_i, by, bz, i_i, col_base, width):
        for s in range(split_store):
            for bi_i, d_i in T.Parallel(BS, width):
                if bi_i < half:
                    staging[bi_i, d_i] = acc[bi_i + s * half, d_i]
            for bi_i, d_i in T.Parallel(half, width // 4):
                T.atomic_addx4(
                    dKV[by, Indices[by, s_i, bz // NH, i_i * BS + bi_i + s * half], bz // NH, col_base + d_i * 4],
                    staging[bi_i, d_i * 4],
                )

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
            # QdO_c*: stacked [Q ; dO] per D-chunk (rows [0, block_H) = Q, rest = dO)
            QdO_c0 = T.alloc_shared([2 * block_H, D_CHUNK], dtype)
            QdO_c1 = T.alloc_shared([2 * block_H, D_CHUNK], dtype)
            KV_c0 = T.alloc_shared([BS, D_CHUNK], dtype)
            KV_c1 = T.alloc_shared([BS, D_CHUNK], dtype)
            dQ_chunk_shared = T.alloc_shared([block_H, D_CHUNK], dtype)
            Q_tail_shared = T.alloc_shared([block_H, D_tail], dtype)
            KV_tail_shared = T.alloc_shared([BS, D_tail], dtype)
            dQ_tail_shared = T.alloc_shared([block_H, D_tail], dtype)
            mask = T.alloc_fragment([BS], "bool")

            dP_shared_cast = T.alloc_shared([block_H, BS], dtype)
            PdP_shared = T.alloc_shared([2 * block_H, BS], dtype)  # stacked [dP ; P]

            acc_p = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dp = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_pdp = T.alloc_fragment([2 * block_H, BS], accum_dtype)  # [scores ; raw dP]
            acc_p_tail = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dq_c0 = T.alloc_fragment([block_H, D_CHUNK], accum_dtype)
            acc_dq_c1 = T.alloc_fragment([block_H, D_CHUNK], accum_dtype)
            acc_dq_tail = T.alloc_fragment([block_H, D_tail], accum_dtype)
            acc_dkv_c = T.alloc_fragment([BS, D_CHUNK], accum_dtype)
            acc_dkv_tail = T.alloc_fragment([BS, D_tail], accum_dtype)
            acc_dkv_shared = T.alloc_shared([half, D_CHUNK], accum_dtype)
            acc_dkv_tail_shared = T.alloc_shared([half, D_tail], accum_dtype)

            # swizzle to avoid shared-bank conflicts in the fp32 dKV scatter staging
            T.annotate_layout(
                {
                    acc_dkv_shared: make_swizzled_layout(acc_dkv_shared),
                    acc_dkv_tail_shared: make_swizzled_layout(acc_dkv_tail_shared),
                }
            )

            # loop-invariant Q/dO staging (same for every index block)
            T.copy(Q[by, s_i, bz * block_H : (bz + 1) * block_H, D:], Q_tail_shared)
            for h_i, d_i in T.Parallel(block_H, D_CHUNK):
                QdO_c0[h_i, d_i] = Q[by, s_i, bz * block_H + h_i, d_i]
                QdO_c1[h_i, d_i] = Q[by, s_i, bz * block_H + h_i, D_CHUNK + d_i]
            for h_i, d_i in T.Parallel(block_H, D_CHUNK):
                QdO_c0[block_H + h_i, d_i] = dO[by, s_i, bz * block_H + h_i, d_i]
                QdO_c1[block_H + h_i, d_i] = dO[by, s_i, bz * block_H + h_i, D_CHUNK + d_i]

            T.clear(acc_dq_c0)
            T.clear(acc_dq_c1)
            T.clear(acc_dq_tail)

            for i_i in T.Pipelined(NS, num_stages=num_stages):
                for bi_i in T.Parallel(BS):
                    mask[bi_i] = Indices[by, s_i, bz // NH, i_i * BS + bi_i] != -1

                # acc_pdp seed: top = masked score (0 / -inf), bottom = 0 for dP
                for r_i, bi_i in T.Parallel(2 * block_H, BS):
                    acc_pdp[r_i, bi_i] = T.if_then_else(
                        r_i < block_H, T.if_then_else(mask[bi_i], 0, -T.infinity(acc_pdp.dtype)), 0
                    )

                for bi_i, d_i in T.Parallel(BS, D_CHUNK):
                    KV_c0[bi_i, d_i] = KV[by, Indices[by, s_i, bz // NH, i_i * BS + bi_i], bz // NH, d_i]
                for bi_i, d_i in T.Parallel(BS, D_CHUNK):
                    KV_c1[bi_i, d_i] = KV[by, Indices[by, s_i, bz // NH, i_i * BS + bi_i], bz // NH, D_CHUNK + d_i]
                for bi_i, d_i in T.Parallel(BS, D_tail):
                    KV_tail_shared[bi_i, d_i] = KV[by, Indices[by, s_i, bz // NH, i_i * BS + bi_i], bz // NH, D + d_i]

                # fused score+dP: stacked-M [Q ; dO] @ KV^T per chunk
                T.gemm(QdO_c0, KV_c0, acc_pdp, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(QdO_c1, KV_c1, acc_pdp, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(
                    Q_tail_shared,
                    KV_tail_shared,
                    acc_p_tail,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = acc_pdp[h_i, bi_i] + acc_p_tail[h_i, bi_i]
                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_dp[h_i, bi_i] = acc_pdp[block_H + h_i, bi_i]
                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = T.exp2(
                        acc_p[h_i, bi_i] * sm_scale_mul_reciprocal_log2 - Lse[by, s_i, bz * block_H + h_i]
                    )
                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_dp[h_i, bi_i] = (
                        acc_p[h_i, bi_i] * (acc_dp[h_i, bi_i] - Delta[by, s_i, bz * block_H + h_i]) * sm_scale
                    )
                T.copy(acc_dp, dP_shared_cast)
                for h_i, bi_i in T.Parallel(block_H, BS):
                    PdP_shared[h_i, bi_i] = acc_dp[h_i, bi_i]
                for h_i, bi_i in T.Parallel(block_H, BS):
                    PdP_shared[block_H + h_i, bi_i] = acc_p[h_i, bi_i]

                # per D-chunk (+ tail): dQ += dP @ KV; fused dKV = [dP ; P]^T @ [Q ; dO]; scatter
                T.gemm(dP_shared_cast, KV_c0, acc_dq_c0, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(
                    PdP_shared, QdO_c0, acc_dkv_c, transpose_A=True, policy=T.GemmWarpPolicy.FullCol, clear_accum=True
                )
                scatter_dkv(dKV, Indices, acc_dkv_c, acc_dkv_shared, s_i, by, bz, i_i, 0, D_CHUNK)

                T.gemm(dP_shared_cast, KV_c1, acc_dq_c1, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(
                    PdP_shared, QdO_c1, acc_dkv_c, transpose_A=True, policy=T.GemmWarpPolicy.FullCol, clear_accum=True
                )
                scatter_dkv(dKV, Indices, acc_dkv_c, acc_dkv_shared, s_i, by, bz, i_i, D_CHUNK, D_CHUNK)

                T.gemm(dP_shared_cast, KV_tail_shared, acc_dq_tail, policy=T.GemmWarpPolicy.FullCol)
                T.clear(acc_dkv_tail)
                T.gemm(dP_shared_cast, Q_tail_shared, acc_dkv_tail, transpose_A=True, policy=T.GemmWarpPolicy.FullCol)
                scatter_dkv(dKV, Indices, acc_dkv_tail, acc_dkv_tail_shared, s_i, by, bz, i_i, D, D_tail)

            T.copy(acc_dq_c0, dQ_chunk_shared)
            for h_i, d_i in T.Parallel(block_H, D_CHUNK):
                dQ[by, s_i, bz * block_H + h_i, d_i] = dQ_chunk_shared[h_i, d_i]
            T.copy(acc_dq_c1, dQ_chunk_shared)
            for h_i, d_i in T.Parallel(block_H, D_CHUNK):
                dQ[by, s_i, bz * block_H + h_i, D_CHUNK + d_i] = dQ_chunk_shared[h_i, d_i]
            T.copy(acc_dq_tail, dQ_tail_shared)
            T.copy(dQ_tail_shared, dQ[by, s_i, bz * block_H : (bz + 1) * block_H, D:])

    return sparse_mla_bwd_kernel


def sparse_mla_bwd(q, kv, o, do, indices, lse, sm_scale=None, is_causal=True, return_kernel=False, delta=None):
    q = q.unsqueeze(0)
    kv = kv.unsqueeze(0)
    o = o.unsqueeze(0)
    do = do.unsqueeze(0)
    indices = indices.unsqueeze(0)
    lse = lse.unsqueeze(0)

    if not q.is_contiguous():
        raise ValueError("q must be contiguous")
    if not kv.is_contiguous():
        raise ValueError("kv must be contiguous")
    if not indices.is_contiguous():
        raise ValueError("indices must be contiguous")
    if not lse.is_contiguous():
        raise ValueError("lse must be contiguous")
    B, S, H, dim_plus_tail_dim = q.shape
    _, S_kv, kv_group, _ = kv.shape
    if kv.shape[-1] != dim_plus_tail_dim:
        raise ValueError("kv dimension mismatch")
    if kv.shape[0] != B:
        raise ValueError("kv batch size mismatch")
    D = o.shape[-1]
    D_tail = dim_plus_tail_dim - D
    topk = indices.shape[-1]
    if indices.shape != (B, S, kv_group, topk):
        raise ValueError("indices shape mismatch")
    if lse.shape != (B, S, H):
        raise ValueError("lse shape mismatch")

    preprocess_kernel = preprocess(B, S, H, D)
    bwd_kernel = bwd(B, S, S_kv, H, D, D_tail, topk, kv_group, sm_scale, is_causal)
    postprocess_kernel = postprocess(B, S_kv, D, D_tail, kv_group)

    if delta is None:
        delta = preprocess_kernel(o, do)
    dkv = torch.zeros_like(kv, dtype=torch.float32)
    dq = bwd_kernel(q, kv, do, indices, lse, delta, dkv)
    dkv = postprocess_kernel(dkv)

    return dq.squeeze(0), dkv.squeeze(0)
