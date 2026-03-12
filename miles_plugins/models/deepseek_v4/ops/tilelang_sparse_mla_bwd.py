# ruff: noqa
import tilelang
from tilelang import language as T
import torch

@tilelang.jit(out_idx=[-1])
def preprocess(
    B,
    S,
    H,
    D,
    block_ND=32,
    num_stages=5,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32
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
    B,
    S_kv,
    D,
    block_N=64,
    threads=128,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32
    dkv_shape = [B, S_kv, D]

    @T.prim_func
    def postprocess_kernel(
        dKV: T.Tensor(dkv_shape, accum_dtype),
        dKV_out: T.Tensor(dkv_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(S_kv, block_N), B, threads=threads) as (bx, bz):
            T.copy(
                dKV[bz, bx * block_N : (bx + 1) * block_N, :],
                dKV_out[bz, bx * block_N : (bx + 1) * block_N, :],
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
    B,
    S,
    S_kv,
    H,
    D,
    topk,
    sm_scale=None,
    is_causal=True,
    block_size=32,
    num_stages=0,
    threads=256,
    indices_dtype=T.int32,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    assert is_causal == True, "non-casual is not supported now"
    assert topk % block_size == 0, "otherwise will load some index=0 thus causing wrong kv to be loaded"
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32
    assert indices_dtype == T.int32

    if sm_scale is None:
        sm_scale = D ** (-0.5)
    sm_scale_mul_reciprocal_log2 = sm_scale * 1.44269504

    q_shape = [B, S, H, D]
    kv_shape = [B, S_kv, D]
    o_shape = [B, S, H, D]
    indices_shape = [B, S, topk]
    delta_shape = [B, S, H]
    lse_shape = [B, S, H]
    attn_sink_shape = [H]
    assert indices_dtype == T.int32
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32

    padded_H = max(tilelang.math.next_power_of_2(H), 16)
    block_H = min(64, padded_H)
    assert padded_H % block_H == 0
    NH = padded_H // block_H
    BS = block_size
    NS = tilelang.cdiv(topk, block_size)

    split_store = 2

    @T.prim_func
    def sparse_mla_bwd_kernel(
        Q: T.Tensor(q_shape, dtype),
        KV: T.Tensor(kv_shape, dtype),
        AttnSink: T.Tensor(attn_sink_shape, accum_dtype),
        dO: T.Tensor(o_shape, dtype),
        Indices: T.Tensor(indices_shape, indices_dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
        Delta: T.Tensor(delta_shape, accum_dtype),
        dQ: T.Tensor(q_shape, dtype),
        dKV: T.Tensor(kv_shape, accum_dtype),
        dAttnSink: T.Tensor(attn_sink_shape, accum_dtype),
    ):
        with T.Kernel(S, B, NH, threads=threads) as (s_i, by, bz):
            Q_shared = T.alloc_shared([block_H, D], dtype)
            KV_shared = T.alloc_shared([BS, D], dtype)
            dO_shared = T.alloc_shared([block_H, D], dtype)
            mask = T.alloc_fragment([BS], "bool")
            attn_sink_local = T.alloc_fragment([block_H], accum_dtype)

            P_shared_cast = T.alloc_shared([block_H, BS], dtype)
            dP_shared_cast = T.alloc_shared([block_H, BS], dtype)
            dQ_shared = T.alloc_shared([block_H, D], dtype)

            acc_p = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dp = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dq = T.alloc_fragment([block_H, D], accum_dtype)
            acc_dkv = T.alloc_fragment([BS, D], accum_dtype)
            acc_dkv_shared = T.alloc_shared([BS // split_store, D], accum_dtype)

            T.copy(Q[by, s_i, bz * block_H : (bz + 1) * block_H, :], Q_shared)
            T.copy(dO[by, s_i, bz * block_H : (bz + 1) * block_H, :], dO_shared)
            T.copy(AttnSink[bz * block_H : (bz + 1) * block_H], attn_sink_local)

            T.clear(acc_dq)

            for i_i in T.Pipelined(NS, num_stages=num_stages):
                for bi_i in T.Parallel(BS):
                    mask[bi_i] = Indices[by, s_i, i_i * BS + bi_i] != -1

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_p.dtype))

                for bi_i, d_i in T.Parallel(BS, D):
                    KV_shared[bi_i, d_i] = T.if_then_else(
                        mask[bi_i], KV[by, Indices[by, s_i, i_i * BS + bi_i], d_i], 0
                    )

                T.gemm(Q_shared, KV_shared, acc_p, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = T.exp2(acc_p[h_i, bi_i] * sm_scale_mul_reciprocal_log2 - Lse[by, s_i, bz * block_H + h_i])

                T.copy(acc_p, P_shared_cast)

                T.gemm(dO_shared, KV_shared, acc_dp, transpose_B=True, policy=T.GemmWarpPolicy.FullCol, clear_accum=True)

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_dp[h_i, bi_i] = acc_p[h_i, bi_i] * (acc_dp[h_i, bi_i] - Delta[by, s_i, bz * block_H + h_i]) * sm_scale

                T.copy(acc_dp, dP_shared_cast)
                T.gemm(dP_shared_cast, KV_shared, acc_dq, policy=T.GemmWarpPolicy.FullCol)

                T.gemm(dP_shared_cast, Q_shared, acc_dkv, transpose_A=True, policy=T.GemmWarpPolicy.FullCol, clear_accum=True)
                T.gemm(P_shared_cast, dO_shared, acc_dkv, transpose_A=True, policy=T.GemmWarpPolicy.FullCol)

                for s in range(split_store):
                    for bi_i, d_i in T.Parallel(BS, D):
                        if bi_i < BS // split_store:
                            acc_dkv_shared[bi_i, d_i] = acc_dkv[bi_i + s * (BS // split_store), d_i]

                    for bi_i, d_i in T.Parallel(BS // split_store, D // 4):
                        raw_idx = Indices[by, s_i, i_i * BS + bi_i + s * (BS // split_store)]
                        if raw_idx != -1:
                            T.atomic_addx4(
                                dKV[by, raw_idx, d_i * 4],
                                acc_dkv_shared[bi_i, d_i * 4],
                            )

            T.copy(acc_dq, dQ_shared)
            T.copy(dQ_shared, dQ[by, s_i, bz * block_H : (bz + 1) * block_H, :])

            log2_e = 1.44269504
            for h_i in T.Parallel(block_H):
                T.atomic_add(
                    dAttnSink[bz * block_H + h_i],
                    -Delta[by, s_i, bz * block_H + h_i] * T.exp2(attn_sink_local[h_i] * log2_e - Lse[by, s_i, bz * block_H + h_i]),
                )

    return sparse_mla_bwd_kernel


def sparse_mqa_bwd_interface(q, kv, attn_sink, o, do, indices, lse, sm_scale=None, is_casual=True, delta=None, block_size=32):
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous() and lse.is_contiguous()
    batch, seq_len, heads, dim = q.shape
    _, seq_len_kv, kv_dim = kv.shape
    assert kv_dim == dim == 512
    assert kv.shape[0] == batch
    assert attn_sink.shape == (heads,)

    topk = indices.shape[-1]
    assert indices.shape == (batch, seq_len, topk)
    assert lse.shape == (batch, seq_len, heads)

    padded_topk = (topk + block_size - 1) // block_size * block_size
    if padded_topk != topk:
        padded_indices = torch.full((batch, seq_len, padded_topk), -1, dtype=indices.dtype, device=indices.device)
        padded_indices[:, :, :topk] = indices
        indices = padded_indices
        topk = padded_topk

    padded_heads = max(tilelang.math.next_power_of_2(heads), 64)
    if padded_heads != heads:
        q_padded = torch.zeros(batch, seq_len, padded_heads, dim, dtype=q.dtype, device=q.device)
        q_padded[:, :, :heads, :] = q
        q = q_padded
        o_padded = torch.zeros(batch, seq_len, padded_heads, dim, dtype=o.dtype, device=o.device)
        o_padded[:, :, :heads, :] = o
        o = o_padded
        do_padded = torch.zeros(batch, seq_len, padded_heads, dim, dtype=do.dtype, device=do.device)
        do_padded[:, :, :heads, :] = do
        do = do_padded
        lse_padded = torch.zeros(batch, seq_len, padded_heads, dtype=lse.dtype, device=lse.device)
        lse_padded[:, :, :heads] = lse
        lse = lse_padded
        attn_sink_padded = torch.full((padded_heads,), float('-inf'), dtype=attn_sink.dtype, device=attn_sink.device)
        attn_sink_padded[:heads] = attn_sink
        attn_sink = attn_sink_padded

    preprocess_kernel = preprocess(batch, seq_len, padded_heads, dim)
    bwd_kernel = bwd(batch, seq_len, seq_len_kv, padded_heads, dim, topk, sm_scale, is_casual, block_size=block_size)
    postprocess_kernel = postprocess(batch, seq_len_kv, dim)

    if delta is None:
        delta = preprocess_kernel(o, do)
    dkv = torch.zeros_like(kv, dtype=torch.float32)
    d_attn_sink = torch.zeros(padded_heads, dtype=torch.float32, device=attn_sink.device)
    dq = bwd_kernel(q, kv, attn_sink, do, indices, lse, delta, dkv, d_attn_sink)
    dkv = postprocess_kernel(dkv)

    if padded_heads != heads:
        dq = dq[:, :, :heads, :].contiguous()
        d_attn_sink = d_attn_sink[:heads].contiguous()

    return dq, dkv, d_attn_sink
