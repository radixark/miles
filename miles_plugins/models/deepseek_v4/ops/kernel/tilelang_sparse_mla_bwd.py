# ruff: noqa
# Adapted from miles_plugins/models/glm5/ops/tilelang_sparse_mla_bwd.py for DeepSeek-V4.
# Key differences from GLM-5:
#   - attn_sink: gradient computation for learnable per-head scalar
#   - Single-head KV: kv shape [B, S_kv, D] (no kv_group, no D/D_tail split)
#   - Index shape: [B, S, topk] (no kv_group dim)
#   - Outputs: dQ [B, S, H, D], dKV [B, S_kv, D], dAttnSink [H]
import tilelang
import torch
from tilelang import language as T


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
        with T.Kernel(T.ceildiv(S_kv, block_N), B, threads=threads) as (bx, by):
            T.copy(
                dKV[by, bx * block_N : (bx + 1) * block_N, :],
                dKV_out[by, bx * block_N : (bx + 1) * block_N, :],
            )

    return postprocess_kernel


@tilelang.jit(
    out_idx=[-3],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: False,
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
    block_size=32,
    num_stages=0,
    threads=128,
    split_store=2,
    stage_dq_through_shared=True,
    max_block_H=None,
    indices_dtype=T.int32,
    dtype=T.bfloat16,
    accum_dtype=T.float32,
):
    assert topk % block_size == 0, f"topk ({topk}) must be divisible by block_size ({block_size})"
    assert block_size % split_store == 0, f"block_size ({block_size}) must be divisible by split_store ({split_store})"
    assert dtype == T.bfloat16
    assert accum_dtype == T.float32

    if sm_scale is None:
        sm_scale = D ** (-0.5)
    sm_scale_mul_reciprocal_log2 = sm_scale * 1.44269504  # log2(e)

    q_shape = [B, S, H, D]
    kv_shape = [B, S_kv, D]
    o_shape = [B, S, H, D]
    indices_shape = [B, S, topk]
    delta_shape = [B, S, H]
    lse_shape = [B, S, H]
    attn_sink_shape = [H]

    padded_H = max(tilelang.math.next_power_of_2(H), 16)
    if max_block_H is None:
        is_hip = getattr(torch.version, "hip", None)
        if is_hip:
            # Split large HIP head tiles to reduce LDS use.
            max_block_H = 32 if padded_H >= 64 else 64
        else:
            max_block_H = 64
    block_H = min(max_block_H, padded_H)
    assert padded_H % block_H == 0
    NH = padded_H // block_H
    BS = block_size
    NS = tilelang.cdiv(topk, block_size)

    @T.prim_func
    def sparse_mqa_bwd_kernel(
        Q: T.Tensor(q_shape, dtype),
        KV: T.Tensor(kv_shape, dtype),
        dO: T.Tensor(o_shape, dtype),
        AttnSink: T.Tensor(attn_sink_shape, accum_dtype),
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

            P_shared_cast = T.alloc_shared([block_H, BS], dtype)
            dP_shared_cast = T.alloc_shared([block_H, BS], dtype)
            if stage_dq_through_shared:
                dQ_shared = T.alloc_shared([block_H, D], dtype)

            acc_p = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dp = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dq = T.alloc_fragment([block_H, D], accum_dtype)
            acc_dkv = T.alloc_fragment([BS, D], accum_dtype)
            acc_dkv_shared = T.alloc_shared([BS // split_store, D], accum_dtype)

            T.copy(Q[by, s_i, bz * block_H : (bz + 1) * block_H, :D], Q_shared)
            T.copy(dO[by, s_i, bz * block_H : (bz + 1) * block_H, :D], dO_shared)

            T.clear(acc_dq)

            for i_i in T.Pipelined(NS, num_stages=num_stages):
                for bi_i in T.Parallel(BS):
                    mask[bi_i] = Indices[by, s_i, i_i * BS + bi_i] != -1

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_p.dtype))

                for bi_i, d_i in T.Parallel(BS, D):
                    KV_shared[bi_i, d_i] = KV[by, Indices[by, s_i, i_i * BS + bi_i], d_i]

                T.gemm(Q_shared, KV_shared, acc_p, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

                # P = exp2(scores * sm_scale_log2e - LSE)
                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = T.exp2(
                        acc_p[h_i, bi_i] * sm_scale_mul_reciprocal_log2 - Lse[by, s_i, bz * block_H + h_i]
                    )

                T.copy(acc_p, P_shared_cast)

                # dP = P * (dO @ KV^T - Delta)
                T.gemm(
                    dO_shared, KV_shared, acc_dp, transpose_B=True, policy=T.GemmWarpPolicy.FullCol, clear_accum=True
                )

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_dp[h_i, bi_i] = (
                        acc_p[h_i, bi_i] * (acc_dp[h_i, bi_i] - Delta[by, s_i, bz * block_H + h_i]) * sm_scale
                    )

                T.copy(acc_dp, dP_shared_cast)

                # dQ += dP @ KV
                T.gemm(dP_shared_cast, KV_shared, acc_dq, policy=T.GemmWarpPolicy.FullCol)

                # dKV += dP^T @ Q + P^T @ dO
                T.gemm(
                    dP_shared_cast,
                    Q_shared,
                    acc_dkv,
                    transpose_A=True,
                    policy=T.GemmWarpPolicy.FullCol,
                    clear_accum=True,
                )
                T.gemm(P_shared_cast, dO_shared, acc_dkv, transpose_A=True, policy=T.GemmWarpPolicy.FullCol)

                # Atomic store dKV with split to reduce register pressure
                for s in range(split_store):
                    for bi_i, d_i in T.Parallel(BS, D):
                        if bi_i < BS // split_store:
                            acc_dkv_shared[bi_i, d_i] = acc_dkv[bi_i + s * (BS // split_store), d_i]

                    for bi_i, d_i in T.Parallel(BS // split_store, D // 4):
                        T.atomic_addx4(
                            dKV[
                                by,
                                Indices[by, s_i, i_i * BS + bi_i + s * (BS // split_store)],
                                d_i * 4,
                            ],
                            acc_dkv_shared[bi_i, d_i * 4],
                        )

            # Store dQ
            if stage_dq_through_shared:
                T.copy(acc_dq, dQ_shared)
                T.copy(dQ_shared, dQ[by, s_i, bz * block_H : (bz + 1) * block_H, :D])
            else:
                T.copy(acc_dq, dQ[by, s_i, bz * block_H : (bz + 1) * block_H, :D])

            # dAttnSink[h] = -sum_{b,s}( Delta[b,s,h] * p_sink[b,s,h] )
            # where p_sink = exp(attn_sink[h]) / Z = exp2(attn_sink[h]*log2e - LSE)
            # attn_sink is a pre-scaled logit, so only convert to log2 base (no sm_scale)
            for h_i in T.Parallel(block_H):
                T.atomic_add(
                    dAttnSink[bz * block_H + h_i],
                    -Delta[by, s_i, bz * block_H + h_i]
                    * T.exp2(AttnSink[bz * block_H + h_i] * 1.44269504 - Lse[by, s_i, bz * block_H + h_i]),
                )

    return sparse_mqa_bwd_kernel


# tilelang rejects an unbuildable tiling in three different places -- the shared-memory check at
# codegen, warp partitioning inside T.gemm, and the vectorizer on the atomic store loop -- so all
# three have to be treated as "try the next candidate" rather than as a hard error.
RETRYABLE_BUILD_ERRORS = ("exceeds device limit", "Divide by zero", "is_scalar")

# (block_size, threads, split_store, stage_dq_through_shared, max_block_H), in decreasing order of
# expected performance. The first entry is the shipped tiling, sized for gfx950's 160 KiB LDS.
#
# The second is the only one that builds on gfx942's 64 KiB, and every part of it is forced:
#   - block_size and max_block_H both at 16, because Q_shared/dO_shared [block_H, D] and KV_shared
#     [block_size, D] are 16 KiB each at D=512 and three of them is already 48 KiB;
#   - stage_dq_through_shared off, because dQ_shared is a fourth [block_H, D] buffer and 64 KiB does
#     not hold four of them plus the accumulators;
#   - threads=64, because a 16x16 gemm output cannot be split across two wave64s under
#     GemmWarpPolicy.FullCol (N=8 is below MFMA's 16 and tilelang divides by zero);
#   - split_store == block_size, because at threads=64 anything wider than a single row in
#     acc_dkv_shared makes the atomic_addx4 store loop fail tilelang's vectorization pass.
# Total shared memory is 52224 B, 80% of the limit. Costs: one head and one KV slot per tile
# instead of 32, a single wave per workgroup, and 16 separate atomic store passes per KV block.
_FALLBACK_TILINGS = (
    (32, 128, 2, True, None),
    (16, 64, 16, False, 16),
)
_fitted_tiling: dict[tuple, tuple] = {}


def bwd_within_shared_mem(B, S, S_kv, H, D, topk, sm_scale=None):
    """Build the backward kernel with the best tiling this GPU can host.

    Which tiling fits is not worth predicting: tilelang merges and pads shared buffers, and two of
    the three failure modes are not about size at all. The only reliable check is asking it to
    build. Memoized per shape, so the cost is at most one wasted compile per shape.
    """
    key = (B, S, S_kv, H, D, topk, sm_scale)
    if key in _fitted_tiling:
        block_size, threads, split_store, stage_dq, max_block_H = _fitted_tiling[key]
        return bwd(
            B,
            S,
            S_kv,
            H,
            D,
            topk,
            sm_scale,
            block_size=block_size,
            threads=threads,
            split_store=split_store,
            stage_dq_through_shared=stage_dq,
            max_block_H=max_block_H,
        )

    # `.with_traceback(None)` is not cosmetic: an exception carried out of its `except` block keeps
    # the frame chain of every caller alive, which here means the model forward's activations, in a
    # cycle only the cyclic collector can break. Only the message survives, which is all
    # `raise last_error` needs.
    last_error = None
    for candidate in _FALLBACK_TILINGS:
        block_size, threads, split_store, stage_dq, max_block_H = candidate
        if topk % block_size != 0:
            continue
        try:
            kernel = bwd(
                B,
                S,
                S_kv,
                H,
                D,
                topk,
                sm_scale,
                block_size=block_size,
                threads=threads,
                split_store=split_store,
                stage_dq_through_shared=stage_dq,
                max_block_H=max_block_H,
            )
        except Exception as e:  # tilelang raises RuntimeError or tvm InternalError
            if not any(marker in str(e) for marker in RETRYABLE_BUILD_ERRORS):
                raise
            last_error = e.with_traceback(None)
            continue
        _fitted_tiling[key] = candidate
        if candidate is not _FALLBACK_TILINGS[0]:
            print(
                f"[sparse_mqa_bwd] shared memory forced a smaller tiling on this GPU: "
                f"block_size={block_size} threads={threads} split_store={split_store} "
                f"stage_dq_through_shared={stage_dq} max_block_H={max_block_H}",
                flush=True,
            )
        return kernel
    raise last_error


def sparse_mqa_bwd_interface(q, kv, attn_sink, o, do, topk_idxs, lse, sm_scale=None):
    """Backward interface for V4 sparse MQA attention.

    Args:
        q:         [B, S, H, D] bf16
        kv:        [B, S_kv, D] bf16
        attn_sink: [H] fp32
        o:         [B, S, H, D] bf16 (forward output)
        do:        [B, S, H, D] bf16 (grad of output)
        topk_idxs: [B, S, topk] int32
        lse:       [B, S, H] fp32 (log-sum-exp from forward)
        sm_scale:  float or None

    Returns:
        dq:         [B, S, H, D] bf16
        dkv:        [B, S_kv, D] bf16
        d_attn_sink: [H] fp32
    """
    assert q.is_contiguous() and kv.is_contiguous()
    assert topk_idxs.is_contiguous() and lse.is_contiguous()
    B, S, H, D = q.shape
    _, S_kv, _ = kv.shape
    topk = topk_idxs.shape[-1]

    # Pad topk to next multiple of block_size (kernel requires divisibility)
    block_size = 32
    padded_topk = (topk + block_size - 1) // block_size * block_size
    if padded_topk != topk:
        pad = torch.full((B, S, padded_topk - topk), -1, device=topk_idxs.device, dtype=topk_idxs.dtype)
        topk_idxs = torch.cat([topk_idxs, pad], dim=-1).contiguous()
        topk = padded_topk

    preprocess_kernel = preprocess(B, S, H, D)
    bwd_kernel = bwd_within_shared_mem(B, S, S_kv, H, D, topk, sm_scale)
    postprocess_kernel = postprocess(B, S_kv, D)

    delta = preprocess_kernel(o, do)
    dkv = torch.zeros_like(kv, dtype=torch.float32)
    d_attn_sink = torch.zeros_like(attn_sink)
    dq = bwd_kernel(q, kv, do, attn_sink, topk_idxs, lse, delta, dkv, d_attn_sink)
    dkv = postprocess_kernel(dkv)

    return dq, dkv, d_attn_sink
