# Copyright (c) Huawei Technologies Co., Ltd. 2026.
#
# Port of miles (radixark/miles) miles_plugins/models/glm5/ops/tilelang_indexer_fwd.py
# (and the V4 variant at miles_plugins/models/deepseek_v4/ops/kernel/tilelang_indexer_fwd.py)
# to Ascend NPU via mlir-ascend.
#
# Note on dtype: miles' wrapper uses bf16 (NOT FP8) even though the upstream
# tile-ai/tilelang examples/deepseek_v32/fp8_lighting_indexer.py uses FP8.
# We follow miles. FP8 path is not needed for this kernel as miles consumes it.
#
# Algorithm: lighting indexer
#   For each (seq_pos i, kv_pos j):
#     score s_h = IndexK[j] @ IndexQ[i*H+h]^T   for each head h
#     score s_h = max(s_h, 0)                   (ReLU)
#     logits[i, j] = sum_h s_h * W[i, h]
#
# Adaptations vs upstream / miles' CUDA version:
#   * is_npu=True (single block dim) — our grid is SEQ // block_Q (already 1-axis).
#   * b_transpose=True (Ascend gemm) replaces transpose_B= / clear_accum= +
#     policy=T.GemmWarpPolicy.FullCol.
#   * NPU vector intrinsics (vbrc/vmul/vmax) replace upstream T.Parallel
#     elementwise patterns.
#   * T.alloc_var (CUDA scalar var) replaced with vbrc-zeroed accumulators.
#   * The 3-deep T.Parallel(BN, BQ, H) for weight-broadcast is unrolled to
#     serial loops — fragment-wise scatter.
#   * **Tile-level T.copy for global memory writes** — per-scalar Logits[i,j]
#     stores trigger MTE aicore exceptions on Ascend. Always tile-copy via
#     a shared buffer. (KB §12.3 R-KA-7.)
#   * No cu_seqlen causal-mask plumbing yet. Upstream's
#     `cu_k_e_max - cu_k_s_min` runtime extent doesn't lower on Ascend; first
#     port assumes full [0, seq_len_kv) is valid. Causal masking done by
#     caller (matches miles' batched_indexer_fwd Python wrapper).
import os
import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1], target="npuir")
def lighting_indexer_fwd(
    seq_len,
    seq_len_kv,
    heads,
    index_dim,
    block_N=64,
    block_Q=None,
    num_stages=1,
):
    """Lighting indexer forward.

    Computes Logits[i, j] = sum_h max(IndexK[j] @ IndexQ[i*H+h]^T, 0) * W[i, h]
    """
    if block_Q is None:
        block_Q = max(1, 128 // heads)
    dtype = "bfloat16"  # miles uses bf16 on the GPU path too
    accum_dtype = "float32"

    NK = (seq_len_kv + block_N - 1) // block_N
    NQ = (seq_len + block_Q - 1) // block_Q
    assert seq_len_kv % block_N == 0, "first port: aligned seq_len_kv only"
    assert seq_len % block_Q == 0, "first port: aligned seq_len only"

    index_q_shape = [seq_len * heads, index_dim]
    index_k_shape = [seq_len_kv, index_dim]
    weights_shape = [seq_len, heads]
    logits_shape = [seq_len, seq_len_kv]

    @T.prim_func
    def main(
        IndexQ: T.Tensor(index_q_shape, dtype),
        IndexK: T.Tensor(index_k_shape, dtype),
        Weights: T.Tensor(weights_shape, accum_dtype),
        Logits: T.Tensor(logits_shape, accum_dtype),
    ):
        with T.Kernel(NQ, is_npu=True) as (bq, _):
            seq_i = bq * block_Q
            q_shared = T.alloc_shared([block_Q * heads, index_dim], dtype)
            k_shared = T.alloc_shared([block_N, index_dim], dtype)
            weights_BH = T.alloc_shared([block_Q, heads], accum_dtype)
            weights_frag = T.alloc_fragment([block_Q, heads], accum_dtype)

            scores = T.alloc_fragment([block_N, block_Q * heads], accum_dtype)
            scores_relu = T.alloc_fragment([block_N, block_Q * heads], accum_dtype)
            scores_weighted = T.alloc_fragment([block_N, block_Q * heads], accum_dtype)
            zeros = T.alloc_fragment([block_N, block_Q * heads], accum_dtype)
            weights_bn_bqh = T.alloc_fragment([block_N, block_Q * heads], accum_dtype)

            logits_local = T.alloc_fragment([block_N, block_Q], accum_dtype)
            logits_local_T = T.alloc_fragment([block_Q, block_N], accum_dtype)
            logits_shared_T = T.alloc_shared([block_Q, block_N], accum_dtype)
            value_zero = 0

            T.copy(
                IndexQ[seq_i * heads : seq_i * heads + block_Q * heads, 0:index_dim],
                q_shared,
            )
            T.copy(Weights[seq_i : seq_i + block_Q, 0:heads], weights_BH)
            T.copy(weights_BH, weights_frag)

            for k in T.Pipelined(NK, num_stages=num_stages):
                T.copy(
                    IndexK[k * block_N : (k + 1) * block_N, 0:index_dim],
                    k_shared,
                )

                # GEMM: scores[BN, BQ*H] = K @ Q^T
                T.gemm(k_shared, q_shared, scores, initC=True, b_transpose=True)

                # ReLU
                T.vbrc(value_zero, zeros)
                T.vmax(scores, zeros, scores_relu)

                # Broadcast weights[BQ, H] over BN rows into weights_bn_bqh[BN, BQ*H]
                # (NPU vbrc doesn't broadcast a 2D fragment without rank tile,
                # so do a 2-deep serial scatter; this lowers cleanly.)
                for bn_i in T.serial(block_N):
                    for bqh in T.serial(block_Q * heads):
                        weights_bn_bqh[bn_i, bqh] = weights_frag[
                            bqh // heads, bqh % heads
                        ]

                # Apply head weights
                T.vmul(scores_relu, weights_bn_bqh, scores_weighted)

                # Reduce over head dim: logits[bn, bq] = sum_h scores_weighted[bn, bq*H+h]
                T.vbrc(value_zero, logits_local)
                for h_idx in T.serial(heads):
                    for bq_idx in T.serial(block_Q):
                        for bn_i in T.serial(block_N):
                            logits_local[bn_i, bq_idx] = (
                                logits_local[bn_i, bq_idx]
                                + scores_weighted[bn_i, bq_idx * heads + h_idx]
                            )

                # Transpose [BN, BQ] -> [BQ, BN] then tile-copy to global
                # (per-scalar global stores cause MTE aicore exception; tile-copy
                # via a shared buffer is the working pattern. KB §12.3 R-KA-7.)
                for bq_idx in T.serial(block_Q):
                    for bn_i in T.serial(block_N):
                        logits_local_T[bq_idx, bn_i] = logits_local[bn_i, bq_idx]
                T.copy(logits_local_T, logits_shared_T)
                T.copy(
                    logits_shared_T,
                    Logits[seq_i : seq_i + block_Q, k * block_N : (k + 1) * block_N],
                )

    return main


def _ref_indexer(q_flat, kv, weights):
    """fp32 CPU reference matching miles V4 test fixture's ref_compute_index_scores."""
    SEQ_x_H, D = q_flat.shape
    SKV, _ = kv.shape
    SEQ, H = weights.shape
    q_3d = q_flat.reshape(SEQ, H, D).float()
    scores = torch.einsum("shd,td->sht", q_3d, kv.float())  # [SEQ, H, SKV]
    scores = scores.clamp(min=0)
    scores = scores * weights.float().unsqueeze(-1)
    logits = scores.sum(dim=1)  # [SEQ, SKV]
    return logits


def test_lighting_indexer_fwd_small():
    torch.npu.set_device(0)
    SEQ, SKV, H, D = 8, 16, 8, 32
    BN, BQ = 16, 4

    torch.manual_seed(0)
    q = torch.randn(SEQ, H, D, dtype=torch.bfloat16, device="npu") * 0.1
    kv = torch.randn(SKV, D, dtype=torch.bfloat16, device="npu") * 0.1
    weights = torch.randn(SEQ, H, dtype=torch.float32, device="npu") * 0.5

    q_flat = q.reshape(SEQ * H, D).contiguous()
    print(
        f"compile lighting_indexer_fwd(SEQ={SEQ},SKV={SKV},H={H},D={D},BN={BN},BQ={BQ}) ..."
    )
    k = lighting_indexer_fwd(SEQ, SKV, H, D, block_N=BN, block_Q=BQ)
    print("compile OK; running ...")
    out = k(q_flat, kv.contiguous(), weights.contiguous())
    print(f"out shape: {tuple(out.shape)}")

    ref = _ref_indexer(q_flat, kv, weights)
    err = (out.cpu().float() - ref.cpu()).abs().max().item()
    print(f"max abs err: {err:.6f}")
    print(f"out[0,:4] = {out[0, :4].cpu().tolist()}")
    print(f"ref[0,:4] = {ref[0, :4].cpu().tolist()}")
    assert err < 1e-2, f"Numerical error too high: {err}"
    print("PASS")


if __name__ == "__main__":
    os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
    test_lighting_indexer_fwd_small()
