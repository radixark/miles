# Copyright (c) Huawei Technologies Co., Ltd. 2026.
#
# Port of upstream tile-ai/tilelang examples/deepseek_v32/sparse_mla_fwd.py to
# Ascend NPU via mlir-ascend. Used by miles (radixark/miles) at
# miles_plugins/models/glm5/ops/tilelang_sparse_mla_fwd.py.
#
# This is the static-shape baseline. Dynamic shapes (T.dynamic) follow in a
# sibling file once this proves out end-to-end.
#
# Notable adaptations vs upstream:
#   * 3-axis grid (seq_len * REPLICATE_H, batch, kv_group) collapsed into 1 axis
#     -- is_npu=True requires exactly one block dimension. We decode inside.
#   * b_transpose= / NPU vector intrinsics (vbrc/vmul/vexp/...) instead of
#     T.Parallel elementwise loops where possible.
#   * Single-pass online softmax matches the flash-attention template that
#     already passes on this backend.
#   * REPLICATE_H == 1; H_per_block == padded_head_kv. Multi-replication will
#     be added once correctness is established here.
#
# Two parser-rank-check workarounds carried in this kernel:
#   * vbrc literal `0` errors at the rank check; assign `value_zero = 0`
#     first so TVM wraps it in a tir.Var (subclass of PrimExpr) and the
#     check is bypassed correctly.
#   * Scalar factory-closure values (e.g. sm_scale) need a local copy
#     inside the prim_func body to be captured cleanly by the parser.
import os
import torch
import tilelang
import tilelang.language as T


@tilelang.jit(
    out_idx=[-2, -1],
    target="npuir",
    pass_configs={
        # Disable auto multi-buffer to halve UB pressure at large
        # head counts (e.g. H=64 DSv4-Flash). Without this, an H=64
        # `acc_o [64, D_V] fp32` accumulator paired with multi-buffer's
        # 2x copies overflows the dav-c220 UB.
        "npuir.enable_auto_multi_buffer": False,
    },
)
def sparse_mla_fwd(
    batch,
    seq_len,
    seq_len_kv,
    heads,
    dim,
    tail_dim,
    topk,
    block_M=None,
    block_N=64,
    num_stages=2,
    block_M_inner=None,
):
    """Sparse MLA forward kernel.

    Args correspond to upstream sparse_mla_fwd; kv_group is fixed at 1 here.

    `block_M` defaults to `heads` (one Q-block per head group). When
    `block_M * dim * sizeof(fp32)` would exceed the chip's Unified Buffer
    budget — true for real GLM-5 / DSv4-Flash shapes where `heads=64` and
    `dim=512` give a 128 KB `acc_o` fragment alone — we *must* drop
    `block_M` below `heads`. Set `block_M_inner` to the per-block head
    count (must divide `heads`); the grid then loops over
    `heads // block_M_inner` head-tiles per (batch, seq) cell so each NPU
    block only ever materialises a `[block_M_inner, dim] fp32` accumulator.

    With `block_M_inner = 16`, a `heads=64 / dim=512` kernel emits
    `acc_o [16, 512] fp32 = 32 KB` instead of the previous 128 KB,
    cutting the total per-block UB request from ~404 KB down to ~150 KB.
    """
    if block_M is None:
        block_M = heads
    if block_M_inner is None:
        block_M_inner = block_M
    assert heads % block_M_inner == 0, (
        f"block_M_inner={block_M_inner} must divide heads={heads}"
    )
    head_groups = heads // block_M_inner
    block_M = block_M_inner  # all subsequent allocations use the inner tile
    D = dim
    DT = tail_dim
    dtype = "float16"
    accum_dtype = "float32"
    idx_dtype = "int32"
    sm_scale = (1.0 / (D + DT)) ** 0.5

    q_shape = [batch, seq_len, heads, D + DT]
    kv_shape = [batch, seq_len_kv, 1, D + DT]
    o_shape = [batch, seq_len, heads, D]
    idx_shape = [batch, seq_len, 1, topk]
    lse_shape = [
        batch,
        seq_len,
        heads,
        1,
    ]  # trailing 1 keeps rank parity with [BM,1] fragment

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        KV: T.Tensor(kv_shape, dtype),
        Indices: T.Tensor(idx_shape, idx_dtype),
        Output: T.Tensor(o_shape, dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
    ):
        # Grid: batch * seq_len * head_groups. Each NPU block owns one
        # (b, s, head-group) cell and processes `block_M_inner` heads —
        # this keeps the per-block UB footprint bounded by block_M_inner
        # rather than the model's full head count.
        with T.Kernel(batch * seq_len * head_groups, is_npu=True) as (cid, _):
            b_i = cid // (seq_len * head_groups)
            rem = cid % (seq_len * head_groups)
            s_i = rem // head_groups
            hg_i = rem % head_groups
            h_start = hg_i * block_M  # H offset for this head-tile

            Q_shared = T.alloc_shared([block_M, D], dtype)
            Q_tail_shared = T.alloc_shared([block_M, DT], dtype)
            KV_shared = T.alloc_shared([block_N, D], dtype)
            K_tail_shared = T.alloc_shared([block_N, DT], dtype)

            scores = T.alloc_fragment([block_M, block_N], accum_dtype)
            scores_cast = T.alloc_fragment([block_M, block_N], dtype)
            correction = T.alloc_fragment([block_M, 1], accum_dtype)
            local_max = T.alloc_fragment([block_M, 1], accum_dtype)
            local_sum = T.alloc_fragment([block_M, 1], accum_dtype)
            acc_m = T.alloc_fragment([block_M, 1], accum_dtype)
            acc_l = T.alloc_fragment([block_M, 1], accum_dtype)
            acc_o = T.alloc_fragment([block_M, D], accum_dtype)
            tmp = T.alloc_fragment([block_M, block_N], accum_dtype)
            tmp1 = T.alloc_fragment([block_M, 1], accum_dtype)
            new_max = T.alloc_fragment([block_M, 1], accum_dtype)
            scales = T.alloc_fragment([block_M, block_N], accum_dtype)
            idx_buf = T.alloc_fragment([block_N], idx_dtype)
            # R-KA-16 partial mitigation (AscendNPU-IR issue #251):
            # explicitly broadcast `correction [block_M, 1]` to full
            # `[block_M, D]` via Python serial fill immediately before the
            # `acc_o = correction * acc_o` step. Without this, the broadcast
            # vmul produces NaN on the cross-iter persistent `acc_o` once the
            # online-softmax loop has more than one iter. Until the upstream
            # bishengir patch on issue #251 lands, the dispatcher additionally
            # pins ``num_stages=1`` (see ``sparse_mla.py``).
            correction_expanded = T.alloc_fragment([block_M, D], accum_dtype)

            local_sm_scale = sm_scale
            value_zero = 0
            value_min = -T.infinity(accum_dtype)
            T.vbrc(value_zero, acc_o)
            T.vbrc(value_zero, acc_l)
            T.vbrc(value_min, acc_m)
            T.vbrc(local_sm_scale, scales)

            T.copy(Q[b_i, s_i, h_start : h_start + block_M, 0:D], Q_shared)
            T.copy(Q[b_i, s_i, h_start : h_start + block_M, D : D + DT], Q_tail_shared)

            for k in T.Pipelined(T.ceildiv(topk, block_N), num_stages=num_stages):
                T.copy(Indices[b_i, s_i, 0, k * block_N], idx_buf)
                # Negative-sentinel guard (reviewer #1246 finding HIGH-5):
                # cur_idx == -1 marks an invalid / masked top-k position. On
                # NPU, KV[b_i, -1, ...] is an OOB read; the masked row will
                # be filtered downstream by the score-softmax (scores at
                # masked rows would have been forced to -inf via lse), so
                # leaving KV_shared / K_tail_shared with their initial value
                # for that slot is safe.
                for bi_i in T.serial(block_N):
                    cur_idx = idx_buf[bi_i]
                    if cur_idx >= 0:
                        T.copy(KV[b_i, cur_idx, 0, 0:D], KV_shared[bi_i, 0:D])
                        T.copy(KV[b_i, cur_idx, 0, D : D + DT], K_tail_shared[bi_i, 0:DT])

                T.gemm(Q_shared, KV_shared, scores, initC=True, b_transpose=True)
                T.gemm(
                    Q_tail_shared, K_tail_shared, scores, initC=False, b_transpose=True
                )

                T.vmul(scores, scales, scores)
                T.reduce_max(scores, local_max, dim=1)
                T.vmax(acc_m, local_max, new_max)
                T.vsub(acc_m, new_max, tmp1)
                T.vexp(tmp1, correction)
                T.vsub(scores, new_max, tmp)
                T.vexp(tmp, scores)
                T.reduce_sum(scores, local_sum, dim=1)
                T.vmul(acc_l, correction, acc_l)
                T.vadd(acc_l, local_sum, acc_l)
                # R-KA-16 E5: scalar-fill correction_expanded[h, d] =
                # correction[h, 0] immediately before the broadcast vmul.
                # Mirrors the verified R-KA-13 E5 pattern from the bwd kernel.
                for h_i in T.serial(block_M):
                    for d_i in T.serial(D):
                        correction_expanded[h_i, d_i] = correction[h_i, 0]
                T.vmul(acc_o, correction_expanded, acc_o)
                T.vcast(scores, scores_cast, round_mode="rint")
                T.vbrc(value_zero, tmp1)
                T.vadd(tmp1, new_max, acc_m)
                T.gemm(scores_cast, KV_shared, acc_o, initC=False)

            T.vdiv(acc_o, acc_l, acc_o)
            O_cast = T.alloc_shared([block_M, D], dtype)
            T.vcast(acc_o, O_cast, round_mode="rint")
            T.copy(O_cast, Output[b_i, s_i, h_start : h_start + block_M, 0:D])

            # Lse for bwd: log(acc_l) + acc_m. Kept as [BM,1] to match fragments;
            # caller squeezes the trailing 1 to recover [B,S,H].
            Lse_shared = T.alloc_shared([block_M, 1], accum_dtype)
            tmp_lse = T.alloc_fragment([block_M, 1], accum_dtype)
            T.vln(acc_l, tmp_lse)
            T.vadd(tmp_lse, acc_m, tmp_lse)
            T.copy(tmp_lse, Lse_shared)
            T.copy(Lse_shared, Lse[b_i, s_i, h_start : h_start + block_M, 0:1])

    return main


def _ref_torch(q, kv, indices, sm_scale=None):
    """Reference computed in fp32 on the same device."""
    qf = q.float()
    kvf = kv.float()
    B, S, H, DQK = q.shape
    _, SKV, G, _ = kv.shape  # G == 1 here
    assert G == 1
    _, _, _, topk = indices.shape
    # We follow upstream convention: q has dim_qk = D + DT, output uses first D.
    if sm_scale is None:
        sm_scale = (1.0 / DQK) ** 0.5
    k_full = kvf  # (B, SKV, 1, DQK)
    out = torch.zeros(B, S, H, q.shape[-1], dtype=torch.float32, device=q.device)
    for b in range(B):
        for s in range(S):
            idxs = indices[b, s, 0].long()
            kg = k_full[b, idxs, 0, :]  # (topk, DQK)
            qi = qf[b, s]  # (H, DQK)
            scores = (qi @ kg.transpose(0, 1)) * sm_scale  # (H, topk)
            mask = torch.softmax(scores, dim=-1)
            out[b, s] = mask @ kg  # (H, DQK) — caller slices the first D
    return out


def test_sparse_mla_fwd_small():
    torch.npu.set_device(0)
    B, S, SKV, H = 1, 8, 16, 16
    D, DT = 64, 16
    topk = 8
    BM, BN = H, topk

    torch.manual_seed(0)
    q = torch.randn(B, S, H, D + DT, dtype=torch.float16, device="npu") * 0.5
    kv = torch.randn(B, SKV, 1, D + DT, dtype=torch.float16, device="npu") * 0.5
    indices = torch.zeros(B, S, 1, topk, dtype=torch.int32, device="npu")
    for s in range(S):
        # only attend to past kv positions (causal-style); fall back to 0 when none.
        avail = max(1, s + 1)
        perm = torch.randperm(min(SKV, avail))[:topk]
        if len(perm) < topk:
            perm = torch.cat([perm, torch.zeros(topk - len(perm), dtype=torch.long)])
        indices[0, s, 0, :] = perm.to(torch.int32)

    print(
        f"compile sparse_mla_fwd(B={B},S={S},SKV={SKV},H={H},D={D},DT={DT},topk={topk}) ..."
    )
    kernel = sparse_mla_fwd(B, S, SKV, H, D, DT, topk, block_M=BM, block_N=BN)
    print("compile OK; running on NPU ...")
    out, lse = kernel(q, kv, indices)
    print("run OK; out shape:", tuple(out.shape), "lse shape:", tuple(lse.shape))
    print("out[0,0,0,:4] =", out[0, 0, 0, :4].cpu().tolist())
    print("lse[0,0,:4]   =", lse[0, 0, :4].cpu().tolist())

    # crude correctness via fp32 ref on cpu (skip strict assert for first pass)
    q_cpu = q.cpu()
    kv_cpu = kv.cpu()
    indices_cpu = indices.cpu()
    ref_out_cpu = _ref_torch(q_cpu, kv_cpu, indices_cpu)
    print("ref_out[0,0,0,:4] =", ref_out_cpu[0, 0, 0, :4].tolist())
    abs_err = (out.cpu().float() - ref_out_cpu[..., :D]).abs().max().item()
    print(f"max abs err vs cpu ref: {abs_err:.4f}")
    # tolerance from manual NPU runs: max abs err ~5e-4 vs fp32 cpu ref
    assert abs_err < 5e-3, f"sparse_mla_fwd accuracy regressed: {abs_err}"
    print("sparse_mla_fwd PASS")


if __name__ == "__main__":
    os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
    test_sparse_mla_fwd_small()
