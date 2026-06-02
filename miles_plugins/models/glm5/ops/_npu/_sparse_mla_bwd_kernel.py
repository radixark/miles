# Copyright (c) Huawei Technologies Co., Ltd. 2026.
#
# Port of upstream tile-ai/tilelang examples/deepseek_v32/sparse_mla_bwd.py
# to Ascend NPU via mlir-ascend. Used by miles (radixark/miles) at
# miles_plugins/models/glm5/ops/tilelang_sparse_mla_bwd.py.
#
# Three kernels match upstream:
#   1. preprocess: delta = sum_d(O[..,d] * dO[..,d]) over D axis
#   2. bwd: main grad — produces dQ + dKV (fp32 accumulator)
#   3. postprocess: cast dKV fp32 -> bf16/fp16
#
# Adaptations vs upstream:
#   * 3-axis grid (S, B, kv_group*NH) -> 1-axis (is_npu=True). With kv_group=1
#     and NH=1 the decomposition is trivial: cid encodes (b_i, s_i).
#   * NPU vector intrinsics (vbrc/vmul/vexp/...) replace T.Parallel loops.
#   * b_transpose= / a_transpose= for T.gemm (Ascend signature).
#   * T.atomic_addx4 path kept identical — Ascend exposes npuir_atomic_addx4.
#
# Carry-overs from sparse_mla_fwd that apply here too:
#   * scalar literals -> bind to local Var before T.vbrc
#   * 3D outputs that need [BM,1] fragment slicing -> use [B,S,H,1] shape
import os
import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1], target="npuir")
def sparse_mla_bwd_preprocess(
    batch,
    seq_len,
    heads,
    dim,
    block_ND=32,
    num_stages=2,
):
    """Compute Delta[b,s,h] = sum_d O[b,s,h,d] * dO[b,s,h,d].

    Upstream uses 3-axis grid (H, ceil(S/block_ND), B). We collapse to a
    single axis: each tile owns one (b_i, s_block, h_i) cell, iterates
    over D via Pipelined; output has trailing 1 for rank parity.
    """
    dtype = "float16"  # NPU starting point; bf16 follows
    accum_dtype = "float32"
    shape = [batch, seq_len, heads, dim]
    delta_shape = [batch, seq_len, heads, 1]
    NS = (seq_len + block_ND - 1) // block_ND
    ND = dim // block_ND
    assert dim % block_ND == 0, f"dim {dim} must be divisible by block_ND {block_ND}"

    @T.prim_func
    def preprocess(
        O: T.Tensor(shape, dtype),
        dO: T.Tensor(shape, dtype),
        Delta: T.Tensor(delta_shape, accum_dtype),
    ):
        with T.Kernel(batch * heads * NS, is_npu=True) as (cid, _):
            # cid = b_i * (heads * NS) + h_i * NS + s_block
            b_i = cid // (heads * NS)
            rem = cid % (heads * NS)
            h_i = rem // NS
            s_block = rem % NS

            o = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            do = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            o_f16 = T.alloc_shared([block_ND, block_ND], dtype)
            do_f16 = T.alloc_shared([block_ND, block_ND], dtype)
            acc = T.alloc_fragment([block_ND, block_ND], accum_dtype)
            delta = T.alloc_fragment([block_ND, 1], accum_dtype)
            delta_shared = T.alloc_shared([block_ND, 1], accum_dtype)
            value_zero = 0
            T.vbrc(value_zero, acc)

            for k in T.Pipelined(ND, num_stages=num_stages):
                T.copy(
                    O[b_i, s_block * block_ND : (s_block + 1) * block_ND, h_i,
                      k * block_ND : (k + 1) * block_ND],
                    o_f16,
                )
                T.copy(
                    dO[b_i, s_block * block_ND : (s_block + 1) * block_ND, h_i,
                       k * block_ND : (k + 1) * block_ND],
                    do_f16,
                )
                T.vcast(o_f16, o, round_mode="rint")
                T.vcast(do_f16, do, round_mode="rint")
                T.vmul(o, do, o)  # in-place: o = o * do
                T.vadd(acc, o, acc)

            T.reduce_sum(acc, delta, dim=1)
            T.copy(delta, delta_shared)
            T.copy(delta_shared,
                   Delta[b_i, s_block * block_ND : (s_block + 1) * block_ND, h_i, 0:1])

    return preprocess


@tilelang.jit(out_idx=[-1], target="npuir")
def sparse_mla_bwd_postprocess(
    batch,
    seq_len_kv,
    dim_plus_tail,
    block_N=64,
):
    """Cast accumulator dKV (fp32) -> dtype (fp16). kv_group=1 baked in."""
    dtype = "float16"
    accum_dtype = "float32"
    shape = [batch, seq_len_kv, 1, dim_plus_tail]
    NS = (seq_len_kv + block_N - 1) // block_N
    assert seq_len_kv % block_N == 0, "first port requires aligned seq_len_kv"

    @T.prim_func
    def postprocess(
        dKV_acc: T.Tensor(shape, accum_dtype),
        dKV_out: T.Tensor(shape, dtype),
    ):
        with T.Kernel(batch * NS, is_npu=True) as (cid, _):
            b_i = cid // NS
            s_block = cid % NS
            acc_buf = T.alloc_shared([block_N, dim_plus_tail], accum_dtype)
            out_buf = T.alloc_shared([block_N, dim_plus_tail], dtype)
            tmp = T.alloc_fragment([block_N, dim_plus_tail], accum_dtype)
            T.copy(
                dKV_acc[b_i, s_block * block_N : (s_block + 1) * block_N, 0, :],
                acc_buf,
            )
            # cast via fragment so vcast can lower
            T.copy(acc_buf, tmp)
            T.vcast(tmp, out_buf, round_mode="rint")
            T.copy(
                out_buf,
                dKV_out[b_i, s_block * block_N : (s_block + 1) * block_N, 0, :],
            )

    return postprocess


@tilelang.jit(
    target="npuir",
    pass_configs={
        # Disable auto multi-buffer to reduce live state for this complex
        # bwd (7+ GEMMs + atomic scatter). Mirrors the working pattern in
        # the lighting_indexer_bwd kernel.
        "npuir.enable_auto_multi_buffer": False,
    },
)
def sparse_mla_bwd_main(
    batch,
    seq_len,
    seq_len_kv,
    heads,
    dim,
    tail_dim,
    topk,
    block_size=32,
    num_stages=1,
    block_H_inner=None,
):
    """Main bwd kernel (kv_group=1, NH=1 first pass).

    Inputs: Q, KV, dO, Indices, Lse, Delta + pre-zeroed dKV (fp32 accumulator).
    Outputs: dQ (fp16), dKV (fp32, in-place via atomic_addx4).

    First pass omits split_store (writes the full BS block in one atomic
    sweep) and keeps a single Pipelined loop without aggressive shared-mem
    merge. Optimization passes can layer in once correctness is proven.
    """
    D = dim
    DT = tail_dim
    dtype = "float16"
    accum_dtype = "float32"
    idx_dtype = "int32"
    sm_scale = (1.0 / (D + DT)) ** 0.5
    # NOTE: the companion fwd kernel uses natural-base exp / log (vexp / vln), so
    # Lse from fwd is in natural log. We MUST stay consistent in bwd —
    # otherwise acc_p collapses to ~0 and all gradients vanish.
    # Upstream CUDA fwd uses exp2 + log2 (log2-base everywhere); we don't.
    sm_scale_mul_log2e = sm_scale  # no log2 conversion; keep natural base
    use_natural_base = True  # cf. fwd kernel's Lse convention
    BS = block_size
    if block_H_inner is None:
        block_H_inner = heads
    assert heads % block_H_inner == 0, (
        f"block_H_inner={block_H_inner} must divide heads={heads}"
    )
    head_groups = heads // block_H_inner
    block_H = block_H_inner  # per-block head tile — UB-bounded
    NS = (topk + BS - 1) // BS
    assert topk % BS == 0, f"topk {topk} must be divisible by block_size {BS}"

    q_shape = [batch, seq_len, heads, D + DT]
    k_shape = [batch, seq_len_kv, 1, D + DT]
    o_shape = [batch, seq_len, heads, D]
    idx_shape = [batch, seq_len, 1, topk]
    delta_shape = [batch, seq_len, heads, 1]
    lse_shape = [batch, seq_len, heads, 1]

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        KV: T.Tensor(k_shape, dtype),
        dO: T.Tensor(o_shape, dtype),
        Indices: T.Tensor(idx_shape, idx_dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
        Delta: T.Tensor(delta_shape, accum_dtype),
        dQ: T.Tensor(q_shape, dtype),
        dKV: T.Tensor(k_shape, accum_dtype),
    ):
        # Grid: batch * seq_len * head_groups. See sparse_mla_fwd_kernel.py
        # for the rationale — keeps per-block UB footprint bounded by
        # block_H_inner so acc_dq [block_H_inner, D] fp32 fits within
        # the 192 KB dav-c220 UB.
        with T.Kernel(batch * seq_len * head_groups, is_npu=True) as (cid, _):
            b_i = cid // (seq_len * head_groups)
            rem = cid % (seq_len * head_groups)
            s_i = rem // head_groups
            hg_i = rem % head_groups
            h_start = hg_i * block_H

            Q_shared = T.alloc_shared([block_H, D], dtype)
            Q_tail_shared = T.alloc_shared([block_H, DT], dtype)
            KV_shared = T.alloc_shared([BS, D], dtype)
            KV_tail_shared = T.alloc_shared([BS, DT], dtype)
            dO_shared = T.alloc_shared([block_H, D], dtype)
            P_shared_cast = T.alloc_shared([block_H, BS], dtype)
            dP_shared_cast = T.alloc_shared([block_H, BS], dtype)
            dQ_shared = T.alloc_shared([block_H, D], dtype)
            dQ_tail_shared = T.alloc_shared([block_H, DT], dtype)
            Lse_shared = T.alloc_shared([block_H, 1], accum_dtype)
            Delta_shared = T.alloc_shared([block_H, 1], accum_dtype)

            acc_p = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dp = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dq = T.alloc_fragment([block_H, D], accum_dtype)
            acc_dq_tail = T.alloc_fragment([block_H, DT], accum_dtype)
            acc_dkv = T.alloc_fragment([BS, D], accum_dtype)
            acc_dkv_tail = T.alloc_fragment([BS, DT], accum_dtype)
            acc_dkv_shared = T.alloc_shared([BS, D], accum_dtype)
            acc_dkv_tail_shared = T.alloc_shared([BS, DT], accum_dtype)
            idx_buf = T.alloc_fragment([BS], idx_dtype)

            lse_frag = T.alloc_fragment([block_H, 1], accum_dtype)
            delta_frag = T.alloc_fragment([block_H, 1], accum_dtype)
            delta_scaled = T.alloc_fragment([block_H, 1], accum_dtype)
            sm_scale_BH1 = T.alloc_fragment([block_H, 1], accum_dtype)
            neg_delta_frag = T.alloc_fragment([block_H, 1], accum_dtype)
            neg_one_frag_H1 = T.alloc_fragment([block_H, 1], accum_dtype)
            lse_expanded = T.alloc_fragment([block_H, BS], accum_dtype)
            delta_expanded = T.alloc_fragment([block_H, BS], accum_dtype)
            tmp_HB = T.alloc_fragment([block_H, BS], accum_dtype)
            sm_scale_buf = T.alloc_fragment([block_H, BS], accum_dtype)
            sm_log2e = sm_scale_mul_log2e
            sm_scale_local = sm_scale
            value_zero = 0
            T.vbrc(sm_scale_local, sm_scale_buf)
            T.vbrc(value_zero, acc_dq)
            T.vbrc(value_zero, acc_dq_tail)

            T.copy(Q[b_i, s_i, h_start : h_start + block_H, 0:D], Q_shared)
            T.copy(Q[b_i, s_i, h_start : h_start + block_H, D : D + DT], Q_tail_shared)
            T.copy(dO[b_i, s_i, h_start : h_start + block_H, 0:D], dO_shared)
            T.copy(Lse[b_i, s_i, h_start : h_start + block_H, 0:1], Lse_shared)
            T.copy(Delta[b_i, s_i, h_start : h_start + block_H, 0:1], Delta_shared)
            T.copy(Lse_shared, lse_frag)
            T.copy(Delta_shared, delta_frag)
            neg_one_val = -1.0
            T.vbrc(neg_one_val, neg_one_frag_H1)
            T.vbrc(sm_scale_local, sm_scale_BH1)

            for k in T.Pipelined(NS, num_stages=num_stages):
                # Gather KV via indices.
                # Negative-sentinel guard (reviewer #1246 finding HIGH-3):
                # idx_buf[bi_i] can be -1 for masked positions; reading
                # KV[b_i, -1, ...] is OOB. KV_shared / KV_tail_shared are
                # carried across iters but only meaningfully consumed for
                # valid (>=0) rows; leaving the masked slot with stale data
                # is correct because the corresponding acc_p row will be
                # exp(-inf)≈0 via the lse subtraction (lse came from the
                # same masked-aware fwd path).
                T.copy(Indices[b_i, s_i, 0, k * BS], idx_buf)
                for bi_i in T.serial(BS):
                    cur_idx = idx_buf[bi_i]
                    if cur_idx >= 0:
                        T.copy(KV[b_i, cur_idx, 0, 0:D], KV_shared[bi_i, 0:D])
                        T.copy(KV[b_i, cur_idx, 0, D : D + DT], KV_tail_shared[bi_i, 0:DT])

                # 1) compute attention scores acc_p = Q @ K^T (split D+DT)
                T.gemm(Q_shared, KV_shared, acc_p, initC=True, b_transpose=True)
                T.gemm(Q_tail_shared, KV_tail_shared, acc_p, initC=False, b_transpose=True)
                # acc_p = exp(acc_p * sm_scale - Lse)
                # R-KA-13 E5 schedule-locality: keep the broadcast scalar-mul
                # immediately before the lse vsub to preserve register-layout adjacency.
                T.vbrc(sm_log2e, tmp_HB)
                T.vmul(acc_p, tmp_HB, acc_p)
                for h_i in T.serial(block_H):
                    for bi_i in T.serial(BS):
                        lse_expanded[h_i, bi_i] = lse_frag[h_i, 0]
                T.vsub(acc_p, lse_expanded, acc_p)
                T.vexp(acc_p, acc_p)
                T.vcast(acc_p, P_shared_cast, round_mode="rint")

                # 2) compute dP = dO @ KV^T  (only D channel; tail dropped per upstream)
                T.gemm(dO_shared, KV_shared, acc_dp, initC=True, b_transpose=True)
                # 3) acc_dp = (acc_dp - delta) * sm_scale * acc_p
                # R-KA-13 WORKAROUND (E5, verified PASS): Python-loop fill
                # delta_expanded RIGHT BEFORE the vsub, mirroring the working
                # lse pattern. Originally vsub(acc_dp, delta_frag, acc_dp) silently
                # zeroed; placing the scalar-fill of delta_expanded[h,b]=delta_frag[h,0]
                # immediately before vsub keeps the scheduler in the same
                # fragment-register-layout iteration as the gemm output, so the
                # subtraction takes effect. Result: dQ cosine 0.93 vs autograd
                # (was 0.53 with vsub omitted).
                for h_i in T.serial(block_H):
                    for bi_i in T.serial(BS):
                        delta_expanded[h_i, bi_i] = delta_frag[h_i, 0]
                T.vsub(acc_dp, delta_expanded, acc_dp)
                T.vmul(acc_dp, sm_scale_buf, acc_dp)
                T.vmul(acc_dp, acc_p, acc_dp)
                T.vcast(acc_dp, dP_shared_cast, round_mode="rint")

                # 4) dQ += dP @ K  (split D+DT path)
                T.gemm(dP_shared_cast, KV_shared, acc_dq, initC=False)
                T.gemm(dP_shared_cast, KV_tail_shared, acc_dq_tail, initC=False)

                # 5) acc_dkv = dP^T @ Q (clear) ; acc_dkv += P^T @ dO
                T.gemm(dP_shared_cast, Q_shared, acc_dkv, initC=True, a_transpose=True)
                T.gemm(P_shared_cast, dO_shared, acc_dkv, initC=False, a_transpose=True)
                # 6) acc_dkv_tail = dP^T @ Q_tail
                T.gemm(dP_shared_cast, Q_tail_shared, acc_dkv_tail, initC=True, a_transpose=True)

                # 7) atomic_addx4 dKV (single full-BS block, no split_store)
                # Negative-sentinel guard (reviewer #1246 finding HIGH-4):
                # cur_idx == -1 must NOT trigger a scatter-add to dKV[-1, ...],
                # which would corrupt the last row or raise an AICore
                # exception. The masked positions contribute zero gradient
                # mathematically (their attention weight was zero), so
                # skipping the atomic add is the correct behavior, not a
                # workaround.
                T.copy(acc_dkv, acc_dkv_shared)
                T.copy(acc_dkv_tail, acc_dkv_tail_shared)
                # write back to global dKV at the indices we gathered from
                for bi_i in T.serial(BS):
                    cur_idx = idx_buf[bi_i]
                    if cur_idx >= 0:
                        # size=[4]: without it, atomic_addx4 only fires for 1
                        # element per call (single-index dst infers extent=[1]).
                        for d_i in T.serial(D // 4):
                            T.npuir_atomic_addx4(
                                dKV[b_i, cur_idx, 0, d_i * 4],
                                acc_dkv_shared[bi_i, d_i * 4],
                                size=[4],
                            )
                        for d_i in T.serial(DT // 4):
                            T.npuir_atomic_addx4(
                                dKV[b_i, cur_idx, 0, D + d_i * 4],
                                acc_dkv_tail_shared[bi_i, d_i * 4],
                                size=[4],
                            )

            # NOTE: topk=16 (NS=2) currently fails bisheng AICore-resource exit 70.
            # Live-state inventory analysis 2026-05-19:
            #   - acc_dq [16,D] + acc_dq_tail [16,DT] persistent across iters (initC=False)
            #   - acc_dkv / acc_dkv_tail re-init each iter (initC=True) — already minimal
            #   - 4 broadcast [block_H, BS] fragments (lse_expanded, delta_expanded,
            #     tmp_HB, sm_scale_buf) — candidates for compaction
            # Recipe to try at NS=2 (record-only; NPU validation required before commit):
            #   1. Replace `T.vbrc(sm_log2e, tmp_HB); T.vmul(acc_p, tmp_HB, acc_p)`
            #      with scalar `T.vmul(acc_p, sm_log2e, acc_p)` (per customize_npuir.py
            #      docstring: B can be scalar). Saves ~512 bytes live state.
            #   2. Same for sm_scale_buf if we can do `T.vmul(acc_dp, sm_scale_local, acc_dp)`.
            #   3. Last resort: skip pipelining at NS=2 by setting num_stages=1 explicitly
            #      (already default in this kernel — verify caller passes num_stages=1).
            # Tracked in task #251.
            #
            # DIAG: write acc_dq (dP@K result, expected non-zero if dP is non-zero)
            # to dQ[..., 0:D] and acc_dq_tail (P@K_tail result, expected non-zero
            # always since P is known non-zero) to dQ[..., D:D+DT].
            # If dQ[..., 0:D] is zero but dQ[..., D:D+DT] is non-zero, dP_shared_cast=0.
            T.vcast(acc_dq, dQ_shared, round_mode="rint")
            T.vcast(acc_dq_tail, dQ_tail_shared, round_mode="rint")
            T.copy(dQ_shared, dQ[b_i, s_i, h_start : h_start + block_H, 0:D])
            T.copy(dQ_tail_shared, dQ[b_i, s_i, h_start : h_start + block_H, D : D + DT])

    return main


def _smoke_preprocess_and_postprocess():
    """Compile-only smoke for the two helper kernels."""
    torch.npu.set_device(0)
    print("compile preprocess ...")
    pp = sparse_mla_bwd_preprocess(batch=1, seq_len=8, heads=16, dim=64)
    print("preprocess compile OK")
    print("compile postprocess ...")
    pq = sparse_mla_bwd_postprocess(batch=1, seq_len_kv=16, dim_plus_tail=80, block_N=16)
    print("postprocess compile OK")

    O = torch.randn(1, 8, 16, 64, dtype=torch.float16, device="npu")
    dO = torch.randn(1, 8, 16, 64, dtype=torch.float16, device="npu")
    delta = pp(O, dO)
    print("preprocess run OK; delta shape:", tuple(delta.shape))

    dKV_acc = torch.randn(1, 16, 1, 80, dtype=torch.float32, device="npu")
    dKV_out = pq(dKV_acc)
    print("postprocess run OK; dKV_out shape:", tuple(dKV_out.shape), "dtype:", dKV_out.dtype)


def _smoke_main_bwd():
    """End-to-end bwd smoke: call all 3 kernels, compare dQ + dKV against
    PyTorch autograd on a fp32 dense-attention reference."""
    torch.npu.set_device(0)
    B, S, SKV, H = 1, 8, 16, 16
    D, DT = 64, 16
    topk = 8  # NS=1; topk=16 (NS=2) currently fails bisheng codegen — needs
              # resource analysis (HIVM IR fails at AICore register limit).
    BS = 8

    torch.manual_seed(42)
    q = torch.randn(B, S, H, D + DT, dtype=torch.float16, device="npu") * 0.5
    kv = torch.randn(B, SKV, 1, D + DT, dtype=torch.float16, device="npu") * 0.5
    indices = torch.zeros(B, S, 1, topk, dtype=torch.int32, device="npu")
    for s in range(S):
        avail = max(1, s + 1)
        perm = torch.randperm(min(SKV, avail))[:topk]
        if len(perm) < topk:
            perm = torch.cat([perm, torch.zeros(topk - len(perm), dtype=torch.long)])
        indices[0, s, 0, :] = perm.to(torch.int32)
    dO = torch.randn(B, S, H, D, dtype=torch.float16, device="npu") * 0.1

    # Use the fwd we already validated to get O + Lse.
    from examples.deepseek_v4.example_sparse_mla_fwd_kernel import sparse_mla_fwd

    print("compile fwd to obtain O + Lse ...")
    fwd_k = sparse_mla_fwd(B, S, SKV, H, D, DT, topk, block_M=H, block_N=BS)
    O, Lse = fwd_k(q, kv, indices)
    print("fwd O shape:", tuple(O.shape), "Lse shape:", tuple(Lse.shape))

    print("compile bwd preprocess ...")
    pp_k = sparse_mla_bwd_preprocess(B, S, H, D, block_ND=8)
    print("compile bwd postprocess ...")
    pq_k = sparse_mla_bwd_postprocess(B, SKV, D + DT, block_N=SKV)
    print("compile bwd main ...")
    bwd_k = sparse_mla_bwd_main(
        B, S, SKV, H, D, DT, topk, block_size=BS, num_stages=1,
    )
    print("compile all 3 bwd kernels OK")

    # Run preprocess to get Delta
    Delta = pp_k(O, dO)
    print("preprocess Delta shape:", tuple(Delta.shape))
    print(f"  Delta[0,0,0,0] = {Delta[0,0,0,0].item():.6f}")
    print(f"  Delta[0,0,:4,0] = {Delta[0,0,:4,0].cpu().tolist()}")
    # Reference Delta = sum_d O * dO
    Delta_ref = (O.cpu().float() * dO.cpu().float()).sum(dim=-1, keepdim=True)
    print(f"  Delta_ref[0,0,:4,0] = {Delta_ref[0,0,:4,0].tolist()}")

    # Allocate ALL outputs externally (no out_idx; per R-KA-12).
    dQ = torch.zeros_like(q)
    dKV_acc = torch.zeros(B, SKV, 1, D + DT, dtype=torch.float32, device="npu")

    # Run main bwd
    print("running main bwd ...")
    bwd_k(q, kv, dO, indices, Lse, Delta, dQ, dKV_acc)
    print("dQ shape:", tuple(dQ.shape), "dKV_acc finite ratio:",
          (torch.isfinite(dKV_acc).sum() / dKV_acc.numel()).item())
    dQ_out = dQ

    # Run postprocess to cast dKV
    dKV_out = pq_k(dKV_acc)
    print("dKV cast shape:", tuple(dKV_out.shape), "dtype:", dKV_out.dtype)
    print("dQ_out[0,0,0,:4]  =", dQ_out[0, 0, 0, :4].cpu().tolist())
    print("dQ_out[0,0,0,D:D+4] =", dQ_out[0, 0, 0, 64:68].cpu().tolist())

    # Quantitative bias check on CPU (autograd) — quick comparison only.
    q_r = q.detach().cpu().float().requires_grad_(True)
    kv_r = kv.detach().cpu().float().requires_grad_(True)
    indices_cpu = indices.cpu()
    dO_cpu = dO.cpu().float()
    scores = torch.einsum("bshd,bkgd->bshk", q_r, kv_r) * (1.0 / (D + DT)) ** 0.5
    mask = torch.zeros(B, S, H, SKV)
    for b in range(B):
        for s in range(S):
            for k_pos in range(topk):
                kv_idx = indices_cpu[b, s, 0, k_pos].item()
                if 0 <= kv_idx < SKV:
                    mask[b, s, :, kv_idx] = 1.0
    scores_masked = scores.masked_fill(mask == 0, float("-inf"))
    P_softmax = scores_masked.softmax(dim=-1)
    v_dim = kv_r[:, :, :, :D]
    O_ref = torch.einsum("bshk,bkgd->bshd", P_softmax, v_dim)
    loss = (O_ref * dO_cpu).sum()
    loss.backward()

    dQ_ref = q_r.grad
    dKV_ref = kv_r.grad
    dq_kernel = dQ_out[..., :D].cpu().float()
    err_q_d = (dq_kernel - dQ_ref[..., :D]).abs().max().item()
    ref_max = dQ_ref[..., :D].abs().max().item()
    rel_err_d = err_q_d / (ref_max + 1e-8)
    cos_sim_d = torch.nn.functional.cosine_similarity(
        dq_kernel.flatten(), dQ_ref[..., :D].flatten(), dim=0
    ).item()
    print(f"\n=== dQ bias check (D channels only; R-KA-13.E5 experiment: vsub with delta_expanded scalar-fill immediately before) ===")
    print(f"dQ kernel max abs:  {dq_kernel.abs().max().item():.5f}")
    print(f"dQ ref    max abs:  {ref_max:.5f}")
    print(f"max abs err:        {err_q_d:.5f}  (rel: {rel_err_d:.3f})")
    print(f"cosine similarity:  {cos_sim_d:.4f}")
    print(f"dq_kernel/ref ratio (max abs): {dq_kernel.abs().max().item() / (ref_max + 1e-8):.3f}")
    print("dKV_out[0,0,0,:4] =", dKV_out[0, 0, 0, :4].cpu().tolist())


if __name__ == "__main__":
    os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
    _smoke_preprocess_and_postprocess()
    print("---")
    _smoke_main_bwd()
