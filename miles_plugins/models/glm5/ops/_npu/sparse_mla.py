# Copyright (c) Huawei Technologies Co., Ltd. 2026.
#
# NPU dispatch for miles' sparse MLA ops (fwd / bwd).
#
# Contracts (match miles' GPU `*_interface` signatures exactly):
#   * npu_sparse_mla_fwd_interface(q, kv, indices, sm_scale=None,
#                                   return_p_sum=False, d_v=512, ...)
#       -> (out [seq, heads, d_v], lse [seq, heads])
#   * npu_sparse_mla_bwd(q, kv, o, do, indices, lse, sm_scale=None,
#                       is_casual=True, return_kernel=False, delta=None)
#       -> (dq, dkv)
#
# Adaptations vs the GPU path:
#   * Miles' kernel signatures take inputs with an implicit batch=1; we wrap to
#     match our kernel's 4D B=1 layout.
#   * Our bwd uses the R-KA-13 E5 workaround already baked into the kernel.
import torch

from ._sparse_mla_fwd_kernel import sparse_mla_fwd as _npu_sparse_mla_fwd
from ._sparse_mla_bwd_kernel import (
    sparse_mla_bwd_preprocess as _npu_preprocess,
    sparse_mla_bwd_postprocess as _npu_postprocess,
    sparse_mla_bwd_main as _npu_bwd_main,
)


def npu_sparse_mla_fwd_interface(
    q,
    kv,
    indices,
    sm_scale=None,
    return_p_sum: bool = False,
    d_v: int = 512,
    block_I: int = 64,
    num_stages: int = 2,
    threads: int = 256,
):
    """Drop-in for miles' sparse_mla_fwd_interface on NPU.

    Input shapes (miles convention):
        q:       [seq, heads, d_v + tail_dim]  fp16
        kv:      [seq_kv, kv_group, d_v + tail_dim]  fp16
        indices: [seq, kv_group, topk]  int32

    Output (squeezed batch):
        out:  [seq, heads, d_v]  fp16
        lse:  [seq, heads]       fp32
    """
    assert return_p_sum is False, "NPU path supports return_p_sum=False only"
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()

    # Inject batch dim to match our kernel's [B, S, H, ...] layout.
    q4 = q.unsqueeze(0)
    kv4 = kv.unsqueeze(0)
    idx4 = indices.unsqueeze(0)

    batch, seq_len, heads, dim_plus_tail = q4.shape
    _, seq_len_kv, kv_group, _ = kv4.shape
    assert kv4.shape[-1] == dim_plus_tail
    assert idx4.shape == (batch, seq_len, kv_group, idx4.shape[-1])
    topk = idx4.shape[-1]
    tail_dim = dim_plus_tail - d_v

    # block_N must divide topk; default cap 64 but tests use smaller topk
    block_N = min(block_I, topk)
    # Pick block_M_inner so the per-block [block_M_inner, d_v] fp32 acc_o
    # fragment stays within ~25% of UB; 16 heads * 512 dim * 4 B = 32 KB
    # leaves room for KV_shared / Q_shared and scores buffers. Must divide
    # heads. For 64 heads we get 4 head-groups; for 16 heads (small smoke
    # tests) we just use the full head count (head_groups=1).
    block_M_inner = 16 if heads % 16 == 0 and heads > 16 else heads
    # num_stages=1 is required for sparse_mla_fwd on NPU: at NS=topk/block_N
    # > 1, the software-pipelined T.Pipelined loop interacts with the online
    # softmax accumulator state and produces NaN output. The R-KA-16 E5 fix
    # in the kernel (correction_expanded) handles the broadcast vmul; the
    # num_stages=1 here handles the multi-buffer scheduling.
    kernel = _npu_sparse_mla_fwd(
        batch=batch,
        seq_len=seq_len,
        seq_len_kv=seq_len_kv,
        heads=heads,
        dim=d_v,
        tail_dim=tail_dim,
        topk=topk,
        block_N=block_N,
        num_stages=1,
        block_M_inner=block_M_inner,
    )
    out4, lse4 = kernel(q4, kv4, idx4)
    out = out4.squeeze(0)
    # lse from kernel is [B, S, H, 1]; miles' upstream lse is [B, S, H]
    lse = lse4.squeeze(0).squeeze(-1)
    return out, lse


def npu_sparse_mla_bwd(
    q,
    kv,
    o,
    do,
    indices,
    lse,
    sm_scale=None,
    is_casual: bool = True,
    return_kernel: bool = False,
    delta=None,
    d_v: int | None = None,
):
    """Drop-in for miles' sparse_mla_bwd on NPU.

    Input shapes (miles convention; batch=1 implicit):
        q:       [seq, heads, d_v + tail_dim]
        kv:      [seq_kv, kv_group, d_v + tail_dim]
        o:       [seq, heads, d_v]
        do:      [seq, heads, d_v]
        indices: [seq, kv_group, topk]
        lse:     [seq, heads]
    """
    # Inject batch dim.
    q4 = q.unsqueeze(0).contiguous()
    kv4 = kv.unsqueeze(0).contiguous()
    o4 = o.unsqueeze(0).contiguous()
    do4 = do.unsqueeze(0).contiguous()
    idx4 = indices.unsqueeze(0).contiguous()
    lse4 = lse.unsqueeze(0).unsqueeze(-1).contiguous()  # [B, S, H, 1]

    B, S, H, dim_plus_tail = q4.shape
    _, S_kv, kv_group, _ = kv4.shape
    # Infer d_v from caller-supplied output shape (`o`) when not given. miles'
    # GPU path uses d_v=512 by convention but we accept any.
    if d_v is None:
        d_v = o.shape[-1]
    D_tail = dim_plus_tail - d_v
    topk = idx4.shape[-1]

    # preprocess kernel: computes delta = sum_d(O * dO)
    if delta is None:
        preprocess_kernel = _npu_preprocess(B, S, H, d_v)
        # our preprocess output is [B, S, H, 1] (trailing 1 for rank parity)
        delta = preprocess_kernel(o4, do4)
    # main bwd kernel computes dq, accumulates dkv in fp32 via atomic_addx4.
    # Our kernel signature is (batch, seq_len, seq_len_kv, heads, dim, tail_dim,
    # topk, block_size=32, num_stages=1) — sm_scale + is_casual + kv_group are
    # not parameters (kv_group=1 baked in; sm_scale auto from D+DT; causal mask
    # via indices ordering). block_size must divide topk.
    #
    # block_size (BS) drives most of the UB pressure:
    #   acc_dkv [BS, d_v] fp32         = 4 * BS * d_v bytes
    #   acc_dkv_shared [BS, d_v] fp32  = 4 * BS * d_v bytes
    #   KV_shared [BS, d_v] fp16       = 2 * BS * d_v bytes
    # For real DSv4 (d_v=512), at BS=32 these three alone are 160 KB and the
    # total kernel goes to ~289 KB (well past the 192 KB UB). T3 measured:
    #   BS=32 -> 289280 B (FAIL CheckUBBudget, FAIL bishengir ub overflow)
    #   BS=16 -> 189888 B (above 80% soft budget, but bishengir still compiles)
    #   BS=8  -> 140192 B (below 80% soft budget — comfortable margin)
    # We pick BS=8 whenever d_v >= 512 to stay under the soft budget; smaller
    # d_v keeps BS=32 for less kernel-launch overhead. The minimum BS is 4
    # (cube-native granularity), never go below.
    if d_v >= 512:
        block_size = min(8, topk)
    else:
        block_size = min(32, topk)
    while topk % block_size != 0 and block_size > 4:
        block_size //= 2
    # Pick block_H_inner to match the fwd kernel's UB-fitting split.
    block_H_inner = 16 if H % 16 == 0 and H > 16 else H
    bwd_kernel = _npu_bwd_main(
        B, S, S_kv, H, d_v, D_tail, topk,
        block_size=block_size, block_H_inner=block_H_inner,
    )
    dkv = torch.zeros_like(kv4, dtype=torch.float32)
    # Kernel signature: Q, KV, dO, Indices, Lse, Delta, dQ, dKV (8 args, no out_idx).
    # dQ must be pre-allocated by the caller.
    dq = torch.zeros_like(q4)
    bwd_kernel(q4, kv4, do4, idx4, lse4, delta, dq, dkv)
    # postprocess cast dkv fp32 -> dtype. Signature: (B, S_kv, dim_plus_tail,
    # block_N=64). block_N must divide S_kv (first-port aligned assumption).
    # UB pressure ~ (4 + 4 + 2) * block_N * (d_v + D_tail) — 10 bytes per
    # row × block_N rows. At d_v=512, D_tail=64, block_N=64 → 368 KB (FAIL).
    # Cap block_N so the cast kernel fits under ~80 KB.
    pp_block_N = 64
    dim_plus_tail = d_v + D_tail
    while pp_block_N > 1 and 10 * pp_block_N * dim_plus_tail > 80 * 1024:
        pp_block_N //= 2
    while S_kv % pp_block_N != 0 and pp_block_N > 1:
        pp_block_N //= 2
    postprocess_kernel = _npu_postprocess(B, S_kv, d_v + D_tail, block_N=pp_block_N)
    dkv = postprocess_kernel(dkv)

    return dq.squeeze(0), dkv.squeeze(0)
