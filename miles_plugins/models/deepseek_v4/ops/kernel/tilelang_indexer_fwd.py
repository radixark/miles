# ruff: noqa
# Adapted from miles_plugins/models/glm5/ops/tilelang_indexer_fwd.py for DeepSeek-V4.
# Key differences from GLM-5:
#   - Operates on [seqlen, batch, heads, dim] (SBHD) layout, batch handled externally
#   - Uses causal mask via cu_seqlens instead of variable-length packed sequences
#   - Supports compressed KV (seq_len_kv = seq_len_q / compress_ratio)
import tilelang
import torch
import torch.nn.functional as F
from tilelang import language as T


_INDEXER_BLOCK_N = 256


def _get_indexer_padded_lengths(seq_len, seq_len_kv, heads, block_n=_INDEXER_BLOCK_N):
    """Return workspace lengths that keep every TileLang load inside its buffers.

    The forward kernel loads a full ``block_n`` KV rows for its last range block,
    while the cleanup kernel iterates in larger blocks and relies on a runtime
    column guard. One extra KV tile is reserved as a scratch tail; callers still
    receive the original logical shape.
    """
    if seq_len < 0 or seq_len_kv < 0:
        raise ValueError(f"sequence lengths must be non-negative, got {seq_len=} {seq_len_kv=}")
    if block_n <= 0:
        raise ValueError(f"block_n must be positive, got {block_n=}")
    block_q = max(1, 128 // heads)
    padded_seq_len = ((seq_len + block_q - 1) // block_q) * block_q if seq_len else 0
    padded_seq_len_kv = ((seq_len_kv + block_n - 1) // block_n + 1) * block_n if seq_len_kv else 0
    return padded_seq_len, padded_seq_len_kv


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def tl_indexer_fwd_impl(
    heads,
    index_dim,
    block_N=256,
    num_stages=3,
    threads=512,
    block_Q=None,
):
    if block_Q is None:
        block_Q = 128 // heads
    dtype = T.bfloat16
    accum_dtype = T.float32
    index_dtype = T.int32

    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    index_q_shape = [seq_len * heads, index_dim]
    index_k_shape = [seq_len_kv, index_dim]
    logits_shape = [seq_len, seq_len_kv]

    @T.prim_func
    def tl_indexer_fwd_kernel(
        IndexQ: T.Tensor(index_q_shape, dtype),  # type: ignore
        IndexK: T.Tensor(index_k_shape, dtype),  # type: ignore
        Logits: T.Tensor(logits_shape, accum_dtype),  # type: ignore
        Weights: T.Tensor([seq_len, heads], accum_dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], index_dtype),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], index_dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len, block_Q), threads=threads) as bx:
            index_q_shared = T.alloc_shared([block_Q * heads, index_dim], dtype)
            index_k_shared = T.alloc_shared([block_N, index_dim], dtype)
            s = T.alloc_fragment([block_N, block_Q * heads], accum_dtype)
            s_reshaped = T.reshape(s, (block_N, block_Q, heads))
            logits = T.alloc_fragment([block_N, block_Q], accum_dtype)
            weights = T.alloc_fragment([block_Q, heads], accum_dtype)

            seq_len_i = bx * block_Q

            cu_k_s_min = T.alloc_var(index_dtype)
            cu_k_e_max = T.alloc_var(index_dtype)

            cu_k_s_min = 2147483647
            cu_k_e_max = -2147483648

            for bq_i in T.serial(block_Q):
                cu_k_s_min = T.min(cu_k_s_min, T.min(CuSeqLenKS[seq_len_i + bq_i], seq_len_kv))
            for bq_i in T.serial(block_Q):
                cu_k_e_max = T.max(cu_k_e_max, T.min(CuSeqLenKE[seq_len_i + bq_i], seq_len_kv))

            T.copy(IndexQ[seq_len_i * heads, 0], index_q_shared)
            T.copy(Weights[seq_len_i, 0], weights)

            for nbn_i in T.Pipelined(T.ceildiv(cu_k_e_max - cu_k_s_min, block_N), num_stages=num_stages):
                T.copy(IndexK[cu_k_s_min + nbn_i * block_N, 0], index_k_shared)

                T.gemm(
                    index_k_shared,
                    index_q_shared,
                    s,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )

                for bn_i, bq_i, h_i in T.Parallel(block_N, block_Q, heads):
                    s_reshaped[bn_i, bq_i, h_i] = T.max(s_reshaped[bn_i, bq_i, h_i], 0) * weights[bq_i, h_i]

                T.reduce_sum(s_reshaped, logits, dim=-1, clear=True)

                for bq_i, bn_i in T.Parallel(block_Q, block_N):
                    Logits[seq_len_i + bq_i, cu_k_s_min + nbn_i * block_N + bn_i] = logits[bn_i, bq_i]

    return tl_indexer_fwd_kernel


@tilelang.jit
def clean_logits_(
    threads: int = 512,
    block_K: int = 4096,
):
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    dtype = T.float
    indices_dtype = T.int32

    @T.prim_func
    def clean_logits_kernel(
        Logits: T.Tensor([seq_len, seq_len_kv], dtype),  # type: ignore
        CuSeqLenKS: T.Tensor([seq_len], indices_dtype),  # type: ignore
        CuSeqLenKE: T.Tensor([seq_len], indices_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len, threads=threads) as bx:
            tx = T.thread_binding(0, threads, thread="threadIdx.x")
            cu_k_s = CuSeqLenKS[bx]
            cu_k_e = CuSeqLenKE[bx]

            for n_i in T.Pipelined(T.ceildiv(seq_len_kv, block_K)):
                for k_i in T.serial(block_K // threads):
                    idx = n_i * block_K + k_i * threads + tx
                    if idx < seq_len_kv and (idx < cu_k_s or idx >= cu_k_e):
                        Logits[bx, idx] = -T.infinity(dtype)

    return clean_logits_kernel


def _make_causal_cu_seqlens(seq_len_q, seq_len_kv, compress_ratio, device):
    """Generate cu_seqlens for causal masking on compressed KV positions.

    For query at position p, valid compressed groups are [0, (p+1) // compress_ratio).
    """
    positions = torch.arange(seq_len_q, device=device, dtype=torch.int32)
    cu_seqlen_ks = torch.zeros(seq_len_q, device=device, dtype=torch.int32)
    cu_seqlen_ke = ((positions + 1) // compress_ratio).to(torch.int32)
    return cu_seqlen_ks, cu_seqlen_ke


def indexer_fwd_interface(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=True):
    """Forward interface matching GLM-5's API but for a single batch element.

    Args:
        q: [seq_len, heads, index_dim] bf16
        kv: [seq_len_kv, index_dim] bf16
        weights: [seq_len, heads] fp32
        cu_seqlen_ks: [seq_len] int32 — start of valid KV range per query
        cu_seqlen_ke: [seq_len] int32 — end of valid KV range per query

    Returns:
        logits: [seq_len, seq_len_kv] fp32
    """
    seq_len, heads, index_dim = q.shape
    seq_len_kv = kv.shape[0]
    logical_seq_len = seq_len
    logical_seq_len_kv = seq_len_kv

    padded_seq_len, padded_seq_len_kv = _get_indexer_padded_lengths(seq_len, seq_len_kv, heads)
    if padded_seq_len == 0 or padded_seq_len_kv == 0:
        return torch.empty([seq_len, seq_len_kv], device=q.device, dtype=torch.float32)

    cu_seqlen_ks = cu_seqlen_ks.to(device=q.device, dtype=torch.int32).clamp(0, logical_seq_len_kv)
    cu_seqlen_ke = cu_seqlen_ke.to(device=q.device, dtype=torch.int32).clamp(0, logical_seq_len_kv)

    if padded_seq_len != logical_seq_len:
        pad_len = padded_seq_len - logical_seq_len
        q = torch.cat((q, q.new_zeros((pad_len, heads, index_dim))), dim=0).contiguous()
        weights = torch.cat((weights, weights.new_zeros((pad_len, heads))), dim=0).contiguous()
        pad_rows = torch.zeros(
            padded_seq_len - logical_seq_len,
            device=q.device,
            dtype=cu_seqlen_ks.dtype,
        ).fill_(logical_seq_len_kv)
        cu_seqlen_ks = torch.cat((cu_seqlen_ks, pad_rows), dim=0)
        cu_seqlen_ke = torch.cat((cu_seqlen_ke, pad_rows), dim=0)
    else:
        q = q.contiguous()
        weights = weights.contiguous()

    if padded_seq_len_kv != logical_seq_len_kv:
        kv = F.pad(kv, (0, 0, 0, padded_seq_len_kv - logical_seq_len_kv), value=0).contiguous()
    else:
        kv = kv.contiguous()

    clean_logits_kernel = clean_logits_()
    tl_indexer_fwd_kernel = tl_indexer_fwd_impl(heads=heads, index_dim=index_dim)

    logits = torch.empty([padded_seq_len, padded_seq_len_kv], device=q.device, dtype=torch.float32)
    tl_indexer_fwd_kernel(
        q.view(padded_seq_len * heads, index_dim),
        kv,
        logits,
        weights.float(),
        cu_seqlen_ks,
        cu_seqlen_ke,
    )
    if clean_logits:
        clean_logits_kernel(logits, cu_seqlen_ks, cu_seqlen_ke)
    return logits[:logical_seq_len, :logical_seq_len_kv]


def batched_indexer_fwd(q, k, weights, cu_seqlen_ks, cu_seqlen_ke):
    """Batched forward: loops over batch dim.

    Args:
        q: [seqlen, batch, heads, dim] bf16
        k: [seqlen_kv, batch, dim] bf16
        weights: [seqlen, batch, heads] fp32
        cu_seqlen_ks: [seqlen] int32
        cu_seqlen_ke: [seqlen] int32

    Returns:
        logits: [batch, seqlen, seqlen_kv] fp32
    """
    seqlen, batch, heads, dim = q.shape
    seq_len_kv = k.shape[0]

    all_logits = torch.empty([batch, seqlen, seq_len_kv], device=q.device, dtype=torch.float32)
    for b in range(batch):
        all_logits[b] = indexer_fwd_interface(
            q[:, b, :, :].contiguous(),
            k[:, b, :].contiguous(),
            weights[:, b, :].contiguous(),
            cu_seqlen_ks,
            cu_seqlen_ke,
        )
    return all_logits
