import torch
import triton
import triton.language as tl


@triton.jit
def _store_neg_inf(out_rows, row_ok, col_start, col_stop, BLOCK_N: tl.constexpr, BLOCK_Q: tl.constexpr):
    neg_inf = tl.full([BLOCK_Q, BLOCK_N], -float("inf"), tl.float32)
    for n_start in range(col_start, col_stop, BLOCK_N):
        cols = n_start + tl.arange(0, BLOCK_N)
        tl.store(out_rows + cols[None, :], neg_inf, mask=row_ok[:, None] & (cols < col_stop)[None, :])


@triton.jit
def _indexer_fwd_kernel(
    q_ptr,
    k_ptr,
    weights_ptr,
    ks_ptr,
    ke_ptr,
    out_ptr,
    seq_len,
    seq_len_kv,
    stride_out,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CLEAN_LOGITS: tl.constexpr,
):
    q_start = tl.program_id(0) * BLOCK_Q
    rows = q_start + tl.arange(0, BLOCK_Q)
    row_ok = rows < seq_len
    ks = tl.load(ks_ptr + rows, mask=row_ok, other=seq_len_kv)
    ke = tl.load(ke_ptr + rows, mask=row_ok, other=0)
    ks = tl.minimum(ks, seq_len_kv)
    ke = tl.minimum(ke, seq_len_kv)
    lo = tl.min(ks, 0) // BLOCK_N * BLOCK_N
    hi = tl.maximum(tl.max(ke, 0), lo)

    head_rows = tl.arange(0, BLOCK_Q * HEADS)
    head_row_ok = q_start + head_rows // HEADS < seq_len
    dims = tl.arange(0, DIM)
    q = tl.load(
        q_ptr + (q_start * HEADS + head_rows).to(tl.int64)[:, None] * DIM + dims[None, :],
        mask=head_row_ok[:, None],
        other=0.0,
    )
    weights = tl.load(weights_ptr + q_start * HEADS + head_rows, mask=head_row_ok, other=0.0)

    out_rows = out_ptr + rows.to(tl.int64)[:, None] * stride_out
    for n_start in range(lo, hi, BLOCK_N):
        cols = n_start + tl.arange(0, BLOCK_N)
        col_ok = cols < seq_len_kv
        k = tl.load(k_ptr + cols.to(tl.int64)[:, None] * DIM + dims[None, :], mask=col_ok[:, None], other=0.0)
        scores = tl.maximum(tl.dot(k, tl.trans(q)), 0.0) * weights[None, :]
        logits = tl.trans(tl.sum(tl.reshape(scores, [BLOCK_N, BLOCK_Q, HEADS]), axis=2))
        if CLEAN_LOGITS:
            in_range = (cols[None, :] >= ks[:, None]) & (cols[None, :] < ke[:, None])
            logits = tl.where(in_range, logits, -float("inf"))
        tl.store(out_rows + cols[None, :], logits, mask=row_ok[:, None] & col_ok[None, :])
    if CLEAN_LOGITS:
        _store_neg_inf(out_rows, row_ok, 0, lo, BLOCK_N, BLOCK_Q)
        _store_neg_inf(out_rows, row_ok, hi, seq_len_kv, BLOCK_N, BLOCK_Q)


def indexer_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    out: torch.Tensor,
    *,
    clean_logits: bool,
    block_rows: int,
    block_n: int,
    num_warps: int,
    num_stages: int,
) -> torch.Tensor:
    """Writes the [ks, ke) scores of every query into out [T, T_kv] (row stride free, columns contiguous).
    Columns outside a query's range hold -inf when clean_logits, otherwise whatever out held or partial
    scores of its query block. block_rows = queries per program x heads, the MMA's N dimension."""
    assert q.is_contiguous() and k.is_contiguous() and weights.is_contiguous() and out.stride(-1) == 1
    seq_len, heads, dim = q.shape
    block_q = max(1, block_rows // heads)
    _indexer_fwd_kernel[(triton.cdiv(seq_len, block_q),)](
        q,
        k,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        out,
        seq_len,
        k.shape[0],
        out.stride(0),
        HEADS=heads,
        DIM=dim,
        BLOCK_Q=block_q,
        BLOCK_N=block_n,
        CLEAN_LOGITS=clean_logits,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
