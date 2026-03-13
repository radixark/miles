import torch

from . import tilelang_sparse_mla_bwd as sparse_mla_bwd
from . import tilelang_sparse_mla_fwd as sparse_mla_fwd


def sparse_attn_torch(q, kv, attn_sink, topk_idxs, sm_scale=None):
    """
    Args:
        q: (b, m, h, d)
        kv: (b, n, d)
        attn_sink: (h,)
        topk_idxs: (b, m, topk)
        sm_scale: float
    Returns:
        o: (b, m, h, d)
    """
    q = q.float()
    kv = kv.float()

    b, m, h, d = q.shape
    k_len = kv.shape[1]
    _, _, topk = topk_idxs.shape

    assert (topk_idxs < k_len).all(), f"topk_idxs should be smaller than length of k: {k_len}, but got {topk_idxs}"

    if sm_scale is None:
        sm_scale = (1.0 / d) ** 0.5

    mask = topk_idxs != -1
    safe_idxs = topk_idxs.masked_fill(~mask, 0)

    batch_idx = torch.arange(b, device=q.device).view(b, 1, 1)

    kv_gathered = kv[batch_idx, safe_idxs]

    scores = torch.einsum("bmhd,bmkd->bmhk", q, kv_gathered)

    scores = scores * sm_scale
    mask_expanded = mask.unsqueeze(2).expand(-1, -1, h, -1)
    scores = scores.masked_fill(~mask_expanded, float("-inf"))

    scores = scores.to(torch.float32)
    scores_max = scores.max(dim=-1).values
    exp_scores = torch.exp(scores - scores_max.unsqueeze(-1))

    numerator = torch.einsum("bmhk,bmkd->bmhd", exp_scores, kv_gathered.to(torch.float32))

    sum_exp = exp_scores.sum(dim=-1)
    sink_term = torch.exp(attn_sink.view(1, 1, h) - scores_max)
    denominator = sum_exp + sink_term

    assert exp_scores.dtype == torch.float32
    assert scores_max.dtype == torch.float32
    assert sum_exp.dtype == torch.float32
    assert sink_term.dtype == torch.float32

    o = numerator / denominator.unsqueeze(-1)

    return o.to(q.dtype)


def dense_attn_torch(q, kv, attn_sink, topk_idxs, sm_scale=None):
    """
    Dense GEMM implementation: converts topk_idxs to a sparse mask and computes
    full Q @ K^T attention, then applies mask. No gather operations.
    """
    b, m, h, d = q.shape
    n = kv.shape[1]
    _, _, topk = topk_idxs.shape

    if sm_scale is None:
        sm_scale = (1.0 / d) ** 0.5

    attn_mask = torch.zeros(b, m, n, device=q.device, dtype=torch.bool)

    batch_idx = torch.arange(b, device=q.device).view(b, 1, 1).expand(b, m, topk)
    seq_idx = torch.arange(m, device=q.device).view(1, m, 1).expand(b, m, topk)

    valid_mask = topk_idxs != -1

    valid_batch = batch_idx[valid_mask]
    valid_seq = seq_idx[valid_mask]
    valid_kv_idx = topk_idxs[valid_mask].long()
    attn_mask[valid_batch, valid_seq, valid_kv_idx] = True

    scores = torch.einsum("bmhd,bnd->bmhn", q, kv).to(torch.float32) * sm_scale

    attn_mask_expanded = attn_mask.unsqueeze(2).expand(-1, -1, h, -1)
    scores = scores.masked_fill(~attn_mask_expanded, float("-inf"))

    scores_max = scores.max(dim=-1, keepdim=True).values
    scores_max = scores_max.clamp(min=-1e30)

    exp_scores = torch.exp(scores - scores_max)

    numerator = torch.einsum("bmhn,bnd->bmhd", exp_scores, kv.float())

    sum_exp = exp_scores.sum(dim=-1)
    sink_term = torch.exp(attn_sink.view(1, 1, h) - scores_max.squeeze(-1))
    denominator = sum_exp + sink_term

    o = numerator / denominator.unsqueeze(-1)

    return o.to(q.dtype)


class DeepSeekV4SparseAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, kv, attn_sink, topk_idxs, sm_scale=None):
        o, lse = sparse_mla_fwd.sparse_mqa_fwd_interface(q, kv, attn_sink, topk_idxs, sm_scale=sm_scale)

        ctx.save_for_backward(q, kv, attn_sink, topk_idxs, o.clone(), lse)
        ctx.sm_scale = sm_scale

        return o

    @staticmethod
    def backward(ctx, do):
        q, kv, attn_sink, topk_idxs, o, lse = ctx.saved_tensors
        sm_scale = ctx.sm_scale

        dq, dkv, d_attn_sink = sparse_mla_bwd.sparse_mqa_bwd_interface(
            q, kv, attn_sink, o, do, topk_idxs, lse, sm_scale=sm_scale
        )

        return dq, dkv, d_attn_sink, None, None


def sparse_attn_tilelang(q, kv, attn_sink, topk_idxs, sm_scale=None):
    return DeepSeekV4SparseAttention.apply(q, kv, attn_sink, topk_idxs, sm_scale)
