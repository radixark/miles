# ruff: noqa
"""Reference (autograd) implementation of DeepSeek-V4 sparse MQA attention.

Used where the tilelang backward kernel cannot be built. On gfx942 (MI300X/MI308X) that kernel's
three unavoidable shared-memory buffers — Q_shared and dO_shared at [block_H, D] plus KV_shared at
[block_size, D] — already total the whole 64 KiB LDS budget at the smallest tiling tilelang will
compile, so it cannot be made to fit without tiling the D dimension.

Semantics follow tilelang_sparse_mla_fwd.sparse_mqa_fwd exactly: scores are q@kv^T scaled by
sm_scale, index -1 marks a masked slot, and attn_sink[h] is a pre-scaled logit that joins the softmax
denominator. That is the same as a softmax over the topk logits plus one extra sink logit whose value
vector is zero, which is how it is expressed here so autograd can differentiate it.

This is orders of magnitude slower and far more memory-hungry than the fused kernel: it materializes
[B, S, H, topk] scores and gathers [B, S, topk, D] of KV.
"""

import torch


def sparse_attn_torch(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """q: [B, S, H, D] bf16, kv: [B, S_kv, D] bf16, attn_sink: [H] fp32, topk_idxs: [B, S, topk]."""
    B, S, H, D = q.shape
    if sm_scale is None:
        sm_scale = D**-0.5

    valid = topk_idxs != -1
    safe_idx = topk_idxs.masked_fill(~valid, 0).long()
    topk = safe_idx.shape[-1]

    # Gather from kv itself, [B, S_kv, D], rather than from a [B, S, S_kv, D] `expand` of it. The
    # forward is identical either way -- expand is a view and gather reads it strided -- but
    # gather's backward scatters into a freshly zeroed tensor shaped like its INPUT, and that shape
    # is 2048 GiB at B=1, S=16384, S_kv=131072, D=512. Flattened, the same values accumulate into
    # [B, S_kv, D] = 128 MiB.
    flat_idx = safe_idx.reshape(B, S * topk)
    kv_gathered = torch.gather(kv, 1, flat_idx.unsqueeze(-1).expand(-1, -1, D))
    kv_gathered = kv_gathered.view(B, S, topk, D).float()  # [B, S, topk, D]

    scores = torch.einsum("bshd,bstd->bsht", q.float(), kv_gathered) * sm_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))

    # The sink is an extra logit with a zero value vector, so it only inflates the denominator.
    sink = attn_sink.float().view(1, 1, H, 1).expand(B, S, H, 1)
    probs = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]

    out = torch.einsum("bsht,bstd->bshd", probs, kv_gathered)
    return out.to(q.dtype)
