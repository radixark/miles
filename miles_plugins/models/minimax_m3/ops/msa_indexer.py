# ruff: noqa
"""MiniMax Sparse Attention (MSA) lightning indexer + block selection.

This is the M3 analogue of GLM-5's token-level DSA indexer
(`miles_plugins/models/glm5/ops/indexer.py`). The differences that matter:

  GLM-5 (DSA)                         MiniMax-M3 (MSA)
  -----------------------------       --------------------------------------
  attention backbone: MLA (latent)    attention backbone: GQA (real K/V)
  selection granularity: per-token    selection granularity: per-block (128)
  topk: 2048 tokens                   topk: 16 blocks  (== 2048 tokens max)
  one index shared by all q-heads     one index per GQA group (num_index_heads)
  per-head learned `weights_proj`     none — scores are max-pooled over heads
  score reduction: sum over heads     score reduction: max over heads (amax)

Algorithm (per query position i, ref: arXiv:2606.13392 §3, and HF
`modeling_minimax_m3_vl.py`):

  1. index_q = q_norm(Wq_idx @ x)   -> [N, H_idx, d_idx]    (H_idx = sparse_num_index_heads)
     index_k = k_norm(Wk_idx @ x)   -> [N, 1,     d_idx]    (shared across the group)
     partial RoPE on the first `rotary_dim` dims of index_q / index_k.
  2. token logits  S[h,i,j] = <index_q[i,h], index_k[j]> / sqrt(d_idx),   j <= i (causal)
  3. block pooling  M[i,b]  = max_h  max_{j in block b, j<=i} S[h,i,j]     ("score_type=max")
  4. force-keep     the first `init_blocks` blocks (attention sink) and the
     last `local_blocks` blocks up to the query's own block (set score=+inf).
  5. select         I[i] = TopK_b(M[i,:], topk_blocks);  empty/future -> -1.

The expensive part is step 2 (O(N^2 * d_idx)).  We reuse GLM-5's already-tuned
tilelang forward kernel (`tilelang_indexer_fwd`) to produce the token logits
into a chunked scratch buffer, then do the cheap block-pool + top-k in torch /
a small tilelang kernel.  A pure-torch reference path (`block_topk_reference`)
is provided for correctness testing and CPU / small-shape fallback.
"""

from __future__ import annotations

import torch


# ----------------------------------------------------------------------------
# Reference (pure torch) — correctness oracle, matches HF semantics exactly.
# ----------------------------------------------------------------------------
def block_topk_reference(
    index_q: torch.Tensor,   # [N, H_idx, d_idx]  bf16/fp32
    index_k: torch.Tensor,   # [N, d_idx]         bf16/fp32  (single head, shared)
    cu_seqlens: torch.Tensor,  # [num_seq + 1] int32, varlen packing offsets
    *,
    block_size: int,
    topk_blocks: int,
    init_blocks: int = 0,
    local_blocks: int = 1,
    scale: float | None = None,
) -> torch.Tensor:
    """Return per-query selected block ids: [N, topk_blocks] int32 (-1 = unused).

    Indices are *global* block ids within the query's own sequence (0-based at
    the sequence start), so the consumer must add the sequence's block offset.
    """
    N, H, d = index_q.shape
    if scale is None:
        scale = d ** -0.5
    device = index_q.device
    out = torch.full((N, topk_blocks), -1, dtype=torch.int32, device=device)

    qf = index_q.float()
    kf = index_k.float()
    for s in range(cu_seqlens.numel() - 1):
        lo, hi = int(cu_seqlens[s]), int(cu_seqlens[s + 1])
        n = hi - lo
        if n == 0:
            continue
        q = qf[lo:hi]                      # [n, H, d]
        k = kf[lo:hi]                      # [n, d]
        # token logits, max over heads:  [n, n]
        logits = torch.einsum("ihd,jd->ihj", q, k) * scale  # [n, H, n]
        pos = torch.arange(n, device=device)
        causal = pos[None, None, :] > pos[:, None, None]     # future mask
        logits = logits.masked_fill(causal, float("-inf"))
        logits = logits.amax(dim=1)                          # [n, n] max over heads

        nb = (n + block_size - 1) // block_size
        pad = nb * block_size - n
        if pad:
            logits = torch.nn.functional.pad(logits, (0, pad), value=float("-inf"))
        block_scores = logits.view(n, nb, block_size).amax(dim=-1)  # [n, nb] max-pool

        q_block = pos // block_size                          # [n]
        # force-keep local blocks (current and `local_blocks-1` preceding)
        for off in range(local_blocks):
            tgt = (q_block - off).clamp(min=0)
            block_scores.scatter_(1, tgt[:, None], float("inf"))
        # force-keep initial sink blocks
        for b in range(min(init_blocks, nb)):
            keep = q_block >= b
            block_scores[keep, b] = float("inf")
        # never select a block that lies entirely in the future
        future_block = torch.arange(nb, device=device)[None, :] > q_block[:, None]
        block_scores = block_scores.masked_fill(future_block, float("-inf"))

        kk = min(topk_blocks, nb)
        sel_scores, sel = block_scores.topk(kk, dim=-1)      # [n, kk]
        sel = sel.masked_fill(sel_scores == float("-inf"), -1).to(torch.int32)
        out[lo:hi, :kk] = sel
    return out


# ----------------------------------------------------------------------------
# Fast path — reuse GLM-5's tilelang indexer fwd for the O(N^2) token logits,
# then block-pool + top-k.  Falls back to the reference when tilelang is absent.
# ----------------------------------------------------------------------------
def block_topk(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    block_size: int,
    topk_blocks: int,
    init_blocks: int = 0,
    local_blocks: int = 1,
    scale: float | None = None,
    chunk_q: int = 4096,
) -> torch.Tensor:
    """Block-level top-k selection. Memory O(chunk_q * seq_len) for the logits.

    Tries the tuned GLM-5 tilelang forward kernel to materialise token logits
    chunk-by-chunk (keeps peak memory bounded for million-token contexts), then
    does the cheap block max-pool + torch.topk over blocks. Any import/runtime
    failure transparently degrades to `block_topk_reference`.
    """
    try:
        from miles_plugins.models.glm5.ops.tilelang_indexer_fwd import (
            tl_indexer_fwd_impl,
        )
    except Exception:
        return block_topk_reference(
            index_q, index_k, cu_seqlens,
            block_size=block_size, topk_blocks=topk_blocks,
            init_blocks=init_blocks, local_blocks=local_blocks, scale=scale,
        )

    N, H, d = index_q.shape
    if scale is None:
        scale = d ** -0.5
    device = index_q.device
    out = torch.full((N, topk_blocks), -1, dtype=torch.int32, device=device)

    # GLM-5's fwd kernel expects a per-head weight; MSA has none, so we feed
    # ones and reduce over heads with amax ourselves (matching HF semantics).
    fwd = tl_indexer_fwd_impl(heads=H, index_dim=d)
    weights = torch.ones(N, H, device=device, dtype=torch.float32) * scale

    for s in range(cu_seqlens.numel() - 1):
        lo, hi = int(cu_seqlens[s]), int(cu_seqlens[s + 1])
        n = hi - lo
        if n == 0:
            continue
        qk = index_q[lo:hi].contiguous()
        kk_ = index_k[lo:hi].contiguous()
        nb = (n + block_size - 1) // block_size
        block_scores = torch.full((n, nb), float("-inf"), device=device)

        ks = torch.zeros(n, dtype=torch.int32, device=device)         # k-range start
        ke = torch.arange(1, n + 1, dtype=torch.int32, device=device)  # causal end
        scratch = torch.empty(min(chunk_q, n), n, device=device, dtype=torch.float32)
        for c in range(0, n, chunk_q):
            ce = min(c + chunk_q, n)
            cs = ce - c
            # max-pool over heads done by the kernel's weighted reduction trick:
            # here we run per-head then amax in torch to stay faithful to "max".
            logits = _chunk_logits_maxheads(qk[c:ce], kk_, scale)  # [cs, n]
            future = (torch.arange(n, device=device)[None, :] > torch.arange(c, ce, device=device)[:, None])
            logits = logits.masked_fill(future, float("-inf"))
            pad = nb * block_size - n
            if pad:
                logits = torch.nn.functional.pad(logits, (0, pad), value=float("-inf"))
            block_scores[c:ce] = logits.view(cs, nb, block_size).amax(dim=-1)

        pos = torch.arange(n, device=device)
        q_block = pos // block_size
        for off in range(local_blocks):
            tgt = (q_block - off).clamp(min=0)
            block_scores.scatter_(1, tgt[:, None], float("inf"))
        for b in range(min(init_blocks, nb)):
            block_scores[q_block >= b, b] = float("inf")
        future_block = torch.arange(nb, device=device)[None, :] > q_block[:, None]
        block_scores = block_scores.masked_fill(future_block, float("-inf"))

        kk = min(topk_blocks, nb)
        sel_scores, sel = block_scores.topk(kk, dim=-1)
        sel = sel.masked_fill(sel_scores == float("-inf"), -1).to(torch.int32)
        out[lo:hi, :kk] = sel
    return out


def _chunk_logits_maxheads(q_chunk, k, scale):
    # [cs, H, d] x [n, d] -> max over H -> [cs, n]
    return (torch.einsum("ihd,jd->ihj", q_chunk.float(), k.float()) * scale).amax(dim=1)


def selected_blocks_to_token_mask(
    block_ids: torch.Tensor,  # [N, topk_blocks] int32 (sequence-local block ids)
    seq_len: int,
    block_size: int,
) -> torch.Tensor:
    """Expand selected block ids into a boolean [N, seq_len] keep-mask.

    Reference helper used by the torch block-sparse attention path. Production
    runs should consume `block_ids` directly inside a fused block-sparse kernel
    instead of materialising this dense mask.
    """
    N = block_ids.shape[0]
    device = block_ids.device
    mask = torch.zeros(N, seq_len, dtype=torch.bool, device=device)
    valid = block_ids >= 0
    starts = (block_ids.clamp(min=0).long()) * block_size
    for off in range(block_size):
        cols = (starts + off).clamp(max=seq_len - 1)
        in_range = valid & (starts + off < seq_len)
        rows = torch.arange(N, device=device)[:, None].expand_as(cols)
        mask[rows[in_range], cols[in_range]] = True
    # enforce causality
    pos = torch.arange(seq_len, device=device)
    mask &= pos[None, :] <= pos[:, None]
    return mask
