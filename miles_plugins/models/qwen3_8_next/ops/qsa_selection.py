"""Qwen3.8-Next QSA (Qwen Sparse Attention) selection math, free of any training backend.

The indexer scores compressed key blocks per query token, keeps ``indexer_budget``
tokens, and the attention adds the query's own partial block. The Megatron layer and
the torchtitan one both select here, so their sparse attention reads the same keys
that sglang's rollout read.
"""

import math

import torch
from torch import Tensor


def _indexer_acc_dtype(x):
    return x.dtype if x.dtype in (torch.float32, torch.float64) else torch.float32


def gemma_rmsnorm_last_dim(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """RMSNorm over the last dim with a Gemma-style ``1 + weight`` scale."""
    acc = _indexer_acc_dtype(x)
    xa = x.to(acc)
    var = xa.pow(2).mean(dim=-1, keepdim=True)
    return ((xa * torch.rsqrt(var + eps)) * (1.0 + weight.to(acc))).to(x.dtype)


def compress_keys_by_mean(token_k: Tensor, compress_ratio: int) -> Tensor:
    """``[T, head_dim] -> [ceil(T / r), head_dim]`` by averaging each run of ``r``."""
    tokens, dim = token_k.shape
    blocks = -(-tokens // compress_ratio)
    padded = blocks * compress_ratio
    if padded != tokens:
        pad = token_k.new_zeros(padded - tokens, dim)
        counts = token_k.new_ones(padded, 1)
        counts[tokens:] = 0
        summed = torch.cat([token_k, pad], dim=0).view(blocks, compress_ratio, dim).sum(1)
        denom = counts.view(blocks, compress_ratio, 1).sum(1).clamp_min(1)
        return summed / denom
    return token_k.view(blocks, compress_ratio, dim).mean(dim=1)


def apply_indexer_rope(x: Tensor, rotary_pos_emb: Tensor, positions: Tensor) -> Tensor:
    """Neox RoPE on the leading ``rotary_dim`` dims of ``x`` ``[T, heads, head_dim]``.

    ``rotary_pos_emb`` is Megatron's non-interleaved ``[max_pos, 1, 1, rotary_dim]`` fp32
    table, ``cat(freqs, freqs)``. Mirrors sglang's eager ``apply_rotary_emb``: cos/sin
    rounded to ``x.dtype`` first, then per-op rounding.
    """
    freqs = rotary_pos_emb.flatten(1)
    half = freqs.shape[-1] // 2
    # cos/sin over the contiguous table, then gather, like sglang's cache: strided or gathered cos can move an ulp
    freqs = freqs[:, :half].contiguous()
    cos = freqs.cos()[positions.long()].unsqueeze(-2).to(x.dtype)
    sin = freqs.sin()[positions.long()].unsqueeze(-2).to(x.dtype)
    x1 = x[..., :half]
    x2 = x[..., half : 2 * half]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, x[..., 2 * half :]], dim=-1)


def block_causal_mask(query_positions: Tensor, num_blocks: int, compress_ratio: int) -> Tensor:
    """``[T, num_blocks]`` bool: which compressed blocks a query may attend to."""
    blocks = torch.arange(num_blocks, device=query_positions.device)
    first_invalid = (query_positions + 1) // compress_ratio
    return blocks.unsqueeze(0) < first_invalid.unsqueeze(1)


class PackedBlockLayout:
    """Per-sequence compressed-block grid for a packed (thd) batch.

    The rollout engine scores one request at a time, so each sequence's block grid
    must start at its own first token.
    """

    def __init__(self, cu_seqlens: Tensor, positions: Tensor, compress_ratio: int):
        device = positions.device
        cu = cu_seqlens.to(device=device, dtype=torch.long)
        lengths = cu[1:] - cu[:-1]
        blocks_per_seq = -(-lengths // compress_ratio)
        self.num_blocks = int(blocks_per_seq.sum())

        seq_block_start = torch.cat([blocks_per_seq.new_zeros(1), blocks_per_seq.cumsum(0)[:-1]])
        seq_token_start = cu[:-1]

        total = positions.numel()
        seg = torch.zeros(total, dtype=torch.long, device=device)
        seg[seq_token_start[1:]] = 1
        seg = seg.cumsum(0)

        # per token
        self.token_seq = seg
        self.token_block = seq_block_start[seg] + positions // compress_ratio
        self.token_block_start = seq_block_start[seg]
        self.token_start = seq_token_start[seg]
        # per block
        self.block_seq = torch.repeat_interleave(torch.arange(blocks_per_seq.numel(), device=device), blocks_per_seq)
        self.block_local = torch.arange(self.num_blocks, device=device) - seq_block_start[self.block_seq]
        self.block_first_token = seq_token_start[self.block_seq] + self.block_local * compress_ratio
        self.block_last_token = torch.minimum(self.block_first_token + compress_ratio, cu[1:][self.block_seq]) - 1


def compress_keys_by_mean_packed(token_k: Tensor, layout: PackedBlockLayout) -> Tensor:
    """``[T, head_dim] -> [layout.num_blocks, head_dim]``, averaging inside each sequence."""
    acc = _indexer_acc_dtype(token_k)
    dim = token_k.shape[-1]
    summed = torch.zeros(layout.num_blocks, dim, dtype=acc, device=token_k.device)
    summed.index_add_(0, layout.token_block, token_k.to(acc))
    counts = torch.zeros(layout.num_blocks, 1, dtype=acc, device=token_k.device)
    counts.index_add_(0, layout.token_block, torch.ones(token_k.shape[0], 1, dtype=acc, device=token_k.device))
    return (summed / counts.clamp_min(1)).to(token_k.dtype)


def packed_block_causal_mask(query_positions: Tensor, layout: PackedBlockLayout, compress_ratio: int) -> Tensor:
    """``[T, num_blocks]`` bool, restricted to each query's own sequence."""
    blocks = torch.arange(layout.num_blocks, device=query_positions.device).unsqueeze(0)
    lo = layout.token_block_start.unsqueeze(1)
    first_invalid = lo + ((query_positions + 1) // compress_ratio).unsqueeze(1)
    return (blocks >= lo) & (blocks < first_invalid)


def project_indexer_qk(
    qk: Tensor,
    q_layernorm: Tensor,
    k_layernorm: Tensor,
    positions: Tensor,
    rotary_pos_emb: Tensor,
    *,
    n_heads: int,
    head_dim: int,
    compress_ratio: int,
    eps: float,
    layout: PackedBlockLayout | None = None,
) -> tuple[Tensor, Tensor]:
    """Indexer projection output ``[T, (n_heads + 1) * head_dim]`` -> ``(q [T, n_heads, head_dim], block_k [B, head_dim])``."""
    split = n_heads * head_dim
    q_raw, token_k = qk[..., :split], qk[..., split:]

    q = gemma_rmsnorm_last_dim(q_raw.reshape(-1, head_dim), q_layernorm, eps).reshape(-1, n_heads, head_dim)

    token_k = token_k.reshape(-1, head_dim)
    if layout is None:
        block_k = compress_keys_by_mean(token_k, compress_ratio)
    else:
        block_k = compress_keys_by_mean_packed(token_k, layout)
    block_k = gemma_rmsnorm_last_dim(block_k, k_layernorm, eps)

    if layout is None:
        block_local = torch.arange(block_k.shape[0], device=positions.device)
    else:
        block_local = layout.block_local
    # sglang rotates a compressed key at the position of its block's first token
    block_positions = block_local * compress_ratio
    q = apply_indexer_rope(q, rotary_pos_emb, positions)
    block_k = apply_indexer_rope(block_k.unsqueeze(1), rotary_pos_emb, block_positions).squeeze(1)
    return q, block_k


def score_indexer_blocks(
    q: Tensor,
    block_k: Tensor,
    query_positions: Tensor,
    *,
    head_dim: int,
    compress_ratio: int,
    layout: PackedBlockLayout | None = None,
) -> Tensor:
    """``[T, num_blocks]`` fp32 logits, invalid blocks at ``-inf``."""
    scores = torch.einsum("mhd,nd->mnh", q.float(), block_k.float())
    logits = torch.relu(scores).sum(dim=-1) / math.sqrt(head_dim)
    if layout is None:
        valid = block_causal_mask(query_positions, block_k.shape[0], compress_ratio)
    else:
        valid = packed_block_causal_mask(query_positions, layout, compress_ratio)
    return logits.masked_fill(~valid, float("-inf"))


def select_indexer_tokens(
    logits: Tensor,
    positions: Tensor,
    *,
    block_topk: int,
    token_topk: int,
    compress_ratio: int,
    layout: PackedBlockLayout | None = None,
) -> Tensor:
    """Block logits -> ``[T, token_topk]`` int32 token indices, absolute in the packed buffer, ``-1`` where unused."""
    k = min(block_topk, logits.shape[-1])
    block_scores, block_idx = torch.topk(logits, k, dim=-1)
    block_idx = block_idx.masked_fill(block_scores == float("-inf"), -1)

    offsets = torch.arange(compress_ratio, device=block_idx.device)
    if layout is None:
        tokens = block_idx.unsqueeze(-1) * compress_ratio + offsets
    else:
        block_local = block_idx - layout.token_block_start.unsqueeze(-1)
        tokens = layout.token_start.unsqueeze(-1).unsqueeze(-1) + (
            block_local.unsqueeze(-1) * compress_ratio + offsets
        )
    tokens = tokens.masked_fill(block_idx.unsqueeze(-1) < 0, -1).flatten(-2)
    limit = positions if layout is None else layout.token_start + positions
    tokens = tokens.masked_fill(tokens > limit.unsqueeze(-1), -1)
    if layout is not None:
        tokens = tokens.masked_fill(tokens < layout.token_start.unsqueeze(-1), -1)
    return tokens[..., :token_topk].to(torch.int32)


def merge_tail_and_clip(selection: Tensor, positions: Tensor, compress_ratio: int) -> tuple[Tensor, Tensor]:
    """Add each query's own partial block to the indexer's selection, keep it inside the query's sequence.

    Returns ``(selection, seq_start)``: ``[T, budget + ratio]`` pack indices (``-1`` pad) and
    each query's sequence start in the packed buffer.
    """
    seq = positions.shape[0]
    seq_start = torch.arange(seq, device=positions.device) - positions
    r = compress_ratio
    tail_in_seq = (positions + 1) // r * r
    offs = torch.arange(r, device=positions.device)
    tail_pos = tail_in_seq.unsqueeze(1) + offs.unsqueeze(0)  # in-seq
    tail_idx = seq_start.unsqueeze(1) + tail_pos  # pack index
    tail_idx = torch.where(
        tail_pos <= positions.unsqueeze(1),
        tail_idx,
        torch.full_like(tail_idx, -1),
    )
    merged = torch.cat([selection, tail_idx.to(selection.dtype)], dim=1)
    pack_pos = torch.arange(seq, device=positions.device).unsqueeze(1)
    seg_lo = seq_start.unsqueeze(1)
    ok = (merged >= seg_lo) & (merged <= pack_pos) & (merged >= 0)
    return torch.where(ok, merged, torch.full_like(merged, -1)), seq_start


def build_block_form(
    selection: Tensor,
    positions: Tensor,
    seq_start: Tensor,
    cu_seqlens: Tensor | None,
    compress_ratio: int,
) -> tuple:
    """Selection rows -> the block form the tensor-core kernel consumes.

    Block ids come from the indexer's per-sequence grid (``PackedBlockLayout``); a
    ceil(seq_start / ratio) base collides across sequences for some lengths.
    """
    seq = positions.shape[0]
    ratio = compress_ratio
    cu = cu_seqlens
    if cu is None or cu.numel() <= 2:
        cu = torch.tensor([0, seq], device=selection.device)
    layout = PackedBlockLayout(cu, positions, ratio)
    blk_base = layout.token_block_start.to(torch.int32)
    tok_base = layout.token_start.to(torch.int32)
    num_blocks = layout.num_blocks

    sel_bitmap = torch.zeros(seq, num_blocks, dtype=torch.uint8, device=selection.device)
    valid = selection >= 0
    if bool(valid.any()):
        rows = torch.arange(seq, device=selection.device).unsqueeze(1).expand_as(selection)
        blk = blk_base.unsqueeze(1).long() + (selection - tok_base.unsqueeze(1)).div(ratio, rounding_mode="floor")
        sel_bitmap[rows[valid], blk[valid].long().clamp_(0, num_blocks - 1)] = 1
    lo = tok_base
    hi = (positions + seq_start).to(torch.int32)
    block_first = layout.block_first_token.to(torch.int32)
    block_last = layout.block_last_token.to(torch.int32)
    return sel_bitmap, lo, hi, blk_base, tok_base, block_first, block_last, ratio
