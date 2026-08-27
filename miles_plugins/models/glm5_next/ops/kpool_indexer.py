"""GLM-5.3 pooled-key (kpool) indexer selection, torch reference.

Reproduces sglang ``dsa_indexer_kpool.py``'s selection semantics on the packed
training stream, without the serving-only fp8/Hadamard machinery:

1. Keys are pooled ``kpool`` -> 1 at sequence-local positions (``pool = pos // kpool``,
   complete pools only). Compression is a learned per-feature softmax over the
   ``kpool`` slots: ``score[s, d] = gate[t0 + s, d] + ape[s, d]``, computed in fp32
   on the post-``wk``+``k_norm`` key, then cast back to bf16.
2. Pool logits use the standard DSA MQA formula (per-head ReLU, head-weighted
   fp32 sum) via the existing tilelang indexer forward, scoring pooled keys with
   per-token eligible-pool ranges ``floor((t + 1) / kpool)``.
3. The top ``index_topk // kpool`` pools are selected per token and expanded to
   ``index_topk`` token indices. Tokens whose causal prefix fits in ``index_topk``
   skip the top-k entirely and select every token (sglang's short-sequence
   shortcut).
4. ``always_select_tail``: the deterministic tail -- the ``(t + 1) % kpool``
   tokens after the last complete pool -- is appended on top of the budget, but
   only for non-shortcut tokens (shortcut rows already contain it). The result
   is ``-1``-padded to a multiple of the SparseMLA block size (64).

The serving side scores Hadamard-rotated fp8 pooled keys with a racy radix
top-k, so this bf16 rescorer cannot byte-match it; exact parity comes from R3
indexer replay (the replay-manager hook below), where the recorded/replayed
tensor is the ``index_topk``-wide token expansion of the pool budget and the
tail is reconstructed deterministically on this side.

Selection carries no gradient: pool logits are computed under ``torch.no_grad``
on detached inputs, matching the glm5 plugin where indexer scores are discarded.
"""

import torch

from miles.utils.replay_base import indexer_replay_manager
from miles_plugins.models.glm5.ops.tilelang_indexer_fwd import indexer_fwd_interface

SPARSE_MLA_BLOCK = 64


def pool_boundaries(cu_seqlens: torch.Tensor, kpool: int) -> torch.Tensor:
    """Cumulative complete-pool counts per sequence, aligned with ``cu_seqlens``."""
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    pool_counts = torch.div(seq_lens, kpool, rounding_mode="floor")
    pool_cu_seqlens = torch.zeros_like(cu_seqlens)
    pool_cu_seqlens[1:] = torch.cumsum(pool_counts, dim=0)
    return pool_cu_seqlens


def build_pooled_keys(
    index_k: torch.Tensor,
    gate_score: torch.Tensor,
    ape: torch.Tensor,
    cu_seqlens: torch.Tensor,
    kpool: int,
) -> torch.Tensor:
    """Compress complete ``kpool``-token groups of ``index_k`` into pooled keys.

    ``index_k``/``gate_score`` are ``[total_tokens, index_head_dim]`` on the packed
    stream; ``ape`` is the fp32 ``[kpool, index_head_dim]`` additive positional
    bias. Returns ``[total_pools, index_head_dim]`` in ``index_k``'s dtype.
    """
    ape = ape.float()
    pooled = []
    boundaries = cu_seqlens.tolist()
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        num_pools = (end - start) // kpool
        if num_pools == 0:
            continue
        span = slice(start, start + num_pools * kpool)
        keys = index_k[span].float().view(num_pools, kpool, -1)
        scores = gate_score[span].float().view(num_pools, kpool, -1) + ape
        weights = torch.softmax(scores, dim=1)
        pooled.append((weights * keys).sum(dim=1))
    if not pooled:
        return index_k.new_zeros((0, index_k.shape[-1]))
    return torch.cat(pooled, dim=0).to(index_k.dtype)


def _pool_topk_to_token_fn(seq_token_base, pool_base, local_positions, shortcut, kpool):
    def topk_fn(pool_logits: torch.Tensor, topk: int) -> torch.Tensor:
        num_tokens, num_pools = pool_logits.shape
        device = pool_logits.device
        tokens = torch.full((num_tokens, topk), -1, dtype=torch.int64, device=device)

        group_topk = min(topk // kpool, num_pools)
        if group_topk > 0:
            scores, pools = torch.topk(pool_logits.float(), group_topk, dim=-1)
            valid = torch.isfinite(scores)
            token_base = seq_token_base.unsqueeze(1) + (pools - pool_base.unsqueeze(1)) * kpool
            candidates = token_base.unsqueeze(-1) + torch.arange(kpool, device=device)
            candidates = torch.where(valid.unsqueeze(-1), candidates, -1)
            tokens[:, : group_topk * kpool] = candidates.reshape(num_tokens, group_topk * kpool)

        if shortcut.any():
            offsets = torch.arange(topk, device=device)
            shortcut_tokens = seq_token_base.unsqueeze(1) + offsets
            shortcut_valid = offsets.unsqueeze(0) <= local_positions.unsqueeze(1)
            shortcut_tokens = torch.where(shortcut_valid, shortcut_tokens, -1)
            tokens = torch.where(shortcut.unsqueeze(1), shortcut_tokens, tokens)
        return tokens.to(torch.int32)

    return topk_fn


def append_tail_and_pad(
    tokens: torch.Tensor,
    seq_token_base: torch.Tensor,
    local_positions: torch.Tensor,
    shortcut: torch.Tensor,
    kpool: int,
    pad_multiple: int = SPARSE_MLA_BLOCK,
) -> torch.Tensor:
    """Append the deterministic ``always_select_tail`` positions and ``-1``-pad.

    The tail of token ``t`` is ``[((t + 1) // kpool) * kpool, t]`` in its sequence,
    a pure function of ``t``, so it is reconstructed here rather than replayed.
    Shortcut rows already contain their tail and get ``-1`` padding instead.
    """
    num_tokens = tokens.shape[0]
    device = tokens.device
    slots = torch.arange(kpool - 1, device=device)
    tail_start = seq_token_base + torch.div(local_positions + 1, kpool, rounding_mode="floor") * kpool
    tail_len = (local_positions + 1) % kpool
    tail = tail_start.unsqueeze(1) + slots
    tail_valid = (slots.unsqueeze(0) < tail_len.unsqueeze(1)) & ~shortcut.unsqueeze(1)
    tail = torch.where(tail_valid, tail, tail.new_full((), -1))
    out = torch.cat([tokens.long(), tail], dim=1)

    width = out.shape[1]
    padded_width = (width + pad_multiple - 1) // pad_multiple * pad_multiple
    if padded_width != width:
        pad = out.new_full((num_tokens, padded_width - width), -1)
        out = torch.cat([out, pad], dim=1)
    return out.to(torch.int32)


def kpool_select_topk(
    index_q: torch.Tensor,
    pooled_k: torch.Tensor,
    head_weights: torch.Tensor,
    cu_seqlens: torch.Tensor,
    pool_cu_seqlens: torch.Tensor,
    index_topk: int,
    kpool: int,
) -> torch.Tensor:
    """Token-level top-k selection, ``[total_tokens, 1, padded_topk]`` int32."""
    num_tokens = index_q.shape[0]
    device = index_q.device
    token_ids = torch.arange(num_tokens, device=device)
    seq_indices = torch.searchsorted(cu_seqlens, token_ids, right=True) - 1
    seq_token_base = cu_seqlens[seq_indices]
    pool_base = pool_cu_seqlens[seq_indices]
    local_positions = token_ids - seq_token_base
    eligible_pools = torch.div(local_positions + 1, kpool, rounding_mode="floor")
    shortcut = (local_positions + 1) <= index_topk

    if pooled_k.shape[0] > 0:
        with torch.no_grad():
            pool_logits = indexer_fwd_interface(
                index_q,
                pooled_k,
                head_weights,
                pool_base.to(torch.int32),
                (pool_base + eligible_pools).to(torch.int32),
                clean_logits=True,
            )
    else:
        pool_logits = torch.full((num_tokens, 1), float("-inf"), dtype=torch.float32, device=device)

    topk_fn = indexer_replay_manager.get_topk_fn(
        _pool_topk_to_token_fn(seq_token_base, pool_base, local_positions, shortcut, kpool),
        return_probs=False,
    )
    tokens = topk_fn(pool_logits, index_topk)
    tokens = append_tail_and_pad(tokens, seq_token_base, local_positions, shortcut, kpool)
    return tokens.unsqueeze(1)
