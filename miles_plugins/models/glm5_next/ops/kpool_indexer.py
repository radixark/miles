import torch

from miles.kernels.attention.dsa.kpool import append_tail_and_pad, pool_topk_to_token_fn, select_expand_tail
from miles.utils.replay_base import indexer_replay_manager


def kpool_select_topk(
    index_q: torch.Tensor,
    pooled_k: torch.Tensor,
    head_weights: torch.Tensor,
    cu_seqlens: torch.Tensor,
    pool_cu_seqlens: torch.Tensor,
    index_topk: int,
    kpool: int,
) -> torch.Tensor:
    num_tokens = index_q.shape[0]
    device = index_q.device
    token_ids = torch.arange(num_tokens, device=device)
    seq_indices = torch.searchsorted(cu_seqlens, token_ids, right=True) - 1
    seq_token_base = cu_seqlens[seq_indices].to(torch.int32)
    pool_base = pool_cu_seqlens[seq_indices].to(torch.int32)
    local_positions = (token_ids - seq_token_base).to(torch.int32)
    eligible_pools = torch.div(local_positions + 1, kpool, rounding_mode="floor")

    if pooled_k.shape[0] > 0:
        # tilelang is GPU-only; importing it here keeps the pure-torch helpers above
        # importable on CPU, where tests/fast pins the per-sequence pool invariants.
        from miles.kernels.attention.dsa.glm5.tilelang_indexer_fwd import indexer_fwd

        with torch.no_grad():
            pool_logits = indexer_fwd(
                index_q,
                pooled_k,
                head_weights,
                pool_base.to(torch.int32),
                (pool_base + eligible_pools).to(torch.int32),
                clean_logits=True,
            )
    else:
        pool_logits = torch.full((num_tokens, 1), float("-inf"), dtype=torch.float32, device=device)

    if indexer_replay_manager.enabled:
        topk_fn = indexer_replay_manager.get_topk_fn(
            pool_topk_to_token_fn(seq_token_base, pool_base, local_positions, kpool),
            return_probs=False,
        )
        tokens = topk_fn(pool_logits, index_topk)
        shortcut = (local_positions + 1) <= index_topk
        tokens = append_tail_and_pad(tokens, seq_token_base, local_positions, shortcut, kpool)
    else:
        tokens = select_expand_tail(pool_logits, seq_token_base, pool_base, local_positions, index_topk, kpool)
    return tokens.unsqueeze(1)
