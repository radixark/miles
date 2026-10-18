"""Step5's ``linear_qkv`` layout: TP-rank blocks of grouped Q/K/V rows, each followed by that rank's gate rows.

Megatron groups Q/K/V per KV group (the group's query heads, then its key and value head) and,
with ``head_wise_attn_gate``, appends each rank's per-head gate rows after its Q/K/V rows. The
full tensor used by the bridge and by weight sync is the rank blocks concatenated, which is also
what gathering the TP shards along dim 0 yields.
"""

import torch


def fuse_qkv_and_gate(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    num_query_groups: int,
    tp_size: int,
) -> torch.Tensor:
    hidden = query.shape[-1]
    grouped_qkv = torch.cat(
        [
            query.view(num_query_groups, -1, hidden),
            key.view(num_query_groups, -1, hidden),
            value.view(num_query_groups, -1, hidden),
        ],
        dim=1,
    ).view(-1, hidden)
    rank_blocks = [
        torch.cat([qkv, rank_gate])
        for qkv, rank_gate in zip(grouped_qkv.chunk(tp_size), gate.chunk(tp_size), strict=True)
    ]
    return torch.cat(rank_blocks).contiguous()


def split_qkv_and_gate(
    fused: torch.Tensor,
    *,
    num_query_groups: int,
    num_attention_heads: int,
    head_dim: int,
    tp_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The inverse of ``fuse_qkv_and_gate``: ``(q_proj, k_proj, v_proj, g_proj)`` weights."""
    hidden = fused.shape[-1]
    gate_rows = num_attention_heads // tp_size
    qkv_blocks, gate_blocks = zip(
        *(block.split([block.shape[0] - gate_rows, gate_rows]) for block in fused.chunk(tp_size)), strict=True
    )
    grouped = torch.cat(qkv_blocks).view(num_query_groups, -1, head_dim, hidden)
    query, key, value = grouped.split([num_attention_heads // num_query_groups, 1, 1], dim=1)
    return query.reshape(-1, hidden), key.reshape(-1, hidden), value.reshape(-1, hidden), torch.cat(gate_blocks)
