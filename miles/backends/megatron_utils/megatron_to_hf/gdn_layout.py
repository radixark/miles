"""Megatron row layout of the delta-rule q/k/v projection and short conv.

HF stores them flat, ``[Q_all, K_all, V_all]``. Megatron shards the output dimension in contiguous
chunks, so the Megatron copy is *group-major*: one block per key head ``g`` holding ``[q_g, k_g, v_g]``
with ``R = num_v_heads // num_k_heads`` value heads in ``v_g``. A TP chunk is then exactly the heads
that rank owns, for any TP size, and the projection output feeds the conv without a copy.
"""

import torch

from miles.kernels.attention.delta_rule import DeltaRuleHeads


def gdn_heads(hf_config) -> DeltaRuleHeads:
    """Head layout of a Qwen3.5 / Qwen3-Next GDN layer; VLM configs nest it under ``text_config``."""
    text_config = getattr(hf_config, "text_config", hf_config)
    return DeltaRuleHeads(
        num_k_heads=text_config.linear_num_key_heads,
        num_v_heads=text_config.linear_num_value_heads,
        head_k_dim=text_config.linear_key_head_dim,
        head_v_dim=text_config.linear_value_head_dim,
    )


def group_major(blocks: list[torch.Tensor], heads: DeltaRuleHeads) -> torch.Tensor:
    """Interleave head-major blocks (dim 0 is ``num_k_heads * rows_per_group``) group by group."""
    rest = blocks[0].shape[1:]
    grouped = [block.reshape(heads.num_k_heads, -1, *rest) for block in blocks]
    return torch.cat(grouped, dim=1).reshape(-1, *rest).contiguous()


def split_group_major(weight: torch.Tensor, rows_per_group: list[int], heads: DeltaRuleHeads) -> list[torch.Tensor]:
    """Inverse of :func:`group_major`."""
    rest = weight.shape[1:]
    parts = weight.reshape(heads.num_k_heads, -1, *rest).split(rows_per_group, dim=1)
    return [part.reshape(-1, *rest).contiguous() for part in parts]


def _qkv_rows(heads: DeltaRuleHeads) -> list[int]:
    return [heads.head_k_dim, heads.head_k_dim, heads.group_value_dim]


def qkv_flat_to_group_major(weight: torch.Tensor, heads: DeltaRuleHeads) -> torch.Tensor:
    """``[Q_all, K_all, V_all]`` (HF) -> group-major ``[q, k, v]``."""
    assert weight.shape[0] == heads.qkv_dim, (weight.shape, heads)
    return group_major(list(weight.split([heads.key_dim, heads.key_dim, heads.value_dim])), heads)


def qkv_group_major_to_flat(weight: torch.Tensor, heads: DeltaRuleHeads) -> torch.Tensor:
    assert weight.shape[0] == heads.qkv_dim, (weight.shape, heads)
    return torch.cat(split_group_major(weight, _qkv_rows(heads), heads))
