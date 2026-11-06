"""Causal attention over packed sequences, for torch_tpu.

torch_tpu runs FlexAttention as one dense L x L mask, so this splits the queries into blocks and runs SDPA per
block against only the keys up to the block's end.
"""

import torch
import torch.nn.functional as F

_BLOCK_Q = 1024  # fastest on v7x


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    segment_ids: torch.Tensor,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Causal attention within each segment. q: [B, Hq, L, D]; k, v: [B, Hkv, L, D]; segment_ids: [B, L]."""
    seq_len = q.shape[2]
    positions = torch.arange(seq_len, device=q.device, dtype=torch.int32)
    blocks = []
    for start in range(0, seq_len, _BLOCK_Q):
        end = min(start + _BLOCK_Q, seq_len)
        causal = positions[None, :end] <= positions[start:end, None]
        same_segment = segment_ids[:, start:end, None] == segment_ids[:, None, :end]
        blocks.append(
            F.scaled_dot_product_attention(
                q[:, :, start:end],
                k[:, :, :end],
                v[:, :, :end],
                attn_mask=(same_segment & causal).unsqueeze(1),
                scale=scale,
                enable_gqa=enable_gqa,
            )
        )
    return torch.cat(blocks, dim=2)
