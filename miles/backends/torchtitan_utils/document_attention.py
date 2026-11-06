"""Packed-document causal attention as query blocks over SDPA, for platforms without a sparse FlexAttention.

torch_tpu runs FlexAttention as one dense L x L mask (torch_tpu/_internal/compile/flex_attention_redirect.py);
per query block, SDPA sees only the keys at or before the block's end. Torch only, so CPU CI can check it.
"""

import torch
import torch.nn.functional as F

_Q_BLOCK = 1024  # fastest on v7x


def document_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    doc_ids: torch.Tensor,
    *,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """q: (B, Nq, L, H); k, v: (B, Nkv, L, H); doc_ids: (B, L). Skips tiles above the diagonal."""
    seq_len = q.shape[2]
    token_index = torch.arange(seq_len, device=q.device, dtype=torch.int32)
    blocks = []
    for start in range(0, seq_len, _Q_BLOCK):
        end = min(start + _Q_BLOCK, seq_len)
        causal = token_index[None, :end] <= token_index[start:end, None]
        same_doc = doc_ids[:, start:end, None] == doc_ids[:, None, :end]
        blocks.append(
            F.scaled_dot_product_attention(
                q[:, :, start:end],
                k[:, :, :end],
                v[:, :, :end],
                attn_mask=(same_doc & causal).unsqueeze(1),
                scale=scale,
                enable_gqa=enable_gqa,
            )
        )
    return torch.cat(blocks, dim=2)


def document_ids(positions: torch.Tensor) -> torch.Tensor:
    """Each token's document, as the index of the document's first token: `arange(L) - positions`.

    Positions restart at 0 in every document and the padding tail, so this needs no cumsum, which libtpu
    (>= 0.0.44) miscompiles for lengths not a multiple of 1024.
    """
    token_index = torch.arange(positions.shape[-1], device=positions.device, dtype=torch.int32)
    return token_index - positions.to(torch.int32)
