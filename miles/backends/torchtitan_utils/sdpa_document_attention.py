"""Packed-document causal attention as query blocks over SDPA, for platforms without a sparse FlexAttention.

torch_tpu runs FlexAttention as one dense L x L mask (torch_tpu/_internal/compile/flex_attention_redirect.py);
per query block, SDPA sees only the keys at or before the block's end.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchtitan.models.common.attention import InnerAttention

_Q_BLOCK = 1024  # measured on v7x: ties 2048 at L=8192, 512 is ~13% slower


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

    Every packed document and the padding tail restart positions at 0 and count up by 1, so this names the
    same documents as `(positions == 0).cumsum(-1)` without a cumulative op: libtpu's associative-scan
    emitter (>= 0.0.44) miscompiles cumsum for lengths that are not a multiple of 1024.
    """
    token_index = torch.arange(positions.shape[-1], device=positions.device, dtype=torch.int32)
    return token_index - positions.to(torch.int32)


class BlockedSdpaInnerAttention(InnerAttention):
    """Document-causal inner attention on SDPA; its metadata is each token's document id."""

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        @property
        def is_full_attention(self) -> bool:
            return True

        def build_attention_metadata(
            self,
            positions: torch.Tensor,
            *,
            padding_mask: torch.Tensor | None = None,
            max_num_documents: int | None = None,
            max_context_length: int | None = None,
        ) -> torch.Tensor:
            del padding_mask, max_num_documents, max_context_length
            return document_ids(positions)

    def __init__(self, config: Config) -> None:
        super().__init__()

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        *,
        attention_metadata: torch.Tensor,
        score_mod=None,
        scale: float | None = None,
        enable_gqa: bool = False,
        out_transform=None,
        **kwargs,
    ) -> torch.Tensor:
        if score_mod is not None or out_transform is not None or kwargs:
            raise NotImplementedError("blocked SDPA attention takes no score_mod, out_transform or other options")
        if not isinstance(attention_metadata, torch.Tensor):
            raise TypeError(f"expected document ids as attention_metadata, got {type(attention_metadata).__name__}")
        out_1HTV = document_causal_attention(
            q_THK.transpose(0, 1).unsqueeze(0),
            k_THK.transpose(0, 1).unsqueeze(0),
            v_THV.transpose(0, 1).unsqueeze(0),
            attention_metadata.unsqueeze(0),
            scale=scale,
            enable_gqa=enable_gqa,
        )
        return out_1HTV.squeeze(0).transpose(0, 1)
