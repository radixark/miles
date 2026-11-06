"""torchtitan's inner attention over `document_attention`'s blocked SDPA."""

from dataclasses import dataclass

import torch
from torchtitan.models.common.attention import InnerAttention

from miles.backends.torchtitan_utils.document_attention import document_causal_attention, document_ids


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
