"""TorchTitan integration for Qwen3.5 gated delta-net on packed documents."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchtitan.models.common.attention import InnerAttention

from miles.backends.torchtitan_utils.document_attention import document_ids
from miles.kernels.tpu.gated_delta import document_gated_delta, document_short_conv


class DocumentGatedDeltaInnerAttention(InnerAttention):
    """torchtitan's `InnerGDN` (short convs, decay gate, delta rule) on document ids, without attn_gym."""

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
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
        query_TC: torch.Tensor,
        key_TC: torch.Tensor,
        value_TC: torch.Tensor,
        a_TH: torch.Tensor,
        b_TH: torch.Tensor,
        conv_q_weight_C1W: torch.Tensor,
        conv_k_weight_C1W: torch.Tensor,
        conv_v_weight_C1W: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_H: torch.Tensor,
        *,
        key_head_dim: int,
        value_head_dim: int,
        attention_metadata: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(attention_metadata, torch.Tensor):
            raise TypeError(f"expected document ids as attention_metadata, got {type(attention_metadata).__name__}")
        num_tokens = query_TC.shape[0]

        def conv_heads(x_TC: torch.Tensor, weight_C1W: torch.Tensor, head_dim: int) -> torch.Tensor:
            return document_short_conv(x_TC, weight_C1W, attention_metadata).reshape(num_tokens, -1, head_dim)

        g_TH = -torch.exp(A_log_H.float()) * F.softplus(a_TH.float() + dt_bias_H)
        return document_gated_delta(
            conv_heads(query_TC, conv_q_weight_C1W, key_head_dim),
            conv_heads(key_TC, conv_k_weight_C1W, key_head_dim),
            conv_heads(value_TC, conv_v_weight_C1W, value_head_dim),
            g_TH,
            torch.sigmoid(b_TH),
            attention_metadata,
        )
