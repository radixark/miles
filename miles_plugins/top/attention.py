"""Attention bound to the rollout's kernel.

Delegate to SGLang's TOP FA4 varlen forward. Unsupported paths refuse.
"""

from __future__ import annotations

import torch
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule


class TopAttention(MegatronModule):
    """core_attention delegating the rollout's shared FA4 forward operation."""

    def __init__(
        self,
        config,
        layer_number,
        attn_mask_type,
        attention_type,
        softmax_scale=None,
        cp_comm_type=None,
        model_comm_pgs=None,
        pg_collection=None,
        **kwargs,
    ):
        super().__init__(config=config)
        # Optional serving dependency; importing the spec alone needs no CUDA.
        from sglang.srt.true_on_policy.attention import flash_attn_varlen_func

        self._attention_forward = flash_attn_varlen_func
        if getattr(
            config, "multi_latent_attention", False
        ):  # config-access-exempt: TransformerConfig versions and model families differ in optional multi_latent_attention
            raise NotImplementedError(
                "[top] dense varlen attention cannot execute sparse MLA; bind its shared operation"
            )
        self.config = config
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        kv = config.kv_channels
        self.head_dim = kv
        self.softmax_scale = softmax_scale if softmax_scale is not None else kv**-0.5
        self.heads_per_partition = config.num_attention_heads  # TOP runs TP1

    def forward(
        self,
        query,
        key,
        value,
        attention_mask,
        attn_mask_type=None,
        attention_bias=None,
        packed_seq_params=None,
        **kwargs,
    ):
        if attention_bias is not None:
            raise NotImplementedError("[top] attention_bias not supported")
        mask_type = attn_mask_type or self.attn_mask_type
        if mask_type not in (None, AttnMaskType.causal):
            raise NotImplementedError(f"[top] only causal attention; got {mask_type}")

        if packed_seq_params is not None:
            cu_q = packed_seq_params.cu_seqlens_q.to(torch.int32)
            cu_k = packed_seq_params.cu_seqlens_kv.to(torch.int32)
            max_q = int(packed_seq_params.max_seqlen_q)
            max_k = int(packed_seq_params.max_seqlen_kv)
            q, k, v = (t.squeeze(1) if t.dim() == 4 else t for t in (query, key, value))
        else:
            s, b, h, d = query.shape
            sk = key.shape[0]
            q = query.transpose(0, 1).reshape(b * s, h, d)
            k = key.transpose(0, 1).reshape(b * sk, key.shape[2], d)
            v = value.transpose(0, 1).reshape(b * sk, value.shape[2], d)
            cu_q = torch.arange(0, (b + 1) * s, s, device=query.device, dtype=torch.int32)
            cu_k = torch.arange(0, (b + 1) * sk, sk, device=query.device, dtype=torch.int32)
            max_q, max_k = s, sk

        out = self._attention_forward(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            softmax_scale=self.softmax_scale,
        )
        if isinstance(out, tuple):
            out = out[0]

        from miles_plugins.top.binding import canon_dump

        if packed_seq_params is not None:
            res = out.reshape(out.shape[0], -1)
        else:
            s, b = query.shape[0], query.shape[1]
            res = out.reshape(b, s, -1).transpose(0, 1).contiguous()
        canon_dump("top_attn_core_out", res)
        return res
