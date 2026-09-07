"""Attention bound to the rollout's kernel.

No autograd Function needed: FA3's varlen entry point has a backward. Unsupported paths refuse.
"""

from __future__ import annotations

import torch
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import divide

try:
    from flash_attn_interface import flash_attn_varlen_func as _fa3_varlen
except ImportError:  # pragma: no cover
    try:
        from flash_attn_3.flash_attn_interface import flash_attn_varlen_func as _fa3_varlen
    except ImportError:
        _fa3_varlen = None


class TopAttention(MegatronModule):
    """core_attention that runs FA3 varlen -- the kernel the rollout is pinned to."""

    def __init__(self, config, layer_number, attn_mask_type, attention_type,
                 softmax_scale=None, cp_comm_type=None, model_comm_pgs=None,
                 pg_collection=None, **kwargs):
        super().__init__(config=config)
        if _fa3_varlen is None:
            raise RuntimeError(
                "[top] FA3 not importable; the program pins attention to fa3 and the trainer "
                "cannot honour it. Install flash_attn_interface or change the program."
            )
        self.cp_size = config.context_parallel_size
        if self.cp_size > 1:
            # CP lives AT this seam: TE handles it inside TEDotProductAttention and megatron's local
            # DotProductAttention handles it inside its own forward, so nothing above us transposes.
            # Only Ulysses (a2a) can be bitwise -- ring merges chunks with an online softmax, which
            # is a different reduction from the rollout's single call. The contract refuses ring, so
            # by here the comm type is a2a; assert rather than trust.
            cp_comm_type = cp_comm_type if cp_comm_type is not None else config.cp_comm_type
            if isinstance(cp_comm_type, (list, tuple)):
                cp_comm_type = cp_comm_type[layer_number - 1]
            if cp_comm_type != "a2a":
                raise NotImplementedError(
                    f"[top] context parallelism needs cp_comm_type='a2a' (Ulysses); got "
                    f"{cp_comm_type!r}. Ring/p2p cannot be bitwise against a cp=1 rollout."
                )
            from miles_plugins.top.cp_layout import UlyssesCPLayout

            if pg_collection is None:
                # Compatibility for callers predating Megatron's explicit process groups.
                from megatron.core import parallel_state

                cp_group = parallel_state.get_context_parallel_group()
            else:
                cp_group = pg_collection.cp
            if cp_group.size() != self.cp_size:
                raise ValueError("[top] CP process-group size does not match context_parallel_size")
            self.cp_layout = UlyssesCPLayout(cp_group, self.cp_size)
        else:
            self.cp_layout = None
        self.config = config
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        kv = config.kv_channels
        self.head_dim = kv
        self.softmax_scale = softmax_scale if softmax_scale is not None else kv ** -0.5
        tp = getattr(config, "tensor_model_parallel_size", 1)
        self.heads_per_partition = divide(config.num_attention_heads, tp)

    def forward(self, query, key, value, attention_mask, attn_mask_type=None,
                attention_bias=None, packed_seq_params=None, **kwargs):
        if attention_bias is not None:
            raise NotImplementedError("[top] attention_bias not supported")
        mask_type = attn_mask_type or self.attn_mask_type
        if mask_type not in (None, AttnMaskType.causal):
            raise NotImplementedError(f"[top] only causal attention; got {mask_type}")

        if self.cp_size > 1 and packed_seq_params is None:
            raise NotImplementedError(
                "[top] context parallelism requires packed_seq_params: the CP transforms need the "
                "GLOBAL cu_seqlens, and the unpacked path derives cu_seqlens from the LOCAL shard "
                "length, which would silently describe the wrong sequence."
            )

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

        local_tokens, q_heads = q.shape[0], q.shape[-2]
        if self.cp_size > 1:
            from miles_plugins.top.cp_layout import replicate_kv_heads_for_cp

            # Under GQA the KV tensors carry num_query_groups heads, which would cap cp at that
            # count. Replicating lifts the bound to num_q_heads and changes no arithmetic: every
            # replicated head holds the same values.
            k = replicate_kv_heads_for_cp(k, self.cp_size)
            v = replicate_kv_heads_for_cp(v, self.cp_size)
            q = self.cp_layout.sequence_to_head_parallel(q, cu_q)
            k = self.cp_layout.sequence_to_head_parallel(k, cu_k)
            v = self.cp_layout.sequence_to_head_parallel(v, cu_k)

        # num_splits is PINNED, not inherited. sglang pins it to 1 under
        # enable_deterministic_inference; FA3's library default happens to be 1 too, so the trainer
        # matches by accident rather than declaration. Under the heuristic (0) a shard's
        # per-row output can differ from the full call's, so CP would break silently.
        out = _fa3_varlen(q, k, v, cu_q, cu_k, max_q, max_k,
                          causal=True, softmax_scale=self.softmax_scale, num_splits=1)
        if isinstance(out, tuple):
            out = out[0]

        if self.cp_size > 1:
            out = self.cp_layout.head_to_sequence_parallel(out, cu_q, local_tokens, q_heads)

        from miles_plugins.top.spec import _canon_dump

        if packed_seq_params is not None:
            res = out.reshape(out.shape[0], -1)
        else:
            s, b = query.shape[0], query.shape[1]
            res = out.reshape(b, s, -1).transpose(0, 1).contiguous()
        _canon_dump("top_attn_core_out", res)
        return res
