from __future__ import annotations

import inspect
import math
import os

import torch
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import divide
from torch import Tensor

from .cp_layout import SGLangUlyssesCPLayout

try:
    from flash_attn_interface import flash_attn_varlen_func as fa3_varlen_func

    HAVE_FA3_VARLEN = True
except ImportError:
    try:
        from flash_attn_3.flash_attn_interface import flash_attn_varlen_func as fa3_varlen_func

        HAVE_FA3_VARLEN = True
    except ImportError:
        HAVE_FA3_VARLEN = False
        fa3_varlen_func = None


class SGLangFlashAttention(MegatronModule):
    """SGLang-compatible FA3 attention path with packed-sequence support."""

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float | None = None,
        softmax_scale: float | None = None,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
    ) -> None:
        super().__init__(config=config)

        self.config = config
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type
        self.current_max_attn_logits = None
        self.cp_size = config.context_parallel_size
        self.cp_comm_type = cp_comm_type

        if self.cp_size > 1:
            assert cp_comm_type == "a2a", (
                "SGLangFlashAttention currently supports packed CP only with Ulysses "
                f"(cp_comm_type='a2a'), got {cp_comm_type!r}"
            )
            assert pg_collection is not None and hasattr(
                pg_collection, "cp"
            ), "ProcessGroupCollection must provide a CP group for SGLangFlashAttention"
            self.cp_layout = SGLangUlyssesCPLayout(pg_collection.cp, self.cp_size)
        else:
            self.cp_layout = None

        kv_channels = config.kv_channels
        assert kv_channels is not None, "kv_channels must be set"
        projection_size = kv_channels * config.num_attention_heads
        tp_world_size = pg_collection.tp.size() if pg_collection is not None else 1

        self.hidden_size_per_partition = divide(projection_size, tp_world_size)
        self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = divide(config.num_attention_heads, tp_world_size)
        self.num_query_groups_per_partition = divide(config.num_query_groups, tp_world_size)

        self.softmax_scale = (
            1.0 / math.sqrt(self.hidden_size_per_attention_head) if softmax_scale is None else softmax_scale
        )
        if config.apply_query_key_layer_scaling:
            self.softmax_scale /= self.layer_number

        self.attention_dropout = config.attention_dropout if attention_dropout is None else attention_dropout

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType = None,
        attention_bias: Tensor = None,
        packed_seq_params: PackedSeqParams | None = None,
    ) -> Tensor:
        del attention_mask

        assert attention_bias is None, "Attention bias is not supported for SGLangFlashAttention"
        assert (
            attn_mask_type is None or attn_mask_type == AttnMaskType.causal
        ), "Only causal attention is supported for SGLangFlashAttention"
        if not HAVE_FA3_VARLEN or fa3_varlen_func is None:
            raise ImportError("Flash Attention 3 varlen is required for SGLangFlashAttention")

        is_packed = packed_seq_params is not None
        input_ndim = query.dim()
        head_dim_idx = -2 if is_packed and input_ndim >= 3 else 2
        if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:
            repeat_factor = self.num_attention_heads_per_partition // self.num_query_groups_per_partition
            key = key.repeat_interleave(repeat_factor, dim=head_dim_idx)
            value = value.repeat_interleave(repeat_factor, dim=head_dim_idx)

        query = query.to(torch.bfloat16)
        key = key.to(torch.bfloat16)
        value = value.to(torch.bfloat16)

        if is_packed:
            if input_ndim == 3:
                total_tokens, _, _ = query.shape
            else:
                total_tokens, batch_size, _, _ = query.shape
                assert batch_size == 1, f"Packed sequences should use batch=1, got {batch_size}"
                query = query.squeeze(1)
                key = key.squeeze(1)
                value = value.squeeze(1)

            cu_seqlens_q = packed_seq_params.cu_seqlens_q
            cu_seqlens_k = packed_seq_params.cu_seqlens_kv
            max_seqlen_q = packed_seq_params.max_seqlen_q
            max_seqlen_k = packed_seq_params.max_seqlen_kv
            if cu_seqlens_q.dtype != torch.int32:
                cu_seqlens_q = cu_seqlens_q.to(torch.int32)
            if cu_seqlens_k.dtype != torch.int32:
                cu_seqlens_k = cu_seqlens_k.to(torch.int32)
        else:
            seq_len, batch_size, num_heads, head_dim = query.shape
            key_seq_len = key.shape[0]
            query = query.transpose(0, 1).reshape(batch_size * seq_len, num_heads, head_dim)
            key = key.transpose(0, 1).reshape(batch_size * key_seq_len, num_heads, head_dim)
            value = value.transpose(0, 1).reshape(batch_size * key_seq_len, num_heads, head_dim)
            cu_seqlens_q = torch.arange(0, (batch_size + 1) * seq_len, seq_len, dtype=torch.int32, device=query.device)
            cu_seqlens_k = torch.arange(
                0,
                (batch_size + 1) * key_seq_len,
                key_seq_len,
                dtype=torch.int32,
                device=query.device,
            )
            max_seqlen_q = seq_len
            max_seqlen_k = key_seq_len

        if self.cp_size > 1:
            assert is_packed, "SGLang Ulysses CP currently requires packed THD inputs."
            assert self.cp_layout is not None
            local_tokens, local_heads, _ = query.shape
            query = self.cp_layout.sequence_to_head_parallel(query, cu_seqlens_q)
            key = self.cp_layout.sequence_to_head_parallel(key, cu_seqlens_k)
            value = self.cp_layout.sequence_to_head_parallel(value, cu_seqlens_k)

        sig = inspect.signature(fa3_varlen_func)
        fa3_kwargs = {
            "q": query,
            "k": key,
            "v": value,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
            "softmax_scale": self.softmax_scale,
            "causal": True,
        }
        if "dropout_p" in sig.parameters:
            fa3_kwargs["dropout_p"] = self.attention_dropout if self.training else 0.0
        if "window_size" in sig.parameters:
            fa3_kwargs["window_size"] = (-1, -1)
        if "softcap" in sig.parameters:
            fa3_kwargs["softcap"] = 0.0
        if "return_attn_probs" in sig.parameters:
            fa3_kwargs["return_attn_probs"] = False
        if "return_softmax_lse" in sig.parameters:
            fa3_kwargs["return_softmax_lse"] = False
        if "num_splits" in sig.parameters:
            fa3_kwargs["num_splits"] = 1
        if (
            "deterministic" in sig.parameters
            and os.environ.get("MEGATRON_TRUE_ON_POLICY_FA3_DETERMINISTIC_BWD") == "1"
        ):
            fa3_kwargs["deterministic"] = True

        output = fa3_varlen_func(**fa3_kwargs)
        if isinstance(output, tuple):
            output = output[0]

        if self.cp_size > 1:
            assert self.cp_layout is not None
            output = self.cp_layout.head_to_sequence_parallel(output, cu_seqlens_q, local_tokens, local_heads)

        if is_packed:
            if input_ndim == 3:
                return output.view(total_tokens, self.hidden_size_per_partition)
            return output.view(total_tokens, 1, self.hidden_size_per_partition)

        output = output.view(batch_size, seq_len, num_heads, head_dim)
        output = output.transpose(0, 1)
        return output.reshape(seq_len, batch_size, self.hidden_size_per_partition)


class SGLangCoreAttention(MegatronModule):
    """Core-attention wrapper used by the SGLang backend."""

    def __init__(self, *args, **kwargs) -> None:
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        super().__init__(config=config)
        self.impl = SGLangFlashAttention(*args, **kwargs)
        self._current_max_attn_logits = getattr(self.impl, "current_max_attn_logits", None)

    @property
    def current_max_attn_logits(self):
        return getattr(self.impl, "current_max_attn_logits", self._current_max_attn_logits)

    @current_max_attn_logits.setter
    def current_max_attn_logits(self, value):
        self._current_max_attn_logits = value
        if hasattr(self.impl, "current_max_attn_logits"):
            self.impl.current_max_attn_logits = value

    def forward(self, *args, **kwargs):
        return self.impl(*args, **kwargs)
