"""CanonicalLoRA support for attention-output-gated fused QKV (radixark/miles#2008).

Megatron-Bridge's ``CanonicalLoRA`` sizes the ``linear_qkv`` adapter for grouped
Q/K/V rows; providers with ``attention_output_gate=True`` (Qwen3.5/3.6, gated
``linear_qkv`` of Qwen3-Next) pack Q/Gate/K/V rows, so the adapter is narrower than
the wrapped projection and the first forward fails with a shape mismatch. The patch
gives ``adapter_q`` per-head [Q; gate] rows (the HF ``q_proj`` layout, as in
``miles_plugins/mbridge/qwen3_5.py``) and packs adapter output per query group as
[Q heads, gate heads, K, V] (Megatron's fused layout). Non-gated modules keep the
original behavior. Like upstream's non-gated ``_interleave_qkv``, the interleave uses
global head counts, so fused-QKV canonical LoRA remains TP=1-only for now. Drop once
the Megatron-Bridge pin includes the fix upstream.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def interleave_gated_qkv(
    query_and_gate: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    num_attention_heads: int,
    num_query_groups: int,
    head_size: int,
) -> torch.Tensor:
    """Pack per-head [Q; gate] adapter output plus K/V into Megatron's grouped Q/Gate/K/V order."""
    if num_attention_heads % num_query_groups != 0:
        raise ValueError("num_attention_heads must be divisible by num_query_groups.")
    if query_and_gate.size(-1) != 2 * num_attention_heads * head_size:
        raise ValueError("Gated query projection size must equal 2 * num_attention_heads * head_size.")

    heads_per_group = num_attention_heads // num_query_groups
    leading_shape = query_and_gate.shape[:-1]
    query_and_gate = query_and_gate.reshape(-1, num_attention_heads, 2, head_size)
    query = query_and_gate[:, :, 0, :]
    gate = query_and_gate[:, :, 1, :]
    key = key.reshape(-1, num_query_groups, head_size)
    value = value.reshape(-1, num_query_groups, head_size)

    qkv_chunks = []
    for group in range(num_query_groups):
        q_group = query[:, group * heads_per_group : (group + 1) * heads_per_group, :]
        gate_group = gate[:, group * heads_per_group : (group + 1) * heads_per_group, :]
        qkv_chunks.extend([q_group, gate_group, key[:, group : group + 1, :], value[:, group : group + 1, :]])

    return torch.cat(qkv_chunks, dim=1).reshape(*leading_shape, -1)


def install_canonical_lora_gated_qkv_patch() -> None:
    """Teach ``CanonicalLoRA`` the Q/Gate/K/V fused-QKV contract. Idempotent."""
    from megatron.bridge.peft import canonical_lora as _canonical_lora
    from megatron.bridge.peft.utils import (
        ParallelLinearAdapter,
        get_adapter_attributes_from_linear,
        get_effective_lora_dim,
        is_modelopt_linear,
    )
    from torch import nn

    if getattr(_canonical_lora.CanonicalLoRA, "_miles_gated_qkv_patch_installed", False):
        return

    class LoRALinearSplitGatedQKV(_canonical_lora.LoRALinearSplitQKV):
        """``linear_qkv`` wrapper for gated attention: ``adapter_q`` carries Q and gate rows."""

        def forward(self, x, *args, **kwargs):
            linear_output, bias, layernorm_output = self.base_linear_forward(x, *args, **kwargs)
            if not self._adapter_enabled:
                return linear_output, bias
            query_and_gate = self.adapter_forward(self.adapter.adapter_q, layernorm_output, *args, **kwargs)
            key = self.adapter_forward(self.adapter.adapter_k, layernorm_output, *args, **kwargs)
            value = self.adapter_forward(self.adapter.adapter_v, layernorm_output, *args, **kwargs)

            config = self.to_wrap.config
            adapter_output = interleave_gated_qkv(
                query_and_gate,
                key,
                value,
                num_attention_heads=config.num_attention_heads,
                num_query_groups=config.num_query_groups,
                head_size=config.kv_channels,
            )
            return linear_output + adapter_output, bias

    _original_transform = _canonical_lora.CanonicalLoRA.transform

    def transform(self, m, name=None, prefix=None):
        already_wrapped = isinstance(
            m,
            (
                _canonical_lora.LinearAdapter,
                _canonical_lora.LoRALinear,
                _canonical_lora.LoRALinearSplitQKV,
                _canonical_lora.LoRALinearSplitFC1UpGate,
                _canonical_lora.LoRATopKRouter,
            ),
        )
        gated = bool(getattr(getattr(m, "config", None), "attention_output_gate", False))
        plain_linear = isinstance(m, nn.Linear) and not is_modelopt_linear(m)
        if name != "linear_qkv" or already_wrapped or plain_linear or not gated:
            return _original_transform(self, m, name, prefix)

        if (ans := self.match(m, name, prefix)) is None:
            return m
        match, full_name = ans

        # linear_qkv is never an expert linear, so the expert adapter variants do not apply.
        attrs = get_adapter_attributes_from_linear(m, is_expert=False)
        dim = get_effective_lora_dim(m, dim=self.dim, normalize_moe_lora=self.normalize_moe_lora, is_expert=False)
        adapter_kwargs = dict(
            dim=dim,
            base_linear_name=full_name,
            activation="identity",
            column_init_method=self.lora_A_init_method,
            row_init_method=self.lora_B_init_method,
            input_is_parallel=attrs.input_is_parallel,
            dropout=self.dropout,
            dropout_position=self.dropout_position,
            model_parallel_config=m.config,
            alpha=self.alpha,
            base_linear_is_parallel=attrs.base_linear_is_parallel,
            is_expert=False,
            disable_tensor_parallel_comm=attrs.disable_tensor_parallel_comm,
            disable_sequence_parallel_comm=attrs.disable_sequence_parallel_comm,
        )

        canonical_submodules = self.canonical_mapping[match]
        logger.info(f"Adding lora to: {full_name} ({canonical_submodules}, attention_output_gate)")
        q_out_features = 2 * m.config.kv_channels * m.config.num_attention_heads
        kv_out_features = m.config.kv_channels * m.config.num_query_groups
        adapter_q, adapter_k, adapter_v = None, None, None
        if "linear_q" in canonical_submodules:
            adapter_q = ParallelLinearAdapter(attrs.in_features, q_out_features, **adapter_kwargs)
        if "linear_k" in canonical_submodules:
            adapter_k = ParallelLinearAdapter(attrs.in_features, kv_out_features, **adapter_kwargs)
        if "linear_v" in canonical_submodules:
            adapter_v = ParallelLinearAdapter(attrs.in_features, kv_out_features, **adapter_kwargs)
        adapters = _canonical_lora.ModuleDict({"adapter_q": adapter_q, "adapter_k": adapter_k, "adapter_v": adapter_v})
        return LoRALinearSplitGatedQKV(m, adapters)

    _canonical_lora.CanonicalLoRA.transform = transform
    _canonical_lora.CanonicalLoRA._miles_gated_qkv_patch_installed = True
    logger.info("Patched CanonicalLoRA.transform for attention_output_gate fused QKV")
