"""Qwen3 dense model math required by the formal true-on-policy contract."""

import torch
from transformers.models.qwen3 import modeling_qwen3


_HFQwen3Attention = modeling_qwen3.Qwen3Attention
_HFQwen3DecoderLayer = modeling_qwen3.Qwen3DecoderLayer
_HFQwen3RMSNorm = modeling_qwen3.Qwen3RMSNorm
_HFQwen3RotaryEmbedding = modeling_qwen3.Qwen3RotaryEmbedding


def _fp32_rms_norm(module, hidden_states):
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + module.variance_epsilon)
    return module.weight.to(torch.float32) * hidden_states


class Qwen3RMSNorm(_HFQwen3RMSNorm):
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if input_dtype == torch.float32:
            return self.weight * hidden_states.to(self.weight.dtype)
        return self.weight.to(torch.float32) * hidden_states.to(input_dtype)


class Qwen3FinalRMSNorm(_HFQwen3RMSNorm):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight.to(torch.bfloat16) * hidden_states.to(torch.bfloat16)


class Qwen3RotaryEmbedding(_HFQwen3RotaryEmbedding):
    def forward(self, x, position_ids):
        return super().forward(x.to(torch.float32), position_ids)


class Qwen3Attention(_HFQwen3Attention):
    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        **kwargs,
    ):
        hidden_states = hidden_states.to(self.q_proj.weight.dtype)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = modeling_qwen3.apply_rotary_pos_emb(query_states, key_states, cos, sin)
        query_states = query_states.to(value_states.dtype)
        key_states = key_states.to(value_states.dtype)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface = modeling_qwen3.ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation,
            modeling_qwen3.eager_attention_forward,
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights


class Qwen3DecoderLayer(_HFQwen3DecoderLayer):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        position_embeddings=None,
        **kwargs,
    ):
        residual = hidden_states.to(torch.float32)
        hidden_states = _fp32_rms_norm(self.input_layernorm, residual)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        residual = residual + hidden_states.to(torch.float32)

        hidden_states = _fp32_rms_norm(self.post_attention_layernorm, residual)
        hidden_states = hidden_states.to(self.mlp.gate_proj.weight.dtype)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states.to(torch.float32)


def apply_qwen3_dense_true_on_policy_patch(model) -> bool:
    base_model = getattr(model, "model", model)
    final_norm = getattr(base_model, "norm", None)
    if not isinstance(final_norm, _HFQwen3RMSNorm):
        raise TypeError("Expected a Qwen3 model with a final Qwen3RMSNorm")

    was_patched = (
        isinstance(base_model.rotary_emb, Qwen3RotaryEmbedding)
        and isinstance(final_norm, Qwen3FinalRMSNorm)
        and all(
            isinstance(layer, Qwen3DecoderLayer)
            and isinstance(layer.self_attn, Qwen3Attention)
            and isinstance(layer.input_layernorm, Qwen3RMSNorm)
            and isinstance(layer.post_attention_layernorm, Qwen3RMSNorm)
            and isinstance(layer.self_attn.q_norm, Qwen3RMSNorm)
            and isinstance(layer.self_attn.k_norm, Qwen3RMSNorm)
            for layer in base_model.layers
        )
    )

    base_model.embed_tokens.to(torch.float32)
    base_model.rotary_emb.__class__ = Qwen3RotaryEmbedding
    for layer in base_model.layers:
        layer.input_layernorm.to(torch.float32)
        layer.post_attention_layernorm.to(torch.float32)
        layer.self_attn.q_norm.to(torch.float32)
        layer.self_attn.k_norm.to(torch.float32)
        layer.input_layernorm.__class__ = Qwen3RMSNorm
        layer.post_attention_layernorm.__class__ = Qwen3RMSNorm
        layer.self_attn.q_norm.__class__ = Qwen3RMSNorm
        layer.self_attn.k_norm.__class__ = Qwen3RMSNorm
        layer.self_attn.__class__ = Qwen3Attention
        layer.__class__ = Qwen3DecoderLayer
    final_norm.__class__ = Qwen3FinalRMSNorm

    if getattr(model.config, "tie_word_embeddings", False):
        model._fsdp_sync_dtype_overrides = {"lm_head.weight": torch.bfloat16}

    return not was_patched
