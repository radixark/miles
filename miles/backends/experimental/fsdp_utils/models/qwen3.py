"""Align HF Qwen3 with SGLang's dense true-on-policy precision contract.

SGLang keeps decoder residuals, decoder norms, and normalized Q/K in fp32 while dense matmuls and
the final norm/head stay bf16. Stock HF keeps the residual path in bf16, so its log-probs diverge.
These class-compatible replacements preserve the checkpoint/state-dict layout while matching that
cast order.
"""

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
    _fsdp_preserve_forward_input_dtype = True

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


def apply_true_on_policy_patch_for_qwen3() -> None:
    modeling_qwen3.Qwen3RMSNorm = Qwen3RMSNorm
    modeling_qwen3.Qwen3RotaryEmbedding = Qwen3RotaryEmbedding
    modeling_qwen3.Qwen3Attention = Qwen3Attention
    modeling_qwen3.Qwen3DecoderLayer = Qwen3DecoderLayer
