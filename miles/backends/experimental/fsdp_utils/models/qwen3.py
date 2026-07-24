"""Qwen3 dense model math required by the formal true-on-policy contract."""

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm


class Qwen3FinalRMSNorm(Qwen3RMSNorm):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight.to(torch.bfloat16) * hidden_states.to(torch.bfloat16)


_FP32_SYNC_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".self_attn.q_norm.weight",
    ".self_attn.k_norm.weight",
)


def resolve_qwen3_dense_sync_dtype(name: str, checkpoint_dtype: torch.dtype) -> torch.dtype:
    if name == "model.embed_tokens.weight" or name.endswith(_FP32_SYNC_SUFFIXES):
        return torch.float32
    return checkpoint_dtype


def apply_qwen3_dense_true_on_policy_patch(model) -> bool:
    base_model = getattr(model, "model", model)
    final_norm = getattr(base_model, "norm", None)
    if isinstance(final_norm, Qwen3FinalRMSNorm):
        return False
    if not isinstance(final_norm, Qwen3RMSNorm):
        raise TypeError("Expected a Qwen3 model with a final Qwen3RMSNorm")
    final_norm.__class__ = Qwen3FinalRMSNorm
    return True
