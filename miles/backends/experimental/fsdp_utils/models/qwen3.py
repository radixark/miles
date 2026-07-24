"""Qwen3 dense model math required by the formal true-on-policy contract."""

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm


class Qwen3FinalRMSNorm(Qwen3RMSNorm):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight.to(torch.bfloat16) * hidden_states.to(torch.bfloat16)


def apply_qwen3_dense_true_on_policy_patch(model) -> bool:
    base_model = getattr(model, "model", model)
    final_norm = getattr(base_model, "norm", None)
    if isinstance(final_norm, Qwen3FinalRMSNorm):
        return False
    if not isinstance(final_norm, Qwen3RMSNorm):
        raise TypeError("Expected a Qwen3 model with a final Qwen3RMSNorm")
    final_norm.__class__ = Qwen3FinalRMSNorm
    return True
