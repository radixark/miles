"""Step5's backend-neutral layer arithmetic: per-attention-type RoPE geometry, the SwiGLU clamp and
the zero-centred RMSNorm, shared by the Megatron and torchtitan models."""

from dataclasses import dataclass

import torch

SWIGLU_LIMIT = 7.0


@dataclass(frozen=True)
class RopeGeometry:
    theta: float
    rotary_percent: float


SLIDING_WINDOW_ROPE = RopeGeometry(theta=1e4, rotary_percent=1.0)
FULL_ATTENTION_ROPE = RopeGeometry(theta=1e7, rotary_percent=1 / 3)


def zero_centered_rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``x / rms(x) * (1 + w)`` in fp32; the caller rounds the result."""
    hidden_states = hidden_states.float()
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    return hidden_states * torch.rsqrt(variance + eps) * (weight.float() + 1.0)
